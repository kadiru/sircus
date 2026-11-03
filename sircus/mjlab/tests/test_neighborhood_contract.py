import copy
import unittest
import numpy as np
from sircus.mjlab import neighborhood_contract as protocol


def row(name,gap,hit=True):
    return dict(id=name,context_id='c',gap_m=gap,role='development',phase='coarse',
        native_release=True,route_success=hit,invalid='',route_quality=1 if hit else 0,
        board_bounce=True,closest_opening_distance_m=.01,max_retention_s=1. if hit else 0.,
        external_contact_steps=0,single_bounce_success=hit,values=[.47,-.27,.44,32.,gap])


class NeighborhoodContractTests(unittest.TestCase):
    def test_balanced_local_grid_and_actual_anchor_gaps(self):
        c=protocol.contexts();p=protocol.coarse_plan()
        self.assertEqual(len(c),75);self.assertEqual(len(p),1275)
        self.assertEqual(protocol.MAX_REQUESTS,2580)
        for a in protocol.ANCHORS:
            exact=[r for r in c if r['board_height_m']==a['height'] and r['tilt_deg']==a['tilt']]
            self.assertEqual(len(exact),1)
            self.assertIn(a['gap'],[r['gap_m'] for r in p if r['context_id']==exact[0]['id']])

    def test_refinement_rejects_heldout_and_retains_disconnected_evidence(self):
        rows=[row(str(i),.1+.01*i,i in (4,5,11)) for i in range(17)]
        gaps=protocol.refine_gaps(rows);self.assertEqual(len(set(gaps)),6)
        self.assertTrue(set(gaps).isdisjoint(r['gap_m'] for r in rows))
        self.assertTrue(any(.13<g<.16 for g in gaps))
        bad=copy.deepcopy(rows);bad[0]['role']='test'
        with self.assertRaises(ValueError):protocol.refine_gaps(bad)
        self.assertEqual(protocol.refine_gaps([row('x',.2,False)]),[])

    def test_selection_never_uses_untested_midpoint_or_heldout(self):
        records=[row('lo',.17),row('hi',.18),row('miss',.19,False)]
        chosen=protocol._best_context_row(records)
        self.assertIn(chosen['source']['gap_m'],(.17,.18))
        with self.assertRaises(ValueError):protocol.select_recipes([dict(records[0],role='test')])

    def test_repeat_requests_are_paired_and_unclipped(self):
        source=row('s',.02);r=dict(id='recipe-00',source=source)
        plan=protocol.repeat_plan([r]);self.assertEqual(len(plan),23)
        self.assertEqual(sum(x['phase']=='nominal' for x in plan),3)
        self.assertTrue(any(p['gap_m']<.02 for p in plan))
        self.assertTrue(any(p['values'][2]!=source['values'][2] and p['values'][3]!=source['values'][3] for p in plan[3:]))
        self.assertEqual(plan,protocol.repeat_plan([r]))

    def test_sensitivity_gate_requires_both_capture_reliability_and_nominal_position(self):
        rows=[];positions={}
        for i in range(23):
            for step in protocol.SUBSTEPS:
                r=dict(id=f'{i}-{step}',repeat_id=str(i),substeps=step,phase='nominal' if i<3 else 'fresh',
                    route_success=True,native_release=True)
                rows.append(r);positions[r['id']]=np.zeros((960,3))
        self.assertTrue(protocol.sensitivity(rows,positions)['paired_engineering_gate'])
        positions['0-128'][83,0]=.006
        self.assertFalse(protocol.sensitivity(rows,positions)['paired_engineering_gate'])
        positions['0-128'][83,0]=0
        rows[2]['route_success']=False
        self.assertFalse(protocol.sensitivity(rows,positions)['paired_engineering_gate'])
