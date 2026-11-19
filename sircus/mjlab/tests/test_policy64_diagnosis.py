import csv
import json
from pathlib import Path
import unittest

from sircus.mjlab import policy64_diagnosis as D


def fixture():
    path=Path(__file__).resolve().parents[3]/'baselines/mjlab-policy64-20261105/test-positions.csv'
    result=[]
    for raw in csv.DictReader(path.read_text().splitlines()):
        if raw['method'] not in ('bc','ppo'):continue
        result.append(dict(id=raw['id'],context_id=raw['context_id'],role='test',method=raw['method'],
            bucket_xy_m=[float(raw[k]) for k in ('bucket_x_m','bucket_y_m')],
            values=[float(raw[k]) for k in D.PARAMETERS],
            native_release=raw['native_release']=='True',route_success=raw['capture']=='True',
            route_quality=2 if raw['clean']=='True' else 1 if raw['capture']=='True' else 0,
            invalid=raw['invalid'] or None,invalid_physics=raw['invalid_physics']=='True',learning_reward=float(raw['learning_reward']),
            source_teacher_id='same-test-anchor',effective_source_hint=dict(values=[0,0,0,0,0]),observation=[0.]*9,
            q10=['never-export-raw-joints'],q14=['never-export-raw-joints']))
    return result


class DiagnosisTests(unittest.TestCase):
    def test_published_paired_counts_keep_every_rejection(self):
        r=D.paired_analysis(fixture());p=r['paired_counts']
        self.assertEqual((p['layouts'],p['both_capture'],p['bc_only'],p['ppo_only'],p['neither']),(50,13,7,5,25))
        self.assertEqual((p['bc']['captures'],p['bc']['native'],p['bc']['rejected']),(20,43,7))
        self.assertEqual((p['ppo']['captures'],p['ppo']['native'],p['ppo']['rejected'],p['ppo']['native_misses']),(18,39,11,21))
        self.assertEqual((p['jointly_admitted']['layouts'],p['jointly_admitted']['bc_captures'],p['jointly_admitted']['ppo_captures']),(37,15,18))
        self.assertEqual(sum(t['count'] for t in p['transitions']),50)
        self.assertEqual(r['actor_shift'],dict(board_height_lower_count=50,release_gap_higher_count=44))

    def test_action_deltas_and_exact_lost_layouts(self):
        r=D.paired_analysis(fixture());lost=[p for p in r['pairs'] if p['bc']['capture'] and not p['ppo']['capture']]
        self.assertEqual([p['context_id'] for p in lost],D.LOSS_IDS)
        self.assertEqual(sum(p['ppo']['outcome']=='rejected' for p in lost),5)
        p=lost[0];self.assertAlmostEqual(p['delta_values'][2]*1000,-8.09823955,places=6)
        self.assertAlmostEqual(p['delta_release_height_m'],p['delta_values'][2]+p['delta_values'][4])
        self.assertNotIn('never-export-raw-joints',json.dumps(r))

    def test_complete_unique_matched_cohort_required(self):
        rows=fixture()
        with self.assertRaises(ValueError):D.paired_analysis(rows[:-1])
        with self.assertRaises(ValueError):D.paired_analysis(rows+[rows[0]])
        rows[0]['bucket_xy_m'][0]+=.01
        with self.assertRaises(ValueError):D.paired_analysis(rows)

    def test_physics_failure_is_not_a_capture(self):
        row=fixture()[0];row.update(native_release=True,route_success=True,invalid_physics=True,invalid=None)
        self.assertFalse(D.captured(row));self.assertEqual(D.outcome(row),'invalid')
        row.update(native_release=False,invalid_physics=False,invalid='finite_ik_unresolved')
        self.assertFalse(D.captured(row));self.assertEqual(D.outcome(row),'rejected')

    def test_clearance_evidence_does_not_claim_unreachability(self):
        proof=dict(reason='strict_endpoint_clearance_or_pose_rejected',passed=False,
            endpoint_attempts=dict(valid_ik_branches=[11,4],pair_attempts=[dict(checked=dict(strict_geometry=dict(minimum_clearance_m=.003832,closest_pair=['wrist','table'])))]))
        e=D.geometry_evidence(proof)
        self.assertEqual(e['checked_branch_pairs'],1);self.assertEqual(e['best_checked_clearance_m'],.003832)
        self.assertEqual(e['minimum_required_clearance_m'],.005);self.assertEqual(e['limiting_pairs'],[['wrist','table']])
        self.assertIn('not a proof of unreachability',D.geometry_evidence(dict(reason='finite_ik_unresolved'))['interpretation'])

    def test_model_options_are_proposal_only(self):
        r=D.model_options();self.assertEqual(r['status'],'proposed_not_trained');self.assertEqual(len(r['sources']),3)
        self.assertIn('future holdouts',r['evaluation']);self.assertIn('separate',r['feasibility'])


if __name__=='__main__':unittest.main()
