"""Transfer evidence, diversity, bounded actions and fresh-repeat invariants."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sircus.mjlab.benchmark_verify import digest, sha
from sircus.mjlab.sweep_worker import atomic
from sircus.mjlab.nearby_transfer_plan import (
    CASE_IDS, DELTAS, SEARCH_RULE, audited_source, captured, proposal_menu,
    preparation_selection, candidate_selection, reserved_sources,
)
from sircus.mjlab.nearby_transfer_study import repeat_schedule, schedule, verify_preflight


def request(identity='old'):
    return dict(id=identity,values=[.60,-.04,.40,28.,.25],bucket_xy_m=[1.,.025],
        board_heading_offset_deg=3.,board_twist_deg=180.,selected_q10=[.1]*6,selected_q14=[.2]*6,
        intended_route='table',prepared_connection_file='proof.json',prepared_connection_sha256='a'*64)


def bank(count=12):
    result=[]
    for i in range(count):
        r=request('seed-'+str(i));r['bucket_xy_m'][0]+=.01*i;r['values'][1]+=.005*i
        result.append(dict(id='source:'+str(i),source_request=r,bucket_xy_m=r['bucket_xy_m'],
            strength='confirmed_3_of_3' if i>=8 else 'single_capture'))
    return result


def case(identity=CASE_IDS[0]):
    r=request();r['bucket_xy_m']=[1.03,-.02]
    return dict(id=identity,original=r,directory='native-'+identity[-2:])


LOW=[.3,-.4,.2,10.,.08]
HIGH=[1.1,.4,.6,50.,.55]


class NearbyTransferTests(unittest.TestCase):
    def test_deterministic_transforms_reaim_and_keep_source_evidence_unchanged(self):
        c=case();seeds=bank(1);before=copy.deepcopy((c,seeds))
        proposals=proposal_menu(c,seeds,LOW,HIGH)
        self.assertEqual(proposals,proposal_menu(c,seeds,LOW,HIGH))
        self.assertEqual((c,seeds),before)
        self.assertEqual(len(proposals),3*len(DELTAS))
        originals={p['transfer_mode']:p for p in proposals if p['delta_index']==0}
        source=seeds[0]['source_request'];delta=[c['original']['bucket_xy_m'][i]-source['bucket_xy_m'][i] for i in (0,1)]
        for mode,fraction in SEARCH_RULE['transforms'].items():
            p=originals[mode]
            self.assertEqual(p['values'][:2],[source['values'][i]+fraction*delta[i] for i in (0,1)])
            self.assertEqual(p['values'][2:],source['values'][2:])
            self.assertEqual(p['bucket_xy_m'],c['original']['bucket_xy_m'])
            self.assertEqual(p['seed_q10'],source['selected_q10'])
            self.assertEqual(p['seed_q14'],source['selected_q14'])
            self.assertEqual(p['board_heading_offset_deg'],source['board_heading_offset_deg'])
            self.assertNotIn('selected_q10',p);self.assertNotIn('prepared_connection_file',p)

    def test_proposals_preserve_bounds_and_positive_board_bucket_distance(self):
        c=case();seeds=bank();before=copy.deepcopy(seeds)
        for row in proposal_menu(c,seeds,LOW,HIGH):
            self.assertTrue(all(lo<=v<=hi for lo,v,hi in zip(LOW,row['values'],HIGH)))
            self.assertLessEqual(abs(row['parameter_delta']['height_m']),.04)
            self.assertLessEqual(abs(row['parameter_delta']['tilt_deg']),6)
            self.assertLessEqual(abs(row['parameter_delta']['gap_m']),.08)
            self.assertLessEqual(abs(row['parameter_delta']['heading_deg']),8)
        self.assertEqual(seeds,before)

    def test_preparation_reserves_confirmed_sources_even_when_nominal_score_is_poor(self):
        c=case();seeds=bank();proposals=proposal_menu(c,seeds,LOW,HIGH)
        nominal={p['id']:{'score_m':-10. if p['source_strength']=='confirmed_3_of_3' else .1} for p in proposals}
        wrist={p['id']:{'passed':True} for p in proposals}
        chosen=preparation_selection(c,seeds,proposals,nominal,wrist)
        self.assertEqual(len(chosen),96)
        first=chosen[:24]
        self.assertTrue(all(p['delta_index']==0 for p in first))
        self.assertEqual({p['source_bank_id'] for p in first},{s['id'] for s in reserved_sources(c,seeds)})
        self.assertEqual(sum(s['strength']=='confirmed_3_of_3' for s in reserved_sources(c,seeds)),3)
        self.assertEqual({p['transfer_mode'] for p in first},{'retain','blend','translate'})
        self.assertEqual(len(chosen),len({p['id'] for p in chosen}))
        self.assertEqual(chosen,preparation_selection(c,seeds,proposals,nominal,wrist))

    def test_native_selection_reserves_teachers_and_deduplicates_geometry(self):
        c=case();proposals=proposal_menu(c,bank(),LOW,HIGH)
        predictions={p['id']:{'score_m':-.1 if p['source_strength']=='confirmed_3_of_3' else .1} for p in proposals}
        selected=candidate_selection(proposals,predictions)
        self.assertEqual(len(selected),20)
        self.assertGreaterEqual(len({p['source_bank_id'] for p in selected if p['delta_index']==0 and p['source_strength']=='confirmed_3_of_3'}),3)
        self.assertEqual({p['transfer_mode'] for p in selected if p['delta_index']==0},{'retain','blend','translate'})
        duplicate=copy.deepcopy(selected[0]);duplicate['id']='duplicate'
        predictions['duplicate']={'score_m':.5}
        self.assertEqual(len(candidate_selection([selected[0],duplicate],predictions)),1)

    def test_originals_are_preserved_and_maximum_native_budget_is_433(self):
        cases=[]
        for identity in CASE_IDS:
            c=case(identity);c['candidates']=[dict(request(identity+'-new-'+str(i)),context_id=identity) for i in range(20)]
            cases.append(c)
        plan={'cases':cases};before=copy.deepcopy(plan)
        slots=schedule(plan)
        self.assertEqual(len(slots),115)
        rows=[dict(candidate_id=r['request']['candidate_id'],variant='search',native_release=True,
                   invalid_physics=False,route_success=True,shot_route='direct') for r in slots if r['variant']=='search']
        repeats=repeat_schedule(plan,rows)
        self.assertEqual(len(repeats),300);self.assertEqual(len(slots)+len(repeats)+18,433)
        self.assertEqual(plan,before)
        self.assertEqual(len({s['request']['id'] for s in slots+repeats}),415)
        for c in cases:
            original=[s for s in slots if s['case_id']==c['id'] and s['variant']=='original']
            self.assertEqual(len(original),3)
            for s in original:
                for key in ('values','selected_q10','selected_q14','prepared_connection_sha256'):
                    self.assertEqual(s['request'][key],c['original'][key])

    def test_actual_route_controls_confirmation_not_intended_route(self):
        row=dict(native_release=True,invalid_physics=False,route_success=True,shot_route='direct',intended_route='table')
        self.assertTrue(captured(row))
        row.update(shot_route='table',intended_route='direct');self.assertTrue(captured(row))
        for change in ({'invalid_physics':True},{'native_release':False},{'route_success':False},{'shot_route':'arm'}):
            self.assertFalse(captured(dict(row,**change)))

    def test_preflight_must_bind_contract_preparation_and_slots(self):
        from argparse import Namespace
        with TemporaryDirectory() as temp:
            root=Path(temp)
            for file in ('contract.json','preparation.json','slots.json'):atomic(root/file,{'file':file})
            audit=dict(status='passed',version='sircus-nearby-transfer-preparation-audit-v1',
                native_execution_performed=False,checks_count=10,contract_sha256=sha(root/'contract.json'),
                preparation_sha256=sha(root/'preparation.json'),slots_sha256=sha(root/'slots.json'))
            atomic(root/'audit.json',audit);a=Namespace(output=root,preflight_audit=root/'audit.json')
            self.assertEqual(verify_preflight(a),audit)
            atomic(root/'slots.json',{'tampered':True})
            with self.assertRaisesRegex(ValueError,'preflight'):verify_preflight(a)

    def test_source_binding_supports_old_contract_but_detects_mutation(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);contract=dict(version='old-v1',physics_substeps=64)
            contract['fingerprint']=digest(contract);atomic(root/'contract.json',contract)
            atomic(root/'rows.jsonl',[])
            manifest={f:sha(root/f) for f in ('contract.json','rows.jsonl')};atomic(root/'evidence-manifest.json',manifest)
            report=dict(status='completed',phase='completed',evidence_manifest_sha256=sha(root/'evidence-manifest.json'))
            atomic(root/'report.json',report)
            audit=dict(status='passed',checks_count=1,report_sha256=sha(root/'report.json'),evidence_manifest_sha256=sha(root/'evidence-manifest.json'))
            atomic(root/'audit.json',audit)
            self.assertEqual(audited_source(root,root/'audit.json','old-v1',('rows.jsonl',))[1],contract)
            atomic(root/'rows.jsonl',[{'tampered':True}])
            with self.assertRaisesRegex(ValueError,'independent audit'):audited_source(root,root/'audit.json','old-v1',('rows.jsonl',))


if __name__=='__main__':unittest.main()
