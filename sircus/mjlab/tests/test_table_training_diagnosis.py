import tempfile
import json
from pathlib import Path
import unittest
import numpy as np
from sircus.mjlab import table_training_diagnosis as d

class DiagnosticTests(unittest.TestCase):
    def test_events_do_not_include_contacts_after_scoring_terminal(self):
        force=np.zeros((6,6));force[0,0]=1;force[1,2]=2;force[3,4]=3;force[5,3]=4
        events=d.event_summary(force,np.arange(7)*.1,3)
        self.assertEqual([x['category'] for x in events],['board','table'])
        self.assertEqual(events[1]['time_s'],.1)

    def test_case_rule_preserves_empty_stratum_and_ignores_successes(self):
        def case(id,cat,tilt=18,distance=.5,capture=False):return dict(context_id=id,native_release=True,capture=capture,values=[0,0,.3,tilt,.25],board_bucket_distance_m=distance,first_post_board_contact=dict(category=cat))
        cases=[case('c','table'),case('b','table'),case('a','table'),case('d','robot'),case('e','robot'),case('f','bucket',tilt=30),case('g','bucket',distance=.3),case('h','bucket',capture=True)]
        self.assertEqual(d.choose_correction_cases(cases),['a','b','d','e'])
        self.assertEqual(d.choose_correction_cases(list(reversed(cases))),['a','b','d','e'])

    def test_table_contact_is_not_a_success_label(self):
        base=dict(native_release=True,capture=False,table_contact=True,first_post_board_contact=dict(category='table'))
        self.assertEqual(d.classify(base),'table_first_miss')
        self.assertEqual(d.classify(dict(base,capture=True)),'capture_with_table_contact')
        self.assertEqual(d.classify(dict(base,native_release=False,reason='finite_ik_unresolved')),'setup_finite_ik_unresolved')

    def test_unbound_source_cannot_publish_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();(study/'report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Exact source report_sha256'):
                d.build(study,root/'audit.json',root/'output')
            self.assertFalse((root/'output').exists())

    def test_setup_proof_authenticates_pair_and_does_not_claim_penetration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);native=root/'native';native.mkdir()
            geometry=dict(passed=False,reason='unproven_clearance',minimum_clearance_m=0,closest_pair=['base','bucket'],inconsistent_distance_queries=1,unresolved_clearance_pairs=[1],fixed_base_blockers=[1])
            resolution=dict(bucket_xy_m=[1,0],layout_eligible=False,anchor_admitted=False,home_geometry=geometry)
            (native/'resolution.json').write_text(json.dumps(resolution));rh=d.sha(native/'resolution.json')
            connection=dict(bucket_xy_m=[1,0],values=[.5,0,.3,18,.2],resolution_file='resolution.json',resolution_sha256=rh,passed=False,action_substituted=False,reason='fixed_base_clearance_rejected')
            (native/'connection.json').write_text(json.dumps(connection));ch=d.sha(native/'connection.json')
            row=dict(id='row',native_release=False,bucket_xy_m=[1,0],values=connection['values'],connection_file='connection.json',connection_sha256=ch,context_resolution_file='resolution.json',context_resolution_sha256=rh)
            def check(ok,label):
                if not ok:raise ValueError(label)
            evidence={'native/connection.json':ch,'native/resolution.json':rh}
            proof=d.setup_proof(root,row,evidence,check)
            self.assertEqual(proof['home_geometry']['minimum_clearance_m'],0)
            self.assertIn('not evidence of penetration',proof['interpretation'])
            with self.assertRaisesRegex(ValueError,'exact source proof'):
                d.setup_proof(root,row,dict(evidence,**{'native/resolution.json':'0'*64}),check)
            altered=dict(row,values=[.5,0,.4,18,.2])
            with self.assertRaisesRegex(ValueError,'same exact action'):
                d.setup_proof(root,altered,evidence,check)

if __name__=='__main__':unittest.main()
