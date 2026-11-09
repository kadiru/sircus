import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab import imitation_regression_diagnosis as d


class RegressionDiagnosisTests(unittest.TestCase):
    def test_contact_events_use_interval_starts_and_exclude_post_terminal(self):
        force=np.zeros((8,6));force[1:3,2]=[2,3];force[4,2]=1;force[6,4]=9
        events=d.contact_events(force,np.arange(9)*.1,6)
        self.assertEqual([e['category'] for e in events],['table','table'])
        self.assertAlmostEqual(events[0]['time_s'],.1)
        self.assertAlmostEqual(events[0]['end_time_s'],.3)
        self.assertEqual(events[0]['peak_force_N'],3)

    def test_setup_classifies_missing_arm_and_keeps_collision_evidence(self):
        connection=dict(passed=False,reason='finite_ik_unresolved',action_substituted=False,
            endpoint_attempts=dict(valid_ik_branches=[0,9],pair_attempts=[]))
        row=dict(connection_sha256='c',context_resolution_sha256='r')
        self.assertEqual(d.setup_evidence(row,connection,{})['group'],'finite_ik_dropper')
        connection['endpoint_attempts']['valid_ik_branches']=[8,0]
        self.assertEqual(d.setup_evidence(row,connection,{})['group'],'finite_ik_board')
        connection.update(reason='strict_endpoint_clearance_or_pose_rejected',endpoint_attempts=dict(
            valid_ik_branches=[12,6],pair_attempts=[dict(branch10=0,branch14=0,checked=dict(
                reason=4,pairs=[['table','Arm14/link5']],independent_reject_arms=[14]))]))
        result=d.setup_evidence(row,connection,{})
        self.assertEqual(result['endpoint_examples'][0]['collision_pairs'],[['table','Arm14/link5']])
        self.assertIsNone(result['endpoint_examples'][0]['minimum_clearance_m'])

    def test_contact_is_not_causal_or_success_evidence(self):
        row=dict(native_release=True,capture=False,invalid_physics=False,whole_ball_entry=False,
            table_contact_steps=20,forbidden_contact_steps=10,max_retention_s=0)
        self.assertEqual(d.classify(row),'no_whole_ball_entry')
        row.update(whole_ball_entry=True,max_retention_s=.4)
        self.assertEqual(d.classify(row),'entry_without_one_second_retention')
        row.update(max_retention_s=1.)
        self.assertEqual(d.classify(row),'retained_with_forbidden_contact')
        row.update(capture=True,forbidden_contact_steps=0)
        self.assertEqual(d.classify(row),'capture_with_table_contact')

    def test_unbound_source_never_creates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);study=root/'study';study.mkdir()
            (study/'report.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Exact completed source/report_sha256'):
                d.build(study,root/'audit.json',root/'new')
            self.assertFalse((root/'new').exists())

    def test_fresh_external_output_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for output in (root,root/'derived'):
                with self.assertRaisesRegex(ValueError,'Fresh diagnosis directory'):
                    d.build(root,root/'audit.json',output)


if __name__=='__main__':
    unittest.main()
