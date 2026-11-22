"""Bounded heading experiment, selection, repeat and source identity checks."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import remaining_heading_study as study
from sircus.mjlab.benchmark_verify import sha


def original():
    return dict(id=study.SOURCE_CANDIDATE,context_id=study.CASE,bucket_xy_m=[1.226,.1834],
        values=[.5739,-.0084,.4473,20.34,.24983],board_heading_offset_deg=7.025,board_twist_deg=0.,
        selected_q10=[0.]*6,selected_q14=[.1]*6)


class RemainingHeadingTests(unittest.TestCase):
    def test_primary_changes_heading_only_and_preserves_source(self):
        source=original();before=copy.deepcopy(source);menu=study.primary_menu(source)
        self.assertEqual(source,before)
        self.assertEqual(len(menu),13)
        self.assertEqual([r['board_heading_offset_deg'] for r in menu],list(study.HEADINGS))
        self.assertNotIn(source['board_heading_offset_deg'],study.HEADINGS)
        for row in menu:
            self.assertEqual(row['values'],source['values'])
            self.assertEqual(row['bucket_xy_m'],source['bucket_xy_m'])
            self.assertEqual(row['board_twist_deg'],source['board_twist_deg'])
            self.assertEqual(row['seed_q10'],source['selected_q10'])
            self.assertEqual(row['seed_q14'],source['selected_q14'])

    def test_shifted_centers_require_complete_failed_setup_evidence(self):
        source=original();menu=study.primary_menu(source)
        records=[dict(proposal_id=p['id'],admitted=int(i!=0),reason=None if i else 'finite_ik_unresolved') for i,p in enumerate(menu)]
        shifted=study.translated_menu(source,records)
        self.assertEqual(len(shifted),8)
        self.assertEqual({r['board_heading_offset_deg'] for r in shifted},{0.})
        self.assertEqual({tuple(r['xy_offset_m']) for r in shifted},set(study.OFFSETS))
        for row in shifted:self.assertEqual(row['values'][2:],source['values'][2:])
        self.assertEqual(study.translated_menu(source,records[:-1]),[])
        records[0]['reason']='case_preparation_time_limit'
        self.assertEqual(study.translated_menu(source,records),[])

    def test_complete_worst_case_is_below_setup_cap(self):
        source=original();primary=study.primary_menu(source)
        rows=[dict(proposal_id=p['id'],admitted=0,reason='finite_ik_unresolved') for p in primary]
        self.assertEqual(len(primary)+len(study.translated_menu(source,rows))+study.RULE['retry_limit'],141)
        self.assertLessEqual(141,study.RULE['maximum_setup_checks'])

    def test_selection_keeps_heading_coverage_and_one_joint_branch_per_geometry(self):
        source=original();primary=study.primary_menu(source)
        shifted=[study.proposal(source,0.,(i*.001,.02),100+i,'center_clearance') for i in range(50)]
        admitted=[dict(p,id=p['id']+'-b0') for p in primary+shifted]
        duplicate=dict(admitted[0],id='another-joint-branch')
        admitted.append(duplicate)
        predictions={r['id']:dict(score_m=0.1 if r['group']=='center_clearance' else -.1) for r in admitted}
        predictions[duplicate['id']]=dict(score_m=.02)
        result=study.select_candidates(admitted,predictions)
        self.assertEqual(len(result),40)
        self.assertEqual([r['board_heading_offset_deg'] for r in result[:13]],list(study.HEADINGS))
        self.assertEqual(result[0]['id'],duplicate['id'])
        self.assertEqual(len({study.native.geometry_key(r) for r in result}),len(result))

    def test_native_schedule_reuses_original_and_repeats_only_qualifying_capture(self):
        source=original();candidates=[dict(r,id=r['id']+'-b0') for r in study.primary_menu(source)[:2]]
        plan=dict(cases=[dict(id=study.CASE,original=source,candidates=candidates,directory='native-19')])
        slots=study.native.schedule(plan)
        self.assertEqual(len(slots),5)
        self.assertEqual([s['request']['board_heading_offset_deg'] for s in slots[:3]],[7.025]*3)
        rows=[dict(variant='search',candidate_id=candidates[0]['id'],native_release=True,invalid_physics=False,route_success=True),
              dict(variant='search',candidate_id=candidates[1]['id'],native_release=True,invalid_physics=False,route_success=False)]
        repeats=study.native.repeat_schedule(plan,rows)
        self.assertEqual(len(repeats),3)
        self.assertEqual([r['repeat'] for r in repeats],[1,2,3])
        self.assertTrue(all(s['request']['values']==candidates[0]['values'] for s in repeats))

    def test_cpu_prepare_rejects_visible_gpu_before_initialization(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaisesRegex(ValueError,'Hide CUDA'):study.prepare(argparse.Namespace())

    def test_bound_source_authenticates_arm_free_observation_and_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);(root/'native-19').mkdir();(root/'native-19'/'trace.npz').write_bytes(b'fixture trace')
            source=original();row=dict(id=source['id']+'-search-0',candidate_id=source['id'],values=source['values'],
                bucket_xy_m=source['bucket_xy_m'],q10=source['selected_q10'],q14=source['selected_q14'],
                native_release=True,invalid_physics=False,route_success=False,experimental_route=dict(robot_contact_steps=0),
                native_cache_file='trace.npz',native_cache_sha256=sha(root/'native-19'/'trace.npz'))
            plan=dict(cases=[dict(id=study.CASE,directory='native-19',candidates=[source])])
            report=dict(status='completed',cases=[dict(id=study.CASE,results=dict(search=dict(rows=[row])))])
            for name,value in (('preparation.json',plan),('report.json',report),('evidence-manifest.json',{})):
                (root/name).write_text(json.dumps(value))
            receipt=dict(status='passed',report_sha256=sha(root/'report.json'),plan_sha256=sha(root/'preparation.json'),evidence_manifest_sha256=sha(root/'evidence-manifest.json'))
            (root/'audit.json').write_text(json.dumps(receipt));args=argparse.Namespace(recovery_study=root,recovery_audit=root/'audit.json')
            with patch.object(study,'SOURCE_REPORT',receipt['report_sha256']):
                found,binding,_=study.bound_source(args)
                self.assertEqual(found['original'],source);self.assertEqual(binding['trace_sha256'],row['native_cache_sha256'])
                (root/'native-19'/'trace.npz').write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError,'trace changed'):study.bound_source(args)


if __name__=='__main__':unittest.main()
