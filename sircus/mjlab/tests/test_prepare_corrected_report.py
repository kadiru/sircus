import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sircus.mjlab.prepare_corrected_report import prepare,watch


class CorrectedDisplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.base=self.root/'base.json';self.study=self.root/'study';self.study.mkdir();self.out=self.root/'view.json'
        self.original=dict(status='completed',frozen_evaluation={'captures':1},contact_repeatability_evaluation={'verification':{'status':'passed'}})
        self.base.write_text(json.dumps(self.original))
    def tearDown(self):self.tmp.cleanup()
    def test_pending_keeps_all_prior_evidence_and_does_not_create_source(self):
        value=prepare(self.base,self.study,self.out)
        self.assertIsNone(value['corrected_grid_evaluation'])
        for k,v in self.original.items():self.assertEqual(value[k],v)
        self.assertFalse((self.study/'report.json').exists())
    def test_counts_connections_and_study_matched_audit(self):
        report=dict(version='new',status='waiting',phase='waiting_for_4090',contract_fingerprint='exact',counts={'grid_positions':35,'native_attempts':0},
            transit_attempts=[{'cell_id':'a','passed':True}],
            cells=[dict(id='a',bucket_xy_m=[.1,-.6],geometry={'eligible':2},search={'attempts':1,'captures':1},confirmations=[{'source_id':'x','qualified':False}],private='omit')])
        source=self.study/'report.json';source.write_text(json.dumps(report));before=source.read_bytes()
        value=prepare(self.base,self.study,self.out)['corrected_grid_evaluation']
        self.assertEqual(value['counts']['native_attempts'],0);self.assertTrue(value['cells'][0]['motion']['connected']);self.assertNotIn('private',value['cells'][0])
        audit=self.study/'audit-corrected.json';audit.write_text(json.dumps(dict(status='passed',contract_fingerprint='wrong',checks=[{'passed':True}],failures=[])))
        self.assertEqual(prepare(self.base,self.study,self.out)['corrected_grid_evaluation']['verification']['status'],'failed')
        audit.write_text(json.dumps(dict(status='passed',contract_fingerprint='exact',checks=[{'passed':True}],failures=[])))
        self.assertEqual(prepare(self.base,self.study,self.out)['corrected_grid_evaluation']['verification']['status'],'pending')
        supplemental=self.study/'audit-corrected-supplement.json'
        supplemental.write_text(audit.read_text())
        complete=prepare(self.base,self.study,self.out)['corrected_grid_evaluation']
        self.assertEqual(complete['verification']['status'],'passed')
        self.assertTrue(all(a['identity_matched'] for a in complete['audits'].values()))
        (self.study/'audit-corrected-supplement.json').write_text(json.dumps(dict(status='failed',contract_fingerprint='exact',checks=[{'passed':False}],failures=['coverage'])))
        self.assertEqual(prepare(self.base,self.study,self.out)['corrected_grid_evaluation']['verification']['status'],'failed')
        self.assertEqual(source.read_bytes(),before)
    def test_protected_paths_and_finite_watcher(self):
        for output in [self.base,self.study/'nested/out.json']:
            with self.assertRaises(ValueError):prepare(self.base,self.study,output)
        for seconds in [-1,30001,float('nan')]:
            with self.assertRaises(ValueError):watch(self.base,self.study,self.out,seconds)
        pending={'corrected_grid_evaluation':{'status':'completed','verification':{'status':'pending'}}}
        done={'corrected_grid_evaluation':{'status':'completed','verification':{'status':'passed'}}}
        with mock.patch('sircus.mjlab.prepare_corrected_report.prepare',side_effect=[pending,done]),mock.patch('sircus.mjlab.prepare_corrected_report.time.monotonic',side_effect=[0,0]),mock.patch('sircus.mjlab.prepare_corrected_report.time.sleep') as sleep:
            self.assertEqual(watch(self.base,self.study,self.out,10),done);sleep.assert_called_once_with(5)
    def test_live_search_map_uses_only_rows_in_the_atomic_report(self):
        report=dict(status='running',phase='confirmation',records=3,cells=[dict(id='a',status='ready_search'),dict(id='b',status='ready_search')])
        rows=[dict(cell_id='a',phase='search',geometry_eligible=False,status='geometry_rejected'),
              dict(cell_id='a',phase='search',geometry_eligible=True,status='native_observed',native_release=True,route_success=True),
              dict(cell_id='b',phase='search',geometry_eligible=True,status='native_observed',native_release=True,route_success=False),
              dict(cell_id='b',phase='search',geometry_eligible=True,status='native_observed',native_release=True,route_success=True)]
        (self.study/'report.json').write_text(json.dumps(report))
        (self.study/'shots.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows)+'{"unfinished":')
        value=prepare(self.base,self.study,self.out)['corrected_grid_evaluation']
        self.assertEqual(value['cells'][0]['search'],dict(attempts=1,captures=1,invalid=0))
        self.assertEqual(value['cells'][1]['search'],dict(attempts=1,captures=0,invalid=0))
        self.assertEqual([c['status'] for c in value['cells']],['native_capture','no_capture_in_budget'])
        self.assertEqual(value['stage_statistics']['search'],dict(requests=2,releases=2,captures=1))
    def test_phase_counts_keep_rejected_confirmations_in_the_denominator(self):
        report=dict(status='completed',records=3,qualified_sources=[{},{}],
            cells=[dict(id='a',confirmations=[dict(qualified=True),dict(qualified=True)])])
        rows=[dict(phase='search',geometry_eligible=True,status='not_selected_budget'),
              dict(phase='confirmation',geometry_eligible=False,native_release=False,invalid='clearance'),
              dict(phase='confirmation',geometry_eligible=True,native_release=True,route_success=True)]
        (self.study/'report.json').write_text(json.dumps(report))
        (self.study/'shots.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        value=prepare(self.base,self.study,self.out)['corrected_grid_evaluation']
        self.assertEqual(value['stage_statistics']['confirmation'],dict(requests=2,releases=1,captures=1))
        self.assertEqual(value['stage_statistics']['search']['requests'],0)
        self.assertEqual(value['counts']['qualified_settings'],2)
        self.assertEqual(value['counts']['positions_with_multiple_qualified'],1)

if __name__=='__main__':unittest.main()
