import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sircus.mjlab.prepare_neighborhood_report import KEY, compact, prepare, watch


class NeighborhoodDisplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.study=self.root/'study';self.study.mkdir()
        self.base=self.root/'base.json';self.output=self.root/'display.json'
        self.old=dict(active_policy='unchanged',window_study_evaluation={'test':{'sac':{'captures':0}}},corrected_grid_evaluation={'qualified':17})
        self.base.write_text(json.dumps(self.old))
    def tearDown(self):self.tmp.cleanup()

    def test_missing_partial_report_preserves_old_studies_and_creates_no_evidence(self):
        value=prepare(self.base,self.study,self.output)
        self.assertIsNone(value[KEY]);self.assertEqual({k:value[k] for k in self.old},self.old)
        self.assertEqual(list(self.study.iterdir()),[])
        (self.study/'report.json').write_text(json.dumps(dict(status='running',phase='geometry',contexts=[])))
        self.assertEqual(prepare(self.base,self.study,self.output)[KEY]['phase'],'geometry')

    def test_whitelist_preserves_context_geometry_and_native_denominators(self):
        raw=dict(status='running',contract_fingerprint='one',private='omit',settings={'bucket_xy_m':[.1,-.6],'board_xy_m':[.5,-.3],'private':'omit'},
            counts={'requests':20,'native_releases':12,'private':'omit'},contexts=[dict(id='a',source_family='anchor-a',board_height_m=.4,
                tilt_deg=32,board_admitted=True,eligible=10,search_attempts=17,search_native_releases=10,search_captures=7,q14=[1]*6)],
            selected_recipes=[dict(id='recipe',context_id='a',board_height_m=.4,tilt_deg=32,gap_m=.3,q10=[1]*6,
                by_substeps={'32':{'attempts':23,'captures':10,'native_releases':20},'64':{'attempts':23,'captures':19,'native_releases':21},'128':{'attempts':23,'captures':18}},
                sensitivity={'capture_agreement_64_128':{'agreements':22,'attempts':23},'nominal_position_delta_350ms_max_m':.004,'paired_engineering_gate':False,'private':'omit'})])
        p=self.study/'report.json';p.write_text(json.dumps(raw));before=p.read_bytes()
        value=prepare(self.base,self.study,self.output)[KEY]
        self.assertEqual(value['counts'],{'requests':20,'native_releases':12})
        c=value['contexts'][0];self.assertEqual(c['search_attempts'],17);self.assertEqual(c['search_native_releases'],10);self.assertNotIn('q14',c)
        recipe=value['selected_recipes'][0]
        self.assertEqual(recipe['by_substeps']['64']['native_releases'],21)
        self.assertNotIn('native_releases',recipe['by_substeps']['128'])
        self.assertFalse(recipe['sensitivity']['paired_engineering_gate']);self.assertEqual(recipe['sensitivity']['capture_agreement_64_128']['agreements'],22)
        for target in (value,value['settings'],recipe,recipe['sensitivity']):self.assertNotIn('private',target)
        self.assertNotIn('q10',recipe);self.assertEqual(p.read_bytes(),before)

    def test_native_trace_values_are_copied_without_reconstruction_and_invalid_ones_missing(self):
        points=[[.5,-.3,.8],[.5,-.3,.4],[.2,-.6,.2]]
        raw=dict(trace_examples=[dict(record_id='native',context_id='a',substeps=128,position_m=points),
            dict(record_id='bad',position_m=[[0,0,float('nan')],[0,0,0]])])
        value=compact(raw)
        self.assertEqual(len(value['trace_examples']),1);self.assertEqual(value['trace_examples'][0]['position_m'],points)
        self.assertEqual(value['trace_examples'][0]['substeps'],128)

    def test_audit_identity_is_bound_to_exact_report(self):
        (self.study/'report.json').write_text(json.dumps(dict(status='completed',contract_fingerprint='a')))
        path=self.study/'audit-neighborhood.json';path.write_text(json.dumps(dict(status='passed',contract_fingerprint='b',checks_count=30)))
        self.assertEqual(prepare(self.base,self.study,self.output)[KEY]['audit']['status'],'failed')
        path.write_text(json.dumps(dict(status='passed',contract_fingerprint='a',checks_count=30)))
        self.assertEqual(prepare(self.base,self.study,self.output)[KEY]['audit']['status'],'passed')

    def test_write_guards_and_nonfinite_values_keep_existing_display(self):
        for out in (self.base,self.study/'inside.json'):
            with self.assertRaises(ValueError):prepare(self.base,self.study,out)
        prepare(self.base,self.study,self.output);before=self.output.read_bytes()
        (self.study/'report.json').write_text(json.dumps(dict(counts={'requests':float('nan')})))
        with self.assertRaises(ValueError):prepare(self.base,self.study,self.output)
        self.assertEqual(self.output.read_bytes(),before)

    def test_watch_is_finite_and_never_launches_simulation(self):
        for seconds in (-1,float('nan'),30001):
            with self.assertRaises(ValueError):watch(self.base,self.study,self.output,seconds)
        with mock.patch('sircus.mjlab.prepare_neighborhood_report.prepare',side_effect=[{KEY:{'status':'running'}},{KEY:{'status':'completed'}}]),\
            mock.patch('sircus.mjlab.prepare_neighborhood_report.time.monotonic',side_effect=[0,0]),\
            mock.patch('sircus.mjlab.prepare_neighborhood_report.time.sleep') as sleep:
            value=watch(self.base,self.study,self.output,10)
            self.assertEqual(value[KEY]['status'],'completed');sleep.assert_called_once_with(5)


if __name__=='__main__':unittest.main()
