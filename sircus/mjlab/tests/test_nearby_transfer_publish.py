import copy
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab.benchmark_verify import sha
from sircus.mjlab.nearby_transfer_publish import CASES, VERSION, publish


class NearbyPublisherTests(unittest.TestCase):
    def fixture(self,path):
        study=path/'study';study.mkdir();paths={}
        for key in ('target','recovery','guarded','parent'):
            root=path/key;root.mkdir();receipt=path/(key+'-audit.json');receipt.write_text('{}')
            paths[key+'_study']=str(root);paths[key+'_audit']=str(receipt)
        contract=dict(source_paths=paths,maximum_total_requests=433)
        cases=[dict(id=i,bucket_xy_m=[1.,0.],results={k:dict(rows=['raw trace'],captures=0)
                                                    for k in ('original','search','repeat')}) for i in CASES]
        report=dict(version=VERSION,status='completed',phase='completed',scope='development',physics_substeps=64,
            native_releases=33,initial_requests=15,confirmation_requests=0,cases=cases,
            source_bank_summary=dict(settings=23,confirmed_settings=8,single_capture_settings=15,sources=[]),
            controls={k:dict(status='passed') for k in ('controls_pre','controls_post')})
        for name,data in (('contract.json',contract),('preparation.json',dict(cases=cases)),
                          ('report.json',report),('source-bank.json',[]),('evidence-manifest.json',{})):
            (study/name).write_text(json.dumps(data))
        audit=path/'audit.json'
        self.receipt(study,audit)
        return study,audit,paths

    def receipt(self,study,audit):
        audit.write_text(json.dumps(dict(status='passed',**{field:sha(study/name) for field,name in
            (('report_sha256','report.json'),('evidence_manifest_sha256','evidence-manifest.json'),
             ('plan_sha256','preparation.json'),('source_bank_sha256','source-bank.json'),('contract_sha256','contract.json'))})))

    def test_publish_preserves_old_results_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);study,audit,_=self.fixture(path)
            before={p.name:p.read_bytes() for p in study.iterdir()}
            display=path/'display.json';display.write_text('{"older":1,"media":{"a":"keep"}}')
            publish(study,audit,display);data=json.loads(display.read_text())
            self.assertEqual(data['older'],1);self.assertEqual(data['media'],dict(a='keep'))
            self.assertEqual(data['nearby_shot_transfer']['audit_status'],'passed')
            self.assertEqual(data['nearby_shot_transfer']['source_bank_summary']['confirmed_settings'],8)
            self.assertNotIn('rows',data['nearby_shot_transfer']['cases'][0]['results']['original'])
            self.assertEqual(before,{p.name:p.read_bytes() for p in study.iterdir()})

    def test_source_roots_receipts_and_symlink_targets_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);study,audit,paths=self.fixture(path)
            targets=[study/'new.json',audit]
            targets += [Path(value)/'new.json' if key.endswith('study') else Path(value) for key,value in paths.items()]
            for target in targets:
                with self.assertRaises(ValueError):publish(study,audit,target)
            display=path/'display.json';display.with_suffix('.json.tmp').symlink_to(Path(paths['recovery_study'])/'new.json')
            with self.assertRaises(ValueError):publish(study,audit,display)

    def test_changed_report_bank_contract_or_preparation_rejected(self):
        for filename in ('report.json','source-bank.json','contract.json','preparation.json'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary);study,audit,_=self.fixture(path)
                with (study/filename).open('a') as stream:stream.write(' ')
                with self.assertRaises(ValueError):publish(study,audit,path/'display.json')

    def test_incomplete_controls_missing_target_and_wrong_counts_rejected(self):
        for change in ('controls','cases','counts'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary);study,audit,_=self.fixture(path)
                report=json.loads((study/'report.json').read_text())
                if change=='controls':report['controls']['controls_post']['status']='failed'
                elif change=='cases':report['cases'].pop()
                else:report['native_releases']+=1
                (study/'report.json').write_text(json.dumps(report));self.receipt(study,audit)
                with self.assertRaises(ValueError):publish(study,audit,path/'display.json')


if __name__=='__main__':unittest.main()
