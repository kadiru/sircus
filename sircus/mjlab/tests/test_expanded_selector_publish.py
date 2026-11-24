"""Publication refuses stale evidence and preserves unrelated live content."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import expanded_selector_publish as publish
from sircus.mjlab.benchmark_verify import digest, sha
from sircus.mjlab.expanded_selector_study import summarize


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.study=self.root/'study';self.study.mkdir()
        self.audit=self.root/'audit.json'
        self.display=self.root/'display.json'
        self.write('current-checkpoint.json', {'frozen':'current'})
        self.write('expanded-checkpoint.json', {'frozen':'expanded'})
        self.write('library-summary.json', {'settings':2})
        self.write('library-manifest.json', {'authentic':'fixture'})
        self.layouts=[dict(id=f'p{i}',bucket_xy_m=[i/100,0]) for i in range(50)]
        self.write('layouts.json', {'layouts':self.layouts})
        for split in ('train','validation'):
            row=dict(split=split,dedup_key=split,capture_probability=1.,request={'bucket_xy_m':[0 if split=='train' else .5,0]})
            (self.study/(split+'-dataset.jsonl')).write_text(json.dumps(row)+'\n')
        frozen={p.name:sha(p) for p in self.study.iterdir()}
        self.contract=dict(methods=['current','expanded'],test_positions=50,physics_substeps=64,
            attempts_per_method_per_position=1,scoring_version='sircus-board-first-native-capture-v4',
            planning_budget_s=30.,frozen_inputs=frozen)
        self.contract['fingerprint']=digest(self.contract)
        self.write('contract.json',self.contract)
        self.slots=[];self.rows=[];self.plans=[]
        for i,layout in enumerate(self.layouts):
            for method in publish.METHODS:
                identity=f'{layout["id"]}-{method}'
                admitted=i<49
                slot=dict(id=identity,context_id=layout['id'],method=method,bucket_xy_m=layout['bucket_xy_m'],request={'id':identity} if admitted else None)
                self.slots.append(slot)
                self.plans.append(dict(variant=method,site=dict(choice=slot,decision_latency_s=10.,request_wall_s=11.)))
                if admitted:self.rows.append(dict(id=identity,adapter_directory='native-'+method,native_release=True,
                    board_first_scalar_audit_passed=True,board_first_score=dict(scoring_version=self.contract['scoring_version'],invalid_physics=False,route_success=i<40)))
        for i in range(18):self.rows.append(dict(id=f'control{i}',adapter_directory='native-controls'))
        self.write('plans.json',self.plans)
        bindings=dict(frozen,**{'contract.json':sha(self.study/'contract.json'),'plans.json':sha(self.study/'plans.json')})
        self.write('choices.json',dict(slots=self.slots,bindings=bindings))
        self.write('native-started.json',dict(bindings=dict(bindings,**{'choices.json':sha(self.study/'choices.json')})))
        (self.study/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        self.write('evidence-manifest.json',{p.name:sha(p) for p in self.study.iterdir()})
        comparison=summarize(self.layouts,self.slots,self.rows,self.plans)
        self.report=dict(version='sircus-expanded-selector-comparison-v1',status='completed',phase='completed',native_completed=True,
            comparison=comparison,controls={p:dict(status='passed',board_first_passed=True) for p in ('controls_pre','controls_post')})
        self.write('report.json',self.report)
        self.receipt=dict(version='sircus-expanded-selector-independent-audit-v1',status='passed',checks_count=100,
            frozen_before_test=True,comparison=comparison,checkpoint_sha256={m:sha(self.study/(m+'-checkpoint.json')) for m in publish.METHODS})
        for key,name in [('report_sha256','report.json'),('evidence_manifest_sha256','evidence-manifest.json'),('contract_sha256','contract.json'),('choices_sha256','choices.json'),('native_started_sha256','native-started.json')]:self.receipt[key]=sha(self.study/name)
        self.audit.write_text(json.dumps(self.receipt))

    def write(self,name,value):
        (self.study/name).write_text(json.dumps(value))

    def tearDown(self):self.temp.cleanup()

    def test_publish_preserves_existing_ui_and_exact_denominator(self):
        original={'planner':'hybrid','other_panel':{'video':'keep.mp4'}}
        self.display.write_text(json.dumps(original))
        section=publish.publish(self.study,self.audit,self.display)
        after=json.loads(self.display.read_text())
        self.assertEqual({k:after[k] for k in original},original)
        self.assertEqual(section['methods'][0]['captures'],40)
        self.assertEqual(section['methods'][0]['setup_rejections'],1)
        self.assertEqual(section['methods'][0]['released_misses'],9)
        self.assertEqual(len(section['results']),50)
        self.assertEqual(section['results'][-1]['expanded'],'setup refused')

    def test_changed_model_or_native_rows_cannot_publish(self):
        for filename in ('expanded-checkpoint.json','rows.jsonl','library-manifest.json'):
            path=self.study/filename;old=path.read_bytes()
            path.write_bytes(old+b' ')
            with self.assertRaises(ValueError):publish.compose(self.study,self.audit)
            path.write_bytes(old)

    def test_stale_or_partial_audit_cannot_publish(self):
        for change in ({'status':'pending'},{'frozen_before_test':False},{'checks_count':0},{'report_sha256':'0'*64}):
            self.audit.write_text(json.dumps(dict(self.receipt,**change)))
            with self.assertRaises(ValueError):publish.compose(self.study,self.audit)

    def test_cannot_replace_evidence_with_display(self):
        with self.assertRaises(ValueError):publish.publish(self.study,self.audit,self.study/'display.json')
        with self.assertRaises(ValueError):publish.publish(self.study,self.audit,self.audit)

    def test_invalid_contact_order_is_not_promoted_to_capture(self):
        slot=self.slots[0];native={r['id']:copy.deepcopy(r) for r in self.rows}
        native[slot['id']]['board_first_score'].update(invalid_physics=True,route_success=True)
        with self.assertRaises(ValueError):publish.outcome(slot,native)


if __name__=='__main__':unittest.main()
