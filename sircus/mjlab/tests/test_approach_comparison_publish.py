"""Only audited frozen fifty-position outcomes may be published as final results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab import approach_comparison_publish as pub
from sircus.mjlab.benchmark_verify import digest,sha


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.study=self.root/'study';self.study.mkdir();self.audit=self.root/'audit.json';self.display=self.root/'display.json'
        self.layouts=[dict(id=f'fresh-{i:02d}',bucket_xy_m=[i/100,.05]) for i in range(50)]
        self.write('layouts.json',dict(layouts=self.layouts))
        for m in pub.METHODS:self.write(m+'-checkpoint.json',dict(frozen=m))
        frozen={p.name:sha(p) for p in self.study.iterdir()}
        self.contract=dict(version=pub.STUDY_VERSION,training_performed=False,methods=list(pub.METHODS),test_positions=50,physics_substeps=64,attempts_per_method_per_position=1,
            scoring_version='sircus-board-first-native-capture-v4',planning_budget_s=30.,frozen_inputs=frozen)
        self.contract['fingerprint']=digest(self.contract);self.write('contract.json',self.contract)
        self.slots=[];self.rows=[];self.plans=[];success={}
        for i,l in enumerate(self.layouts):
            for method in pub.METHODS:
                identity=l['id']+'-'+method;admitted=i<(48 if method=='current' else 47)
                capture=(i<35 if method=='current' else i<37 and i!=3)
                invalid=method=='expanded' and i==46
                slot=dict(id=identity,context_id=l['id'],bucket_xy_m=l['bucket_xy_m'],method=method,request={'id':identity} if admitted else None)
                self.slots.append(slot);success[l['id'],method]=capture
                decision=(5+.1*i) if method=='current' else (4+.2*i)
                self.plans.append(dict(variant=method,site=dict(layout=l,decision_latency_s=decision,request_wall_s=decision+1)))
                if admitted:self.rows.append(dict(id=identity,adapter_directory='native-'+method,native_release=True,board_first_scalar_audit_passed=True,
                    board_first_score=dict(scoring_version='sircus-board-first-native-capture-v4',route_success=capture,invalid_physics=invalid)))
        for i in range(18):self.rows.append(dict(id=f'control-{i}',adapter_directory='native-controls'))
        self.write('plans.json',self.plans)
        bindings=dict(frozen,**{'contract.json':sha(self.study/'contract.json'),'plans.json':sha(self.study/'plans.json')})
        self.write('choices.json',dict(slots=self.slots,bindings=bindings))
        self.write('native-started.json',dict(bindings=dict(bindings,**{'choices.json':sha(self.study/'choices.json')})))
        (self.study/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        paired=dict(both=0,current_only=0,expanded_only=0,neither=0)
        for l in self.layouts:
            a,b=[success[l['id'],m] for m in pub.METHODS];paired['both' if a and b else 'current_only' if a else 'expanded_only' if b else 'neither']+=1
        methods={}
        for m in pub.METHODS:
            rows=[r for r in self.rows if r['adapter_directory']=='native-'+m]
            captures=sum(r['board_first_score']['route_success'] for r in rows);invalid=sum(r['board_first_score']['invalid_physics'] for r in rows)
            stats={}
            for field,suffix in (('decision_latency_s','planning_s'),('request_wall_s','request_s')):
                stats.update({k+'_'+suffix:v for k,v in pub.statistics([p['site'][field] for p in self.plans if p['variant']==m]).items()})
            methods[m]=dict(positions=50,captures=captures,releases=len(rows),setup_rejections=50-len(rows),flight_misses=len(rows)-captures-invalid,invalid_physics=invalid,**stats)
        self.report=dict(version=pub.STUDY_VERSION,status='completed',phase='completed',native_completed=True,
            comparison=dict(methods=methods,paired=paired),gpu_wait_s=120.,gpu_lease_s=60.,updated_at_unix=1795795200.,
            controls={p:dict(status='passed',board_first_passed=True) for p in ('controls_pre','controls_post')})
        self.write('report.json',self.report)
        self.write('evidence-manifest.json',{p.name:sha(p) for p in self.study.iterdir() if p.name!='report.json'})
        self.receipt=dict(version=pub.AUDIT_VERSION,status='passed',checks_count=900,frozen_before_test=True,comparison=copy.deepcopy(self.report['comparison']),
            checkpoint_sha256={m:sha(self.study/(m+'-checkpoint.json')) for m in pub.METHODS})
        for key,name in {'report_sha256':'report.json','evidence_manifest_sha256':'evidence-manifest.json','contract_sha256':'contract.json','choices_sha256':'choices.json','native_started_sha256':'native-started.json'}.items():self.receipt[key]=sha(self.study/name)
        self.audit.write_text(json.dumps(self.receipt))

    def tearDown(self):self.temp.cleanup()
    def write(self,name,value):(self.study/name).write_text(json.dumps(value))

    def test_audited_publish_retains_all_prior_sections_and_media(self):
        old={'planner':'hybrid','expanded_selector_replays':{'videos':list(range(17))},'unrelated':{'keep':True}}
        self.display.write_text(json.dumps(old));pub.publish(self.study,self.audit,self.display)
        after=json.loads(self.display.read_text());self.assertEqual({k:after[k] for k in old},old)
        section=after['approach_comparison'];self.assertEqual(section['native_releases'],113)
        self.assertEqual([m['captures'] for m in section['methods']],[35,36]);self.assertEqual(section['paired'],dict(both=34,current_only=1,approach_only=2,neither=13))
        self.assertEqual(section['gpu_wait_s'],120);self.assertEqual(section['gpu_lease_s'],60)
        self.assertEqual(section['methods'][0]['decision_s'],pub.statistics([5+.1*i for i in range(50)]))
        self.assertNotIn(str(self.root),json.dumps(section))

    def test_tampered_native_or_choice_evidence_fails_closed(self):
        for name in ('rows.jsonl','plans.json','expanded-checkpoint.json','choices.json'):
            p=self.study/name;old=p.read_bytes();p.write_bytes(old+b' ')
            with self.assertRaises(ValueError):pub.compose(self.study,self.audit)
            p.write_bytes(old)

    def test_no_final_counts_with_pending_audit_or_wrong_report_binding(self):
        for changes in ({'status':'pending'},{'frozen_before_test':False},{'report_sha256':'0'*64},{'checks_count':0}):
            self.audit.write_text(json.dumps(dict(self.receipt,**changes)))
            with self.assertRaises(ValueError):pub.compose(self.study,self.audit)

    def test_independent_statistics_use_prespecified_nearest_rank_percentile(self):
        self.assertAlmostEqual(pub.percent90([0,10]),10.)
        for values in ([float('nan')],[True],[],[-1]):
            with self.assertRaises(ValueError):pub.statistics(values)

    def test_waiting_snapshot_requires_actual_wait_event_and_shows_no_captures(self):
        self.report.update(status='running',phase='waiting_for_4090',native_completed=False,gpu_wait=dict(status='waiting_for_4090',wait_s=12.,reason=[1234]),progress=dict(prepared=100))
        self.write('report.json',self.report)
        snapshot=pub.pending(self.study);self.assertEqual(snapshot['gpu_wait_s'],120.)
        self.assertEqual(snapshot['phase'],'waiting_for_4090');self.assertNotIn('methods',snapshot)
        self.assertNotIn('comparison',snapshot);self.assertNotIn('1234',json.dumps(snapshot))
        self.report['queue_expired']=True;self.write('report.json',self.report)
        self.assertTrue(pub.pending(self.study)['queue_expired'])
        self.report.pop('gpu_wait');self.write('report.json',self.report)
        with self.assertRaises(ValueError):pub.pending(self.study)

    def test_pending_cannot_replace_final_and_display_cannot_overwrite_evidence(self):
        pub.publish(self.study,self.audit,self.display)
        with self.assertRaises(ValueError):pub.publish(self.study,None,self.display,pending_only=True)
        for dest in (self.study/'display.json',self.audit):
            with self.assertRaises(ValueError):pub.publish(self.study,self.audit,dest)


if __name__=='__main__':unittest.main()
