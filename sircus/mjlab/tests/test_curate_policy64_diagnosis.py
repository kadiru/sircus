"""Synthetic publication fixtures only; these tests do not establish physics."""
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import curate_policy64_diagnosis as c


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')


class DiagnosisCurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.study=self.root/'diagnosis';self.study.mkdir()
        self.report=self.root/'combined.json';self.audit=self.root/'audit.json';self.output=self.root/'public'
        self.pairs=[]
        for i in range(50):
            pair=dict(context_id=f'policy64-test-{i:02d}',bucket_xy_m=[.2+i*.001,-.3])
            for method in ('bc','ppo'):
                native=not(i%7==0 and method=='ppo');capture=native and (i%2==0 if method=='bc' else i%3==0)
                values=[.5,-.2,.3,30.,.4] if method=='bc' else [.499,-.198,.29,31.,.42]
                pair[method]=dict(record_id=f'{method}-test-{i:02d}',record_sha256=c.digest([method,i]),values=values,native_release=native,
                    capture=capture,clean=False,invalid=None if native else 'finite_ik_unresolved',invalid_physics=False,
                    learning_reward=1. if capture else -.1,source_teacher_id=f'teacher-{i:02d}',
                    outcome='rejected' if not native else 'capture' if capture else 'miss',max_retention_s=2. if capture else 0.,
                    board_bounce=native,whole_ball_entry=capture,rim_band_contact=capture,bucket_contact_before_entry=capture,
                    entry_class='rim_assisted' if capture else None,minimum_mouth_clearance_mm=None,
                    first_flight=dict(first_crossing=None,forward_error_m=None,lateral_error_m=None,rebound_position_m=['PRIVATE']),
                    geometry_evidence=dict(reason=None if native else 'finite_ik_unresolved',minimum_required_clearance_m=.005,
                        checked_branch_pairs=2,best_checked_clearance_m=.006,proof_minimum_clearance_m=.006,q14=['PRIVATE']),
                    q10=['PRIVATE'],position_m=[['PRIVATE']],contact_arrays=['PRIVATE'])
            pair['delta_values']=[p-b for b,p in zip(pair['bc']['values'],pair['ppo']['values'])]
            pair['delta_release_height_m']=pair['delta_values'][2]+pair['delta_values'][4]
            pair['shared_anchor']=dict(same_id=True,same_values=True,same_observation=True)
            self.pairs.append(pair)
        self.source=dict(study_version=c.PARENT_VERSION,contract_fingerprint=c.PARENT_FP,report_sha256=c.PARENT_REPORT_SHA,
            manifest_sha256=c.PARENT_MANIFEST_SHA,original_audit_sha256=c.PARENT_AUDIT_SHA,substeps=64,
            bc=dict(samples=0,sha256='a'*64,weight_sha256='b'*64,path='/home/PRIVATE'),
            selected=dict(samples=1024,sha256='c'*64,weight_sha256='d'*64,path='/home/PRIVATE'),host_path='/home/PRIVATE')
        self.frozen=dict(version=c.STUDY_VERSION,status='completed',phase='original_analysis',source=self.source,pairs=self.pairs,
            paired_counts=self.paired_counts(),parameter_order=list(c.PARAMETERS),delta_summary=[],rejection_reasons=self.reasons(),
            actor_shift=dict(board_height_lower_count=50,release_gap_higher_count=50),
            optimization_history=dict(updates=[dict(index=i,samples=(i+1)*32,analytical_kl_mean=.3,analytical_kl_max=.4,
                sampled_ratio_clip_fraction=.8,critic_mse_before=1.,critic_mse_after=.9,gradient_norm_max=.5,
                before_sha256='e'*64,after_sha256='f'*64,buffer='/home/PRIVATE',raw_actions=['PRIVATE']) for i in range(32)],
                bc_retention_penalty=False,mean_post_update_kl=.3,mean_post_update_ratio_clip_fraction=.8,
                metric_definition='Final update likelihood ratio, not invalid physical actions.',caveat='Association, not causal proof.'),
            model_options=dict(status='proposed_not_trained',recommendation='Small forward ensemble.',current='BC and PPO actors.',
                feasibility='Separate geometry.',evaluation='Fresh future holdouts.',wam='World-action model.',inputs=['geometry'],targets=['capture'],
                sources=[dict(title='PETS',url='https://arxiv.org/abs/1805.12114')],credentials='PRIVATE'),
            examples=[dict(position_m=['PRIVATE'],q10=['PRIVATE'])],native_study=dict(status='pending'),credentials='PRIVATE')
        self.rows=[]
        for phase in ('controls_pre','diagnostic','controls_post'):
            if phase!='diagnostic':
                for role in ('positive_control','negative_control'):
                    for repeat in range(3):self.rows.append(self.row(phase,role,role,'control_exact',repeat,'positive_control'==role,True))
            else:
                for context in c.CONTEXTS:
                    for variant in c.VARIANTS:
                        for repeat in range(3 if variant.endswith('_exact') else 1):
                            native=not(variant=='ppo_exact' and context in (c.CONTEXTS[0],c.CONTEXTS[2]))
                            kind='historical_rejection' if not native else 'exact_replay' if variant.endswith('_exact') else 'coordinate_swap'
                            self.rows.append(self.row(phase,'diagnostic',context,variant,repeat,variant=='bc_exact',native,kind))
        self.bind()

    def row(self,phase,role,context,variant,repeat,success,native,kind='exact_replay'):
        return dict(id=f'diag-{phase}-{context}-{variant}-{repeat}',phase=phase,role=role,context_id=context,variant=variant,repeat=repeat,
            kind=kind,substeps=64,selection_or_learning=False,values=[.5,-.2,.3,30.,.4],bucket_xy_m=[.7,-.3],
            native_release=native,route_success=success and native,route_quality=1 if success and native else 0,
            invalid=None if native else 'finite_ik_unresolved',invalid_physics=False,whole_ball_entry=success and native,
            scalar_score_audit_passed=True,q10=['PRIVATE'],q14=['PRIVATE'],position_m=['PRIVATE'],contact_arrays=['PRIVATE'])

    def paired_counts(self):
        result=dict(layouts=50)
        for method in ('bc','ppo'):
            rows=[p[method] for p in self.pairs]
            result[method]=dict(requested=50,native=sum(r['native_release'] for r in rows),captures=sum(r['capture'] for r in rows),
                clean=0,rejected=sum(not r['native_release'] for r in rows),native_misses=sum(r['outcome']=='miss' for r in rows),invalid=0,
                mean_reward=sum(r['learning_reward'] for r in rows)/50)
        for k,fn in (('both_capture',lambda b,p:b and p),('bc_only',lambda b,p:b and not p),('ppo_only',lambda b,p:not b and p),('neither',lambda b,p:not b and not p)):
            result[k]=sum(fn(r['bc']['capture'],r['ppo']['capture']) for r in self.pairs)
        transitions=Counter((r['bc']['outcome'],r['ppo']['outcome']) for r in self.pairs)
        result['transitions']=[dict(bc_outcome=b,ppo_outcome=p,count=n) for (b,p),n in sorted(transitions.items())]
        result['admission_cohorts']=dict(Counter('both_native' if p['bc']['native_release'] and p['ppo']['native_release'] else 'bc_only_native' for p in self.pairs))
        joint=[p for p in self.pairs if p['bc']['native_release'] and p['ppo']['native_release']]
        result['jointly_admitted']=dict(layouts=len(joint),bc_captures=sum(p['bc']['capture'] for p in joint),ppo_captures=sum(p['ppo']['capture'] for p in joint),caveat='/home/PRIVATE')
        return result

    def reasons(self):
        contexts=[p['context_id'] for p in self.pairs if not p['ppo']['native_release']]
        return [dict(method='ppo',reason='finite_ik_unresolved',count=len(contexts),context_ids=contexts)]

    def bind(self):
        save(self.study/'analysis.json',self.frozen)
        text=''.join(json.dumps(r)+'\n' for r in self.rows)
        (self.study/'native').mkdir(exist_ok=True);(self.study/'records.jsonl').write_text(text);(self.study/'native/shots.jsonl').write_text(text)
        self.spec=dict(version=c.NATIVE_VERSION,substeps=64,maximum_requests=96,maximum_acquired_s=3600,training=False,noise=False,
            contexts=[v.removeprefix('policy64-test-') for v in c.CONTEXTS],analysis_sha256=c.sha(self.study/'analysis.json'),plan=copy.deepcopy(self.rows),
            parent_bindings=dict(parent_contract_fingerprint=c.PARENT_FP,parent_report_sha256=c.PARENT_REPORT_SHA,parent_manifest_sha256=c.PARENT_MANIFEST_SHA,
                parent_audit_sha256=c.PARENT_AUDIT_SHA,analysis_sha256=c.sha(self.study/'analysis.json')),
            diagnostic_source_sha256={'sircus/mjlab/producer.py':'a'*64},native_source_sha256={'sircus/mjlab/native.py':'b'*64})
        self.spec['fingerprint']=c.digest(self.spec);save(self.study/'contract.json',self.spec)
        manifest={str(p.relative_to(self.study)):c.sha(p) for p in self.study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
        save(self.study/'evidence-manifest.json',manifest)
        gates={k:c.control_gate([r for r in self.rows if r['phase']==phase]) for k,phase in (('pre','controls_pre'),('post','controls_post'))}
        self.native=dict(version=c.NATIVE_VERSION,status='completed',phase='completed',training=False,substeps=64,
            contract_fingerprint=self.spec['fingerprint'],analysis_sha256=c.sha(self.study/'analysis.json'),counts=c.counts(self.rows),control_gates=gates,
            contexts=[dict(context_id=context,variants={variant:c.counts([r for r in self.rows if r['phase']=='diagnostic' and r['context_id']==context and r['variant']==variant]) for variant in c.VARIANTS}) for context in c.CONTEXTS],
            reported_worker_elapsed_s=123.,evidence_sha256=manifest,
            gate=dict(status='passed',controls_passed=True,complete=True,scope='Diagnostic execution completion only; no new policy success-rate claim'))
        save(self.study/'report.json',self.native)
        combined=copy.deepcopy(self.frozen);combined.update(status='completed',phase='completed',contract_fingerprint=self.spec['fingerprint'],native_study=self.native,evidence_sha256=manifest);save(self.report,combined)
        self.proof=dict(version=c.AUDIT_VERSION,status='passed',checks_count=1,checks=[dict(passed=True,label='Synthetic fixture, not physics')],failures=[],
            contract_fingerprint=self.spec['fingerprint'],report_sha256=c.sha(self.report),analysis_report_sha256=c.sha(self.report),native_report_sha256=c.sha(self.study/'report.json'),
            frozen_analysis_sha256=c.sha(self.study/'analysis.json'),evidence_manifest_sha256=c.sha(self.study/'evidence-manifest.json'),
            source=copy.deepcopy(self.source),source_parent={k:self.source[k] for k in ('contract_fingerprint','report_sha256','manifest_sha256','original_audit_sha256')},
            producer_source_sha256=self.spec['diagnostic_source_sha256'],native_source_sha256=self.spec['native_source_sha256'],auditor_sha256='c'*64,
            requested_trials=96,native_releases=c.counts(self.rows)['native_releases'],counts=c.counts(self.rows),control_gates=gates,paired_counts=self.frozen['paired_counts'])
        save(self.audit,self.proof)

    def run_curate(self):return c.curate(self.report,self.audit,self.study,self.output)

    def test_exports_all_pairs_and56_variant_groups_without_private_data_or_evidence_changes(self):
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()};summary=self.run_curate()
        self.assertEqual(summary['native_study']['counts']['attempts'],96);self.assertEqual(summary['native_study']['diagnostic_counts']['attempts'],84)
        self.assertEqual(summary['native_study']['controls']['pre']['counts']['attempts'],6)
        pairs=list(csv.DictReader((self.output/'pairs.csv').read_text().splitlines()))
        variants=list(csv.DictReader((self.output/'native-variants.csv').read_text().splitlines()))
        for filename in ('pairs.csv','native-variants.csv'):
            self.assertNotIn(b'\r',(self.output/filename).read_bytes())
        self.assertEqual(len(pairs),50);self.assertEqual(len(variants),56);self.assertEqual(sum(int(r['attempts']) for r in variants),84)
        self.assertEqual(pairs[0]['ppo_first_flight_forward_error_m'],'');self.assertEqual(pairs[0]['ppo_max_retention_s'],'0.0')
        for filename,expected in c.read(self.output/'checksums.json').items():self.assertEqual(c.sha(self.output/filename),expected)
        text='\n'.join(p.read_text() for p in self.output.iterdir());self.assertNotIn('PRIVATE',text)
        for forbidden in ('q10','q14','position_m','contact_arrays','rebound_position_m','/home/','/Users/'):
            self.assertNotIn(forbidden,text)
        self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_strict_v1_nonempty_literal_passed_checklist(self):
        original=copy.deepcopy(self.proof)
        for change in (dict(version='sircus-policy64-diagnosis-audit-v2'),dict(checks=[]),dict(checks_count=True),dict(checks_count=2),dict(checks=[dict(passed=1,label='bad')]),dict(failures=['bad'])):
            save(self.audit,dict(original,**change))
            with self.assertRaises(ValueError):self.run_curate()
            self.assertFalse(self.output.exists())

    def test_every_bound_input_and_complete_manifest_is_verified(self):
        for path in (self.report,self.study/'report.json',self.study/'analysis.json',self.study/'records.jsonl',self.study/'evidence-manifest.json'):
            before=path.read_bytes();path.write_bytes(before+b'\n')
            with self.assertRaises(ValueError):self.run_curate()
            path.write_bytes(before)
        (self.study/'unmanifested.bin').write_bytes(b'new')
        with self.assertRaisesRegex(ValueError,'every frozen'):self.run_curate()

    def test_duplicate_pairs_incomplete_requests_and_modified_native_body_are_rejected(self):
        original=copy.deepcopy(self.frozen)
        self.frozen['pairs'][1]['context_id']=self.frozen['pairs'][0]['context_id'];self.bind()
        with self.assertRaisesRegex(ValueError,'50 unique'):self.run_curate()
        self.frozen=original;self.rows.pop();self.bind()
        with self.assertRaisesRegex(ValueError,'96 unique'):self.run_curate()

    def test_combined_merge_parent_identity_and_overwrite_guards(self):
        combined=c.read(self.report);combined['native_study']['counts']['captures']+=1;save(self.report,combined)
        self.proof.update(report_sha256=c.sha(self.report),analysis_report_sha256=c.sha(self.report));save(self.audit,self.proof)
        with self.assertRaisesRegex(ValueError,'exact immutable'):self.run_curate()
        self.bind();self.source['original_audit_sha256']='f'*64;self.bind()
        with self.assertRaisesRegex(ValueError,'parent identity'):self.run_curate()
        self.source['original_audit_sha256']=c.PARENT_AUDIT_SHA;self.bind()
        self.output.mkdir()
        with self.assertRaises(ValueError):self.run_curate()
        for output in (self.study/'public',self.root):
            with self.assertRaises(ValueError):c.curate(self.report,self.audit,self.study,output)


if __name__=='__main__':unittest.main()
