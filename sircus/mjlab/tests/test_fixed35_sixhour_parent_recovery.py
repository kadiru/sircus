"""Fail-closed acceptance of a separate, fully bound parent audit correction."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import fixed35_sixhour_pipeline as P
from sircus.mjlab import fixed35_sixhour_verify as V
from sircus.mjlab import fixed35_sixhour_contract as C
from sircus.mjlab.benchmark_verify import EvidenceError
from sircus.mjlab.tests.test_fixed35_sixhour_pipeline import write
from sircus.mjlab.tests import test_fixed35_sixhour_pipeline as pipeline_tests


def corrected_fixture(root, **ignored):
    study,auditpath=root/'broad-study',root/'broad-audit'
    source=Path(P.__file__).parent
    sources={key:P.sha(source/name) for key,name in (
        ('original_auditor_source_sha256','fixed35_broad_verify.py'),
        ('adapter_source_sha256','certificate_roundoff_verify.py'),
        ('wrapper_source_sha256','fixed35_broad_audit_v2.py'))}
    spec=dict(version=P.PARENT_VERSION,source_sha256={'sircus/mjlab/fixed35_broad_verify.py':sources['original_auditor_source_sha256']})
    spec['fingerprint']=P.digest(spec);write(study/'contract.json',spec)
    (study/'rows.jsonl').write_text('')
    manifest={name:P.sha(study/name) for name in ('contract.json','rows.jsonl')}
    write(study/'evidence-manifest.json',manifest)
    report=dict(version=P.PARENT_VERSION,status='completed',phase='completed',fresh_search_releases=P.PARENT_QUOTAS,
        gate=dict(search_quota_completed=True,controls_passed=True,preservation_passed=True),
        contract_fingerprint=spec['fingerprint'],evidence_sha256=manifest,training=False,
        historical_policy_captures=24,parent_positions=35,actual_requests=0)
    write(study/'report.json',report)
    original=study.with_name(study.name+'-audit.json')
    write(original,dict(version=P.PARENT_AUDIT_VERSION,status='failed',failures=['EvidenceError: v3 numeric exception restricted to near-zero normal angle']))
    pipeline=study.with_suffix('.pipeline.json')
    write(pipeline,dict(status='audit_failed',output=str(study),study_exit_code=0,audit_exit_code=1,finished_unix_s=900.))
    correction=dict(version=P.CORRECTION_VERSION,kind='diagnostic_certificate_roundoff_only',
        original_audit_file=str(original),original_audit_sha256=P.sha(original),
        original_pipeline_file=str(pipeline),original_pipeline_sha256=P.sha(pipeline),
        original_auditor_version=P.PARENT_AUDIT_VERSION,original_audit_failure=P.read(original),**sources,
        report_sha256=P.sha(study/'report.json'),evidence_manifest_sha256=P.sha(study/'evidence-manifest.json'),
        producer_source_sha256=spec['source_sha256'],physics_changed=False,scoring_changed=False,
        admission_changed=False,completed_unix_s=950.)
    proof=dict(version=P.CORRECTED_PARENT_AUDIT_VERSION,status='passed',failures=[],checks_count=1,
        checks=[dict(passed=True)],full_checks_verified=True,native_checks_verified=True,
        report_sha256=P.sha(study/'report.json'),contract_fingerprint=spec['fingerprint'],
        evidence_manifest_sha256=P.sha(study/'evidence-manifest.json'),producer_source_sha256=spec['source_sha256'],
        requested_trials=0,audit_correction=correction)
    write(auditpath,proof)
    return study,auditpath,P.sha(study/'contract.json')


class CorrectedParentTests(unittest.TestCase):
    def authenticate(self,study,audit,contract):
        with patch.object(P.time,'time',return_value=1000.):
            receipt=P.parent_ready(study,audit,contract)
            independent=V.authenticate_broad(study,audit,V.AuditChecks())
        return receipt,independent

    def test_explicit_correction_preserves_original_failures_and_binds_full_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,audit,contract=corrected_fixture(Path(tmp))
            original=study.with_name(study.name+'-audit.json');pipeline=study.with_suffix('.pipeline.json')
            before={p:p.read_bytes() for p in (original,pipeline,study/'report.json',study/'evidence-manifest.json')}
            receipt,parent=self.authenticate(study,audit,contract)
            self.assertEqual(receipt['parent_acceptance'],parent['parent_acceptance'])
            self.assertEqual(receipt['parent_acceptance']['basis'],'independent_corrected_parent_audit')
            self.assertEqual(parent['parent_acceptance']['original_pipeline_sha256'],P.sha(pipeline))
            self.assertEqual({p:p.read_bytes() for p in before},before)
            binding={k:parent[k] for k in ('report_sha256','audit_sha256','contract_fingerprint','evidence_manifest_sha256','checks_count','parent_acceptance')}
            V.authenticate_broad(study,audit,V.AuditChecks(),expected_binding=binding)
            binding['parent_acceptance']['original_audit_sha256']='changed'
            with self.assertRaises(EvidenceError):V.authenticate_broad(study,audit,V.AuditChecks(),expected_binding=binding)

    def test_original_failed_audit_alone_still_cannot_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,_,contract=corrected_fixture(Path(tmp))
            with self.assertRaises(ValueError):P.parent_ready(study,study.with_name(study.name+'-audit.json'),contract)

    def test_changed_failure_worker_evidence_or_correction_refuses_in_both_gates(self):
        changes=(('pipeline','study_exit_code',1),('pipeline','status','failed'),('pipeline','output','/another-study'),
            ('original','status','passed'),('correction','original_audit_failure',{}),('correction','original_audit_sha256','0'*64),
            ('correction','original_pipeline_sha256','0'*64),('correction','physics_changed',True),
            ('correction','scoring_changed',True),('correction','admission_changed',True),
            ('correction','adapter_source_sha256','0'*64),('correction','wrapper_source_sha256','0'*64),
            ('correction','producer_source_sha256',{}),('correction','report_sha256','0'*64),
            ('correction','evidence_manifest_sha256','0'*64),('correction','completed_unix_s',1001.),
            ('correction','completed_unix_s',899.),('audit','version','unknown-audit'),
            ('audit','full_checks_verified',False),('audit','native_checks_verified',False),
            ('audit','checks',[]),('audit','checks',[dict(passed=False)]),('audit','status','failed'))
        for where,key,value in changes:
            with self.subTest(where=where,key=key,value=value),tempfile.TemporaryDirectory() as tmp:
                study,audit,contract=corrected_fixture(Path(tmp));proof=P.read(audit)
                if where=='correction':proof['audit_correction'][key]=value;write(audit,proof)
                elif where=='audit':proof[key]=value;write(audit,proof)
                else:
                    path=study.with_suffix('.pipeline.json') if where=='pipeline' else study.with_name(study.name+'-audit.json')
                    doc=P.read(path);doc[key]=value;write(path,doc)
                with patch.object(P.time,'time',return_value=1000.):
                    with self.assertRaises((ValueError,KeyError)):P.parent_ready(study,audit,contract)
                    with self.assertRaises((EvidenceError,KeyError)):V.authenticate_broad(study,audit,V.AuditChecks())

    def test_rebound_failure_with_valid_hashes_is_not_this_specific_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,audit,contract=corrected_fixture(Path(tmp));original=study.with_name(study.name+'-audit.json')
            failure=P.read(original);failure['failures']=['EvidenceError: invalid native contact'];write(original,failure)
            proof=P.read(audit);proof['audit_correction'].update(original_audit_failure=failure,original_audit_sha256=P.sha(original));write(audit,proof)
            with self.assertRaises(ValueError):P.parent_ready(study,audit,contract)
            with self.assertRaises(EvidenceError):V.authenticate_broad(study,audit,V.AuditChecks())


    def test_audited_but_incomplete_quota_is_never_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,audit,contract=corrected_fixture(Path(tmp));report=P.read(study/'report.json')
            report['fresh_search_releases'][next(iter(P.PARENT_QUOTAS))]-=1;write(study/'report.json',report)
            proof=P.read(audit);proof['report_sha256']=P.sha(study/'report.json');proof['audit_correction']['report_sha256']=proof['report_sha256'];write(audit,proof)
            with self.assertRaises(ValueError):P.parent_ready(study,audit,contract)
            with self.assertRaises(EvidenceError):V.authenticate_broad(study,audit,V.AuditChecks())

    def test_budget_audit_requires_correction_completed_before_new_launch(self):
        for launch_time in (1000.,949.):
            with self.subTest(launch_time=launch_time),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);parentdir,audit,contract=corrected_fixture(root)
                receipt,parent=self.authenticate(parentdir,audit,contract)
                study=root/'new-study';study.mkdir()
                queue=dict(pipeline_source_sha256='pipeline',queue_limit_s=18000,
                    started_unix_s=910.,queue_deadline_unix_s=18910.,global_deadline_unix_s=40510.)
                queuepath=root/'new-study-queue.json';write(queuepath,queue)
                write(root/'new-study-launch.json',dict(parent=receipt,queue_receipt_sha256=P.sha(queuepath),
                    acquired_total_limit_s=21600,requested_unix_s=launch_time))
                budget=dict(limit_s=18000,audit_reserve_s=3600,acquired_total_limit_s=21600,gpu_uuid=P.GPU_UUID,
                    acquired_unix_s=1001.,acquired_deadline_unix_s=22601.,effective_worker_limit_s=18000,
                    global_deadline_unix_s=40510.)
                write(root/'new-study-budget.json',budget)
                write(root/'new-study.pipeline.json',dict(global_deadline_unix_s=40510.))
                report=dict(elapsed_s=2.,deadlines=dict(worker_limit_s=18000,search_limit_s=16200,
                    **{k:budget[k] for k in ('effective_worker_limit_s','global_deadline_unix_s','acquired_deadline_unix_s')}))
                spec=dict(source_sha256={'sircus/mjlab/fixed35_sixhour_pipeline.py':'pipeline'})
                if launch_time==1000.:
                    result=V.audit_budget(study,report,spec,parent,V.AuditChecks())
                    self.assertEqual(result['acquired_total_limit_s'],21600)
                else:
                    with self.assertRaisesRegex(EvidenceError,'before completed independent corrected audit'):
                        V.audit_budget(study,report,spec,parent,V.AuditChecks())


    def test_contract_retains_acceptance_receipt_in_frozen_binding(self):
        acceptance=dict(basis='independent_corrected_parent_audit',correction_sha256='a'*64)
        historical=dict(bindings={'old':'binding'},parent=dict(identity=dict(scene_sha256='scene',asset_sha256='assets')),preserved=[])
        parent=dict(spec=dict(source_bindings=historical['bindings'],identity=historical['parent']['identity']),
            report=dict(status='completed',fresh_search_releases=C.old.QUOTAS),rows=[],parent_acceptance=acceptance,
            **{k:k for k in ('report_sha256','audit_sha256','contract_fingerprint','evidence_manifest_sha256','checks_count')})
        with patch.object(C.parent,'load_inputs',return_value=historical),patch.object(V,'authenticate_broad',return_value=parent),patch.object(V,'confirmed_broad_recipes',return_value=[]):
            result=C.load_inputs(SimpleNamespace(broad_study=Path('/parent'),broad_audit=Path('/audit')),Path('/source'))
        self.assertEqual(result['bindings']['broad']['parent_acceptance'],acceptance)

    def test_fresh_pipeline_launch_keeps_six_hour_budget_and_original_terminal_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('sircus.mjlab.tests.test_fixed35_sixhour_pipeline.parent_fixture',side_effect=corrected_fixture):
                calls,state,_,error=pipeline_tests.PipelineTests().invoke(root)
            self.assertIsNone(error)
            self.assertEqual(state['status'],'completed')
            self.assertEqual([c[0] for c in calls],['worker','audit'])
            self.assertEqual(P.read(root/'broad-study.pipeline.json')['status'],'audit_failed')
            receipt=P.read(root/'study-launch.json')
            self.assertEqual(receipt['acquired_total_limit_s'],21600)
            self.assertEqual(receipt['parent']['parent_acceptance']['basis'],'independent_corrected_parent_audit')
            self.assertEqual(calls[0][2]['worker_limit'],18000)
            self.assertEqual(calls[1][2]['timeout'],3600)



def timing_corrected_fixture(root):
    study,auditpath,_=corrected_fixture(root);source=Path(P.__file__).parent
    spec=P.read(study/'contract.json');spec.pop('fingerprint')
    spec['protocol']=dict(worker_limit_s=43200,search_limit_s=36000,audit_reserve_s=7200)
    spec['fingerprint']=P.digest(spec);write(study/'contract.json',spec)
    manifest=P.read(study/'evidence-manifest.json');manifest['contract.json']=P.sha(study/'contract.json')
    write(study/'evidence-manifest.json',manifest)
    budget=dict(limit_s=43200,audit_reserve_s=7200,gpu_uuid=P.GPU_UUID,acquired_unix_s=1100.,
        effective_worker_limit_s=43200,global_deadline_unix_s=52300.)
    budgetpath=study.with_name(study.name+'-budget.json');write(budgetpath,budget)
    pipelinepath=study.with_suffix('.pipeline.json');state=P.read(pipelinepath)
    state.update(started_unix_s=1000.,global_limit_s=51300,global_deadline_unix_s=52300.,finished_unix_s=30500.)
    write(pipelinepath,state)
    report=P.read(study/'report.json');report.update(elapsed_s=29190.8101,contract_fingerprint=spec['fingerprint'],
        evidence_sha256=manifest,deadlines=dict(worker_limit_s=43200,search_limit_s=36000,
            effective_worker_limit_s=43200,global_deadline_unix_s=52300.))
    write(study/'report.json',report)
    proof=P.read(auditpath);proof.update(version=P.TIMING_PARENT_AUDIT_VERSION,contract_fingerprint=spec['fingerprint'],
        report_sha256=P.sha(study/'report.json'),evidence_manifest_sha256=P.sha(study/'evidence-manifest.json'),request_packets=284)
    proof['audit_correction'].update(original_pipeline_sha256=P.sha(pipelinepath),completed_unix_s=31000.,
        report_sha256=proof['report_sha256'],evidence_manifest_sha256=proof['evidence_manifest_sha256'])
    previous=study.with_name(study.name+'-audit-v2.json')
    failure=dict(version=P.CORRECTED_PARENT_AUDIT_VERSION,status='failed',
        auditor_sha256=P.sha(source/'fixed35_broad_audit_v2.py'),
        original_audit_failure_sha256=proof['audit_correction']['original_audit_sha256'],
        failures=['EvidenceError: Monotone native batch timings within wall budget'])
    write(previous,failure)
    correction=dict(version='sircus-broad-authenticated-worker-clock-v1',kind='authenticated_original_worker_budget_only',
        previous_audit_file=str(previous),previous_audit_sha256=P.sha(previous),previous_audit_failure=failure,
        previous_auditor_version=P.CORRECTED_PARENT_AUDIT_VERSION,
        previous_auditor_source_sha256=P.sha(source/'fixed35_broad_audit_v2.py'),
        adapter_version='sircus-authenticated-packet-worker-clock-audit-v1',
        adapter_source_sha256=P.sha(source/'packet_timing_verify.py'),wrapper_source_sha256=P.sha(source/'fixed35_broad_audit_v3.py'),
        original_budget=dict(worker_receipt_sha256=P.sha(budgetpath),effective_worker_limit_s=43200,global_limit_s=51300,
            reported_elapsed_s=report['elapsed_s'],report_sha256=proof['report_sha256'],contract_sha256=P.sha(study/'contract.json'),
            budget_file=str(budgetpath.resolve()),original_packet_auditor_sha256=P.sha(source/'coverage_training_verify.py'),obsolete_inherited_cap_s=19800),
        report_sha256=proof['report_sha256'],evidence_manifest_sha256=proof['evidence_manifest_sha256'],
        producer_source_sha256=spec['source_sha256'],request_packets=284,
        **{key:False for key in ('physics_changed','scoring_changed','admission_changed','request_bytes_changed','timestamps_changed','physics_rerun')},
        completed_unix_s=32000.)
    proof['timing_audit_correction']=correction;write(auditpath,proof)
    return study,auditpath,P.sha(study/'contract.json')


class TimingCorrectedParentTests(unittest.TestCase):
    def both(self,study,audit,contract):
        with patch.object(P.time,'time',return_value=33000.):
            return P.parent_ready(study,audit,contract),V.authenticate_broad(study,audit,V.AuditChecks())

    def reject_both(self,study,audit,contract):
        with patch.object(P.time,'time',return_value=33000.):
            with self.assertRaises((ValueError,KeyError)):P.parent_ready(study,audit,contract)
            with self.assertRaises((EvidenceError,KeyError)):V.authenticate_broad(study,audit,V.AuditChecks())

    def test_v3_accepts_original_43200_cap_without_editing_failure_or_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,audit,contract=timing_corrected_fixture(Path(tmp))
            paths=[study.with_name(study.name+name) for name in ('-audit.json','-audit-v2.json','-budget.json')]
            paths += [study.with_suffix('.pipeline.json'),study/'report.json',study/'contract.json',study/'evidence-manifest.json']
            before={p:p.read_bytes() for p in paths};receipt,parent=self.both(study,audit,contract)
            self.assertEqual(receipt['parent_acceptance'],parent['parent_acceptance'])
            self.assertEqual(receipt['parent_acceptance']['completed_unix_s'],32000.)
            self.assertEqual(receipt['parent_acceptance']['corrected_audit_version'],P.TIMING_PARENT_AUDIT_VERSION)
            self.assertEqual({p:p.read_bytes() for p in paths},before)
            self.assertEqual(P.read(study/'report.json')['elapsed_s'],29190.8101)
            self.assertEqual(P.read(study.with_name(study.name+'-budget.json'))['limit_s'],43200)
            self.assertEqual(P.WORKER_LIMIT_S,18000)
            self.assertEqual(P.ACQUIRED_LIMIT_S,21600)

    def test_each_v3_provenance_or_semantic_change_fails_closed(self):
        changes=(('previous_audit_sha256','0'*64),('previous_audit_failure',{}),('previous_auditor_source_sha256','0'*64),
            ('adapter_source_sha256','0'*64),('wrapper_source_sha256','0'*64),('request_packets',283),
            ('report_sha256','0'*64),('evidence_manifest_sha256','0'*64),('producer_source_sha256',{}),
            ('physics_changed',True),('scoring_changed',True),('admission_changed',True),
            ('request_bytes_changed',True),('timestamps_changed',True),('physics_rerun',True),
            ('completed_unix_s',30999.),('completed_unix_s',33001.),('original_budget',{}))
        for key,value in changes:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                study,audit,contract=timing_corrected_fixture(Path(tmp));proof=P.read(audit)
                proof['timing_audit_correction'][key]=value;write(audit,proof);self.reject_both(study,audit,contract)

    def test_rehashed_wrong_original_budget_cannot_increase_or_replace_authorized_cap(self):
        for key,value in (('limit_s',50000),('audit_reserve_s',3600),('gpu_uuid','4070'),
                          ('effective_worker_limit_s',50000),('global_deadline_unix_s',60000),('acquired_unix_s',20000)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                study,audit,contract=timing_corrected_fixture(Path(tmp));budgetpath=study.with_name(study.name+'-budget.json')
                budget=P.read(budgetpath);budget[key]=value;write(budgetpath,budget)
                proof=P.read(audit);c=proof['timing_audit_correction'];c['original_budget']['worker_receipt_sha256']=P.sha(budgetpath)
                c['original_budget']['effective_worker_limit_s']=budget['effective_worker_limit_s'];write(audit,proof)
                self.reject_both(study,audit,contract)

    def test_rehashed_unrelated_v2_failure_cannot_use_packet_clock_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            study,audit,contract=timing_corrected_fixture(Path(tmp));path=study.with_name(study.name+'-audit-v2.json')
            failure=P.read(path);failure['failures']=['EvidenceError: unsafe contact'];write(path,failure)
            proof=P.read(audit);proof['timing_audit_correction'].update(previous_audit_failure=failure,previous_audit_sha256=P.sha(path))
            write(audit,proof);self.reject_both(study,audit,contract)

    def test_v3_cannot_omit_original_numeric_correction_or_full_native_audit(self):
        for field in ('audit_correction','timing_audit_correction','full_checks_verified','native_checks_verified'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                study,audit,contract=timing_corrected_fixture(Path(tmp));proof=P.read(audit);proof.pop(field);write(audit,proof)
                self.reject_both(study,audit,contract)


if __name__=='__main__':unittest.main()
