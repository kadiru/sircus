"""Synthetic receipts only; no simulator, network, GPU or server launches."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shot_benchmark_report as report_module
from shot_benchmark_report import (EXPERIMENT_SCHEMA, PROGRESS_SCHEMA, MILESTONES,
    counts, publish, sha, summarize)
from shot_benchmark_server import byte_range, published_file


class BenchmarkReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'native';self.root.mkdir()
        self.source=dict(commit='a'*40,archive_sha256='b'*64,geometry_fingerprint='c'*64,camera_sha256='d'*64,scoring_version='synthetic-test-score-v1')
        self.splits={s:[dict(id=f'{s}-{i}',split=s,bucket_xy_m=[j+i/1000,j/10]) for i in range(n)] for j,(s,n) in enumerate([('train',35),('validation',35),('test',70)])}
        self.write(self.root/'splits.json',self.splits)
        self.exp=dict(schema=EXPERIMENT_SCHEMA,experiment_id='synthetic-fixture-only',source=self.source,
            methods=['sweep','ppo'],seeds=[20261121,20261122,20261123],train_budget_per_run=10000,
            milestones=list(MILESTONES),validation_layout_count=35,test_layout_count=70,
            split_manifest_file='splits.json',split_manifest_sha256=sha(self.root/'splits.json'),
            runs=[dict(method=m,seed=s,path=f'{m}-seed{s}') for m in ('sweep','ppo') for s in (20261121,20261122,20261123)])
        self.save_exp()
    def write(self,path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
    def save_exp(self):self.write(self.root/'experiment.json',self.exp)
    def run_data(self,index=0,complete=False):
        p=self.exp['runs'][index];n=10000 if complete else 1000
        raw=dict(schema=PROGRESS_SCHEMA,experiment_id=self.exp['experiment_id'],method=p['method'],seed=p['seed'],source=self.source,split_manifest_sha256=self.exp['split_manifest_sha256'],status='completed' if complete else 'running',train=dict(attempts=n,captures=n//10,clean=n//20,rejected=n//2,failures={'ik':n//2}),learning_curve=[dict(attempts=0,captures=0,clean=0,rejected=0,mean_reward=None,wall_s=0),dict(attempts=n,captures=n//10,clean=n//20,rejected=n//2,mean_reward=.2,wall_s=10)],checkpoints=[],resources={'headless':{'wall_s':10,'cpu_s':20,'peak_rss_mib':100,'peak_gpu_mib':None,'scope':'Synthetic fixture, not measurement'}},batches=[dict(setup_wall_s=4,score_wall_s=2,filming=False)])
        def evaluation(split,cap,clean,rejected):
            return dict(attempts=len(self.splits[split]),captures=cap,clean=clean,rejected=rejected,failures={'ik':rejected},layouts=[dict(layout_id=p['id'],bucket_xy_m=p['bucket_xy_m'],requested=None,measured=None,outcome=dict(captured=i<cap,clean=i<clean,rejected=cap<=i<cap+rejected,reason=None)) for i,p in enumerate(self.splits[split])])
        for budget in MILESTONES if complete else [0,1000]:
            cp=self.root/p['path']/f'checkpoint-{budget}.json';proof=self.root/p['path']/f'proof-{budget}.json'
            self.write(cp,{'synthetic':True,'budget':budget});self.write(proof,{'synthetic':True,'proof':budget})
            raw['checkpoints'].append(dict(training_attempts=budget,checkpoint_sha256=sha(cp),proof_sha256=sha(proof),checkpoint_file=cp.name,proof_file=proof.name,validation=evaluation('validation',7,3,10),test=evaluation('test',14,6,20) if budget==10000 else None))
        return raw
    def save_run(self,raw,index=0):self.write(self.root/self.exp['runs'][index]['path']/'progress.json',raw)
    def test_missing_runs_are_pending_not_fabricated_zero_results(self):
        d=summarize(self.root);self.assertEqual(d['observed_train_attempts'],0)
        self.assertTrue(all(r['train'] is None and r['status']=='pending' and r['resources']['headless'] is None for r in d['runs']))
    def test_rejects_remain_denominator_resources_and_frozen_splits_separate(self):
        self.save_run(self.run_data());d=summarize(self.root);r=d['runs'][0]
        self.assertEqual(r['train']['capture_rate'],.1);self.assertEqual(r['train']['released'],500)
        self.assertIsNone(r['resources']['render']);self.assertIsNone(r['resources']['headless']['peak_gpu_mib'])
        self.assertEqual(r['batch_timing']['setup_wall_s'],4);self.assertIsNone(r['checkpoints'][0]['test'])
    def test_all_completed_requires_six_exact_budgets_and_frozen_checks(self):
        for i in range(6):self.save_run(self.run_data(i,True),i)
        d=summarize(self.root);self.assertEqual(d['status'],'completed');self.assertEqual(d['observed_train_attempts'],60000)
        summary=next(s for s in d['method_summary'] if s['method']=='ppo' and s['split']=='test');self.assertEqual(summary['measured_seeds'],3);self.assertEqual(summary['mean_capture_rate'],.2);self.assertEqual(summary['seed_sd'],0)
        r=self.run_data(0,True);r['checkpoints'].pop(0);self.save_run(r)
        with self.assertRaisesRegex(ValueError,'checkpoints'):summarize(self.root)
    def test_completed_missing_layout_or_proof_is_rejected(self):
        r=self.run_data(0,True);r['checkpoints'][-1]['test'].pop('layouts');self.save_run(r)
        with self.assertRaisesRegex(ValueError,'per-layout'):summarize(self.root)
        r=self.run_data(0,True);r['checkpoints'][0].pop('proof_file');self.save_run(r)
        with self.assertRaisesRegex(ValueError,'actual model and proof'):summarize(self.root)
    def test_early_test_leakage_rejected(self):
        r=self.run_data();r['checkpoints'][0]['test']=dict(attempts=70,captures=0,clean=0,rejected=0)
        self.save_run(r)
        with self.assertRaisesRegex(ValueError,'only.*final'):summarize(self.root)
    def test_source_camera_split_and_bytes_corruption_rejected(self):
        for field in ('camera_sha256','geometry_fingerprint','scoring_version'):
            r=self.run_data();r['source']=dict(self.source);r['source'][field]='x';self.save_run(r)
            with self.assertRaisesRegex(ValueError,'source|SHA256'):summarize(self.root)
        r=self.run_data();r['source_trial_sha256']={'trials/one.json':'0'*64};self.write(self.root/self.exp['runs'][0]['path']/'trials/one.json',{'dummy':True});self.save_run(r)
        with self.assertRaisesRegex(ValueError,'bytes changed'):summarize(self.root)
    def test_invalid_counts_budget_and_regressing_curves_rejected(self):
        for patch in ({'clean':101},{'rejected':1001},{'captures':True}):
            r=self.run_data();r['train'].update(patch);self.save_run(r)
            with self.assertRaises(ValueError):summarize(self.root)
        r=self.run_data();r['train']['attempts']=10001;self.save_run(r)
        with self.assertRaisesRegex(ValueError,'budget'):summarize(self.root)
        r=self.run_data();r['learning_curve'].append(dict(r['learning_curve'][-1]));self.save_run(r)
        with self.assertRaisesRegex(ValueError,'regressed|duplicated'):summarize(self.root)
    def test_split_coordinate_leakage_rejected_even_if_rehashed(self):
        self.splits['test'][0]['bucket_xy_m']=self.splits['train'][0]['bucket_xy_m'];self.write(self.root/'splits.json',self.splits)
        self.exp['split_manifest_sha256']=sha(self.root/'splits.json');self.save_exp()
        with self.assertRaisesRegex(ValueError,'leakage'):summarize(self.root)
    def test_prediction_membership_and_outcome_counts_checked(self):
        r=self.run_data();layouts=[dict(layout_id=p['id'],bucket_xy_m=p['bucket_xy_m'],requested={'height_m':.3},measured=None,outcome=dict(captured=i<7,clean=i<3,rejected=7<=i<17,reason=None),provenance='Synthetic fixture') for i,p in enumerate(self.splits['validation'])]
        r['checkpoints'][0]['validation']['layouts']=layouts;self.save_run(r);self.assertEqual(len(summarize(self.root)['runs'][0]['checkpoints'][0]['validation']['layouts']),35)
        layouts[0]['layout_id']='unknown';self.save_run(r)
        with self.assertRaisesRegex(ValueError,'split positions'):summarize(self.root)
    def test_traversal_unplanned_runs_and_unprescheduled_video_rejected(self):
        self.exp['runs'][0]['path']='../escape';self.save_exp()
        with self.assertRaisesRegex(ValueError,'Unsafe'):summarize(self.root)
        self.exp['runs'][0]['path']='sweep-seed20261121';self.save_exp();r=self.run_data()
        r['videos']=[dict(file='native.mp4',sha256='0'*64,selection='best-successes',camera_sha256=self.source['camera_sha256'])];self.save_run(r)
        with self.assertRaisesRegex(ValueError,'prescheduled'):summarize(self.root)
    def test_late_atomic_video_index_is_published_without_progress_mutation(self):
        r=self.run_data();self.save_run(r);folder=self.root/self.exp['runs'][0]['path'];movie=folder/'native.mp4';movie.write_bytes(b'synthetic-video-test-only')
        record=dict(file='native.mp4',sha256=sha(movie),selection='prescheduled',camera_sha256=self.source['camera_sha256'],method=r['method'],seed=r['seed'],decode_verified=True)
        self.write(folder/'video-index.json',{'videos':[record]});before=(folder/'progress.json').read_bytes();d=summarize(self.root)
        self.assertEqual(len(d['runs'][0]['videos']),1);self.assertIn('video-index.json',d['runs'][0]['evidence']);self.assertEqual((folder/'progress.json').read_bytes(),before)
    def test_publication_keeps_old_studies_and_never_exposes_raw_files(self):
        out=Path(self.temp.name)/'viewer';d=publish(self.root,out);self.assertTrue((out/'report.json').exists())
        self.assertIsNone(published_file(out,'/experiment.json'));self.assertIsNone(published_file(out,'/%2e%2e/secret'))
        self.assertEqual(published_file(out,'/report.json'),out/'report.json')
        old=dict(d,experiment_id='other');self.write(out/'report.json',old)
        with self.assertRaisesRegex(ValueError,'another study'):publish(self.root,out)
        with self.assertRaisesRegex(ValueError,'outside'):publish(self.root,self.root/'export')
    def test_progress_snapshot_hash_matches_counts_during_atomic_update(self):
        raw=self.run_data();self.save_run(raw);p=self.root/self.exp['runs'][0]['path']/'progress.json';old_sha=sha(p);original=report_module.read_snapshot
        def concurrent(path):
            value,digest=original(path)
            if Path(path).resolve()==p.resolve():
                updated=copy.deepcopy(raw);updated['train']['attempts']=2000;self.save_run(updated)
            return value,digest
        with patch.object(report_module,'read_snapshot',side_effect=concurrent):d=summarize(self.root)
        self.assertEqual(d['runs'][0]['train']['attempts'],1000);self.assertEqual(d['runs'][0]['evidence']['progress.json'],old_sha);self.assertNotEqual(sha(p),old_sha)
    def test_sha_recomputes_same_size_changes_even_with_identical_metadata(self):
        p=self.root/'mutable.txt';p.write_bytes(b'first');a=sha(p);old=p.stat()
        self.assertEqual(sha(p),a)
        # Simulate coarse filesystem timestamps: stat metadata remains identical
        # while different same-size bytes must produce a different digest.
        with patch.object(Path,'stat',return_value=old):
            p.write_bytes(b'other');os.utime(p,ns=(old.st_atime_ns,old.st_mtime_ns))
            b=sha(p);self.assertNotEqual(a,b)
        replacement=self.root/'replacement';replacement.write_bytes(b'third');replacement.replace(p)
        self.assertNotEqual(sha(p),b)
    def test_byte_ranges_for_read_only_video_seeking(self):
        self.assertEqual(byte_range(None,100),(0,99,False));self.assertEqual(byte_range('bytes=10-19',100),(10,19,True));self.assertEqual(byte_range('bytes=-10',100),(90,99,True))
        for value in ('bytes=100-','bytes=0-1,5-6','bytes=-0','garbage'):
            with self.assertRaises(ValueError):byte_range(value,100)

if __name__=='__main__':unittest.main()
