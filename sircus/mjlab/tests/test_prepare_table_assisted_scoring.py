import copy
import hashlib
import json
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sircus.mjlab import prepare_table_assisted_scoring as P

H='a'*64
def dump(path,value):path.write_text(json.dumps(value,allow_nan=False)+'\n')

class TableScoringPublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.study=self.root/'study';self.study.mkdir();self.html=self.root/'index.html'
        self.html.write_text('<html><body><section id="old">Previous results</section><!--TABLE_ASSISTED_VIDEO--><video src="old.mp4"></video></body></html>')
        counts=dict(requested=4,scorable_native=2,rejected=1,unscorable=1,evidence_errors=0,invalid_native=0,old_captures=1,new_captures=2,new_table_assisted_captures=1,losses=0,unchanged_misses=0,old_reward_sum=3.1,new_reward_sum=6.6,reward_count=2)
        events=dict(first_board_contact_s=.137,first_table_contact_s=.1,whole_ball_entry_s=.5,capture_s=2.1)
        common=dict(source_study='saved-study',method='ppo',phase='train',substeps=64)
        self.rows=[dict(id='gained',status='scored',old_score=dict(route_success=False,reward=.1),new_score=dict(route_success=True,reward=3.6),events=events,native_sha256=H,native_row=0,**common),dict(id='kept',status='scored',old_score=dict(route_success=True,reward=3.),new_score=dict(route_success=True,reward=3.),events={},native_sha256=H,native_row=1,**common),dict(id='rejected',status='rejected',**common),dict(id='old',status='unscorable',**common)]
        (self.study/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        contract=dict(version=P.SOURCE_VERSION,no_new_native_steps=True,no_training=True,source_sha256={'table_assisted_score.py':H},old_scoring_version='old-v2',new_scoring_version='table-v3');dump(self.study/'contract.json',contract)
        manifest=dict(inputs={'source/records.jsonl':H,'source/native.npz':'b'*64},artifacts={n:P.sha(self.study/n) for n in ('records.jsonl','contract.json')});dump(self.study/'manifest.json',manifest)
        example=dict(id='gained',old_capture=False,new_capture=True,old_reward=.1,new_reward=3.6,events=events,native_sha256=H,native_row=0,**common)
        summary=dict(version=P.SOURCE_VERSION,status='verified',totals=counts,metadata=dict(rows=5,duplicate_rows=1,unique_requests=4),groups=[dict(counts=counts,**common)],examples=[example],exclusions={'incompatible_legacy_scorer':1},contract_sha256=P.sha(self.study/'contract.json'),manifest_sha256=P.sha(self.study/'manifest.json'));dump(self.study/'summary.json',summary)
        proof=dict(version=P.AUDIT_VERSION,status='passed',full_checks_verified=True,new_trials=0,new_training_requests=0,failures=[],checks=[dict(label='All native rows and inputs checked',passed=True)],checks_count=1,summary_sha256=P.sha(self.study/'summary.json'),manifest_sha256=P.sha(self.study/'manifest.json'),contract_sha256=P.sha(self.study/'contract.json'),totals=counts,groups=summary['groups']);dump(self.study/'audit.json',proof)
    def test_compact_counts_denominators_and_noncausal_event_times(self):
        out=P.compact(self.study);self.assertEqual(out['totals']['new_table_assisted_captures'],1);self.assertEqual(out['totals']['retained_rejections'],1);self.assertEqual(out['totals']['unscorable'],1);self.assertEqual(out['groups'][0]['substeps'],64)
        self.assertEqual(out['examples'][0]['table_contact_s'],.1);self.assertEqual(out['examples'][0]['first_entry_s'],.5);self.assertEqual(out['examples'][0]['capture_s'],2.1);self.assertNotIn(str(self.root),json.dumps(out));self.assertEqual(out['new_trials'],0)
    def test_incomplete_or_mismatched_proof_cannot_publish(self):
        path=self.study/'audit.json';before=P.read(path)
        for change in (lambda d:d.update(summary_sha256=H),lambda d:d.update(manifest_sha256=H),lambda d:d.update(contract_sha256=H),lambda d:d.update(full_checks_verified=False),lambda d:d.update(checks_count=2),lambda d:d['checks'][0].update(passed=False),lambda d:d.update(failures=['bad'])):
            bad=copy.deepcopy(before);change(bad);dump(path,bad)
            with self.assertRaises(ValueError):P.compact(self.study)
        dump(path,before)
    def test_changed_derivative_or_summary_rejected(self):
        path=self.study/'records.jsonl';path.write_text(path.read_text()+'{}\n')
        with self.assertRaises(ValueError):P.compact(self.study)
    def test_accounting_rejects_invalid_as_miss_or_losses(self):
        c=P.read(self.study/'summary.json')['totals'];P.check_counts(c)
        for change in (lambda x:x.update(invalid_native=1),lambda x:x.update(losses=1),lambda x:x.update(new_captures=3),lambda x:x.update(requested=5),lambda x:x.update(reward_count=3)):
            bad=dict(c);change(bad)
            with self.assertRaises(ValueError):P.check_counts(bad)
    def test_own_block_replacement_preserves_history_video_and_escapes_script_content(self):
        old=self.html.read_text();snapshot=P.compact(self.study);snapshot['exclusion_summary']='</script><script>unsafe</script>'
        rendered=P.embed_html(old,snapshot);self.assertIn(old.split('</body>')[0],rendered);self.assertIn('old.mp4',rendered);self.assertNotIn('</script><script>unsafe',rendered);self.assertIn('\\u003c/script',rendered)
        twice=P.embed_html(rendered,snapshot);self.assertEqual(twice,rendered);self.assertEqual(twice.count(P.START),1)
        with self.assertRaises(ValueError):P.embed_html(old+P.START,snapshot)
    def test_embedded_script_is_classic_scoped_and_keeps_module_source_exports(self):
        block=P.render_block(P.compact(self.study));script=re.search(r'<script>([\s\S]+?)</script>',block).group(1)
        self.assertTrue(script.startswith("(function(){'use strict';"));self.assertTrue(script.endswith('})();'))
        self.assertNotRegex(script,r'(?m)^\s*(?:export|import)\b');self.assertNotIn('type="module"',block)
        source=Path(P.__file__).with_name('web').joinpath('table-assisted-scoring.js').read_text()
        self.assertIn('export function renderTableAssistedScoring',source)
    def test_install_requires_exact_original_and_preserves_source_files(self):
        before={p.name:p.read_bytes() for p in self.study.iterdir()};oldhash=P.sha(self.html);P.install(self.html,self.study,oldhash)
        self.assertIn(P.START,self.html.read_text());self.assertIn('old.mp4',self.html.read_text());self.assertEqual(before,{p.name:p.read_bytes() for p in self.study.iterdir()})
        with self.assertRaises(ValueError):P.install(self.html,self.study,oldhash)
    def test_concurrent_html_edit_survives(self):
        original=P.embed_html
        def changed(text,snapshot):self.html.write_text('another writer');return original(text,snapshot)
        with patch.object(P,'embed_html',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'Concurrent'):P.install(self.html,self.study,P.sha(self.html))
        self.assertEqual(self.html.read_text(),'another writer');self.assertEqual(list(self.root.glob('.table-assisted-scoring-*.tmp')),[])
    def test_actual_rescore_output_matches_publisher_without_relabeling(self):
        from sircus.mjlab.tests.test_table_assisted_rescore import SavedRescoreTests
        source=SavedRescoreTests();source.setUp()
        try:
            summary=source.execute();out=P.compact(Path(source.tmp.name)/'result')
            self.assertEqual(out['totals']['new_captures'],summary['totals']['new_captures'])
            self.assertEqual({g['substeps'] for g in out['groups']},{64,128})
            self.assertEqual(len(out['examples']),1);self.assertTrue(out['audit']['full_checks_verified'])
        finally:source.doCleanups()

if __name__=='__main__':unittest.main()
