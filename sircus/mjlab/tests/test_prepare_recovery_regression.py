import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import prepare_recovery_regression as P


def write(path,value):path.write_text(json.dumps(value))

def row(i,release=True,capture=False):
    return dict(record_id=f'r{i}',context_id=f'c{i}',values=[.5,0,.4,25,.2],bucket_xy_m=[.7,0],native_release=release,capture=capture,invalid_physics=False,
        trace=dict(time_s=[j/100 for j in range(241)],position_m=[[.5+j/1000,0,.7-j/1000] for j in range(241)]) if release else None,
        events=[dict(category='board',time_s=.321)])


def diagnosis():
    d=dict(version=P.DIAGNOSIS,status='completed',source=dict(checks_count=283251,full_checks_verified=True,native_checks_verified=True,report_sha256='a'*64,audit_sha256='b'*64,contract_fingerprint='c'*64),
        losses=[dict(context_id=f'c{i}',parent_bc=row(i+100,capture=True),recovery_bc=row(i,release=i<2)) for i in range(5)],
        setup_rejections=[row(i,release=False) for i in range(16)],flight_misses=[row(i) for i in range(17)])
    d['diagnosis_fingerprint']=P.digest(d)
    return d


class Publication(unittest.TestCase):
    def setUp(self):
        mocked=patch.object(P,'corrected_recordings',return_value=[]);mocked.start();self.addCleanup(mocked.stop)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.study=self.root/'study';self.study.mkdir();self.diag=self.root/'diagnosis.json';write(self.diag,diagnosis())
        self.html=self.root/'index.html';self.original='<html><body><div id="history">retained</div></body></html>';self.html.write_text(self.original)
        self.css=self.root/'style.css';self.style='/* historical */ body {color:white}\n';self.css.write_text(self.style)
        self.snapshot=self.root/'snapshot.json';self.audit=self.root/'audit.json'

    def compact(self):return P.compact(self.study,self.diag,audit_path=self.audit,now=123)

    def test_native_samples_are_subset_not_interpolated_and_stay_bounded(self):
        r=row(1);out=P.compact_row(r);self.assertLessEqual(len(out['trace']['time_s']),100)
        self.assertEqual(out['trace']['time_s'][0],0);self.assertEqual(out['trace']['time_s'][-1],2.4)
        for t,p in zip(out['trace']['time_s'],out['trace']['position_m']):self.assertEqual(p,r['trace']['position_m'][r['trace']['time_s'].index(t)])
        self.assertLess(len(P.encoded(self.compact()).encode()),P.MAX_SNAPSHOT_BYTES)
        r['native_release']=False
        with self.assertRaisesRegex(ValueError,'Unflown'):P.compact_row(r)

    def test_diagnosis_hash_native_audit_and_exact_groups_required(self):
        d=diagnosis();d['losses'].pop();write(self.diag,d)
        with self.assertRaises(ValueError):self.compact()
        d['diagnosis_fingerprint']=P.digest({k:v for k,v in d.items() if k!='diagnosis_fingerprint'});write(self.diag,d)
        with self.assertRaisesRegex(ValueError,'five-loss'):self.compact()
        d=diagnosis();d['source']['native_checks_verified']=False;d['diagnosis_fingerprint']=P.digest({k:v for k,v in d.items() if k!='diagnosis_fingerprint'});write(self.diag,d)
        with self.assertRaisesRegex(ValueError,'native audit'):self.compact()

    def completed(self):
        contract=dict(version=P.STUDY,protocol=dict(substeps=64));contract['fingerprint']=P.digest(contract);write(self.study/'contract.json',contract)
        raw=dict(version=P.STUDY,status='completed',phase='completed',expected_requests=63,contract_fingerprint=contract['fingerprint'],rule=dict(diagnosis_sha256=P.sha(self.diag)),counts=dict(attempts=63),cases=[],control_gates={'pre':{'status':'passed'},'post':{'status':'passed'}})
        raw['evidence_sha256']={};write(self.study/'evidence-manifest.json',{});write(self.study/'report.json',raw);write(self.audit,dict(evidence_manifest_sha256=P.sha(self.study/'evidence-manifest.json'),version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,failures=[],checks=[dict(passed=True)],checks_count=1,report_sha256=P.sha(self.study/'report.json'),contract_fingerprint=contract['fingerprint']))

    def test_audit_requires_current_bytes_all_checks_and_native_verification(self):
        self.completed();self.assertEqual(self.compact()['correction']['audit']['status'],'passed')
        p=self.study/'report.json';p.write_bytes(p.read_bytes()+b' ');self.assertEqual(self.compact()['correction']['audit']['status'],'unverified')
        self.completed();a=P.read(self.audit);a['native_checks_verified']=False;write(self.audit,a);self.assertEqual(self.compact()['correction']['audit']['status'],'unverified')

    def test_preserves_every_byte_outside_own_markers_and_escapes_html_css(self):
        raw=self.compact();raw['note']='*/</script><script>bad</script>'
        html=P.embed_html(self.original,raw);css=P.embed_css(self.style,raw)
        self.assertEqual(html.split(P.START)[0]+html.split(P.END)[1],self.original)
        self.assertEqual(css.split(P.CSS_START)[0]+css.split(P.CSS_END)[1],self.style)
        self.assertNotIn('*/</script>',html);self.assertNotIn('*/</script>',css)
        self.assertEqual(P.embed_html(html,raw),html);self.assertEqual(P.embed_css(css,raw),css)
        with self.assertRaisesRegex(ValueError,'Ambiguous'):P.embed_css(css+P.CSS_START,raw)
        with self.assertRaisesRegex(ValueError,'Reversed'):P.embed_html(P.END+P.START,raw)

    def test_refuses_changed_surface_or_source_target_before_writing(self):
        args=dict(snapshot_path=self.snapshot,target_html=self.html,target_css=self.css)
        with self.assertRaisesRegex(ValueError,'HTML changed'):P.refresh(self.study,self.diag,**args,expected_html_sha256='0'*64)
        self.assertFalse(self.snapshot.exists());self.assertEqual(self.css.read_text(),self.style)
        with self.assertRaisesRegex(ValueError,'CSS changed'):P.refresh(self.study,self.diag,**args,expected_css_sha256='0'*64)
        args['snapshot_path']=self.diag
        with self.assertRaisesRegex(ValueError,'immutable'):P.refresh(self.study,self.diag,**args)

    def test_pending_display_does_not_invent_correction(self):
        out=P.refresh(self.study,self.diag,snapshot_path=self.snapshot,target_html=self.html,target_css=self.css,expected_html_sha256=P.sha(self.html),expected_css_sha256=P.sha(self.css),now=123)
        self.assertEqual(out['correction']['status'],'pending');self.assertEqual(out['correction']['cases'],[]);self.assertFalse(out['publication']['terminal'])
        self.assertEqual(P.read(self.snapshot),out)

if __name__=='__main__':unittest.main()
