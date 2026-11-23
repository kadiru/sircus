import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab.benchmark_verify import sha
from sircus.mjlab.link_clearance_publish import AUDIT_VERSION, BINDINGS, CASES, VERSION, publish


class LinkClearancePublishTests(unittest.TestCase):
    def fixture(self, path):
        study = path/'study'; study.mkdir()
        parent = path/'parent'; parent.mkdir()
        parent_audit = path/'parent-audit.json'; parent_audit.write_text('{}')
        external_diagnosis = path/'prior-diagnosis.json'; external_diagnosis.write_text('{}')
        paths = dict(parent_study=str(parent), parent_audit=str(parent_audit), diagnosis=str(external_diagnosis))
        cases = [dict(id=id_, bucket_xy_m=[1.+i*.1, 0.],
                      results={group: dict(rows=['raw']) for group in ('original', 'search', 'repeat')},
                      repeated_settings=[], confirmed_settings=[]) for i, id_ in enumerate(CASES)]
        request = dict(id='shot-1', context_id=CASES[0], bucket_xy_m=[1., 0.], values=[.6, 0., .4, 28., .12],
                       selected_q10=[0.]*6, selected_q14=[0.]*6, board_heading_offset_deg=0., board_twist_deg=0.,
                       context_resolution_file='/private/raw-proof.json')
        setting = dict(candidate_id='shot-1', request=request, repeat_captures=3, repeat_completed=3)
        cases[0]['repeated_settings'] = [setting]; cases[0]['confirmed_settings'] = [setting]
        control = dict(status='passed', direct_captures=3, table_assisted_captures=3, negative_captures=0)
        report = dict(version=VERSION, status='completed', phase='completed', scope='development', physics_substeps=64,
                      cases=cases, initial_requests=8, confirmation_requests=3, native_releases=38,
                      controls={phase: control for phase in ('controls_pre', 'controls_post')},
                      preserved_shots=dict(requested=9, completed=9, captures=9, passed=True, rows=['raw']))
        diagnosis = dict(diagnosis_rows=[dict(id=row['id'], bucket_xy_m=row['bucket_xy_m'], trials=20, robot_contacts=20,
                         first_contact_link_counts={'Arm14/link3/collision/0': 20}, first_contact_time_s=dict(min=.4, max=.6),
                         method='native_force_cpu_reconstruction') for row in cases])
        data = {'report.json':report, 'contract.json':dict(source_paths=paths, maximum_total_requests=289),
                'preparation.json':{}, 'source-bank.json':[], 'evidence-manifest.json':{}, 'diagnosis.json':diagnosis}
        for name, value in data.items(): (study/name).write_text(json.dumps(value))
        audit = path/'audit.json'; self.receipt(study, audit)
        return study, audit, paths

    def receipt(self, study, audit):
        audit.write_text(json.dumps(dict(version=AUDIT_VERSION, status='passed', report_sha256=sha(study/'report.json'),
            evidence_manifest_sha256=sha(study/'evidence-manifest.json'), **{field:sha(study/name) for field,name in BINDINGS.items()})))

    def test_preserves_existing_display_and_sources_and_removes_raw_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); study,audit,_=self.fixture(path)
            before={p.name:p.read_bytes() for p in study.iterdir()}
            display=path/'display.json'; display.write_text('{"nearby_shot_transfer":{"keep":true},"media":{"clip":"unchanged"}}')
            publish(study,audit,display); data=json.loads(display.read_text()); section=data['remaining_link_clearance']
            self.assertEqual(data['nearby_shot_transfer'], dict(keep=True)); self.assertEqual(data['media'], dict(clip='unchanged'))
            self.assertEqual(section['audit_status'],'passed'); self.assertEqual(len(section['diagnosis_rows']),2)
            self.assertNotIn('rows',section['preserved_shots']); self.assertNotIn('rows',section['cases'][0]['results']['search'])
            self.assertNotIn('/private/',json.dumps(section)); self.assertEqual(before,{p.name:p.read_bytes() for p in study.iterdir()})

    def test_stale_report_diagnosis_and_proofs_are_rejected(self):
        for filename in ('report.json','diagnosis.json','preparation.json','contract.json','source-bank.json','evidence-manifest.json'):
            with self.subTest(filename=filename),tempfile.TemporaryDirectory() as directory:
                path=Path(directory);study,audit,_=self.fixture(path)
                with (study/filename).open('a') as stream:stream.write(' ')
                with self.assertRaises(ValueError):publish(study,audit,path/'display.json')

    def test_failed_preservation_controls_and_wrong_scope_are_withheld(self):
        for field in ('preservation','controls','counts','case','substeps','diagnosis','audit_version'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
                path=Path(directory);study,audit,_=self.fixture(path); report=json.loads((study/'report.json').read_text())
                if field=='preservation':report['preserved_shots']['captures']=8
                elif field=='controls':report['controls']['controls_post']['negative_captures']=1
                elif field=='counts':report['native_releases']+=1
                elif field=='case':report['cases'].pop()
                elif field=='substeps':report['physics_substeps']=128
                elif field=='diagnosis':
                    diagnosis=json.loads((study/'diagnosis.json').read_text());diagnosis['diagnosis_rows'][0]['method']='native_link_label'
                    (study/'diagnosis.json').write_text(json.dumps(diagnosis))
                (study/'report.json').write_text(json.dumps(report));self.receipt(study,audit)
                if field=='audit_version':
                    data=json.loads(audit.read_text());data['version']='unrelated';audit.write_text(json.dumps(data))
                with self.assertRaises(ValueError):publish(study,audit,path/'display.json')

    def test_original_evidence_and_symlink_destinations_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);study,audit,paths=self.fixture(path)
            for destination in (study/'display.json',audit,Path(paths['parent_study'])/'display.json',Path(paths['parent_audit']),Path(paths['diagnosis'])):
                with self.assertRaises(ValueError):publish(study,audit,destination)
            display=path/'display.json';display.with_suffix('.json.tmp').symlink_to(study/'new.json')
            with self.assertRaises(ValueError):publish(study,audit,display)


if __name__=='__main__':unittest.main()
