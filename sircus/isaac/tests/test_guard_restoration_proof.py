"""Synthetic source/data fixtures for bounded restored-guard applicability."""

import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
import diverse_pilot_audit as audit
import test_diverse_pilot_audit as audit_fixtures

OLD_CONTROLLER = '''VERSION="fixture-v5"
class SettledTransitShots(ContinuousShots):
 def prepare(self,actions):
  self.spec["cells"]=[dynamic_cell(c,a) for c,a in zip(self.base_cells,actions)]
  invalid=[None]*self.n
  for i,action in enumerate(actions):
   reason=None
   if not self.active_mask[i]:reason="inactive_split"
   elif self.trial_mode!="coverage" and getattr(self,"support_missing",[False]*self.n)[i]:reason="no_geometry_support"
   if reason:invalid[i]=reason
  return np.asarray(releases),normals,records,invalid
'''
HOME_GUARD = '   elif not self.home_readback["path_eligible"][i]:reason="common_home_path_proxy_rejected"\n'
DISTANCE_GUARD = '  for i,cell in enumerate(self.spec["cells"]):\n   if cell["distance_m"] < .28:\n    invalid[i]="board_bucket_separation"\n'
NEW_CONTROLLER = OLD_CONTROLLER.replace('   if reason:', HOME_GUARD+'   if reason:').replace('  return np.asarray', DISTANCE_GUARD+'  return np.asarray')
HOME_SOURCE = '''class ContinuousShots:
 def home(self):
  return dict(path_eligible=[True]*self.n)
'''
RUNNER_SOURCE = '''def run_setup_check(env,bundle,feasibility_path,source):
 for item in prior["batches"][:4]:
  old=read(item)
  env.home_readback=env.home()
  env.active_mask=old["enabled"]
  releases,normals,settled,invalid=env.prepare(old["actions"])
'''
DYNAMIC_SOURCE = '''import copy,math
def dynamic_cell(cell,action):
 c=copy.deepcopy(cell)
 c["board_xy_m"]=[action["board_x_m"],action["board_y_m"]]
 dx,dy=[b-a for a,b in zip(c["board_xy_m"],c["bucket_xy_m"])]
 c["distance_m"]=math.hypot(dx,dy)
 return c
'''


class GuardSourceTests(unittest.TestCase):
    def test_exact_two_nodes_removed_and_whole_ast_equal(self):
        receipt = audit.verify_guard_only_controller_ast(OLD_CONTROLLER, NEW_CONTROLLER)
        self.assertEqual(receipt['removed_guard_nodes'],[1,1])
        self.assertEqual(receipt['old_controller_ast_sha256'],receipt['stripped_new_controller_ast_sha256'])

    def test_guard_threshold_extra_change_duplicate_and_wrong_position_rejected(self):
        cases=[NEW_CONTROLLER.replace('< .28','< .27'),NEW_CONTROLLER.replace('invalid=[None]*self.n','invalid=[None]*(self.n+1)'),
               NEW_CONTROLLER.replace(HOME_GUARD,HOME_GUARD+HOME_GUARD),
               NEW_CONTROLLER.replace(DISTANCE_GUARD,'').replace('  invalid=[None]*self.n',DISTANCE_GUARD+'  invalid=[None]*self.n')]
        for value in cases:
            with self.subTest(value=value),self.assertRaises(ValueError):
                audit.verify_guard_only_controller_ast(OLD_CONTROLLER,value)

    def test_home_flag_is_source_invariant_not_new_observation(self):
        receipt=audit.verify_home_source_invariant(HOME_SOURCE,RUNNER_SOURCE)
        self.assertIn('not an observed',receipt['basis'])
        with self.assertRaises(ValueError):audit.verify_home_source_invariant(HOME_SOURCE.replace('[True]','[False]'),RUNNER_SOURCE)
        changed=RUNNER_SOURCE.replace('  env.home_readback=env.home()\n','').replace('  releases,normals,settled,invalid=env.prepare(old["actions"])','  releases,normals,settled,invalid=env.prepare(old["actions"])\n  env.home_readback=env.home()')
        with self.assertRaises(ValueError):audit.verify_home_source_invariant(HOME_SOURCE,changed)


class GuardApplicabilityTests(unittest.TestCase):
    def fixture(self, root, *, distance_source=DYNAMIC_SOURCE, home_source=HOME_SOURCE):
        old_run,run=audit_fixtures.SetupCheckAuditTests().make_setup(root)
        old_source=root/'snapshot-old'/'sircus'/'isaac';new_source=root/'snapshot-new'/'sircus'/'isaac'
        old_source.mkdir(parents=True);new_source.mkdir(parents=True)
        files={'continuous_pilot.py':home_source,'diverse_pilot.py':RUNNER_SOURCE,
               'film_cylinder_farm.py':'# unchanged native wrapper fixture\n','feasibility_contract.py':distance_source}
        for name,text in files.items():
            (old_source/name).write_text(text);(new_source/name).write_text(text)
        (old_source/'settled_transit.py').write_text(OLD_CONTROLLER)
        (new_source/'settled_transit.py').write_text(NEW_CONTROLLER)
        archive=root/'snapshot-old.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for path in old_source.iterdir():
                data=path.read_bytes();member=tarfile.TarInfo('sircus/isaac/'+path.name);member.size=len(data)
                tar.addfile(member,io.BytesIO(data))
        integrity=dict(archive_sha256=audit.GUARD_PREFLIGHT_ARCHIVE_SHA256,
            source_members_sha256={'sircus/isaac/'+f.name:audit.digest(f) for f in old_source.iterdir()})
        Path(str(run)+'.launch.json').write_text(json.dumps(dict(source_commit=audit.GUARD_PREFLIGHT_COMMIT,integrity=integrity)))
        Path(str(run)+'.supervisor.json').write_text(json.dumps(dict(source_commit=audit.GUARD_PREFLIGHT_COMMIT,status='completed',returncode=0)))
        return old_run,run,old_source,new_source,archive

    def prove(self, data, *, create_prior=True):
        old_run,run,old_source,new_source,archive=data
        actual_digest=audit.digest
        def fake_pinned_fixture_digest(path):
            if Path(path)==old_run/'result.json':return 'cd90e6ec9c1e1bc3729ca5113b67d98c6ebf680f0964cec68065acb1f98138ca'
            if Path(path)==archive:return audit.GUARD_PREFLIGHT_ARCHIVE_SHA256
            return actual_digest(path)
        # Authentication hashes are deliberately mocked ONLY for synthetic
        # fixtures. All source-member byte and guard/data checks remain real.
        with patch.object(audit,'digest',side_effect=fake_pinned_fixture_digest), \
             patch('feasibility_guided_audit.backend_for_run',return_value=SimpleNamespace(validate_study=lambda *args:None)):
            prior=Path(str(run)+'-independent-audit.json')
            if create_prior:prior.write_text(json.dumps(audit.audit_setup_check(run,old_run)))
            return audit.audit_guard_restoration_applicability(run,old_source,new_source,original_run=old_run,old_archive=archive)

    def test_all_100_measured_requests_inert_without_resimulation_or_relabel(self):
        with tempfile.TemporaryDirectory() as name:
            result=self.prove(self.fixture(Path(name).resolve()))
            self.assertEqual(result['checked_active_requests'],100)
            self.assertEqual(result['observed_workcell_count'],35)
            self.assertFalse(result['resimulation'])
            self.assertEqual(result['measured_preflight_source_commit'],audit.GUARD_PREFLIGHT_COMMIT)
            self.assertGreaterEqual(result['recomputed_distance_min_m'],.28)
            self.assertIn('SOURCE INVARIANT',result['scope'])

    def test_other_source_change_and_wrong_native_provenance_rejected(self):
        for key in ('bytes','controller','launch','status','prior'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as name:
                data=self.fixture(Path(name).resolve());old_run,run,old,new,archive=data
                self.prove(data)
                if key=='bytes':(new/'continuous_pilot.py').write_text(HOME_SOURCE+'# even comments must stay byte-identical\n')
                elif key=='controller':(new/'settled_transit.py').write_text(NEW_CONTROLLER.replace('fixture-v5','fixture-v6'))
                elif key=='launch':Path(str(run)+'.launch.json').write_text(json.dumps(dict(source_commit='bad')))
                elif key=='status':Path(str(run)+'.supervisor.json').write_text(json.dumps(dict(source_commit=audit.GUARD_PREFLIGHT_COMMIT,status='completed',returncode=1)))
                else:
                    p=Path(str(run)+'-independent-audit.json');j=json.loads(p.read_text());j['after_admitted']-=1;p.write_text(json.dumps(j))
                with self.assertRaises(ValueError):self.prove(data,create_prior=False)

    def test_real_dynamic_cell_body_must_keep_distance_guard_inert(self):
        changed=DYNAMIC_SOURCE.replace('math.hypot(dx,dy)','.279999')
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError,'recomputed-distance guard'):
                self.prove(self.fixture(Path(name).resolve(),distance_source=changed))


if __name__=='__main__':unittest.main()
