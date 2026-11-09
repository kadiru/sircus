"""Synthetic saved-evidence controls; no simulator, training or GPU access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import shutil

import numpy as np

from sircus.mjlab import calibrated_score as old
from sircus.mjlab import table_assisted_rescore as rescore
from sircus.mjlab.tests.test_calibrated_score import config, trace, ROOT


class SavedRescoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/"evidence"; self.root.mkdir()
        self.study = self.root/"study"; self.study.mkdir(); (self.study/"native").mkdir()
        self.scene = json.loads((ROOT/"sircus/isaac/scene.json").read_text())
        self.cfg = config(self.scene); self.xy = self.scene["bucket"]["center_xy_m"]
        self.contract = dict(scene=self.scene, identity=dict(scene_sha256=rescore.digest(self.scene),
            calibration=dict(score_config=self.cfg,score_config_sha256=rescore.digest(self.cfg))))
        self.contract["fingerprint"] = rescore.digest(self.contract)
        (self.study/"contract.json").write_text(json.dumps(self.contract))
        dt=self.scene["simulation"]["physics_dt_s"]/64
        a = trace(self.scene,dt=dt); b = copy.deepcopy(a); b["forces_N"][round(.1/dt),2] = 2.
        self.ids = ["direct", "table"]
        self.cache = self.study/"native"/"native-0000.npz"
        np.savez_compressed(self.cache,record_ids=np.array(self.ids),time_s=a["time_s"],
            position_m=np.stack([a["position_m"],b["position_m"]],axis=1),
            velocity_m_s=np.stack([a["velocity_m_s"],b["velocity_m_s"]],axis=1),
            forces_N=np.stack([a["forces_N"],b["forces_N"]],axis=1))
        self.rows=[]
        for i,t in enumerate((a,b)):
            r=old.score_trace(self.scene,self.xy,[0.,0.,1.],self.cfg,t)
            r.update(id=self.ids[i],native_release=True,native_cache_file=self.cache.name,
                native_cache_row=i,native_cache_sha256=rescore.sha(self.cache),
                bucket_xy_m=self.xy,board_normal=[0.,0.,1.],values=[.5,0,.3,30,.2],
                phase="test",method="fixture",contact_protocol=dict(substeps=64))
            self.rows.append(r)
        self.rows.append(dict(id="rejected", native_release=False, invalid="finite_ik_unresolved",phase="test",method="fixture",substeps=64))
        self.rows.append(dict(id="legacy",scoring_version="historical-v5",phase="old",method="fixture",substeps=128))
        self.save()

    def save(self, alias=True):
        (self.study/"records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in self.rows))
        if alias:
            (self.study/"native"/"shots.jsonl").write_text("".join(json.dumps(r)+"\n" for r in self.rows[:2]))

    def execute(self, name="result"):
        return rescore.run(self.root,Path(self.tmp.name)/name,sources=["study"],workers=1)

    def test_native_rescore_changes_only_table_credit_and_preserves_denominators(self):
        original=(self.study/"records.jsonl").read_bytes()
        result=self.execute();t=result["totals"]
        self.assertEqual(result["status"],"verified")
        self.assertEqual(result["metadata"],dict(files=2,rows=6,duplicate_rows=2,unique_requests=4))
        self.assertEqual({k:t[k] for k in ("requested","scorable_native","rejected","unscorable","old_captures","new_captures","new_table_assisted_captures","losses")},
            dict(requested=4,scorable_native=2,rejected=1,unscorable=1,old_captures=1,new_captures=2,new_table_assisted_captures=1,losses=0))
        self.assertEqual({g["substeps"] for g in result["groups"]},{64,128})
        self.assertEqual((self.study/"records.jsonl").read_bytes(),original)
        audit=rescore.read(Path(self.tmp.name)/"result/audit.json")
        self.assertTrue(audit["full_checks_verified"])
        self.assertEqual(audit["checks_count"],len(audit["checks"]))
        self.assertTrue(all(c["passed"] for c in audit["checks"]))
        self.assertEqual(audit["new_trials"],0)
        self.assertEqual(audit["summary_sha256"],rescore.sha(Path(self.tmp.name)/"result/summary.json"))

    def test_changed_native_bytes_fail_closed(self):
        with self.cache.open("ab") as f:f.write(b"changed")
        result=self.execute()
        self.assertEqual(result["status"],"completed_with_errors")
        self.assertEqual(result["totals"]["evidence_errors"],2)
        self.assertEqual(result["totals"]["new_captures"],0)

    def test_same_cache_alias_cannot_change_geometry(self):
        aliases=copy.deepcopy(self.rows[:2]);aliases[1]["bucket_xy_m"]=[99.,99.]
        (self.study/"native/shots.jsonl").write_text("".join(json.dumps(r)+"\n" for r in aliases))
        result=self.execute()
        self.assertEqual(result["totals"]["evidence_errors"],1)
        self.assertEqual(result["totals"]["new_table_assisted_captures"],0)

    def test_wrong_archive_record_id_and_old_labels_fail_closed(self):
        self.rows[0]["id"]="wrong-id";self.rows[1]["reward"]+=.5;self.save()
        result=self.execute()
        self.assertEqual(result["totals"]["evidence_errors"],2)

    def test_scene_hash_and_calibration_must_match_contract(self):
        c=copy.deepcopy(self.contract);c["identity"]["scene_sha256"]="0"*64;c.pop("fingerprint");c["fingerprint"]=rescore.digest(c)
        (self.study/"contract.json").write_text(json.dumps(c))
        indexed=rescore.index(self.root,sources=["study"])
        self.assertTrue(all(r["status"]=="evidence_error" for r in indexed["rows"]))

    def test_index_only_does_not_open_dense_arrays_and_missing_old_versions_stay_unknown(self):
        with mock.patch.object(np,"load",side_effect=AssertionError("native read")):
            indexed=rescore.run(self.root,Path(self.tmp.name)/"index",sources=["study"],index_only=True)
        self.assertEqual(indexed["status"],"index_only")
        self.assertEqual(indexed["by_status"]["pending_native"],2)
        self.assertEqual(indexed["pending_archives"],1)

    def test_contact_at_terminal_or_after_does_not_enter_example_evidence(self):
        t=dict(time_s=np.array([0.,1.,2.,3.]),forces_N=np.zeros((3,6)))
        t["forces_N"][1,2]=1.;t["forces_N"][2,4]=1.
        e=rescore.events(t,dict(physics_steps=1,capture_time_s=1.,max_retention_s=1.))
        self.assertIsNone(e["first_table_contact_s"])
        self.assertIsNone(e["first_robot_contact_s"])

    def test_refuses_existing_output(self):
        out=Path(self.tmp.name)/"existing";out.mkdir()
        with self.assertRaises(rescore.EvidenceError):rescore.run(self.root,out,sources=["study"])

    def test_same_named_requests_in_independent_studies_are_not_identity_conflicts(self):
        other=self.root/"other";shutil.copytree(self.study,other)
        cache=other/"native/native-0000.npz"
        with cache.open("ab") as f:f.write(b"independent archive identity")
        for name in ("records.jsonl","native/shots.jsonl"):
            rows=[json.loads(line) for line in (other/name).read_text().splitlines()]
            for r in rows:
                if r.get("native_cache_sha256"):r["native_cache_sha256"]=rescore.sha(cache)
            (other/name).write_text("".join(json.dumps(r)+"\n" for r in rows))
        indexed=rescore.index(self.root)
        self.assertEqual(sum(r["status"]=="pending_native" for r in indexed["rows"]),4)
        self.assertFalse(any(r["status"]=="evidence_error" for r in indexed["rows"]))

    def test_identical_copied_native_cache_across_studies_does_not_multiply_trials(self):
        shutil.copytree(self.study,self.root/"copied")
        indexed=rescore.index(self.root)
        self.assertEqual(sum(r["status"]=="pending_native" for r in indexed["rows"]),2)

    def test_step_specific_native_archive_layout_is_resolved_and_still_hash_checked(self):
        target=self.cache.parent/"64";target.mkdir();self.cache.rename(target/self.cache.name)
        result=self.execute()
        self.assertEqual(result["totals"]["scorable_native"],2)
        self.assertEqual(result["totals"]["new_captures"],2)


if __name__=="__main__":unittest.main()
