"""Saved-evidence controls for new labels; never starts simulator or GPU."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from sircus.mjlab import board_first_rescore as rescore
from sircus.mjlab import table_assisted_score as old
from sircus.mjlab.tests.test_calibrated_score import config,trace,ROOT


class BoardFirstRescoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/"sources";self.study=self.root/"study"
        self.native=self.study/"native";self.native.mkdir(parents=True)
        self.scene=json.loads((ROOT/"sircus/isaac/scene.json").read_text())
        self.cfg=config(self.scene);self.xy=self.scene["bucket"]["center_xy_m"]
        contract=dict(scene=self.scene,identity=dict(scene_sha256=rescore.digest(self.scene),
            calibration=dict(score_config=self.cfg,score_config_sha256=rescore.digest(self.cfg))))
        contract["fingerprint"]=rescore.digest(contract)
        (self.study/"contract.json").write_text(json.dumps(contract))
        self.dt=self.scene["simulation"]["physics_dt_s"]/64
        direct=trace(self.scene,dt=self.dt)
        robot=copy.deepcopy(direct);robot["forces_N"][round(.1/self.dt),4]=2.
        before=copy.deepcopy(direct);before["forces_N"][round(.01/self.dt),2]=2.
        self.traces=[direct,robot,before];self.ids=["direct","robot-after","table-before"]
        self.cache=self.native/"native-0000.npz"
        np.savez_compressed(self.cache,record_ids=np.array(self.ids),time_s=direct["time_s"],
            position_m=np.stack([t["position_m"] for t in self.traces],axis=1),
            velocity_m_s=np.stack([t["velocity_m_s"] for t in self.traces],axis=1),
            forces_N=np.stack([t["forces_N"] for t in self.traces],axis=1))
        self.rows=[]
        for index,t in enumerate(self.traces):
            row=old.score_trace(self.scene,self.xy,[0,0,1],self.cfg,t)
            row.update(id=self.ids[index],native_release=True,native_cache_file=self.cache.name,
                       native_cache_row=index,native_cache_sha256=rescore.sha(self.cache),
                       bucket_xy_m=self.xy,board_normal=[0,0,1],values=[.5,0,.3,30,.2],
                       phase="train",method="fixture",contact_protocol=dict(substeps=64))
            self.rows.append(row)
        self.save()

    def save(self):
        (self.native/"shots.jsonl").write_text("".join(json.dumps(r)+"\n" for r in self.rows))

    def run_study(self,name="result"):
        return rescore.run(self.root,Path(self.temp.name)/name,sources=["study"],workers=1)

    def test_gain_and_loss_both_recorded_original_files_unchanged(self):
        before=(self.native/"shots.jsonl").read_bytes()
        summary=self.run_study()
        self.assertEqual(summary["status"],"verified")
        c=summary["totals"]
        self.assertEqual((c["scorable_native"],c["old_captures"],c["new_captures"],c["gained_captures"],
                          c["lost_captures"],c["unchanged_captures"]),(3,2,2,1,1,1))
        self.assertEqual((self.native/"shots.jsonl").read_bytes(),before)
        self.assertEqual(len(summary["buckets"]),1)
        self.assertEqual({r["id"] for r in summary["changes"]},{"robot-after","table-before"})
        rows=[json.loads(x) for x in (Path(self.temp.name)/"result/records.jsonl").read_text().splitlines()]
        self.assertEqual(rows[1]["original"]["route_success"],False)
        self.assertEqual(rows[1]["new_score"]["route_success"],True)
        audit=json.loads((Path(self.temp.name)/"result/audit.json").read_text())
        self.assertTrue(audit["full_checks_verified"])
        self.assertEqual(audit["summary_sha256"],rescore.sha(Path(self.temp.name)/"result/summary.json"))

    def test_copied_archives_do_not_double_count(self):
        shutil.copytree(self.study,self.root/"copied")
        result=rescore.run(self.root,Path(self.temp.name)/"result",sources=["study","copied"],workers=1)
        self.assertEqual(result["metadata"]["duplicate_rows"],3)
        self.assertEqual(result["totals"]["scorable_native"],3)

    def test_conflicting_alias_fails(self):
        shutil.copytree(self.study,self.root/"copied")
        rows=copy.deepcopy(self.rows);rows[0]["route_success"]=False
        (self.root/"copied/native/shots.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        result=rescore.run(self.root,Path(self.temp.name)/"result",sources=["study","copied"],workers=1)
        self.assertEqual(result["totals"]["evidence_errors"],1)

    def test_corrupted_archive_cannot_gain_credit(self):
        with self.cache.open("ab") as out:out.write(b"corruption")
        result=self.run_study()
        self.assertEqual(result["totals"]["evidence_errors"],3)
        self.assertEqual(result["totals"]["new_captures"],0)

    def test_wrong_record_identity_or_original_score_fails(self):
        self.rows[0]["id"]="wrong"
        self.rows[1]["reward"]+=1
        self.save()
        result=self.run_study()
        self.assertEqual(result["totals"]["evidence_errors"],2)

    def test_missing_archive_and_unsupported_versions_explicit(self):
        self.rows[0]["native_cache_file"]="missing.npz"
        self.rows[1]["scoring_version"]="unsupported"
        self.save()
        result=self.run_study()
        self.assertEqual(result["totals"]["unscorable"],2)
        self.assertEqual(result["totals"]["scorable_native"],1)

    def test_index_only_and_explicit_sources(self):
        with mock.patch.object(np,"load",side_effect=AssertionError("No arrays while indexing")):
            result=rescore.run(self.root,Path(self.temp.name)/"index",sources=["study"],index_only=True)
        self.assertEqual(result["statuses"]["pending"],3)
        with self.assertRaises(ValueError):rescore.index(self.root,[])

    def test_existing_derivative_not_recounted(self):
        out=self.root/"derivative"
        rescore.run(self.root,out,sources=["study"],workers=1)
        result=rescore.run(self.root,Path(self.temp.name)/"second",sources=["study","derivative"],workers=1)
        self.assertEqual(result["totals"]["scorable_native"],3)
        self.assertEqual(result["metadata"]["skipped_derivative_files"],1)


if __name__=="__main__":unittest.main()
