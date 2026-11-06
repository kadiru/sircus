"""Storage/accounting guards only; no simulator, renderer or hardware."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shot_benchmark_native import compressed_json, summary


class EvidenceTests(unittest.TestCase):
    def test_lossless_native_histories_keep_byte_identity(self):
        record=dict(rows=[dict(invalid="endpoint_ik",route_success=False)],
                    settling_history=[[dict(position_mm=1.2,velocity=[0.,.001])]])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"native.json.gz"
            result=compressed_json(p,record)
            original=gzip.decompress(p.read_bytes())
            self.assertEqual(json.loads(original),record)
            self.assertEqual(result["sha256"],hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(result["uncompressed_sha256"],hashlib.sha256(original).hexdigest())
            self.assertFalse(Path(str(p)+".tmp").exists())

    def test_rejects_and_external_misses_remain_in_denominator(self):
        rows=[dict(invalid="ik",route_success=False,route_quality=0,external_contact_steps=0),
              dict(invalid=None,route_success=False,route_quality=0,external_contact_steps=3),
              dict(invalid=None,route_success=True,route_quality=2,external_contact_steps=0)]
        actual=summary(rows,[-1.,-.4,1.6])
        self.assertEqual([actual[k] for k in ("attempts","captures","clean","rejected")],[3,1,1,1])
        self.assertEqual(actual["failures"],dict(ik=1,external_contact=1,captured=1))
        self.assertAlmostEqual(actual["mean_reward"],.2/3)


if __name__=="__main__":unittest.main()
