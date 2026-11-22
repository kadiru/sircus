import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sircus.mjlab import hybrid_miss_video as V


class ReplayContractTests(unittest.TestCase):
    def test_clock_samples_and_terminal_are_exact_no_interpolation(self):
        times=np.arange(20001)/15360
        bundle=dict(time_s=times,scored_until_s=float(times[-1]),events={'robot':float(times[7123])})
        frames=V.frame_plan(bundle)
        self.assertTrue(all(x['source_time_s']==times[x['source_tick']] for x in frames))
        self.assertTrue(all(0<=x['source_tick']<=20000 for x in frames))
        holds=[x for x in frames if x['segment']=='robot_contact_hold']
        self.assertEqual(len(holds),24)
        self.assertTrue(all(x['source_tick']==7123 for x in holds))
        self.assertEqual(frames[-1]['source_tick'],20000)
        slow=[x for x in frames if x['segment']=='slow']
        self.assertEqual(slow[1]['source_tick']-slow[0]['source_tick'],128)

    def test_pending_audit_does_not_render(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name,value in [('report',{'status':'completed'}),('contract',{}),('evidence-manifest',{}),('audit',{'status':'pending'})]:
                (p/(name+'.json')).write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'independently audited'):
                V.verify_envelope(p,p/'audit.json')

    def test_cpu_guard_rejects_nvidia_before_rendering(self):
        for renderer in ('NVIDIA GeForce RTX4090','NVIDIA GeForce RTX4070'):
            with self.assertRaises(ValueError):V.software_backend(renderer,'NVIDIA','')
        with self.assertRaises(ValueError):V.software_backend('llvmpipe','Mesa','GPU-4090')
        self.assertTrue(V.software_backend('llvmpipe','Mesa','')['software_only'])

    def test_outcome_does_not_relabel_contact_as_capture(self):
        b=dict(events={'robot':.4,'bucket':.6},row={'route_success':False})
        self.assertIn('robot',V.stage(dict(source_time_s=.4,segment='slow'),b))
        self.assertEqual(V.stage(dict(source_time_s=1,segment='outcome'),b),'MISS / no accepted capture')

if __name__=='__main__':unittest.main()
