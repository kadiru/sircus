import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from sircus.mjlab import gap_coverage_candidates as gap


def original_bank(anchor_gap=.20):
    anchor = dict(values=[.6, -.3, .28, 28., anchor_gap])
    residual = [.15, -.2, .1, .25, .2]
    items = []
    for i in range(16):
        rr = residual.copy()
        if i == 8: rr[4] -= .3
        if i == 9: rr[4] += .3
        values = (np.asarray(anchor['values']) + np.asarray(rr)*gap.RESIDUAL_SCALE).tolist()
        items.append(dict(id=f'candidate-{i:02d}', values=values, normalized_residual=rr,
                          source_kind={8:'bc_axis_4_-0.3',9:'bc_axis_4_0.3'}.get(i,'original_family'),
                          source_training_ids=['train-a'] if i == 3 else []))
    return items, anchor


class GapCoverageTests(unittest.TestCase):
    def test_only_two_gap_slots_change_and_fourteen_rows_stay_exact(self):
        rows, anchor = original_bank(); before = copy.deepcopy((rows,anchor))
        out = gap.candidates(rows,anchor)
        self.assertEqual(len(out),16)
        self.assertEqual([x['id'] for x in out],[x['id'] for x in rows])
        for i in range(16):
            self.assertEqual(out[i]['values'][:4],rows[i]['values'][:4])
            self.assertEqual(out[i]['normalized_residual'][:4],rows[i]['normalized_residual'][:4])
            if i not in (8,9):self.assertEqual(out[i],rows[i])
        self.assertEqual((rows,anchor),before)
        self.assertEqual(out[8]['values'][4],anchor['values'][4]-.12)
        self.assertEqual(out[9]['values'][4],anchor['values'][4]+.12)
        self.assertEqual(out[8]['normalized_residual'][4],-1.)
        self.assertEqual(out[9]['normalized_residual'][4],1.)
        for i in (8,9):self.assertEqual(out[i]['gap_coverage']['original_candidate'],rows[i])
        json.dumps(out,allow_nan=False)

    def test_nonpositive_endpoint_is_preserved_without_clamping_or_removal(self):
        rows,anchor=original_bank(.08)
        out=gap.candidates(rows,anchor)
        self.assertAlmostEqual(out[8]['values'][4],-.04)
        self.assertEqual(len(out),16)
        self.assertNotIn('native_rejected',out[8])
        self.assertNotIn('capture',out[8])

    def test_copies_are_isolated_and_repeated_calls_deterministic(self):
        rows,anchor=original_bank();a=gap.candidates(rows,anchor);b=gap.candidates(rows,anchor)
        self.assertEqual(a,b)
        a[8]['gap_coverage']['original_candidate']['values'][0]=9
        a[3]['source_training_ids'].append('changed')
        self.assertNotEqual(a,b)
        self.assertNotEqual(rows[8]['values'][0],9)
        self.assertEqual(rows[3]['source_training_ids'],['train-a'])

    def test_requires_literal_source_bank_not_any_sixteen_rows(self):
        for kind in ('count','id','order','family','residual','mismatch','already_applied'):
            rows,anchor=original_bank()
            if kind=='count': rows.pop()
            elif kind=='id': rows[0]['id']='candidate-01'
            elif kind=='order': rows[2],rows[3]=rows[3],rows[2]
            elif kind=='family': rows[8]['source_kind']='unrelated'
            elif kind=='residual': rows[7]['normalized_residual'][0]=1.001
            elif kind=='mismatch': rows[4]['values'][1]+=.001
            else: rows[8]['gap_coverage']={}
            with self.subTest(kind=kind),self.assertRaises(ValueError):gap.candidates(rows,anchor)

    def test_malformed_nonfinite_input_is_not_repaired(self):
        for kind in ('anchor','anchor_size','values','residual','missing','nondict'):
            rows,anchor=original_bank()
            if kind=='anchor':anchor['values'][4]=float('nan')
            elif kind=='anchor_size':anchor['values'].pop()
            elif kind=='values':rows[15]['values'][2]=float('inf')
            elif kind=='residual':rows[2]['normalized_residual'][3]=float('nan')
            elif kind=='missing':rows[4].pop('values')
            else:rows[8]=None
            with self.subTest(kind=kind),self.assertRaises(ValueError):gap.candidates(rows,anchor)

    def test_import_is_cpu_only_and_does_not_load_simulator_or_model(self):
        code=('import sys; from sircus.mjlab import gap_coverage_candidates; '
              'assert not any(x in sys.modules for x in ("torch","mujoco","warp","mjlab"))')
        subprocess.run([sys.executable,'-c',code],check=True,cwd=Path(__file__).resolve().parents[3])


if __name__=='__main__':unittest.main()
