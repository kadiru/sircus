"""Paired denominators, shared resampling and descriptive-only intervals."""
import copy
import unittest

import numpy as np

from sircus.mjlab import physics_ensemble_statistics as stats


def cohort():
    captures={'bc':set(range(19))|set(range(24,28)),
              'ppo':set(range(22))|set(range(28,32)),'ensemble':set(range(24))}
    native={'bc':44,'ppo':45,'ensemble':49};rows=[]
    for method in stats.METHODS:
        for i in range(60):
            released=i<native[method];hit=i in captures[method]
            rows.append(dict(context_id=f'layout-{i:02d}',method=method,bucket_xy_m=[.1+i*.01,-.3],
                record_id=f'{method}-{i}',actual=dict(native_release=released,capture=hit,clean=False,
                    invalid=None if released else 'endpoint_rejected',invalid_physics=False)))
    return rows


class StatisticsTests(unittest.TestCase):
    def test_complete_counts_and_known_pairings_keep_all_rejects_in_denominators(self):
        rows=cohort();before=copy.deepcopy(rows);result=stats.describe(rows)
        self.assertEqual(rows,before);self.assertEqual(result['same_layouts'],60)
        self.assertNotIn('audit',result);self.assertIn('not seed/noise reliability',result['scope'])
        self.assertEqual([(r['counts']['captures'],r['counts']['native_releases'],r['counts']['rejected'])
                         for r in result['test_comparison']],[(23,44,16),(26,45,15),(24,49,11)])
        bc_ens=next(r for r in result['paired_differences'] if (r['a'],r['b'])==('bc','ensemble'))
        self.assertEqual((bc_ens['shared'],bc_ens['a_only'],bc_ens['b_only'],bc_ens['neither']),(19,4,5,32))
        self.assertAlmostEqual(bc_ens['delta_rate'],-1/60);self.assertEqual(bc_ens['exact_mcnemar_p'],1.)
        ppo_ens=next(r for r in result['paired_differences'] if (r['a'],r['b'])==('ppo','ensemble'))
        self.assertEqual((ppo_ens['shared'],ppo_ens['a_only'],ppo_ens['b_only'],ppo_ens['neither']),(22,4,2,32))
        self.assertAlmostEqual(ppo_ens['delta_rate'],2/60);self.assertAlmostEqual(ppo_ens['exact_mcnemar_p'],.6875)

    def test_seeded_shared_layout_bootstrap_and_input_reordering_are_identical(self):
        rows=cohort();a=stats.describe(rows);b=stats.describe(list(reversed(rows)))
        self.assertEqual(a,b)
        by={(r['context_id'],r['method']):r['actual']['capture'] for r in rows}
        contexts=sorted({r['context_id'] for r in rows})
        values=np.asarray([[by[c,m] for m in stats.METHODS] for c in contexts],dtype=float)
        idx=np.random.default_rng(20261651).integers(0,60,size=(10000,60))
        means=values[idx].mean(axis=1)
        expected=np.percentile(means[:,0]-means[:,2],[2.5,97.5])
        np.testing.assert_array_equal(a['paired_differences'][1]['ci95'],expected)
        self.assertIn('delta=a-b',a['paired_differences'][1]['interval_method'])

    def test_wilson_zero_all_and_exact_mcnemar_edges(self):
        self.assertEqual(stats._wilson(0,60)[0],0.)
        self.assertAlmostEqual(stats._wilson(0,60)[1],.06017185214208986)
        self.assertAlmostEqual(stats._wilson(60,60)[0],.9398281478579101)
        self.assertEqual(stats._wilson(60,60)[1],1.)
        self.assertEqual(stats._mcnemar(0,0),1.);self.assertEqual(stats._mcnemar(0,5),.0625)
        self.assertEqual(stats._mcnemar(3,3),1.)

    def test_native_invalid_remains_separate_from_geometry_reject_and_miss(self):
        rows=cohort();r=next(r for r in rows if r['method']=='bc' and r['context_id']=='layout-40')
        r['actual'].update(invalid='outside_calibrated_domain',invalid_physics=True)
        result=stats.describe(rows)['test_comparison'][0]['counts']
        self.assertEqual(result,dict(requested=60,native_releases=44,captures=23,clean=0,rejected=16,native_misses=20,invalid=1))

    def test_missing_duplicate_mismatched_coordinates_and_inconsistent_flags_are_rejected(self):
        edits=[lambda r:r.pop(),lambda r:r.__setitem__(1,copy.deepcopy(r[0])),
               lambda r:r[60].update(bucket_xy_m=[.9,-.3]),
               lambda r:r[0].update(method='other'),lambda r:r[0]['actual'].update(capture=1),
               lambda r:r[59]['actual'].update(capture=True),
               lambda r:r[40]['actual'].update(clean=True),lambda r:r[59]['actual'].update(invalid=None),
               lambda r:r[0].update(bucket_xy_m=[float('nan'),0.]),
               lambda r:r[1].update(record_id=r[0]['record_id'])]
        for edit in edits:
            rows=cohort();edit(rows)
            with self.subTest(edit=edit),self.assertRaises(ValueError):stats.describe(rows)


if __name__=='__main__':unittest.main()
