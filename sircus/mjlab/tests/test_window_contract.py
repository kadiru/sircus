import unittest
from sircus.mjlab.window_contract import contexts,seed_gaps,settings,adaptive_gaps,windows
from sircus.mjlab.admission import REFERENCE_VALUES

class WindowContractTests(unittest.TestCase):
    def test_splits_budgets_and_exact_anchor_seed(self):
        groups=contexts();self.assertEqual([len(v) for v in groups.values()],[24,6,8])
        points=[(r['board_height_m'],r['tilt_deg']) for values in groups.values() for r in values]
        self.assertEqual(len(set(points)),38);self.assertIn(REFERENCE_VALUES[4],seed_gaps());self.assertEqual(len(seed_gaps()),21)
        self.assertEqual(settings()['budgets']['unique_training_requests'],504+504+504)
    def test_adaptation_refines_without_test_data_or_duplicate_requests(self):
        rows=[dict(role='train',gap_m=g,route_success=g==.3,learning_reward=float(g==.3)) for g in [.02,.1,.2,.3,.4,.5,.65]]
        proposed=adaptive_gaps(rows,5);self.assertEqual(len(set(proposed)),5)
        self.assertTrue(all(g not in [r['gap_m'] for r in rows] for g in proposed))
        rows[0]['role']='test'
        with self.assertRaises(ValueError):adaptive_gaps(rows)
    def test_disconnected_windows_and_failures_remain_visible(self):
        rows=[dict(context_id='a',gap_m=g,route_success=hit,native_release=True) for g,hit in [(.1,False),(.2,True),(.3,False),(.4,True),(.5,True),(.6,False)]]
        result=windows(rows)[0];self.assertEqual(len(result['observed_hit_runs']),2)
        self.assertEqual(result['observed_hit_runs'][1]['lower_tested_nonhit_gap_m'],.3)
        self.assertEqual(len(result['points']),6)

if __name__=='__main__':unittest.main()
