import unittest
from sircus.mjlab.recovery_regression_video import choose_records


class Selection(unittest.TestCase):
    def rows(self):
        rows=[]
        for context in ('recovery-test-r0-c5-n1','recovery-test-r0-c6-n1'):
            for method in ('parent_bc','recovery_bc'):
                rows.append(dict(id=method+context,context_id=context,method=method,phase='test',native_release=True,invalid_physics=False,route_success=method=='parent_bc'))
        rows.append(dict(id='representative',context_id='other',method='recovery_bc',phase='test',native_release=True,invalid_physics=False,route_success=False,forbidden_contact_steps=1))
        return rows

    def test_both_exact_parent_recovery_pairs_then_observed_forbidden_miss(self):
        rows=self.rows();clips=choose_records(rows)
        self.assertEqual([r['id'] for r in clips],[r['id'] for r in rows]);self.assertEqual(len(clips),5)
        self.assertIn('Parent capture',clips[0]['label']);self.assertIn('Recovery miss',clips[1]['label'])

    def test_no_substituted_invalid_unflown_training_or_reversed_pair(self):
        for change in ({'invalid_physics':True},{'native_release':False},{'phase':'train'},{'route_success':False}):
            rows=self.rows();rows[0].update(change)
            with self.assertRaises(ValueError):choose_records(rows)

    def test_no_invented_representative(self):
        rows=self.rows();rows[-1]['forbidden_contact_steps']=0
        with self.assertRaisesRegex(ValueError,'Representative'):choose_records(rows)

if __name__=='__main__':unittest.main()
