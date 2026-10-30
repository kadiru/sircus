import unittest
from unittest.mock import patch
from sircus.studies import diagnostic_controls as diagnostic
from sircus.studies import front_study as study


class DiagnosticControlsTests(unittest.TestCase):
    def test_diagnostic_ood_is_explicit_and_budget_bounded(self):
        episode=study.sample_layouts(3,'challenge',1)[0];episode['scene']['bin']['center_xy_mm'][1]=-600
        calls=[]
        def evaluate(scene,action,seed,**kwargs):
            calls.append(kwargs)
            return {'action':action,'result':{'success':False,'outcome':'invalid_action','executed':False},'scope':{'in_distribution':False}}
        with patch.object(study,'evaluate_action',side_effect=evaluate):rows,best=diagnostic.search_task(episode)
        self.assertEqual(len(rows),16);self.assertEqual(len(calls),16)
        self.assertTrue(all(c.get('allow_ood') is True for c in calls))
        self.assertEqual(best['role'],'post_study_search_diagnostic')
        self.assertFalse(best['result']['success'])
        with self.assertRaises(ValueError):study.validate_layout(episode['scene'])


if __name__=='__main__':unittest.main()
