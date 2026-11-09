import unittest
from sircus.mjlab.coverage_training_video import choose_records

class Selection(unittest.TestCase):
    def row(self,method,context,capture,**kw):
        return dict(id=method+'-'+context,method=method,context_id=context,role='test',native_release=True,invalid_physics=False,route_success=capture,**kw)
    def test_matched_gain_then_failure(self):
        rows=[self.row('current_ppo','a',False),self.row('selected_ppo','a',True),self.row('selected_ppo','b',False)]
        self.assertEqual([r['id'] for r in choose_records(rows)],[r['id'] for r in rows])
    def test_table_capture_separate_when_available(self):
        rows=[self.row('current_ppo','a',False),self.row('selected_ppo','a',True),self.row('selected_ppo','b',True,table_assisted_capture=True)]
        self.assertEqual(len(choose_records(rows)),3)
    def test_no_capture_does_not_invent_one(self):
        rows=[self.row('current_ppo','a',False),self.row('selected_ppo','a',False)]
        self.assertTrue(all('miss' in r['label'] for r in choose_records(rows)))
    def test_no_invalid_or_training_examples(self):
        r=self.row('selected_ppo','a',True);r['role']='train'
        with self.assertRaises(ValueError):choose_records([r])
        r['role']='test';r['invalid_physics']=True
        with self.assertRaises(ValueError):choose_records([r])

if __name__=='__main__':unittest.main()
