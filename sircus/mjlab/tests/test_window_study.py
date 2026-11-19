import unittest
import json
from pathlib import Path
import tempfile
import numpy as np

from sircus.mjlab import window_contract as protocol
from sircus.mjlab.window_study import checkpoint_choice, evaluation_requests, train_rows
from sircus.mjlab.window_pipeline import publish


class WindowStudyTests(unittest.TestCase):
    def test_display_publish_preserves_old_studies_and_source_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);study=root/'study';study.mkdir()
            original=dict(status='running',phase='coarse_seed',contract_fingerprint='abc',settings=protocol.settings())
            (study/'report.json').write_text(json.dumps(original))
            display=root/'display.json';display.write_text(json.dumps(dict(active_policy='old',historical={'captures':7})))
            publish(study,display);value=json.loads(display.read_text())
            self.assertEqual(value['active_policy'],'old');self.assertEqual(value['historical'],{'captures':7})
            self.assertEqual(value['window_study_evaluation']['contract_fingerprint'],'abc')
            self.assertEqual(json.loads((study/'report.json').read_text()),json.loads(json.dumps(original)))

    def test_eval_freezes_actions_and_paired_noise_without_resampling(self):
        contexts=protocol.contexts()['test'];gaps=np.full(8,.02)
        a=evaluation_requests(contexts,gaps,role='test',method='sac',milestone=504,checkpoint_hash='s',seed=7)
        b=evaluation_requests(contexts,gaps,role='test',method='adaptive_surrogate',milestone=504,checkpoint_hash='a',seed=7)
        self.assertEqual(len(a),184)
        self.assertEqual([r['gap_noise_m'] for r in a],[r['gap_noise_m'] for r in b])
        self.assertTrue(any(r['gap_m']<.02 for r in a))
        for c in contexts:
            selected=[r for r in a if r['context_id']==c['id']]
            self.assertEqual(sum(r['phase']=='nominal' for r in selected),3)
            self.assertEqual(sum(r['phase']=='fresh' for r in selected),20)
            self.assertEqual(len({r['nominal_gap_m'] for r in selected}),1)

    def test_train_selection_quarantines_other_method_control_and_evaluation(self):
        rows=[dict(id=i,role=role,method=method) for i,(role,method) in enumerate([
            ('train','shared'),('train','sac'),('train','adaptive_surrogate'),('test','sac'),
            ('validation','sac'),('control','control')])]
        self.assertEqual([r['id'] for r in train_rows(rows,'sac')],[0,1])
        self.assertEqual([r['id'] for r in train_rows(rows,'adaptive_surrogate')],[0,2])

    def test_checkpoint_selection_uses_validation_then_reward_then_earliest(self):
        rows=[dict(id=str(i),route_success=i!=0,learning_reward=float(i)) for i in range(4)]
        checkpoints=[dict(method='sac',milestone=m,validation_record_ids=[str(i)]) for i,m in enumerate([504,672,840,1008])]
        self.assertEqual(checkpoint_choice(checkpoints,rows,'sac')['milestone'],1008)
        rows[-1]['learning_reward']=rows[-2]['learning_reward']
        self.assertEqual(checkpoint_choice(checkpoints,rows,'sac')['milestone'],840)
        # An unrelated TEST success can never choose a checkpoint.
        rows.append(dict(id='heldout',role='test',route_success=True,learning_reward=1000.))
        self.assertEqual(checkpoint_choice(checkpoints,rows,'sac')['milestone'],840)


if __name__=='__main__':unittest.main()
