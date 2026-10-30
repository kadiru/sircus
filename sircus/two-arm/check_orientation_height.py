"""Small deterministic checks for the new optimizer contract; no network."""
import copy,json,tempfile
from pathlib import Path
from unittest.mock import patch
import train_orientation_height as learner
from physics import release_point
base=json.loads((Path(__file__).parent/'scene.json').read_text())
xy=release_point(base,{'u_mm':-16.408,'v_mm':-44.821,'gap_mm':161.715})[:2]
for roll,pitch in [(0,30),(10,20),(-10,40)]:
 scene,action=learner.candidate(base,roll,pitch,150,xy)
 actual=release_point(scene,action)
 assert all(abs(a-b)<1e-7 for a,b in zip(actual[:2],xy))
 assert scene['board']['center_mm']==base['board']['center_mm']
 assert scene['bin']==base['bin']
with tempfile.TemporaryDirectory() as tmp:
 out=Path(tmp)/'run'
 def fake_sim(scene,action,seed,frames):
  if scene['simulation']['noise_xy_mm']:
   assert (out/'policy.json').exists(), 'Policy must be frozen before tests'
  return {'reward':1.,'retained':True,'outcome':'test fixture'}
 with patch.object(learner,'feasibility',return_value={'arm_reachable':True,'board_arm_reachable':True}),patch.object(learner,'simulate',side_effect=fake_sim):
  policy,summary=learner.train(base,out,4,2)
 learner.verify_policy(policy,base)
 bad=copy.deepcopy(base);bad['bin']['inner_top_diameter_mm']+=1
 try:learner.verify_policy(policy,bad)
 except ValueError:pass
 else:raise AssertionError('Geometry mismatch accepted')
 altered=copy.deepcopy(policy);altered['release_action']['gap_mm']+=1
 try:learner.verify_policy(altered,base)
 except ValueError:pass
 else:raise AssertionError('Changed action accepted')
 assert len(list(out.glob('train-*.json')))==4 and len(list(out.glob('test-*.json')))==2
print('PASS: fixed world release XY, unchanged board center/bucket, freeze-before-test, fingerprint/action guards and distinct trial files')
