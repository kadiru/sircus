"""Small deterministic checks for the new optimizer contract; no network."""
import asyncio,copy,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock,patch
import train_orientation_height as learner
import replay_orientation_height as native_replay
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
 calls=[]
 def fake_rollout(scene,action,seed,frames):
  if scene['simulation']['noise_xy_mm']:
   assert (out/'policy.json').exists(), 'Policy must be frozen before tests'
  calls.append((seed,scene['simulation']['noise_xy_mm']))
  # A high shaping reward / legacy retained must NOT outrank a 1 s capture.
  success=len(calls)==2 or bool(scene['simulation']['noise_xy_mm'])
  return {'scoring_version':learner.SCORING_VERSION,'physics_version':learner.PHYSICS_VERSION,
          'success':success,'shaping_reward':-100. if success else 100.,
          'retained':True,'reward':1000.,'executed':True,'outcome':'captured' if success else 'no_entry'}
 with patch.object(learner,'feasibility',return_value={'arm_reachable':True,'board_arm_reachable':True}),patch.object(learner,'rollout',side_effect=fake_rollout):
  policy,summary=learner.train(base,out,4,2)
 assert summary['test_captures']==2 and 'test_retained' not in summary
 assert policy['parameters']==json.loads((out/'train-0002.json').read_text())['parameters']
 assert set(x[0] for x in calls[:4]).isdisjoint(x[0] for x in calls[4:])
 assert all(json.loads(f.read_text())['scoring_version']==learner.SCORING_VERSION for f in out.glob('test-*.json'))
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
 for key,value in [('schema','sircus-orientation-height-v1'),('scoring_version','legacy'),('physics_version','legacy')]:
  bad=copy.deepcopy(policy);bad[key]=value
  try:learner.verify_policy(bad,base)
  except ValueError:pass
  else:raise AssertionError('Incompatible policy accepted: '+key)
 try:learner.verify_policy(policy,base,for_native=True)
 except ValueError as exc:assert 'legacy-only' in str(exc)
 else:raise AssertionError('v2 policy sent to legacy native scoring')
 with patch.object(native_replay,'connect',new_callable=AsyncMock) as connect:
  try:asyncio.run(native_replay.run(SimpleNamespace(policy=out/'policy.json',output=None)))
  except ValueError as exc:assert 'legacy-only' in str(exc)
  else:raise AssertionError('Native replay did not reject v2')
  connect.assert_not_awaited()
 # Mutating a source hash must fail even when the scene/action still match.
 changed_manifest=copy.deepcopy(policy['contract'])
 changed_manifest['source_sha256']['physics.py']='0'*64
 with patch.object(learner,'contract_manifest',return_value=changed_manifest):
  try:learner.verify_policy(policy,base)
  except ValueError:pass
  else:raise AssertionError('Source mismatch accepted')
 numeric=copy.deepcopy(base);numeric['board']['rpy_deg'][0]=-0.;numeric['bin']['inner_top_diameter_mm']=177
 assert learner.contract_fingerprint(numeric)==learner.contract_fingerprint(base)
print('PASS: fixed XY, finite-board gap, capture-first selection (ignoring retained/reward), shared train/test versions, freeze-before-test, independent seeds, schema/source/action/native guards and distinct files')
