"""Small observation-based release search; frozen-policy tests are held out."""
import copy
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from physics import simulate, validate_config, release_point
from kinematics import sequence

ROOT = Path(__file__).parent


def fingerprint(config):
    physics_hash = hashlib.sha256((ROOT / 'physics.py').read_bytes()).hexdigest()
    model_hash = hashlib.sha256((ROOT / 'models/xarm850_kinematics.json').read_bytes()).hexdigest()
    payload = {'config': config, 'physics_sha256': physics_hash, 'arm_model_sha256': model_hash}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def train(config, seed=42, budget=20, callback=None):
    validate_config(config)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**30:
        raise ValueError('seed must be an integer from 0 to 2^30-1')
    if isinstance(budget, bool) or not isinstance(budget, int) or not 8 <= budget <= 80:
        raise ValueError('budget must be 8..80')
    rng = random.Random(seed)
    size = config['board']['size_mm']
    u_limit, v_limit = size[0] * .40, size[1] * .40
    history = []
    best = None
    initial = [dict(u_mm=u, v_mm=v, gap_mm=g) for u, v, g in
               [(0, 0, 180), (0, -65, 120), (0, -65, 220), (0, 50, 140),
                (-65, 0, 180), (65, 0, 180), (0, -30, 300), (0, 30, 80)]]
    for i in range(budget):
        if i < len(initial):
            a = initial[i].copy()
            a['u_mm'] = max(-u_limit, min(u_limit, a['u_mm']))
            a['v_mm'] = max(-v_limit, min(v_limit, a['v_mm']))
        elif i < budget // 2 or best is None:
            a = dict(u_mm=rng.uniform(-u_limit, u_limit), v_mm=rng.uniform(-v_limit, v_limit), gap_mm=rng.uniform(60, 380))
        else:
            scale = max(.12, 1 - i / budget)
            a = {k: best['action'][k] + rng.gauss(0, width * scale)
                 for k, width in [('u_mm', 120), ('v_mm', 120), ('gap_mm', 180)]}
            a = dict(u_mm=max(-u_limit, min(u_limit, a['u_mm'])),
                     v_mm=max(-v_limit, min(v_limit, a['v_mm'])), gap_mm=max(40, min(450, a['gap_mm'])))
        feasibility = sequence(config, release_point(config, a))
        if not feasibility['arm_reachable']:
            row = {'trial': i + 1, 'action': a, 'reward': -1000.0,
                   'retained': False, 'outcome': 'unreachable_arm_target', 'arm_reachable': False}
            history.append(row)
            if callback:
                callback(i + 1, budget, best or row, row)
            continue
        result = simulate(config, a, seed=seed * 3, frames=False)
        row = {'trial': i + 1, 'action': a, 'reward': float(result['reward']),
               'retained': result['retained'], 'outcome': result['outcome'], 'arm_reachable': True,
               'impact_uv_mm': result.get('impact_uv_mm'), 'landing_xy_mm': result.get('landing_xy_mm')}
        history.append(row)
        if best is None or row['reward'] > best['reward']:
            best = row
        if callback:
            callback(i + 1, budget, best, row)
    if best is None:
        raise ValueError('No kinematically reachable release found for this scene')
    session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = ROOT / 'runs' / session
    folder.mkdir(parents=True)
    policy = {'schema': 'sircus-compliant-release-v1', 'config_fingerprint': fingerprint(config),
              'best_action': best['action'], 'train_seed': seed, 'budget': budget,
              'physically_verified': False,
              'selection_scope': 'one board pose; synthetic physics; kinematic endpoint checks, no path collision checks',
              'test_spec': {'trials': 10, 'seed': seed * 3 + 1, 'release_xy_mm': 2, 'release_gap_mm': 3}}
    # Save before generating held-out perturbations, never tune on those results.
    (folder / 'policy.json').write_text(json.dumps(policy, indent=2))
    testing = copy.deepcopy(config)
    testing['simulation']['noise_xy_mm'] = 2.0
    testing['simulation']['noise_gap_mm'] = 3.0
    test_rows = []
    for j in range(10):
        result = simulate(testing, best['action'], seed=seed * 3 + 1 + j * 104729, frames=False)
        test_rows.append({k: result.get(k) for k in ('retained', 'outcome', 'reward', 'impact_uv_mm', 'landing_xy_mm')})
    nominal = simulate(config, best['action'], seed=seed * 3, frames=True)
    output = {**policy, 'history': history, 'best_result': nominal,
              'test': {'retained_count': sum(r['retained'] for r in test_rows), 'trials': len(test_rows),
                       'results': test_rows, 'scope': 'synthetic release noise only; no physical or arbitrary-pose validation'},
              'session_id': session}
    (folder / 'result.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    return output
