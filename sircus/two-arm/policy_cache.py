"""Recover a completed policy only for the current scene and dynamics sources."""
import copy
import json
from pathlib import Path
from trial_store import fingerprint
ROOT = Path(__file__).resolve().parent

def load_saved_training(config, archive=None):
    expected = fingerprint(config)
    archive = Path(archive) if archive is not None else ROOT / 'data/archive'
    for path in sorted(archive.glob('*/summary.json'), reverse=True):
        try:
            record = json.loads(path.read_text())
            summary = record['summary']
            frozen = json.loads((path.parent / 'policy.json').read_text())
            policy = frozen['policy']
            if any(value != expected for value in (record.get('config_fingerprint'),
                    summary.get('config_fingerprint'), frozen.get('config_fingerprint'),
                    policy.get('config_fingerprint'))):
                continue
            if summary.get('status') in ('failed', 'cancelled'):
                continue
            if summary.get('schema') != 'sircus-two-arm-release-v1' or summary.get('policy') != policy:
                continue
            if summary.get('best_action') != policy.get('best_action'):
                continue
            if len(summary.get('history', [])) != summary.get('budget'):
                continue
            test = summary.get('test', {})
            if test.get('trials') != 10 or len(test.get('results', [])) != 10:
                continue
            result = copy.deepcopy(summary)
            result['restored_from_archive'] = True
            return result
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None
