"""One bounded JSON request for numerical policy preview; never imports Viam."""
import json
import sys


def main():
    try:
        raw = sys.stdin.read(1_000_001)
        if len(raw.encode('utf-8')) > 1_000_000:
            raise ValueError('Preview input exceeds one megabyte')
        payload = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Finite JSON values required')))
        if not isinstance(payload, dict) or set(payload) != {'checkpoint', 'scene', 'seed'}:
            raise ValueError('Supply checkpoint, scene and seed JSON values only')
        if not isinstance(payload['checkpoint'], dict) or not isinstance(payload['scene'], dict):
            raise ValueError('Checkpoint and scene must be JSON objects')
        if type(payload['seed']) is not int or not 0 <= payload['seed'] < 2**32:
            raise ValueError('Invalid preview seed')
        from scene_study import preview
        value = preview(payload['checkpoint'], payload['scene'], seed=payload['seed'])
        print(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError, KeyError) as exc:
        print(json.dumps({'error': str(exc)[:500]}))
        return 2
    except Exception:
        print(json.dumps({'error': 'Numerical preview failed; no robot runtime was used'}))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
