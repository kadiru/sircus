"""Build a static reach-audit UI bundle; no server, network or training actions.

The snapshot is embedded separately from report.json. Deployment is an explicit
caller action; this builder writes only a new staging directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

MARKER = '<script id="reach-ik-audit-data" type="application/json">null</script>'
SCHEMA = 'sircus-shot-benchmark-reach-audit-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(snapshot_path, output, source_commit, *, sources=None):
    snapshot_path, output = Path(snapshot_path), Path(output)
    sources = Path(sources) if sources is not None else Path(__file__).parent
    if not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('Full UI source commit required')
    data = snapshot_path.read_bytes()
    if len(data) > 8*1024*1024:
        raise ValueError('Audit snapshot exceeds static bundle limit')
    snapshot = json.loads(data)
    if snapshot.get('schema') != SCHEMA or not isinstance(snapshot.get('snapshot'), dict):
        raise ValueError('Explicit versioned reach-audit snapshot required')
    native = snapshot['snapshot'].get('source', {})
    if not re.fullmatch('[0-9a-f]{40}', native.get('commit', '')):
        raise ValueError('Bound native source commit required')
    # Escape markup delimiters even inside inert JSON. No untrusted strings are
    # interpreted as HTML or JavaScript by the panel.
    payload = json.dumps(snapshot, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    html = (sources/'shot_benchmark.html').read_text()
    if html.count(MARKER) != 1:
        raise ValueError('Expected exactly one empty audit JSON slot')
    html = html.replace(MARKER, '<script id="reach-ik-audit-data" type="application/json">'+payload+'</script>')
    panel = (sources/'shot_benchmark_panel.js').read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    files = {'shot_benchmark.html': html.encode(), 'shot_benchmark_panel.js': panel}
    for name, content in files.items():
        (output/name).write_bytes(content)
    receipt = dict(schema='sircus-shot-benchmark-audit-ui-bundle-v1', ui_source_commit=source_commit,
                   native_source=native, snapshot_sha256=sha(data),
                   embedded_snapshot_sha256=sha(payload.encode()),
                   files_sha256={name: sha(content) for name, content in files.items()},
                   scope='Static audit overlay only. Original report bytes and GET/HEAD reader remain independent; no native, policy or service-control actions.')
    (output/'audit-ui-manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.snapshot, args.output, args.source_commit)))
