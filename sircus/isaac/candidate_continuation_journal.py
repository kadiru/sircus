"""Durable new-only learning receipts, with no resume or native dispatch.

One producer owns a new directory for its lifetime. A reservation must be
committed before the caller can obtain choices. Interrupted journals are audit
inputs, never an instruction to retry consumed slots or resume an optimizer.
"""

import copy
import json
import os
from pathlib import Path
import re
import tempfile

from candidate_continuation_contract import JOURNAL_SCHEMA
from candidate_continuation_ledger import canonical_hash

ZERO_SHA256 = "0" * 64


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new_json(path, value):
    """Publish complete finite JSON without replacing an existing destination."""
    path = Path(path)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Journal:
    """Append complete hash-bound events; I/O failure permanently poisons writer."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir()
        _sync_directory(self.directory.parent)
        self.head = ZERO_SHA256
        self.sequence = 0
        self.failed = False

    def append(self, kind, payload):
        if self.failed:
            raise RuntimeError("Journal failed; continuation cannot resume")
        if not isinstance(kind, str) or not kind:
            raise ValueError("Expected event kind")
        try:
            event = dict(schema=JOURNAL_SCHEMA, seq=self.sequence,
                previous_sha256=self.head, kind=kind, payload=copy.deepcopy(payload))
            event["fingerprint"] = canonical_hash(event)
            write_new_json(self.directory/f"event-{self.sequence:06d}.json", event)
        except BaseException:
            self.failed = True
            raise
        self.head = event["fingerprint"]
        self.sequence += 1
        return copy.deepcopy(event)


def read_events(directory):
    """Read a contiguous prefix without hiding partial/unexpected journal files."""
    directory = Path(directory)
    names = sorted(directory.iterdir())
    if any(not re.fullmatch(r"event-[0-9]{6}\.json", p.name) or not p.is_file() for p in names):
        raise ValueError("Unexpected or partial journal files; inspect preserved evidence")
    events = []
    previous = ZERO_SHA256
    for index, path in enumerate(names):
        if path.name != f"event-{index:06d}.json":
            raise ValueError("Journal sequence has a gap")
        value = json.loads(path.read_bytes(), parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("Nonfinite journal JSON")))
        unsigned = {k:v for k,v in value.items() if k != "fingerprint"}
        if set(value) != {"schema", "seq", "previous_sha256", "kind", "payload", "fingerprint"} \
                or value["schema"] != JOURNAL_SCHEMA or type(value["seq"]) is not int \
                or value["seq"] != index or value["previous_sha256"] != previous \
                or canonical_hash(unsigned) != value["fingerprint"]:
            raise ValueError("Journal chain changed")
        events.append(value)
        previous = value["fingerprint"]
    return events
