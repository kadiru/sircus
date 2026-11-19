"""Coordinate-only discovery from fetched published baseline JSON blobs.

Reads local Git objects, never fetches, reads trial/media files, queries a
model, or conditions exclusions on outcomes. Original discovery coordinates
are retained even if their old baseline disappeared from the current trees.
"""

import hashlib
import json
import math
from pathlib import Path
import subprocess

from candidate_fresh_splits import build_history_receipt

MAX_BLOB_BYTES = 32*1024**2
MAX_TOTAL_BYTES = 512*1024**2
SCOPE = ("Known bucket_xy_m coordinates in unique baseline JSON blobs at fetched origin-ref trees and task HEAD, "
         "unioned with the preserved original discovery pool. Only coordinates affect selection; "
         "no claim of exhaustive unpublished or globally unseen experiments.")


def coordinates_only(value):
    points = set()
    def visit(node):
        if type(node) is dict:
            for key, item in node.items():
                if key == "bucket_xy_m":
                    if item is None:  # Explicitly unset UI goal; no recorded position.
                        continue
                    if type(item) not in (list, tuple) or len(item) != 2 or not all(
                            type(x) in (int, float) and math.isfinite(x) for x in item):
                        raise ValueError("Malformed published bucket_xy_m coordinate")
                    points.add(tuple(map(float, item)))
                else:
                    visit(item)
        elif type(node) is list:
            for item in node: visit(item)
    visit(value)
    return sorted(points)


def _git(repo, *arguments, input=None):
    return subprocess.run(["git", "-C", str(repo), *arguments], input=input,
                          check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def discover_history(repo, original_pool_path, *, expected_original_pool_sha256):
    """Return a signed coordinate receipt and a separate discovery inventory."""
    raw = Path(original_pool_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_original_pool_sha256:
        raise ValueError("Original historical-coordinate pool bytes changed")
    original = json.loads(raw)
    old = original["coordinates"]
    if type(old) is not list or not old or sum(s["added_coordinates"] for s in original["sources"]) != len(old):
        raise ValueError("Original coordinate discovery receipt changed")
    old_points = coordinates_only({"points":[{"bucket_xy_m":p} for p in old]})
    if len(old_points) != len(old): raise ValueError("Original coordinate pool has duplicates")
    refs = _git(repo, "for-each-ref", "--format=%(refname)", "refs/remotes/origin").decode().splitlines()
    refs = sorted(set(refs+["HEAD"]))
    objects = {}
    for ref in refs:
        for entry in _git(repo, "ls-tree", "-r", "-z", ref, "--", "baselines").split(b"\0"):
            if not entry: continue
            metadata, path = entry.split(b"\t", 1)
            mode, kind, identifier = metadata.decode().split()
            path = path.decode()
            if kind == "blob" and path.endswith(".json"):
                objects.setdefault(identifier, dict(ref=ref, path=path, blob=identifier))
    identifiers = sorted(objects)
    if not identifiers: raise ValueError("No published baseline JSON blobs were discovered")
    request = ("\n".join(identifiers)+"\n").encode()
    sizes = _git(repo, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)", input=request)
    total = 0
    for line in sizes.decode().splitlines():
        identifier, kind, size = line.split(); size = int(size)
        if kind != "blob" or size > MAX_BLOB_BYTES: raise ValueError("Published discovery blob exceeds bounded JSON allowance")
        total += size
    if total > MAX_TOTAL_BYTES: raise ValueError("Published history discovery exceeds bounded aggregate allowance")
    body = _git(repo, "cat-file", "--batch", input=request)
    offset, points = 0, set(old_points)
    sources, bindings = [], {"original-historical-pool.json":expected_original_pool_sha256,
        "candidate_fresh_history.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for identifier in identifiers:
        newline = body.index(b"\n", offset)
        oid, kind, size = body[offset:newline].decode().split(); size = int(size)
        if oid != identifier or kind != "blob": raise ValueError("Git discovery blob order changed")
        raw = body[newline+1:newline+1+size]
        offset = newline+size+2
        if len(raw) != size or body[offset-1:offset] != b"\n": raise ValueError("Truncated Git discovery object")
        try:
            found = coordinates_only(json.loads(raw))
        except ValueError as error:
            raise ValueError("Published coordinate parsing failed at "+objects[identifier]["path"]+" blob "+identifier) from error
        added = len(set(found)-points)
        points.update(found)
        if found:
            fingerprint = hashlib.sha256(raw).hexdigest()
            bindings["git-blob-"+identifier] = fingerprint
            sources.append(dict(objects[identifier], sha256=fingerprint,
                                coordinates=len(found), added_coordinates=added))
    if offset != len(body): raise ValueError("Unexpected trailing Git discovery bytes")
    receipt = build_history_receipt([list(p) for p in sorted(points)], bindings, scope=SCOPE)
    inventory = dict(schema="sircus-fresh-coordinate-discovery-inventory-v1", scope=SCOPE,
        refs_examined=len(refs), unique_baseline_json_blobs=len(identifiers), total_blob_bytes=total,
        original_pool_coordinates=len(old_points), discovered_union_coordinates=len(points),
        coordinate_sources=sources, outcome_fields_used=False, native_shots=0, optimizer_updates=0)
    return receipt, inventory
