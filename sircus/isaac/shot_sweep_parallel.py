"""Bounded CPU-only parallel shot search with separately sealed worker evidence."""

import argparse
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

VERSION = "sircus-parallel-shot-sweep-v1"


def assignments(workers):
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use one to four CPU search workers")
    return [list(range(i, 35, workers)) for i in range(workers)]


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def merge_worker_manifests(spec, paths, expected_assignments):
    """Select each cell only from its assigned worker; reject stale/foreign input."""
    from expanded_pose_menu import canonical_hash, make_manifest

    flat = [i for group in expected_assignments for i in group]
    if sorted(flat) != list(range(35)) or len(set(flat)) != 35:
        raise ValueError("Worker assignments must partition the original 35 cells")
    if len(paths) != len(expected_assignments):
        raise ValueError("Missing CPU worker artifacts")
    unsigned_spec = dict(spec)
    if unsigned_spec.pop("fingerprint") != canonical_hash(unsigned_spec):
        raise ValueError("Parallel task fingerprint mismatch")
    menus, fits, checks = ([None] * 35 for _ in range(3))
    records, protocol, seed_hash = [], None, None
    for index, (path, assigned) in enumerate(zip(paths, expected_assignments)):
        path = Path(path)
        raw = path.read_bytes()
        value = json.loads(raw)
        unsigned = dict(value)
        if unsigned.pop("fingerprint", None) != canonical_hash(unsigned):
            raise ValueError("CPU worker checksum mismatch")
        if (
            value.get("spec") != spec
            or value.get("source_fingerprint") != spec["fingerprint"]
        ):
            raise ValueError("CPU worker changed the original task or bucket sites")
        if value.get("qualification") != "offline_endpoint_ik":
            raise ValueError("CPU worker must remain offline-only")
        if value.get("summary", {}).get("requested_cells") != assigned:
            raise ValueError("CPU worker assignment mismatch")
        if index == 0:
            protocol = value.get("proposal_protocol")
            seed_hash = value.get("seed_manifest_sha256")
            if not isinstance(protocol, dict) or not isinstance(
                protocol.get("version"), str
            ):
                raise ValueError("CPU worker is missing its proposal protocol")
        elif (
            value.get("proposal_protocol") != protocol
            or value.get("seed_manifest_sha256") != seed_hash
        ):
            raise ValueError("CPU workers used different search or seed protocols")
        for key in ("menus", "fits", "checks"):
            if len(value.get(key, [])) != 35:
                raise ValueError("CPU worker is missing original cell slots")
        for cell in assigned:
            if len(value["menus"][cell]) != len(value["fits"][cell]):
                raise ValueError("CPU worker is missing endpoint fits")
            menus[cell], fits[cell], checks[cell] = (
                value[key][cell] for key in ("menus", "fits", "checks")
            )
        records.append(
            dict(
                worker=index,
                file=path.name,
                sha256=hashlib.sha256(raw).hexdigest(),
                fingerprint=value["fingerprint"],
                assigned_cells=assigned,
            )
        )
    return make_manifest(
        spec,
        menus,
        fits=fits,
        checks=checks,
        qualification="offline_endpoint_ik",
        seed_manifest_sha256=seed_hash,
        search_proposal_version=protocol["version"],
        proposal_protocol=protocol,
        source_original_geometry_fp=spec.get("camera_source_fingerprint"),
        parallel_protocol=dict(
            version=VERSION,
            workers=len(paths),
            threads_per_worker=1,
            worker_artifacts=records,
        ),
        summary=dict(
            accepted_per_cell=[len(m) for m in menus],
            requested_cells=list(range(35)),
            ik_checks_per_cell=[
                sum(c.get("ik_checked", False) for c in group) for group in checks
            ],
        ),
        scope="Merged bounded CPU proposal/strict FK-IK search; each original cell comes only from its assigned worker. Native settling/contact validation required.",
    )


def _worker(spec_path, assets, seed_native, output, cells, options):
    # These variables affect only this owned Python process and its children.
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    from shot_sweep import qualify

    spec = json.loads(Path(spec_path).read_text())
    output = Path(output)
    partial = output.with_suffix(".partial.json")

    def progress(row):
        print(json.dumps(dict(worker=output.stem, **row)), flush=True)

    value = qualify(
        spec,
        assets,
        seed_manifest=seed_native,
        cell_indices=cells,
        progress=progress,
        partial=lambda value: atomic(partial, value),
        **options,
    )
    atomic(output, value)
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--seed-native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-checks", type=int, default=512)
    parser.add_argument("--accepted", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--controls", type=int, default=2)
    parser.add_argument("--workspace", choices=("common", "table"), default="table")
    args = parser.parse_args()
    groups = assignments(args.workers)
    worker_dir = args.output.with_name(args.output.stem + "-workers")
    if args.output.exists() or worker_dir.exists():
        parser.error(
            "Choose a fresh output and worker directory; previous evidence is preserved"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker_dir.mkdir()
    for key in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    options = dict(
        max_checks_per_cell=args.max_checks,
        accepted_per_cell=args.accepted,
        candidates_per_cell=args.candidates,
        controls=args.controls,
        workspace=args.workspace,
    )
    paths = [worker_dir / f"worker-{i}.json" for i in range(args.workers)]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(
                _worker,
                str(args.spec),
                str(args.assets),
                str(args.seed_native),
                str(path),
                cells,
                options,
            )
            for path, cells in zip(paths, groups)
        ]
        for future in as_completed(futures):
            print(json.dumps(dict(completed_worker=future.result())), flush=True)
    value = merge_worker_manifests(json.loads(args.spec.read_text()), paths, groups)
    atomic(args.output, value)
    print(json.dumps(value["summary"]), flush=True)


if __name__ == "__main__":
    main()
