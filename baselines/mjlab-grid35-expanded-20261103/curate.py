"""Copy a compact, audited simulation study without raw arrays or host state."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def curate(study, audit_path, output):
    study, audit_path, output = Path(study), Path(audit_path), Path(output)
    report, audit = read(study / "report.json"), read(audit_path)
    assert report["status"] == "completed", "Study is not complete"
    assert audit["status"] == "passed" and audit["failures"] == []
    for key, name in (("study_manifest_sha256", "manifest.json"),
                      ("report_sha256", "report.json"),
                      ("frozen_selection_sha256", "frozen-selection.json")):
        assert audit[key] == sha(study / name), (key, name)
    assert audit["contract_fingerprint"] == read(study / "contract.json")["fingerprint"]
    manifest = read(study / "manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("contract.json", "report.json", "frozen-selection.json"):
        assert sha(study / name) == manifest[name], name
        (output / name).write_bytes((study / name).read_bytes())
    (output / "original-manifest.json").write_bytes((study / "manifest.json").read_bytes())
    (output / "audit.json").write_bytes(audit_path.read_bytes())
    settings, coverage = [], []
    for cell in report["cells"]:
        selected = cell.get("selected")
        if selected:
            path = study / "results" / (cell["id"] + ".json")
            result = read(path)
            assert sha(path) == manifest[str(path.relative_to(study))], cell["id"]
            assert result["request_id"] == selected["request_id"]
            assert result["bucket_xy_m"] == cell["bucket_xy_m"]
            assert digest(result["configuration"]) == selected["configuration_sha256"]
            settings.append(dict(id=cell["id"], bucket_xy_m=cell["bucket_xy_m"],
                                 selected=selected, configuration=result["configuration"],
                                 result_file_sha256=sha(path), provenance=result["provenance"],
                                 frozen_validation=cell.get("frozen_validation")))
        for stage in ("baseline", "expanded"):
            directory = study / "cells" / cell["id"] / stage
            path = directory / "diversity.json"
            if path.is_file():
                assert sha(path) == manifest[str(path.relative_to(study))]
                value = read(path)
                coverage.append(dict(id=cell["id"], stage=stage,
                                     diversity_file_sha256=sha(path), **value))
    save(output / "selected-settings.json", settings)
    save(output / "diversity-summary.json", coverage)
    save(output / "BUNDLE.json", dict(
        version="sircus-compact-grid35-study-v1",
        scope="Curated simulation evidence; raw trajectories, logs and host state excluded. Original manifest refers to retained external raw evidence.",
        study_manifest_sha256=sha(study / "manifest.json"),
        audit_sha256=sha(audit_path),
        report_sha256=sha(study / "report.json"),
        contract_fingerprint=audit["contract_fingerprint"],
        selected_configurations=len(settings),
        diversified_stage_records=len(coverage)))
    save(output / "manifest.json", {
        p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()
    })
    print(json.dumps(dict(output=str(output), selected=len(settings), coverage=len(coverage))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    curate(args.study, args.audit, args.output)
