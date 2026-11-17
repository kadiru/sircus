"""Create a private preparation package. Never import a simulator or policy."""

import argparse
import hashlib
import json
from pathlib import Path

import location_comparison as comparison

PLACEMENT_SOURCE_FINGERPRINT = "7e75b8b0984aa80eb82412886fdacc6cc484270b93ae9d8beb953b258caaf020"


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def verify_receipt(receipt):
    comparison.require(receipt.get("fingerprint") == comparison.digest(
        {key: value for key, value in receipt.items() if key != "fingerprint"}),
        "Changed private receipt fingerprint")


def placement_from_receipt(receipt):
    verify_receipt(receipt)
    comparison.require(receipt["fingerprint"] == PLACEMENT_SOURCE_FINGERPRINT,
                       "Expected the independently verified placement source receipt")
    comparison.require(receipt.get("schema") == "sircus-seed22-location-comparison-placement-source-v1",
                       "Unexpected placement source receipt")
    comparison.require(receipt["original_split_sha256"] == comparison.SPLIT_SHA,
                       "Placement receipt belongs to another original split")
    comparison.require(receipt["base_mesh"]["sha256"] ==
                       "871d501d23e8164440656331ee0e005d859dbdb7fab1c3725bef423adac9cf0e",
                       "Unexpected base collision mesh")
    comparison.require(receipt["base_mesh"]["collider_type"] == "convex_hull"
                       and receipt["base_mesh"]["collision_from_visuals"] is False,
                       "Unexpected collider source")
    geometry = dict(table_bounds_xy_m=receipt["tabletop_xy_bounds_m"],
        bucket_radius_m=receipt["bucket"]["outer_footprint_radius_m"],
        bases=[dict(id=base["name"], xy_m=base["position_m"][:2],
                    radius_m=receipt["base_mesh"]["conservative_rounded_radius_m"])
               for base in receipt["base_poses"]],
        clearance_m=receipt["comparison_minimum_base_clearance_m"],
        source_sha256=dict(receipt["source_files_sha256"],
                           collision_mesh=receipt["base_mesh"]["sha256"],
                           **receipt["asset_files_sha256"]),
        method="Whole bucket XY circle and outward-rounded source collision mesh XY circles; fixed base/table placement only")
    comparison.validate_placement(geometry)
    return geometry


def prepare_files(original_split, history_path, placement_path, output):
    output = Path(output)
    comparison.require(not output.exists(), "Output already exists; preserve previous preparation")
    split, split_sha = read_json(original_split)
    comparison.require(split_sha == comparison.SPLIT_SHA, "Original split bytes changed")
    history, history_sha = read_json(history_path)
    verify_receipt(history)
    receipt, receipt_sha = read_json(placement_path)
    geometry = placement_from_receipt(receipt)
    train = split["train"]
    points = comparison.coordinates(history["coordinates"] + [cell["bucket_xy_m"] for cell in train])
    bindings = dict(parent_checkpoint=comparison.PARENT_SHA, original_split_file=split_sha,
        continuous_geometry_layer=comparison.GEOMETRY_SHA, original_train=comparison.digest(train),
        historical_coordinates=comparison.digest([list(x) for x in points]), history_file=history_sha,
        placement_source_file=receipt_sha, placement_geometry=comparison.digest(geometry),
        preparation_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        comparison_source=hashlib.sha256(Path(comparison.__file__).read_bytes()).hexdigest())
    plan = comparison.build_comparison(train, points, geometry, bindings)
    comparison.validate_comparison(plan, original_train=train, history=points,
                                   placement=geometry, bindings=bindings)
    expected = [(row["id"], row["admitted"]) for row in receipt["train"]]
    actual = [(row["id"], row["admitted"]) for row in plan["original_placement_audit"]]
    comparison.require(actual == expected, "Placement classification disagrees with source receipt")
    # Only this writer creates files; simulation, policy loading and training
    # are absent. Exclusive creation fails closed without overwriting evidence.
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("comparison.json", plan), ("summary.json", comparison.summarize(plan))):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return comparison.summarize(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-split", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--placement-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = prepare_files(args.original_split, args.history, args.placement_source, args.output)
    print(json.dumps(dict(fingerprint=summary["fingerprint"],
        admitted_original_locations=summary["admitted_original_locations"],
        broader_unique_locations=summary["broader_unique_locations"],
        test_positions=summary["protocol"]["test_positions"], native_requests=0, optimizer_steps=0)))


if __name__ == "__main__":
    main()
