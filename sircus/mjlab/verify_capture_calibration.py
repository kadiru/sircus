"""Independent CPU audit of isolated native contact calibration controls.

No method results, policy files, training data, GPU rollout or service is used.
The supplied prelaunch journal binds producer bytes checked again after the run;
producer caches did not themselves embed source hashes. Native contact geometry
is independently reconstructed using CPU compiled hulls and mj_geomDistance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import numpy as np

VERSION = "sircus-native-contact-calibration-audit-v1"
CALIBRATION_REPORT_VERSION = "sircus-native-contact-calibration-report-v1"
VERIFICATION_REPORT_VERSION = "sircus-native-contact-verification-report-v1"
PRODUCER_VERSION = "sircus-independent-contact-controls-v2"
FIT_RULE = "1.25 * maximum fit contact/inside-plane overlap + 8um; capped below wall thickness and half base thickness"
REQUIRED_PRODUCER_FILES = (
    "sircus/mjlab/capture_calibration.py",
    "sircus/mjlab/calibrated_native_log.py",
)
GEOMETRY_EPS_M = 2e-6
NORMAL_COEFFICIENT_GUARD = 1e-4  # dimensionless; never compared to a metre value
ARRAY_NAMES = {
    "position_m", "velocity_m_s", "forces_N", "native_time_s", "contact_distance_m",
    "contact_geom_ids", "contact_position_m", "contact_normal", "time_s", "dt_s", "record_ids",
}


class CalibrationAuditError(ValueError):
    pass


class Checks:
    def __init__(self):
        self.count = 0

    def require(self, condition, label):
        self.count += 1
        if not bool(condition):
            raise CalibrationAuditError(label)

    def close(self, actual, expected, label, *, atol=1e-12):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                     and np.allclose(a, b, atol=atol, rtol=0), label)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 8 * 1024 * 1024:
        raise CalibrationAuditError("Unsafe or oversized control metadata")
    return json.loads(path.read_text())


def expected_cases(scene, role):
    """Independently restate the predeclared physical input grid, not outcomes."""
    bucket, radius = scene["bucket"], scene["ball"]["radius_m"]
    floor = bucket["assumed_bottom_thickness_m"]
    rim = floor + bucket["assumed_interior_depth_m"]
    answer = []
    def append(label, p, v, xy=(0.0, 0.0)):
        answer.append(dict(id=f"{role}-{label}", position_local_m=list(p), velocity_m_s=list(v), bucket_xy_m=list(xy)))
    if role == "fit":
        for index, height in enumerate((floor + radius, 0.25, 0.75, 1.2875)):
            append(f"vertical-{index}", (0, 0, height), (0, 0, 0))
        speeds = (1.0, 3.0, 6.0)
        heights = (floor + radius + 0.01, rim - radius - 0.01)
        phases = (0.0, math.pi / 64)
    elif role == "verification":
        for index, height in enumerate((floor + radius + 0.001, 0.4, 1.05)):
            append(f"vertical-{index}", (0.006, -0.004, height), (0, 0, 0), (0.31, -0.27))
        speeds = (0.5, 2.0, 4.5, 5.75)
        heights = (floor + radius + 0.007, rim - radius - 0.006)
        phases = (math.pi / 128, 3 * math.pi / 128)
    else:
        raise CalibrationAuditError("Unknown control role")
    for direction in range(4):
        for phase_index, phase in enumerate(phases):
            angle = direction * math.pi / 2 + phase
            for height_index, height in enumerate(heights):
                for speed in speeds:
                    append(f"wall-{direction}-{phase_index}-{height_index}-{speed:g}", (0, 0, height),
                           (speed * math.cos(angle), speed * math.sin(angle), 0),
                           (0, 0) if role == "fit" else (0.31, -0.27))
    for index, offset in enumerate((-0.004, -0.001, 0.001, 0.004)):
        append(f"rim-{index}", (bucket["inner_top_diameter_m"] / 2 - radius + offset, 0, rim + radius + 0.04),
               (-0.08, 0, -0.5), (0, 0) if role == "fit" else (-0.22, 0.18))
    append("outside", (0.20, 0, rim + 0.12), (0, 0, 0))
    return answer


def source_bindings(repository, journal, reference, checks):
    repository = Path(repository).resolve()
    snapshots = journal.get("snapshots") if isinstance(journal, dict) else None
    checks.require(isinstance(snapshots, dict) and repository.name in snapshots,
                   "Supplied prelaunch journal omits producer snapshot")
    declared = snapshots[repository.name]
    checks.require(isinstance(declared, dict) and set(REQUIRED_PRODUCER_FILES) <= set(declared),
                   "Prelaunch journal omits required producer files")
    result = {}
    for name, expected in declared.items():
        path = Path(name)
        checks.require(not path.is_absolute() and ".." not in path.parts and isinstance(expected, str)
                       and re.fullmatch(r"[0-9a-f]{64}", expected), "Unsafe producer journal member")
        actual = repository / path
        checks.require(not actual.is_symlink() and sha(actual) == expected, "Producer source differs from supplied prelaunch journal")
        result[name] = expected
    for name, expected in reference["source_hashes"].items():
        checks.require(sha(repository / name) == expected, "Underlying frozen reference source changed")
    audit_root = Path(__file__).resolve().parents[2]
    for name in ("sircus/mjlab/model.py", "sircus/isaac/geometry.py"):
        checks.require(name in reference["source_hashes"] and sha(audit_root / name) == reference["source_hashes"][name],
                       "CPU physical model builder differs from native reference")
    return dict(source_sha256=result, reference_source_sha256=dict(reference["source_hashes"]),
                binding_scope="Supplied prelaunch hash journal, with post-run source-byte verification; caches did not embed producer source hashes")


def check_fixture_manifest(value, scene, role, checks):
    checks.require(value.get("version") == PRODUCER_VERSION and value.get("role") == role, "Control role/version changed")
    substeps = value.get("substeps")
    checks.require(type(substeps) is int and substeps in (32, 64, 128), "Unsupported control resolution")
    checks.require(value.get("fit_rule") == FIT_RULE and value.get("duration_s") == 4,
                   "Predeclared fit rule/horizon changed")
    checks.close(value.get("input_speed_limit_m_s"), 6.0, "Input speed-domain limit changed")
    checks.close(value.get("verified_speed_limit_m_s"), 6.25, "Verification speed-domain limit changed")
    cases = value.get("cases")
    checks.require(isinstance(cases, list) and value.get("fingerprint") == digest(cases), "Fixture fingerprint changed")
    expected = expected_cases(scene, role)
    ids = [case.get("id") for case in cases]
    checks.require(len(ids) == len(set(ids)) == len(expected) and set(ids) == {case["id"] for case in expected},
                   "Fixture input grid changed or repeated")
    original = {case["id"]: case for case in expected}
    for case in cases:
        reference = original[case["id"]]
        checks.require(set(case) == set(reference), "Fixture fields changed")
        for key in ("position_local_m", "velocity_m_s", "bucket_xy_m"):
            checks.close(case[key], reference[key], "Fixture initial condition changed")
        checks.require(np.linalg.norm(case["velocity_m_s"]) <= 6.0 + 1e-12, "Control input exceeds predeclared speed domain")
    canonical_ids = [case["id"] for case in expected]
    order = "canonical" if ids == canonical_ids else "reversed" if ids == canonical_ids[::-1] else "other-permutation"
    if "fixture_order" in value:
        checks.require(value["fixture_order"] == order, "Declared fixture order is inaccurate")
    return cases, substeps, order


def load_dataset(directory, scene, role, checks):
    directory = Path(directory).resolve()
    checks.require(directory.is_dir() and not directory.is_symlink(), "Control directory unavailable")
    names = {"fixture-manifest.json", "compiled-geometry.json", "component-summary.json", "native-components.npz"}
    if role == "fit":
        names.add("candidate-score-config.json")
    checks.require({p.name for p in directory.iterdir()} == names and all(p.is_file() and not p.is_symlink() for p in directory.iterdir()),
                   "Unexpected/missing control evidence files")
    hashes = {name: sha(directory / name) for name in sorted(names)}
    manifest = read(directory / "fixture-manifest.json")
    cases, substeps, order = check_fixture_manifest(manifest, scene, role, checks)
    geometry, summary = read(directory / "compiled-geometry.json"), read(directory / "component-summary.json")
    checks.require(summary.get("version") == PRODUCER_VERSION and summary.get("role") == role and summary.get("substeps") == substeps,
                   "Control summary role/version/resolution changed")
    checks.require(summary.get("fixture_sha256") == hashes["fixture-manifest.json"] and summary.get("raw_sha256") == hashes["native-components.npz"],
                   "Control raw/fixture binding changed")
    checks.require(summary.get("geometry_sha256") == geometry.get("geometry_sha256"), "Control geometry binding changed")
    checked_geometry = dict(planes=geometry["compiled_planes"], floor=geometry["compiled_floor_m"], rim=geometry["compiled_rim_m"], radius=geometry["ball_radius_m"])
    checks.require(geometry.get("geometry_sha256") == digest(checked_geometry), "Compiled geometry fingerprint changed")
    with np.load(directory / "native-components.npz", allow_pickle=False) as cache:
        checks.require(set(cache.files) == ARRAY_NAMES, "Native component cache schema changed")
        arrays = {key: cache[key] for key in cache.files}
    validate_arrays(arrays, cases, substeps, checks)
    return dict(directory=directory, hashes=hashes, manifest=manifest, cases=cases, substeps=substeps,
                order=order, geometry=geometry, summary=summary, arrays=arrays)


def validate_arrays(arrays, cases, substeps, checks):
    ticks, count = 4 * 240 * substeps, len(cases)
    shapes = dict(position_m=(ticks + 1, count, 3), velocity_m_s=(ticks + 1, count, 3),
                  forces_N=(ticks, count, 6), native_time_s=(ticks + 1, count),
                  contact_distance_m=(ticks, count, 2), contact_geom_ids=(ticks, count, 2, 2),
                  contact_position_m=(ticks, count, 2, 3), contact_normal=(ticks, count, 2, 3),
                  time_s=(ticks + 1,), dt_s=(), record_ids=(count,))
    for key, shape in shapes.items():
        checks.require(arrays[key].shape == shape, "Native dense array shape changed: " + key)
    checks.require(arrays["record_ids"].dtype.kind == "U" and arrays["record_ids"].tolist() == [case["id"] for case in cases], "Native world IDs/order changed")
    for key in ("position_m", "velocity_m_s", "forces_N", "native_time_s", "contact_position_m", "contact_normal"):
        checks.require(arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all(), "Nonfinite/wrong-precision native array: " + key)
    checks.require(arrays["contact_distance_m"].dtype == np.float32 and not np.isnan(arrays["contact_distance_m"]).any()
                   and not np.isneginf(arrays["contact_distance_m"]).any(), "Invalid signed contact distances")
    checks.require(arrays["contact_geom_ids"].dtype == np.int32, "Contact IDs changed precision")
    dt = 1 / (240 * substeps)
    checks.require(arrays["dt_s"].dtype == np.float64 and float(arrays["dt_s"]) == dt, "Native dt changed")
    checks.require(arrays["time_s"].dtype == np.float64 and np.array_equal(arrays["time_s"], np.arange(ticks + 1) * dt),
                   "Canonical tick-derived time changed")
    checks.require((arrays["forces_N"] >= 0).all() and not arrays["forces_N"][:, :, [0, 2, 3, 4, 5]].any(), "Isolated controls contain unexpected force categories")
    clock = arrays["native_time_s"]
    checks.require((clock[0] == 0).all() and (np.diff(clock, axis=0) > 0).all(), "Native GPU clock is not monotone from zero")
    checks.require(float(np.ptp(clock, axis=1).max()) <= 1e-7, "World GPU clocks disagree")
    # A conservative floating-point summation bound, not an independent time sensor.
    drift = float(np.abs(clock.astype(float) - arrays["time_s"][:, None]).max())
    bound = ticks * np.finfo(np.float32).eps * 4.0
    checks.require(drift <= bound, "Native clock drift exceeds float32 accumulation bound")
    for index, case in enumerate(cases):
        expected = np.asarray(case["position_local_m"]) + np.r_[case["bucket_xy_m"], 0.0]
        checks.close(arrays["position_m"][0, index], expected, "Initial ball position changed", atol=2e-7)
        # Native input storage is float32. Compare to that exact quantization,
        # rather than a fixed decimal tolerance that fails at larger speeds.
        checks.require(np.array_equal(arrays["velocity_m_s"][0, index], np.asarray(case["velocity_m_s"], dtype=np.float32)),
                       "Initial ball velocity changed")


def compile_cpu_model(scene, assets, substeps):
    """Use the hash-bound physical builder, not producer extraction/fit functions."""
    import mujoco
    from .model import build_scene
    root = ET.fromstring(build_scene(scene, Path(assets), solref=(0.004, 0.15), contact_friction=0.35))
    world = root.find("worldbody")
    for element in list(world):
        if element.tag != "body" or element.get("name") not in {"bucket", "ball"}:
            world.remove(element)
    for tag in ("actuator", "contact", "sensor", "equality", "keyframe"):
        for element in list(root.findall(tag)):
            root.remove(element)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    expected = dict(timestep=1 / (240 * substeps), solver=2, iterations=100, tolerance=1e-8, cone=0,
                    integrator=3, ccd_iterations=50, ccd_tolerance=1e-6, disableflags=0, enableflags=0)
    for name, value in expected.items():
        setattr(model.opt, name, value)
    model.opt.gravity[:] = (0, 0, -9.81)
    return model, dict(expected, gravity=[0.0, 0.0, -9.81])


def inspect_compiled_geometry(model, scene, recorded, checks):
    """Check every actual hull facet without the producer's angle-based selection."""
    import mujoco
    from scipy.spatial import ConvexHull
    data = mujoco.MjData(model)
    bucket_body = model.body("bucket").id
    data.mocap_pos[int(model.body_mocapid[bucket_body])] = 0
    mujoco.mj_forward(model, data)
    count = scene["bucket"]["segments"]
    checks.require(model.ngeom == count + 2, "Isolated model contains extra geometry or a lid")
    planes = np.asarray(recorded["compiled_planes"], float)
    checks.require(planes.shape == (count, 4), "Wrong recorded wall plane count")
    floor_geom = model.geom("bucket/base").id
    base_axis = data.geom_xmat[floor_geom].reshape(3, 3)
    half_height = float(model.geom_size[floor_geom, 1])
    radial = float(model.geom_size[floor_geom, 0])
    floor = float(data.geom_xpos[floor_geom, 2] + abs(base_axis[2, 2]) * half_height + radial * math.hypot(base_axis[2, 0], base_axis[2, 1]))
    all_vertices, all_facets, inner_top_vertices = [], [], []
    normal_errors, facet_distance_errors = [], []
    wall_ids = []
    radius = scene["ball"]["radius_m"]
    rim_expected = scene["bucket"]["assumed_bottom_thickness_m"] + scene["bucket"]["assumed_interior_depth_m"]
    cavity_reference = np.array([0.0, 0.0, (floor + rim_expected) / 2])
    extent = scene["bucket"]["outer_top_diameter_m"] / 2 + radius
    probes = np.asarray([[x, y, z, 1] for x in (-extent, extent) for y in (-extent, extent)
                         for z in (floor - radius, rim_expected + radius)])
    for index in range(count):
        geom = model.geom(f"bucket/wall{index:02}").id
        wall_ids.append(int(geom))
        mesh_id = int(model.geom_dataid[geom])
        begin, length = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        local_vertices = np.asarray(model.mesh_vert[begin:begin + length], float)
        world_vertices = (data.geom_xmat[geom].reshape(3, 3) @ local_vertices.T).T + data.geom_xpos[geom]
        hull = ConvexHull(world_vertices)
        selected = []
        for equation in hull.equations:
            # A cavity-facing side face separates the central cavity reference
            # from the solid wedge. Radial seams pass through the axis; caps and
            # bottom faces are not near-vertical side faces.
            horizontal = math.hypot(*equation[:2])
            if horizontal > 0.8 and abs(equation[2]) < 0.25 and equation[:3] @ cavity_reference + equation[3] > 0.01:
                selected.append(equation)
        checks.require(bool(selected), "No independent cavity-side hull face")
        for face in selected:
            normal_error = float(np.linalg.norm(face[:3] - planes[index, :3]))
            distance_error = float(np.max(np.abs(probes @ (face - planes[index]))))
            normal_errors.append(normal_error)
            facet_distance_errors.append(distance_error)
            checks.require(normal_error <= NORMAL_COEFFICIENT_GUARD, "Compiled facet normal coefficient guard failed")
            checks.require(distance_error <= GEOMETRY_EPS_M, "Compiled facet/score-plane disagreement exceeds metre envelope")
            all_facets.append(face.copy())
        all_vertices.append(world_vertices)
        top_vertices = world_vertices[world_vertices[:, 2] >= rim_expected - GEOMETRY_EPS_M]
        checks.require(len(top_vertices) >= 2, "Compiled wedge top is missing")
        inner_top_vertices.append(top_vertices[np.argmin(np.linalg.norm(top_vertices[:, :2], axis=1))])
    rim = float(np.concatenate(all_vertices)[:, 2].max())
    ball_geom = model.geom("ball/geom").id
    checks.close(floor, scene["bucket"]["assumed_bottom_thickness_m"], "Compiled floor differs", atol=GEOMETRY_EPS_M)
    checks.close(rim, rim_expected, "Compiled rim differs", atol=GEOMETRY_EPS_M)
    checks.close(recorded["compiled_floor_m"], floor, "Recorded floor differs", atol=GEOMETRY_EPS_M)
    checks.close(recorded["compiled_rim_m"], rim, "Recorded rim differs", atol=GEOMETRY_EPS_M)
    checks.close(float(model.geom_size[ball_geom, 0]), radius, "Compiled sphere radius differs")
    checks.close(recorded["ball_radius_m"], radius, "Recorded sphere radius differs")
    checks.require(recorded["contact_geom_ids"] == dict(ball=int(ball_geom), walls=wall_ids, base=int(floor_geom)), "Native contact geom ID map differs")
    thickness = min((scene["bucket"]["outer_bottom_diameter_m"] - scene["bucket"]["inner_bottom_diameter_m"]) / 2,
                    (scene["bucket"]["outer_top_diameter_m"] - scene["bucket"]["inner_top_diameter_m"]) / 2)
    checks.close(recorded["minimum_radial_wall_thickness_m"], thickness, "Wall thickness changed")
    checks.close(recorded["wall_tolerance_cap_m"], 0.95 * thickness, "Wall thickness cap changed")
    checks.close(recorded["floor_tolerance_cap_m"], scene["bucket"]["assumed_bottom_thickness_m"] / 2, "Floor thickness cap changed")
    expected_contact = dict(solref=[0.004, 0.15], solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
                            friction=[0.35, 0.005, 0.0001], margin=0.0, gap=0.0)
    for key, expected in expected_contact.items():
        actual = getattr(model, "geom_" + key)[ball_geom]
        checks.close(actual, expected, "CPU ball contact parameters changed: " + key)
        checks.close(recorded["ball_contact"][key], expected, "Recorded ball contact parameters changed: " + key)
    return dict(data=data, floor=floor, rim=rim, radius=radius, planes=planes,
                facets=np.asarray(all_facets), wall_ids=wall_ids, base_id=int(floor_geom), ball_id=int(ball_geom),
                top_vertices=np.asarray(inner_top_vertices), max_normal_coefficient_error=max(normal_errors),
                max_facet_distance_error_m=max(facet_distance_errors))


def geometry_queries(model, geometry, scene, checks):
    """Native CPU signed distances at hull-derived offset faces, vertices and caps."""
    import mujoco
    data = geometry["data"]
    ball_q = int(model.jnt_qposadr[model.joint("ball/free").id])
    mocap = int(model.body_mocapid[model.body("bucket").id])
    facets, radius = geometry["facets"], geometry["radius"]
    floor, rim = geometry["floor"], geometry["rim"]
    offsets = (-0.001, -0.00001, 0.0, 0.00001, 0.001)
    translations = ((0.0, 0.0), (0.31, -0.27), (-0.22, 0.18))
    heights = (floor + radius + 0.005, 0.08, 0.11, rim - radius - 0.005)
    records, errors = [], []
    def query(point, xy, candidates):
        data.mocap_pos[mocap] = [*xy, 0]
        data.qpos[ball_q:ball_q + 3] = np.asarray(point) + np.r_[xy, 0]
        data.qpos[ball_q + 3:ball_q + 7] = [1, 0, 0, 0]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        return min(float(mujoco.mj_geomDistance(model, data, geometry["ball_id"], geom, 0.5, None)) for geom in candidates)
    count = scene["bucket"]["segments"]
    for xy in translations:
        for phase, label in ((0.5, "face"), (0.0, "vertex")):
            for index in range(0, count, 4):
                angle = (index + phase) * 2 * math.pi / count
                ray = np.array([math.cos(angle), math.sin(angle)])
                outward = -(facets[:, :2] @ ray)
                relevant = outward > 0.1
                for height in heights:
                    for offset in offsets:
                        limits = (facets[relevant, 2] * height + facets[relevant, 3] - radius - offset) / outward[relevant]
                        point = np.r_[ray * limits.min(), height]
                        actual = query(point, xy, geometry["wall_ids"])
                        error = abs(actual - offset)
                        checks.require(error <= GEOMETRY_EPS_M, "Independent wall signed-distance fixture failed")
                        errors.append(error)
                        records.append((label, index, height, offset, list(xy)))
        for offset in offsets:
            actual = query([0, 0, floor + radius + offset], xy, [geometry["base_id"]])
            checks.require(abs(actual - offset) <= GEOMETRY_EPS_M, "Independent floor signed-distance fixture failed")
            errors.append(abs(actual - offset)); records.append(("floor", offset, list(xy)))
        for index in range(0, count, 8):
            vertex = geometry["top_vertices"][index]
            for offset in offsets:
                point = vertex + np.array([0.0, 0.0, radius + offset])
                actual = query(point, xy, geometry["wall_ids"])
                checks.require(abs(actual - offset) <= GEOMETRY_EPS_M, "Independent rim signed-distance fixture failed")
                errors.append(abs(actual - offset)); records.append(("rim", index, offset, list(xy)))
        open_distance = query([0, 0, rim + radius + 0.01], xy, geometry["wall_ids"] + [geometry["base_id"]])
        checks.require(open_distance > GEOMETRY_EPS_M, "Compiled bucket mouth is not open")
    return dict(queries=len(records), fixture_grid_sha256=digest(records), maximum_signed_distance_error_m=max(errors),
                normal_coefficient_guard=NORMAL_COEFFICIENT_GUARD,
                maximum_facet_normal_coefficient_error=geometry["max_normal_coefficient_error"],
                maximum_facet_plane_distance_bound_m=geometry["max_facet_distance_error_m"],
                geometry_epsilon_m=GEOMETRY_EPS_M,
                scope="CPU nativeCCD geom-distance queries using independently reconstructed compiler hull facets; no producer plane-extraction or fit function")


def component_rows(scene, geometry, cases, arrays, checks):
    """Independent plane-by-plane reduction; never imports producer summary code."""
    radius = scene["ball"]["radius_m"]
    planes = np.asarray(geometry["compiled_planes"], float)
    floor, rim = geometry["compiled_floor_m"], geometry["compiled_rim_m"]
    answer = []
    for index, case in enumerate(cases):
        p = arrays["position_m"][:, index].astype(float)
        x = p[:, 0] - case["bucket_xy_m"][0]
        y = p[:, 1] - case["bucket_xy_m"][1]
        z = p[:, 2]
        gap = np.full(len(p), np.inf)
        for nx, ny, nz, offset in planes:
            gap = np.minimum(gap, nx * x + ny * y + nz * z + offset)
        cavity = (gap >= 0) & (z + radius <= rim + GEOMETRY_EPS_M) & (z >= floor)
        contact_distance = arrays["contact_distance_m"][:, index]
        contact = np.isfinite(contact_distance).any(axis=1)
        producer_domain = cavity[1:] | contact
        speeds = np.sqrt((arrays["velocity_m_s"][:, index].astype(float) ** 2).sum(axis=1))
        domain = cavity | np.r_[contact, False] | np.r_[False, contact]
        def penetration(values):
            finite = values[np.isfinite(values)]
            return max(0.0, -float(finite.min())) if len(finite) else 0.0
        row = dict(id=case["id"], contact_wall_overlap_m=penetration(contact_distance[:, 0]),
                   contact_floor_overlap_m=penetration(contact_distance[:, 1]),
                   cavity_wall_overlap_m=max(0.0, float(np.max(radius - gap[cavity]))) if cavity.any() else 0.0,
                   cavity_floor_overlap_m=max(0.0, float(np.max(floor + radius - z[cavity]))) if cavity.any() else 0.0,
                   max_bucket_domain_speed_m_s=float(speeds[1:][producer_domain].max()) if producer_domain.any() else 0.0,
                   verified_pre_post_bucket_speed_m_s=float(speeds[domain].max()) if domain.any() else 0.0,
                   native_clock_drift_s=float(np.abs(arrays["native_time_s"][:, index].astype(float) - arrays["time_s"]).max()),
                   final_position_local_m=[float(x[-1]), float(y[-1]), float(z[-1])])
        answer.append(row)
    return answer


def validate_contact_observations(dataset, geometry, checks):
    arrays = dataset["arrays"]
    distance, ids = arrays["contact_distance_m"], arrays["contact_geom_ids"]
    active = np.isfinite(distance)
    checks.require((ids[~active] == -1).all(), "Absent contact has a geom ID")
    checks.require((arrays["contact_position_m"][~active] == 0).all() and (arrays["contact_normal"][~active] == 0).all(),
                   "Absent contact has uninitialized geometry")
    for category, allowed in enumerate((set(geometry["wall_ids"]), {geometry["base_id"]})):
        values = ids[:, :, category][active[:, :, category]]
        checks.require(len(values) > 0, "Control grid has no wall/base contact observations")
        ball = geometry["ball_id"]
        checks.require(np.all((values[:, 0] == ball) | (values[:, 1] == ball)), "Contact geom pair omits the ball")
        others = np.where(values[:, 0] == ball, values[:, 1], values[:, 0])
        checks.require(set(others.tolist()) <= allowed, "Contact geom pair belongs to another category")
    normal = arrays["contact_normal"][active].astype(float)
    norms = np.linalg.norm(normal, axis=1)
    # Native CCD can return an exactly zero frame at a numerically tangent
    # contact. Such a frame carries no direction; it is permitted only inside
    # the independently fixed geometry epsilon and separately CPU-audited below.
    tangent_zero = (norms == 0) & (np.abs(distance[active]) <= GEOMETRY_EPS_M)
    checks.require(np.all((np.abs(norms - 1) <= 1e-4) | tangent_zero),
                   "Contact frame is neither unit nor a precision-bounded zero tangency")
    force_active = arrays["forces_N"][:, :, 1] > 0.01
    checks.require(not (force_active & ~active.any(axis=2)).any(), "Bucket force has no native contact geometry")
    # Logged contact position is associated with the pre-integration sphere.
    points = arrays["contact_position_m"].astype(float)
    separation = np.linalg.norm(points - arrays["position_m"][:-1, :, None, :], axis=-1)
    checks.require(np.all(separation[active] <= geometry["radius"] + np.maximum(0, -distance[active]) + GEOMETRY_EPS_M),
                   "Logged contact point is not near its pre-state sphere")


def degenerate_contact_queries(model, geometry, dataset, checks):
    """Independently verify every zero-frame CCD tangency, without direction use."""
    import mujoco
    arrays = dataset["arrays"]
    zero = np.isfinite(arrays["contact_distance_m"]) & ~arrays["contact_normal"].any(axis=-1)
    selected = np.argwhere(zero)
    data = geometry["data"]
    q = int(model.jnt_qposadr[model.joint("ball/free").id])
    mocap = int(model.body_mocapid[model.body("bucket").id])
    references, errors, magnitudes = [], [], []
    for tick, world, category in selected:
        pair = arrays["contact_geom_ids"][tick, world, category].astype(int)
        data.mocap_pos[mocap] = [*dataset["cases"][world]["bucket_xy_m"], 0]
        data.qpos[q:q+3] = arrays["position_m"][tick, world]
        data.qpos[q+3:q+7] = [1, 0, 0, 0]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        native = float(mujoco.mj_geomDistance(model, data, int(pair[0]), int(pair[1]), .05, None))
        logged = float(arrays["contact_distance_m"][tick, world, category])
        error = abs(native-logged)
        checks.require(abs(native) <= GEOMETRY_EPS_M and error <= GEOMETRY_EPS_M,
                       "Zero-frame contact is not a CPU-verified geometric tangency")
        errors.append(error); magnitudes.append(abs(native))
        references.append([int(tick), int(world), int(category), native, logged])
    return dict(zero_frame_contacts=len(selected), maximum_cpu_distance_error_m=max(errors, default=0.),
                maximum_cpu_tangency_distance_m=max(magnitudes, default=0.),
                all_zero_frame_contacts_checked=True, references_sha256=digest(references),
                treatment="No direction inferred from zero frames; only precision-bounded CPU-verified tangencies accepted")


def timing_queries(model, geometry, dataset, checks):
    """Determine pre/post alignment from independent CPU native distance queries."""
    import mujoco
    arrays = dataset["arrays"]
    p, distances = arrays["position_m"], arrays["contact_distance_m"]
    displacement = np.diff(p.astype(float), axis=0)
    normal_motion = np.abs((displacement[:, :, None, :] * arrays["contact_normal"]).sum(axis=-1))
    eligible = np.argwhere(np.isfinite(distances) & (distances < -25 * GEOMETRY_EPS_M) & (normal_motion > 10 * GEOMETRY_EPS_M))
    checks.require(len(eligible) >= 32, "Insufficient dynamic contacts to establish timing")
    selected = eligible[np.unique(np.linspace(0, len(eligible) - 1, min(128, len(eligible)), dtype=int))]
    data = geometry["data"]
    q = int(model.jnt_qposadr[model.joint("ball/free").id])
    mocap = int(model.body_mocapid[model.body("bucket").id])
    pre, post, references = [], [], []
    for tick, world, category in selected:
        pair = arrays["contact_geom_ids"][tick, world, category].astype(int)
        data.mocap_pos[mocap] = [*dataset["cases"][world]["bucket_xy_m"], 0]
        actual = []
        for point in (p[tick, world], p[tick + 1, world]):
            data.qpos[q:q + 3] = point
            data.qpos[q + 3:q + 7] = [1, 0, 0, 0]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            actual.append(float(mujoco.mj_geomDistance(model, data, int(pair[0]), int(pair[1]), 0.05, None)))
        observed = float(distances[tick, world, category])
        pre.append(abs(actual[0] - observed)); post.append(abs(actual[1] - observed))
        references.append([int(tick), int(world), int(category), observed, *actual])
    pre, post = np.asarray(pre), np.asarray(post)
    checks.require(float(pre.max()) <= GEOMETRY_EPS_M, "Contact distance does not match pre-integration geometry")
    checks.require(int(np.count_nonzero(pre < post)) >= math.ceil(0.9 * len(pre)), "Pre-state timing does not dominate post-state timing")
    checks.require(float(np.median(post)) >= max(5 * GEOMETRY_EPS_M, 10 * float(np.median(pre))), "Dynamic contacts do not distinguish pre/post timing")
    return dict(samples=len(pre), eligible_dynamic_contacts=len(eligible), sample_references_sha256=digest(references),
                pre_state_error_max_m=float(pre.max()), pre_state_error_median_m=float(np.median(pre)),
                pre_state_error_p95_m=float(np.quantile(pre, 0.95)),
                post_state_error_max_m=float(post.max()), post_state_error_median_m=float(np.median(post)),
                pre_state_wins=int(np.count_nonzero(pre < post)),
                established_contact_state="position_m[k] (pre-integration)",
                force_observation="Solver force observation for interval [time_s[k],time_s[k+1]], not an instantaneous post-state force",
                position_state="position_m[k+1] is post-integration",
                canonical_time="Exact dt-times-tick convention; native GPU clock accumulation/drift is recorded separately")


def check_summary(dataset, rows, checks):
    claims = dataset["summary"].get("rows")
    checks.require(isinstance(claims, list) and len(claims) == len(rows), "Control summary count changed")
    for actual, claimed in zip(rows, claims):
        checks.require(actual["id"] == claimed.get("id"), "Control summary order changed")
        for key in ("contact_wall_overlap_m", "contact_floor_overlap_m", "cavity_wall_overlap_m", "cavity_floor_overlap_m", "native_clock_drift_s", "final_position_local_m"):
            checks.close(actual[key], claimed.get(key), "Independent control summary disagreement: " + key, atol=2e-9)
        # Producer norms were computed in float32; independently recomputed f64
        # magnitudes can differ by the expected float32 arithmetic precision.
        checks.close(actual["max_bucket_domain_speed_m_s"], claimed.get("max_bucket_domain_speed_m_s"), "Independent speed summary disagreement", atol=2e-6)


def fit_bounds(rows):
    wall = max(max(row["contact_wall_overlap_m"], row["cavity_wall_overlap_m"]) for row in rows)
    base = max(max(row["contact_floor_overlap_m"], row["cavity_floor_overlap_m"]) for row in rows)
    return wall, base, 1.25 * wall + 8e-6, 1.25 * base + 8e-6


def check_score_config(config, geometry, rows, checks, *, fitted):
    from .calibrated_score import validate_calibration
    from .calibrated_score_oracle import validate_calibration as oracle_validate
    validate_calibration(config); oracle_validate(config)
    checks.require(config["compiled_planes"] == geometry["compiled_planes"], "Score configuration uses another compiled plane set")
    for key in ("compiled_floor_m", "compiled_rim_m", "wall_tolerance_cap_m", "floor_tolerance_cap_m"):
        checks.close(config[key], geometry[key], "Score configuration geometry/cap changed: " + key)
    checks.close(config["geometry_epsilon_m"], GEOMETRY_EPS_M, "Geometric epsilon was retuned")
    checks.close(config["max_speed_m_s"], 6.25, "Verified speed limit was retuned")
    checks.close(config["retention_s"], 1.0, "Retention duration was retuned")
    wall, base, wall_rule, base_rule = fit_bounds(rows)
    if fitted:
        checks.close(config["wall_tolerance_m"], wall_rule, "Wall envelope violates predeclared fit rule", atol=2e-9)
        checks.close(config["floor_tolerance_m"], base_rule, "Floor envelope violates predeclared fit rule", atol=2e-9)
    checks.require(config["wall_tolerance_m"] < config["wall_tolerance_cap_m"] and config["floor_tolerance_m"] < config["floor_tolerance_cap_m"],
                   "Contact envelope exceeds predeclared thickness cap")
    checks.require(max(row["verified_pre_post_bucket_speed_m_s"] for row in rows) <= config["max_speed_m_s"] + 1e-9,
                   "Independent bucket contact speed is outside verified domain")
    return dict(maximum_wall_overlap_m=wall, maximum_floor_overlap_m=base,
                maximum_pre_post_bucket_speed_m_s=max(row["verified_pre_post_bucket_speed_m_s"] for row in rows),
                wall_envelope_m=config["wall_tolerance_m"], floor_envelope_m=config["floor_tolerance_m"],
                wall_inside_envelope=wall <= config["wall_tolerance_m"] + GEOMETRY_EPS_M,
                floor_inside_envelope=base <= config["floor_tolerance_m"] + GEOMETRY_EPS_M)


def component_witnesses(scene, config, cases, arrays):
    """Containment-only witnesses, explicitly without a board-route claim."""
    plane = np.asarray(config["compiled_planes"], float)
    radius = scene["ball"]["radius_m"]
    floor, rim = config["compiled_floor_m"], config["compiled_rim_m"]
    times = arrays["time_s"]
    result = {}
    for world, case in enumerate(cases):
        p = arrays["position_m"][:, world].astype(float)
        x, y, z = p[:, 0] - case["bucket_xy_m"][0], p[:, 1] - case["bucket_xy_m"][1], p[:, 2]
        clearance = np.full(len(p), np.inf)
        for a, b, c, d in plane:
            clearance = np.minimum(clearance, a*x + b*y + c*z + d)
        within = (z + radius <= rim + GEOMETRY_EPS_M) & (z - radius >= floor - config["floor_tolerance_m"] - GEOMETRY_EPS_M) & (clearance >= radius - config["wall_tolerance_m"] - GEOMETRY_EPS_M)
        ids = np.arange(len(p))
        previous_outside = np.maximum.accumulate(np.where(~within, ids, -1))
        durations = np.zeros(len(p))
        active = np.flatnonzero(within)
        durations[active] = times[active] - times[previous_outside[active] + 1]
        result[case["id"]] = bool(durations.max() >= 1 - 1e-12)
    return result


def audit_dataset(directory, role, scene, assets, config, source, checks, *, diagnostic=False):
    dataset = load_dataset(directory, scene, role, checks)
    model, options = compile_cpu_model(scene, assets, dataset["substeps"])
    recorded = dataset["geometry"]
    checks.require(set(recorded["effective_options"]) == set(options), "Recorded native option keys differ")
    for key, value in options.items():
        checks.close(recorded["effective_options"][key], value, "Native effective physics option changed: " + key)
    geometry = inspect_compiled_geometry(model, scene, recorded, checks)
    if role == "fit":
        checks.require(dataset["substeps"] == 64 and dataset["order"] == "canonical", "Fit must use the predeclared canonical64 grid")
        config = read(Path(directory) / "candidate-score-config.json")
    validate_contact_observations(dataset, geometry, checks)
    rows = component_rows(scene, recorded, dataset["cases"], dataset["arrays"], checks)
    check_summary(dataset, rows, checks)
    bounds = check_score_config(config, recorded, rows, checks, fitted=role == "fit")
    if not diagnostic:
        checks.require(bounds["wall_inside_envelope"] and bounds["floor_inside_envelope"], "Independent verification exceeds frozen contact envelope")
    static = geometry_queries(model, geometry, scene, checks)
    timing = timing_queries(model, geometry, dataset, checks)
    degeneracy = degenerate_contact_queries(model, geometry, dataset, checks)
    witnesses = component_witnesses(scene, config, dataset["cases"], dataset["arrays"])
    # Only two unambiguous component controls are prespecified as label gates.
    # Rim/high-speed-wall outcomes are reported, not selected or relabeled.
    checks.require(witnesses[role + "-vertical-0"], "Deep central rest control was not retained")
    checks.require(not witnesses[role + "-outside"], "Outside negative has an inside-retention witness")
    for name, expected in dataset["hashes"].items():
        checks.require(sha(Path(directory) / name) == expected, "Control evidence changed during audit")
    clock_drift = max(row["native_clock_drift_s"] for row in rows)
    report = dict(role=role, substeps=dataset["substeps"], fixture_order=dataset["order"], fixture_count=len(rows),
                  evidence_sha256=dataset["hashes"], fixture_fingerprint=dataset["manifest"]["fingerprint"],
                  input_order_sha256=digest([case["id"] for case in dataset["cases"]]),
                  input_case_ids=[case["id"] for case in dataset["cases"]], source=source,
                  geometry_sha256=recorded["geometry_sha256"], geometry=static, timing=timing, contact_frame_degeneracy=degeneracy, envelope=bounds,
                  maximum_native_clock_drift_s=clock_drift,
                  containment_component_witnesses=witnesses,
                  scope="Isolated bucket/ball geometry/contact controls, not completed board-bounce shots",
                  diagnostic_only=bool(diagnostic), effective_options=options)
    return report, config


def _report_payload(version, checks, reference, journal_sha):
    root = Path(__file__).resolve().parents[2]
    return dict(version=version, status="passed", checks=checks.count, failures=[], synthetic_fixture=False,
                auditor_version=VERSION, auditor_sha256=sha(Path(__file__)),
                scorer_source_sha256={name:sha(root/name) for name in ("sircus/mjlab/calibrated_score.py", "sircus/mjlab/calibrated_score_oracle.py")},
                scene_sha256=digest(reference["scene"]), asset_sha256=reference["asset_hashes"],
                prelaunch_journal_sha256=journal_sha,
                provenance_scope="Inputs/geometry and source bytes verified after controlled launches; supplied journal preserves root's prelaunch producer hashes. No producer source manifest was embedded in raw caches.",
                packages={name:importlib.metadata.version(name) for name in ("mujoco", "numpy", "scipy")},
                limitations=["No new physics rollout was performed by this CPU audit.",
                             "Isolated component witnesses are not board-bounce route success or physical validation.",
                             "Canonical time is dt-times-tick; native floating-point clock drift is retained separately.",
                             "No policy/method outcomes were used to fit or verify these contact bounds."])


def audit_fit(args, checks):
    from .sweep_worker import load_reference
    producer = Path(args.producer_repository)
    reference = load_reference(args.reference, args.assets, repository=producer)
    journal = read(args.expected_producer_sha)
    source = source_bindings(producer, journal, reference, checks)
    result, config = audit_dataset(args.fit_dir, "fit", reference["scene"], args.assets, None, source, checks)
    report = _report_payload(CALIBRATION_REPORT_VERSION, checks, reference, sha(args.expected_producer_sha))
    report.update(score_config=config, score_config_sha256=digest(config), fit_rule=FIT_RULE, fit_evidence=result)
    report["checks"] = checks.count
    report["fingerprint"] = digest(report)
    return report


def verify(args, checks):
    from .sweep_worker import load_reference
    calibration = read(args.calibration)
    unsigned = dict(calibration); expected = unsigned.pop("fingerprint", None)
    checks.require(expected == digest(unsigned) and calibration.get("version") == CALIBRATION_REPORT_VERSION
                   and calibration.get("status") == "passed" and type(calibration.get("checks")) is int
                   and calibration["checks"] > 0 and calibration.get("failures") == [], "Calibration report did not pass or changed")
    checks.require(calibration.get("auditor_sha256") == sha(Path(__file__)), "Calibration auditor source changed")
    checks.require(calibration.get("score_config_sha256") == digest(calibration["score_config"]), "Calibration config changed")
    config = calibration["score_config"]
    reference = load_reference(args.reference, args.assets, repository=args.producer_repository)
    checks.require(calibration["scene_sha256"] == digest(reference["scene"]) and calibration["asset_sha256"] == reference["asset_hashes"], "Verification reference differs from calibration")
    journal = read(args.expected_producer_sha)
    checks.require(calibration["prelaunch_journal_sha256"] == sha(args.expected_producer_sha), "Prelaunch source journal changed between calibration and verification")
    for name, expected_hash in calibration["fit_evidence"]["evidence_sha256"].items():
        checks.require(sha(Path(args.fit_dir)/name) == expected_hash, "Calibration fit evidence changed")
    directories = list(args.verification_dir or [])
    producers = list(args.verification_producer_repository or [])
    checks.require(bool(directories) and len(producers) in (0, 1, len(directories)), "Verification directories/source lists differ")
    if not producers:producers = [args.producer_repository] * len(directories)
    elif len(producers) == 1:producers *= len(directories)
    results = []
    for directory, producer in zip(directories, producers):
        source = source_bindings(producer, journal, reference, checks)
        result, _ = audit_dataset(directory, "verification", reference["scene"], args.assets, config, source, checks)
        results.append(result)
    canonical = [r for r in results if r["substeps"] == 64 and r["fixture_order"] == "canonical"]
    fine = [r for r in results if r["substeps"] == 128 and r["fixture_order"] == "canonical"]
    reversed_order = [r for r in results if r["substeps"] == 64 and r["fixture_order"] == "reversed"]
    checks.require(len(canonical) >= 1 and len(fine) >= 1 and len(reversed_order) >= 1, "Required canonical64/fine128/reversed64 verification gates missing")
    checks.require(reversed_order[0]["input_case_ids"] == canonical[0]["input_case_ids"][::-1], "Reverse-order control inputs differ")
    diagnostics = []
    diag_dirs = list(args.diagnostic_dir or [])
    diag_sources = list(args.diagnostic_producer_repository or [])
    checks.require(len(diag_sources) in (0, 1, len(diag_dirs)), "Diagnostic directory/source lists differ")
    if not diag_sources:diag_sources = [args.producer_repository]*len(diag_dirs)
    elif len(diag_sources) == 1:diag_sources *= len(diag_dirs)
    for directory, producer in zip(diag_dirs, diag_sources):
        source = source_bindings(producer, journal, reference, checks)
        result, _ = audit_dataset(directory, "verification", reference["scene"], args.assets, config, source, checks, diagnostic=True)
        checks.require(result["substeps"] == 32, "Only coarse32 controls are diagnostic-only")
        result["supported_by_frozen64_envelope"] = bool(result["envelope"]["wall_inside_envelope"] and result["envelope"]["floor_inside_envelope"])
        diagnostics.append(result)
    report = _report_payload(VERIFICATION_REPORT_VERSION, checks, reference, sha(args.expected_producer_sha))
    report.update(calibration_sha256=sha(args.calibration), score_config_sha256=digest(config),
                  score_config_fingerprint=config["fingerprint"], verification=results, diagnostics=diagnostics,
                  required_gates=dict(canonical64=True, fine128=True, reversed64=True),
                  refitted_from_verification=False,
                  coarse_scope="32-substep observations are diagnostic-only, not pooled or certified as benchmark physics")
    report["checks"] = checks.count
    report["fingerprint"] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("fit-dir", "reference", "assets", "producer-repository", "expected-producer-sha", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    for key in ("verification-dir", "verification-producer-repository", "diagnostic-dir", "diagnostic-producer-repository"):
        parser.add_argument("--" + key, type=Path, action="append")
    args = parser.parse_args()
    protected = [args.fit_dir, *(args.verification_dir or []), *(args.diagnostic_dir or [])]
    if args.output.exists() or any(args.output.resolve().is_relative_to(Path(p).resolve()) for p in protected):
        parser.error("Audit output must be new and outside immutable control evidence")
    checks = Checks()
    start = time.monotonic()
    try:
        result = verify(args, checks) if args.calibration else audit_fit(args, checks)
    except Exception as exc:
        result = dict(version=VERIFICATION_REPORT_VERSION if args.calibration else CALIBRATION_REPORT_VERSION,
                      status="failed", checks=checks.count,
                      failures=[str(exc) if isinstance(exc, CalibrationAuditError) else type(exc).__name__ + ": control audit could not complete"],
                      auditor_version=VERSION, auditor_sha256=sha(Path(__file__)))
    result["elapsed_s"] = time.monotonic() - start
    # elapsed time is included in the final report fingerprint, while calibration
    # SHA binding remains exact and immutable for subsequent verification.
    result.pop("fingerprint", None)
    result["fingerprint"] = digest(result)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:result[k] for k in ("version", "status", "checks", "failures", "elapsed_s")}))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
