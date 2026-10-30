"""Pin this module to one cloud-bound, fake-only replay machine."""
import hashlib
import json
import os
import stat
import time
from pathlib import Path
from uuid import UUID
from build_config import expected_config
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
TEMPLATE_HASH = "0815a2ad801827a0e473ae675f7f0fc43e1c5c023507e5f1d648b13afd096cae"
GRIPPER_HASH = "109af8da38e419a2f2f69e9aaeae06ca0ae3153c9cb4b3dcf959872710ce5518"
ORG = "6d6c7293-bc67-43e4-9c8b-31c76aac27d3"
LOCATION = "ty1cywh7tq"


def binding():
    value = json.loads((ROOT / "binding.json").read_text())
    if value.get("organization_id") != ORG or value.get("location_id") != LOCATION:
        raise RuntimeError("Wrong simulation organization/location binding")
    if value.get("machine_name") != "sircus-sim":
        raise RuntimeError("Wrong simulation machine binding")
    for field in ("machine_id", "part_id"):
        if str(UUID(value[field])) != value[field]:
            raise RuntimeError("Invalid simulation binding")
    return value



def _vector(value):
    if set(value) == {"X", "Y", "Z"}:
        return {axis: value[axis.upper()] for axis in ("x", "y", "z")}
    return value


def _orientation(value):
    kind=value.get("type"); v=value.get("value",{})
    if kind == "quaternion":
        q=[v.get(k,v.get(k.lower(),0)) for k in ("X","Y","Z","W")]
        matrix=Rotation.from_quat(q).as_matrix()
    elif kind == "ov_degrees":
        axis=np.array([v.get(k,0) for k in ("x","y","z")],float)
        # Configuration frames here are Z rotations only; reject other axes.
        if not np.allclose(axis,[0,0,1]): return value
        matrix=Rotation.from_euler('z',v.get('th',0),degrees=True).as_matrix()
    else: return value
    return {"matrix":np.round(matrix,10).tolist()}


def _normalize_frame(value):
    frame = dict(value)
    if frame.get("id") == "":
        frame.pop("id")
    if "translation" in frame:
        frame["translation"] = _vector(frame["translation"])
    if "orientation" in frame:
        frame["orientation"] = _orientation(frame["orientation"])
    if "geometry" in frame:
        shape = dict(frame["geometry"])
        if shape.get("type") == "box":
            for field in ("l", "r"):
                if shape.get(field) == 0:
                    shape.pop(field)
            if shape.get("Label") == "":
                shape.pop("Label")
            if _vector(shape.get("translation", {})) == {"x": 0, "y": 0, "z": 0}:
                shape.pop("translation")
            identity = {"matrix":np.eye(3).tolist()}
            if _orientation(shape.get("orientation", {})) == identity:
                shape.pop("orientation")
        frame["geometry"] = shape
    return frame


def _normalize_resource(resource):
    """Discard only empty serializer defaults and non-behavioral log level."""
    result = dict(resource)
    for key in tuple(result):
        if key not in {"attributes"} and result[key] in (None, "", [], {}):
            result.pop(key)
    if result.get("type") == result.get("api", "").rsplit(":", 1)[-1]:
        result.pop("type")
    if result.get("namespace") == "rdk":
        result.pop("namespace")
    if result.get("log_level") in ("debug", "info", "warn", "error"):
        result.pop("log_level")
    if "frame" in result:
        result["frame"] = _normalize_frame(result["frame"])
    result.setdefault("attributes", {}) if "api" in result else None
    return result


def validate_cached_config(config, bound):
    if config.get("cloud", {}).get("id") != bound["part_id"]:
        raise RuntimeError("Refusing a different cloud machine part")
    for field in ("remotes", "processes", "packages", "fragments", "jobs", "triggers"):
        if config.get(field):
            raise RuntimeError("Additional resources are not allowed in this simulation")
    expected = expected_config()
    for kind in ("components", "services", "modules"):
        items = config.get(kind, [])
        if len(items) != len(expected[kind]):
            raise RuntimeError("Simulation resource set changed")
        actual_by_name = {item.get("name"): _normalize_resource(item) for item in items}
        wanted = {item["name"]: _normalize_resource(item) for item in expected[kind]}
        if actual_by_name != wanted:
            raise RuntimeError("Simulation resource configuration changed: " + kind)
    network = config.get("network", {})
    if network.get("bind_address") != "127.0.0.1:8082":
        raise RuntimeError("Simulation must bind only to loopback port 8082")
    for name, digest in json.loads((ROOT/'model-hashes.json').read_text()).items():
        if hashlib.sha256((ROOT/'models'/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Pinned simulation geometry changed")
    return config


def check_simulation_config():
    if os.environ.get("SIRCUS_LOCAL_TEST") == "1":
        cfg=json.loads((ROOT/'viam-config.test.json').read_text())
        if cfg != expected_config(test=True): raise RuntimeError('Local fake-only test config changed')
        for name,digest in json.loads((ROOT/'model-hashes.json').read_text()).items():
            if hashlib.sha256((ROOT/'models'/name).read_bytes()).hexdigest()!=digest: raise RuntimeError('Model changed')
        return cfg
    bound = binding()
    cache = Path.home() / ".viam" / ("cached_cloud_config_" + bound["part_id"] + ".json")
    launch_path=ROOT/"data/launch.json"
    if not launch_path.exists(): launch_path=ROOT.parent/"viam-cloud/data/launch.json"
    launch = json.loads(launch_path.read_text())
    if launch.get("part_id") != bound["part_id"] or cache.stat().st_mtime_ns < launch["started_ns"] - 2_000_000_000:
        raise RuntimeError("Cloud configuration cache predates this simulation launch")
    if cache.is_symlink() or cache.stat().st_uid != os.getuid() or cache.stat().st_mode & 0o022:
        raise RuntimeError("Unexpected cloud configuration cache ownership/permissions")
    return validate_cached_config(json.loads(cache.read_text()), bound)


def prepare_launch():
    bound = binding()
    secret_path = ROOT.parent / "viam-cloud/.private/viam-connection.json"
    info = secret_path.stat()
    if secret_path.is_symlink() or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise RuntimeError("Connection JSON must be owned by you with mode 600")
    connection = json.loads(secret_path.read_text())
    if set(connection) != {"cloud"} or connection["cloud"].get("id") != bound["part_id"]:
        raise RuntimeError("Connection JSON must contain only the bound simulation cloud connection")
    if not connection["cloud"].get("secret"):
        raise RuntimeError("Simulation connection secret is missing")
    if json.loads((ROOT / "viam-config.cloud.json").read_text()) != expected_config():
        raise RuntimeError("Cloud deployment configuration changed")
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data/launch.json").write_text(json.dumps({"part_id": bound["part_id"], "started_ns": time.time_ns()}))


if __name__ == "__main__":
    prepare_launch()
    print("Bound simulation connection validated; credentials not displayed.")
