"""Generate a standalone local fake-only Viam configuration; never edits cloud machines."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def build(root=ROOT):
    root = Path(root).resolve()
    template = json.loads((ROOT / "viam-config.template.json").read_text())
    def replace(value):
        if isinstance(value, str):
            return value.replace("__ROOT__", str(root))
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return value
    return replace(template)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, help="Optional existing full physics scene JSON")
    args = parser.parse_args()
    from physics import default_config, validate_config
    scene = json.loads(args.scene.read_text()) if args.scene else default_config()
    validate_config(scene)
    scene_path = ROOT / "viam-scene.json"
    if args.scene or not scene_path.exists():
        scene_path.write_text(json.dumps(scene, indent=2) + "\n")
    output = ROOT / "viam-config.local.json"
    output.write_text(json.dumps(build(), indent=2) + "\n")
    print(output)
