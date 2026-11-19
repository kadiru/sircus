"""Version dispatch cannot confuse legacy4-D evidence with expanded5-D proof."""

import copy
import json
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from action_training import (
    task_dispatch,
    training_formats,
    load_expanded_training_menu,
    verify_expanded_binding,
    digest,
)
from action_training_export import verify_action_geometry


class ExpandedTrainingAdapterTests(unittest.TestCase):
    def test_legacy_contract_stays_four_dimensional_with_original_schemas(self):
        module, expanded = task_dispatch(dict(version="sircus-cylinder-film-v2"))
        self.assertFalse(expanded)
        self.assertEqual(len(module.LOW), 4)
        formats = training_formats(dict(version="sircus-cylinder-film-v2"))
        self.assertEqual(formats["action_dim"], 4)
        self.assertEqual(formats["report_schema"], "sircus-action-training-film-v1")
        self.assertEqual(
            formats["audit_schema"], "sircus-action-training-checkpoint-audit-v1"
        )
        self.assertEqual(formats["protocol"], "sircus-fixed-camera-on-policy-film-v2")

    def test_version_is_whitelisted_and_not_inferred_from_a_claimed_dimension(self):
        for spec in (
            {},
            dict(version="future", action_dim=5),
            dict(version="sircus-cylinder-film-v3"),
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                task_dispatch(spec)
        fake = types.SimpleNamespace(LOW=np.zeros(5))
        with patch.dict(sys.modules, {"expanded_farm": fake}):
            module, expanded = task_dispatch(
                dict(version="sircus-expanded-cylinder-film-v1")
            )
            self.assertIs(module, fake)
            self.assertTrue(expanded)
            formats = training_formats(dict(version="sircus-expanded-cylinder-film-v1"))
            self.assertEqual(formats["action_dim"], 5)
            self.assertEqual(formats["clip_schema"], "sircus-action-training-clip-v2")
            self.assertEqual(
                formats["audit_schema"], "sircus-action-training-checkpoint-audit-v2"
            )

    def test_expanded_adapter_requires_two_native_options_for_every_fixed_site(self):
        spec = dict(version="sircus-expanded-cylinder-film-v1")
        menus = [[dict(id=f"{i}-a"), dict(id=f"{i}-b")] for i in range(35)]
        loader = types.SimpleNamespace(
            load_native_menu=lambda path, spec: (menus, dict(file_sha256="proof"))
        )
        with patch.dict(sys.modules, {"expanded_pose_menu": loader}):
            got, audit = load_expanded_training_menu(Path("qualified.json"), spec)
            self.assertEqual(len(got), 35)
            menus[-1].pop()
            with self.assertRaisesRegex(ValueError, "all 35"):
                load_expanded_training_menu(Path("qualified.json"), spec)
            menus[-1].clear()
            with self.assertRaises(ValueError):
                load_expanded_training_menu(Path("qualified.json"), spec)

    def test_fifth_radius_is_explicit_and_measured_from_unchanged_bucket(self):
        cell = dict(bucket_xy_m=[0.1, 0.2])
        row = dict(
            bucket_xy_m=[0.1, 0.2],
            board_x_m=0.7,
            board_y_m=0.2,
            distance_m=0.6,
            radius_m=0.6,
        )
        verify_action_geometry(row, dict(radius_m=0.6), cell, True)
        for name, value in (
            ("radius_m", 0.4),
            ("board_x_m", 0.8),
            ("distance_m", 0.4),
            ("bucket_xy_m", [0.2, 0.2]),
        ):
            changed = dict(row, **{name: value})
            with self.assertRaises(ValueError):
                verify_action_geometry(changed, dict(radius_m=0.6), cell, True)
        with self.assertRaises(ValueError):
            verify_action_geometry(row, dict(radius_m=0.6), cell, False)

    def test_original_camera_source_and_new_experiment_binding_are_distinct(self):
        spec = dict(
            version="sircus-expanded-cylinder-film-v1", fingerprint="new-experiment"
        )
        camera = dict(
            source_fingerprint="original-camera-scene",
            eye_m=[0, 0, 1],
            target_m=[1, 0, 0],
        )
        binding = dict(
            source_fingerprint="original-camera-scene",
            experiment_fingerprint="new-experiment",
            camera_sha256="exact-original-bytes",
        )
        fake = types.SimpleNamespace(
            LOW=np.zeros(5),
            bind_camera=lambda path, spec, origins: (
                copy.deepcopy(camera),
                copy.deepcopy(binding),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"expanded_farm": fake}
        ):
            run = Path(tmp)
            (run / "camera-binding.json").write_text(json.dumps(binding))
            report = dict(
                camera=camera,
                origins_m=[[0, 0, 0]] * 35,
                camera_source_fingerprint=camera["source_fingerprint"],
                camera_binding_sha256=digest(run / "camera-binding.json"),
                task_version=spec["version"],
                action_dim=5,
            )
            self.assertEqual(
                verify_expanded_binding(run, spec, report)["binding"], binding
            )
            report["camera_source_fingerprint"] = "new-experiment"
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                verify_expanded_binding(run, spec, report)
            report["camera_source_fingerprint"] = camera["source_fingerprint"]
            (run / "camera-binding.json").write_text(
                json.dumps(dict(binding, experiment_fingerprint="old-experiment"))
            )
            report["camera_binding_sha256"] = digest(run / "camera-binding.json")
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                verify_expanded_binding(run, spec, report)


@unittest.skipUnless(
    importlib.util.find_spec("torch"),
    "Torch CPU is required for five-dimensional checkpoint verification",
)
class ExpandedCheckpointAdapterTests(unittest.TestCase):
    def test_five_dimensional_cpu_updates_and_strict_checkpoint_shape(self):
        import torch
        import expanded_farm as task
        from farm_policy import Policy, update, weight_hash
        from action_training_audit import verify_checkpoint_chain
        from action_training import training_formats

        torch.set_num_threads(2)
        torch.manual_seed(31)
        scene = json.loads((SOURCE / "scene.json").read_text())
        spec = task.make_spec(scene, 10)
        formats = training_formats(spec)
        extra = dict(
            task_version=spec["version"],
            action_dim=5,
            camera_source_fingerprint=spec["camera_source_fingerprint"],
            camera_binding_sha256="fixture-camera-binding",
        )
        # These are synthetic geometry fixtures. The native-proof verifier is
        # tested separately; this fixture isolates actual CPU likelihood replay.
        menus = []
        for cell in spec["cells"]:
            a = task.shot(cell, task.WARM)
            b = task.shot(cell, task.WARM + np.array([0, 0, 0.02, 0, 0]))
            menus.append([dict(a, id="fixture-a"), dict(b, id="fixture-b")])
        menu_audit = dict(file_sha256="fixture-native-menu")
        policy = Policy(task.LOW, task.HIGH, task.WARM, task.LOGSTD).cpu()
        optimizer = torch.optim.Adam(policy.parameters(), lr=spec["lr"])
        obs = torch.tensor(
            [task.observation(c) for c in spec["cells"]], dtype=torch.float32
        )
        first = weight_hash(policy)
        changes = []
        summaries = []
        buffer = []
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "contract.json").write_text(json.dumps(spec))
            for index in range(10):
                before = weight_hash(policy)
                name = f"policy-{index:02}.pt"
                torch.save(
                    dict(
                        state_dict=policy.state_dict(),
                        optimizer=optimizer.state_dict(),
                        source_fingerprint=spec["fingerprint"],
                        updates=index // 2,
                        menu_sha256=menu_audit["file_sha256"],
                        weight_hash=before,
                        **extra,
                    ),
                    run / name,
                )
                with torch.no_grad():
                    dist = policy.distribution(obs)
                    latent = dist.sample()
                    logp = dist.log_prob(latent).sum(-1)
                returns = torch.sin(latent[:, 0])
                buffer.append((obs, latent, logp, returns))
                actions = task.project_actions(spec["cells"], latent.numpy(), menus)
                rows = [
                    dict(
                        a,
                        cell_id=c["id"],
                        bucket_xy_m=c["bucket_xy_m"],
                        distance_m=a["radius_m"],
                    )
                    for c, a in zip(spec["cells"], actions)
                ]
                clip = dict(
                    schema=formats["clip_schema"],
                    batch=index,
                    checkpoint_file=name,
                    checkpoint_sha256=digest(run / name),
                    source_fingerprint=spec["fingerprint"],
                    menu_sha256=menu_audit["file_sha256"],
                    policy_hash=before,
                    observation=obs.tolist(),
                    latent=latent.tolist(),
                    old_logp=logp.tolist(),
                    release_enabled=[True] * 35,
                    actions=actions,
                    rollout=rows,
                    **extra,
                )
                filename = f"clip-{index:02}.json"
                (run / filename).write_text(json.dumps(clip))
                summaries.append(dict(batch=index, file=filename, policy_hash=before))
                if index % 2:
                    changes.append(
                        update(
                            policy,
                            optimizer,
                            *[torch.cat(v) for v in zip(*buffer)],
                            spec,
                        )
                    )
                    buffer.clear()
            final = weight_hash(policy)
            torch.save(
                dict(
                    state_dict=policy.state_dict(),
                    optimizer=optimizer.state_dict(),
                    contract=spec,
                    menu_sha256=menu_audit["file_sha256"],
                    weight_hash=final,
                    **extra,
                ),
                run / "policy-final.pt",
            )
            report = dict(
                schema=formats["report_schema"],
                source_fingerprint=spec["fingerprint"],
                fingerprint=spec["fingerprint"],
                workcells=35,
                articulations=70,
                initial_hash=first,
                final_hash=final,
                menu_sha256=menu_audit["file_sha256"],
                menu_provenance=menu_audit,
                batches=summaries,
                updates=changes,
                final_checkpoint_sha256=digest(run / "policy-final.pt"),
                **extra,
            )
            (run / "result.json").write_text(json.dumps(report))
            with patch(
                "action_training.load_expanded_training_menu",
                return_value=(menus, menu_audit),
            ), patch(
                "action_training.verify_expanded_binding",
                return_value=dict(sha256="fixture-camera-binding"),
            ):
                audit = verify_checkpoint_chain(run)
                self.assertEqual(
                    audit["schema"], "sircus-action-training-checkpoint-audit-v2"
                )
                self.assertEqual(audit["action_dim"], 5)
                self.assertEqual(audit["latent_actions_checked"], 350)
                self.assertLess(audit["latent_logprob_max_abs_error"], 1e-6)
                path = run / "clip-00.json"
                original_clip = path.read_text()
                clip = json.loads(original_clip)
                clip["latent"][0].pop()
                path.write_text(json.dumps(clip))
                with self.assertRaisesRegex(ValueError, "sampled latents"):
                    verify_checkpoint_chain(run)
                # Updating the serialized file checksum cannot make an old
                # four-output network compatible with the new five-output one.
                import cylinder_farm as legacy

                saved = torch.load(
                    run / "policy-00.pt", map_location="cpu", weights_only=True
                )
                old_policy = Policy(legacy.LOW, legacy.HIGH, legacy.WARM, legacy.LOGSTD)
                saved["state_dict"] = old_policy.state_dict()
                torch.save(saved, run / "policy-00.pt")
                clip = json.loads(original_clip)
                clip["checkpoint_sha256"] = digest(run / "policy-00.pt")
                path.write_text(json.dumps(clip))
                with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                    verify_checkpoint_chain(run)


if __name__ == "__main__":
    unittest.main()
