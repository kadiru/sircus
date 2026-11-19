"""CPU-only live reuse checks; synthetic outcomes are never capture evidence."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import interactive_fast_planner as live
from sircus.mjlab.fast_planning_plan import incremental_fast_plan


_engine = None


def fake_engine(directory):
    base = SimpleNamespace(directory=Path(directory), _resolved={"old": 1}, _connected={"old": 1})
    resolver = SimpleNamespace(base=base, pending=deque(), connections={"old": 1})
    adapter = SimpleNamespace(directory=Path(directory), env=None, resolver=resolver,
                              _ids={"old"}, packet=7, orientation_by_id={"old": 1})
    return SimpleNamespace(directory=Path(directory), adapter=adapter, prepared_ids={"old"},
                           _anchors={"old": 1}, model=object())


def toy_initialize(directory):
    global _engine
    _engine = fake_engine(directory)


def toy_job(layout, method, directory, *, publish, deadline, profile):
    live._bind_fresh_request(_engine, Path(directory) / "requests" / layout["id"])
    if layout.get("fail"):
        raise ValueError("injected failure")
    if layout.get("hang"):
        while True:
            time.sleep(.01)
    data = dict(layout_id=layout["id"], bucket=layout["bucket_xy_m"], pid=os.getpid())
    path = _engine.directory / "proof.json"
    path.write_text(json.dumps(data))
    publish(dict(layout=layout, choice=dict(request=data), completed_proposal_ids=[layout["id"]]))
    return "completed"


def lane(directory, budget=.3):
    planner = live.LiveReusablePlanner("warm_early", {}, {}, {}, {}, {}, reference={}, assets=directory,
                                      directory=directory, engine_kwargs={}, budget_s=budget)
    planner.worker.function = toy_job
    planner.worker.common_args = (Path(directory),)
    planner.worker.initializer = toy_initialize
    planner.worker.initargs = (Path(directory),)
    return planner


def layout(name, **kwargs):
    return dict(id=name, bucket_xy_m=[.4, -.3], stratum="test", **kwargs)


class LivePlannerTests(unittest.TestCase):
    def test_rebinding_discards_query_state_preserves_models_and_old_proofs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = fake_engine(root)
            model = engine.model
            live._bind_fresh_request(engine, root / "one")
            (root / "one" / "proof").write_text("old immutable evidence")
            engine.adapter.resolver.base._resolved["same-xy"] = "old proof"
            live._bind_fresh_request(engine, root / "two")
            self.assertIs(engine.model, model)
            self.assertEqual(engine.adapter.resolver.base._resolved, {})
            self.assertEqual(engine.adapter.resolver.base._connected, {})
            self.assertEqual(engine.prepared_ids, set())
            self.assertEqual(engine.adapter._ids, set())
            self.assertEqual(engine.adapter.orientation_by_id, {})
            self.assertEqual(engine.adapter.packet, 0)
            self.assertEqual(engine.directory, root / "two")
            self.assertEqual(engine.adapter.directory, root / "two")
            self.assertEqual(engine.adapter.resolver.base.directory, root / "two")
            self.assertEqual((root / "one" / "proof").read_text(), "old immutable evidence")
            with self.assertRaises(FileExistsError):
                live._bind_fresh_request(engine, root / "one")

    def test_native_or_pending_state_cannot_rebind(self):
        for pending, env in ((deque([1]), None), (deque(), object())):
            engine = fake_engine("unused")
            engine.adapter.resolver.pending = pending
            engine.adapter.env = env
            with self.assertRaises(RuntimeError):
                live._bind_fresh_request(engine, "must-not-create")

    def test_repeated_xy_reuses_worker_with_separate_immutable_evidence(self):
        with tempfile.TemporaryDirectory() as directory, lane(directory) as planner:
            first = planner.plan(layout("first"))
            proof = Path(directory) / "requests/first/proof.json"
            original = proof.read_bytes()
            second = planner.plan(layout("second"))
            self.assertFalse(first["worker_reused"])
            self.assertTrue(second["worker_reused"])
            self.assertEqual(first["worker_pid"], second["worker_pid"])
            self.assertEqual(proof.read_bytes(), original)
            self.assertNotEqual(original, (Path(directory) / "requests/second/proof.json").read_bytes())
            self.assertNotEqual(first["journal_directory"], second["journal_directory"])
            with self.assertRaises(FileExistsError):
                planner.plan(layout("second"))
            self.assertIsNone(planner.worker.process)

    def test_timeout_failure_recovery_and_close(self):
        with tempfile.TemporaryDirectory() as directory:
            planner = lane(directory, .1)
            try:
                timed = planner.plan(layout("timeout", hang=True))
                self.assertEqual(timed["status"], "deadline")
                self.assertIsNone(timed["choice"]["request"])
                self.assertIsNone(planner.worker.process)
                recovered = planner.plan(layout("recovered"))
                self.assertFalse(recovered["worker_reused"])
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    planner.plan(layout("failure", fail=True))
                self.assertIsNone(planner.worker.process)
                final = planner.plan(layout("final"))
                self.assertFalse(final["worker_reused"])
            finally:
                planner.close()
            self.assertIsNone(planner.worker.process)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                planner.plan(layout("after-close"))

    def test_concurrent_request_refused_without_discarding_active_worker(self):
        with tempfile.TemporaryDirectory() as directory, lane(directory) as planner:
            planner._live_lock.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "one live"):
                    planner.plan(layout("concurrent"))
                self.assertFalse((Path(directory) / "requests").exists())
            finally:
                planner._live_lock.release()

    def test_unchanged_frozen_algorithm_is_called_after_scoped_rebinding(self):
        engine = fake_engine("unused")
        with tempfile.TemporaryDirectory() as directory:
            with patch("sircus.mjlab.parallel_capture_grid_workers._engine", engine), \
                    patch.object(live, "incremental_fast_plan", return_value="completed") as frozen:
                item = layout("request")
                self.assertEqual(live._isolated_plan(item, "warm_early", directory, "input", publish="cb"), "completed")
                frozen.assert_called_once_with(item, "warm_early", "input", publish="cb")
                self.assertEqual(engine.directory, Path(directory) / "requests/request")
        self.assertIs(live.incremental_fast_plan, incremental_fast_plan)

    def test_tampered_published_artifacts_block_before_any_planner_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["report.json", "choices.json", "evidence-manifest.json", "audit"]
            hashes = {}
            for name in names:
                (root / name).write_text(name)
                hashes[name] = hashlib.sha256(name.encode()).hexdigest()
            with patch.object(live, "PUBLISHED", hashes):
                live._verify_published(root, root / "audit")
                for name in names:
                    (root / name).write_text("changed")
                    with self.assertRaisesRegex(ValueError, "evidence changed"):
                        live.verified_inputs(SimpleNamespace(fast_study=root, fast_audit=root / "audit"), None)
                    (root / name).write_text(name)


if __name__ == "__main__":
    unittest.main()
