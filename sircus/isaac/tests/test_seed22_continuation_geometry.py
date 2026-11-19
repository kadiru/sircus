"""Synthetic wrapper/delegation tests; no model, native simulator or shot runs."""

import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seed22_continuation_geometry as subject


def signed(value):
    value = copy.deepcopy(value)
    value.pop("fingerprint", None)
    return dict(value, fingerprint=subject.geometry.canonical_hash(value))


def result(cell, original, classification="unchanged", delta=.002):
    action = copy.deepcopy(original)
    if classification == "adjusted":
        values = [action[k] for k in subject.geometry.ACTION_NAMES]
        values[2] += delta
        action = subject.geometry.decode_action(cell, values, original["id"])
    accepted = classification != "rejected"
    proof = None if not accepted else dict(accepted=True, release_authorized=False,
        action=action, qualification=dict(ik={n:dict(q=[.25] * 6) for n in subject.ARMS}))
    return signed(dict(schema=subject.layer.VERSION, classification=classification,
        accepted=accepted, action=action if accepted else None,
        original_action=copy.deepcopy(original), final_preflight=proof,
        release_authorized=False, bucket_impossibility_proven=False,
        native_shots=0, training_updates=0,
        binding=dict(protocol=subject.layer._protocol(True, 24, 512, .10)),
        scope="Explicitly synthetic filter result, not native or geometry evidence"))


class FakeEnv:
    n = 35
    trial_mode = "train"

    def __init__(self):
        self.cfg = dict(synthetic=True)
        self.base_cells = [dict(id=f"synthetic-{i}", split="train", bucket_xy_m=[.27+i*.001, -.1])
                           for i in range(35)]
        self.active_mask = np.array([True, True] + [False] * 33)
        self.support_missing = [False] * 35
        self.proposal_ik_seeds = None
        self.continuous_links = {n: [] for n in subject.ARMS}
        self.arms = {n:SimpleNamespace(data=SimpleNamespace(joint_pos=
            np.arange(35*8, dtype=float).reshape(35, 8) / 1000 + j))
            for j, n in enumerate(subject.ARMS)}
        self.calls = []

    def actions(self):
        return [subject.geometry.decode_action(c, [0., .4, .35, .24, math.radians(29)],
                                              f"proposal-{i}") for i, c in enumerate(self.base_cells)]

    def prepare(self, actions):
        # Observe delegation and the original gate inputs, without simulating
        # arm movement. The existing native-controller tests verify its physics.
        seen = dict(actions=copy.deepcopy(actions), support=copy.deepcopy(self.support_missing),
                    seeds=copy.deepcopy(self.proposal_ik_seeds), active=self.active_mask.copy())
        self.calls.append(seen)
        return seen


class GeometryHookTests(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.actions = self.env.actions()
        self.filter_calls = []
        self.classifications = ["unchanged"] * 35

    def fake_filter(self, scene, cell, action, links, starts, **kwargs):
        self.filter_calls.append(copy.deepcopy(dict(cell=cell, action=action, starts=starts, kwargs=kwargs)))
        index = int(cell["id"].rsplit("-", 1)[1])
        return result(cell, action, self.classifications[index])

    def attach(self, **kwargs):
        return subject.install(self.env, filter_fn=kwargs.pop("filter_fn", self.fake_filter), **kwargs)

    def test_unchanged_paid_actions_delegate_once_without_proposal_changes(self):
        before = copy.deepcopy(self.actions)
        self.attach()
        seen = self.env.prepare(self.actions)
        self.assertEqual(len(self.filter_calls), 2)
        self.assertEqual(len(self.env.calls), 1)
        self.assertEqual(self.actions, before)
        self.assertEqual(self.env.proposed_actions, before)
        self.assertEqual(self.env.geometry_actions, before)
        self.assertEqual(seen["support"], [False] * 35)
        self.assertEqual(self.filter_calls[0]["kwargs"], dict(repair_max_nfev=24,
            repair_max_evaluations=512, max_normalized_adjustment=.10))
        self.assertTrue(self.env.geometry_filter_synthetic)
        self.assertIsNone(self.env.geometry_records[2])
        self.assertIsNone(self.env.geometry_starts[2])

    def test_adjusted_action_mutates_same_dictionary_and_preserves_original(self):
        self.classifications[0] = "adjusted"
        before = copy.deepcopy(self.actions)
        identity = id(self.actions[0])
        self.attach()
        seen = self.env.prepare(self.actions)
        self.assertEqual(id(self.actions[0]), identity)
        self.assertNotEqual(self.actions[0], before[0])
        self.assertEqual(self.actions[1:], before[1:])
        self.assertEqual(self.env.proposed_actions, before)
        self.assertEqual(seen["actions"], self.env.geometry_actions)
        self.assertEqual(seen["seeds"][0], {n:dict(q=[.25]*6) for n in subject.ARMS})

    def test_actual_measured_starts_are_fresh_each_batch(self):
        self.attach()
        self.env.prepare(self.actions)
        initial = copy.deepcopy(self.env.geometry_starts)
        for arm in self.env.arms.values():
            arm.data.joint_pos += .125
        self.env.prepare(self.env.actions())
        for n in subject.ARMS:
            self.assertTrue(np.allclose(self.env.geometry_starts[0][n],
                                        np.asarray(initial[0][n])+.125))
        self.assertEqual(self.filter_calls[2]["starts"], self.env.geometry_starts[0])

    def test_declined_paid_slot_keeps_paid_mask_and_blocks_original_native_gate(self):
        self.classifications[0] = "rejected"
        before = copy.deepcopy(self.actions)
        self.attach()
        seen = self.env.prepare(self.actions)
        self.assertTrue(seen["active"][0])
        self.assertTrue(seen["support"][0])
        self.assertIsNone(seen["seeds"][0])
        self.assertEqual(self.actions, before)
        self.assertEqual(self.env.geometry_records[0]["classification"], "rejected")

    def test_next_batch_does_not_inherit_previous_geometry_decline(self):
        self.classifications[0] = "rejected"
        self.attach()
        self.env.prepare(self.actions)
        self.classifications[0] = "unchanged"
        seen = self.env.prepare(self.env.actions())
        self.assertFalse(seen["support"][0])
        self.assertIsNotNone(seen["seeds"][0])

    def test_raw_parent_bypasses_filter_and_clears_previous_geometry_state(self):
        self.classifications[0] = "rejected"
        self.attach()
        self.env.prepare(self.actions)
        self.env.trial_mode = "raw_parent"
        actions = self.env.actions()
        before = copy.deepcopy(actions)
        seen = self.env.prepare(actions)
        self.assertEqual(len(self.filter_calls), 2)
        self.assertEqual(actions, before)
        self.assertEqual(seen["support"], [False]*35)
        self.assertIsNone(seen["seeds"])
        self.assertFalse(self.env.geometry_enabled_this_batch)
        self.assertEqual(self.env.geometry_records, [None]*35)

    def test_disabled_wrapper_never_reads_joints_or_filters(self):
        self.attach(enabled=False)
        del self.env.arms
        self.env.prepare(self.actions)
        self.assertEqual(self.filter_calls, [])
        self.assertFalse(self.env.geometry_enabled_this_batch)
        self.assertEqual(self.env.geometry_wall_s, 0.)
        self.assertEqual(self.env.geometry_process_cpu_s, 0.)

    def test_geometry_cost_excludes_delegated_native_prepare(self):
        def native(actions):
            self.assertEqual(self.env.geometry_wall_s, 2.75)
            self.assertEqual(self.env.geometry_process_cpu_s, 1.25)
            return "synthetic native result"
        self.env.prepare = native
        self.attach()
        with patch.object(subject.time, "perf_counter", side_effect=[10., 12.75]) as wall, \
                patch.object(subject.time, "process_time", side_effect=[20., 21.25]) as cpu:
            self.assertEqual(self.env.prepare(self.actions), "synthetic native result")
            self.assertEqual(wall.call_count, 2)
            self.assertEqual(cpu.call_count, 2)

    def test_failed_geometry_still_records_its_cost_without_native_call(self):
        def failed(*args, **kwargs):
            raise RuntimeError("synthetic filter interruption")
        self.attach(filter_fn=failed)
        with patch.object(subject.time, "perf_counter", side_effect=[10., 11.5]), \
                patch.object(subject.time, "process_time", side_effect=[20., 20.75]):
            with self.assertRaisesRegex(RuntimeError, "filter interruption"):
                self.env.prepare(self.actions)
        self.assertEqual(self.env.geometry_wall_s, 1.5)
        self.assertEqual(self.env.geometry_process_cpu_s, .75)
        self.assertEqual(self.env.calls, [])

    def test_raw_parent_resets_previous_batch_geometry_cost_to_zero(self):
        self.attach()
        with patch.object(subject.time, "perf_counter", side_effect=[10., 12.]), \
                patch.object(subject.time, "process_time", side_effect=[20., 21.]):
            self.env.prepare(self.actions)
        self.assertEqual(self.env.geometry_wall_s, 2.)
        self.env.trial_mode = "raw_parent"
        with patch.object(subject.time, "perf_counter", side_effect=AssertionError("No geometry timer")), \
                patch.object(subject.time, "process_time", side_effect=AssertionError("No geometry timer")):
            self.env.prepare(self.env.actions())
        self.assertEqual(self.env.geometry_wall_s, 0.)
        self.assertEqual(self.env.geometry_process_cpu_s, 0.)

    def test_production_filter_path_revalidates_each_original_proof_before_native(self):
        # Explicitly mocked original-filter/validator entry points. This verifies
        # wiring only; it does not certify a numerical or native result.
        production_filter = Mock(side_effect=self.fake_filter)
        with patch.object(subject.layer, "filter_shot", production_filter), \
                patch.object(subject.layer, "validate_result") as validator:
            subject.install(self.env, filter_fn=production_filter)
            self.env.prepare(self.actions)
            self.assertEqual(validator.call_count, 2)
            self.assertFalse(self.env.geometry_filter_synthetic)
            first = validator.call_args_list[0].args
            self.assertEqual(first[2], self.env.base_cells[0])
            self.assertEqual(first[3], self.env.proposed_actions[0])
            self.assertEqual(first[5], self.env.geometry_starts[0])

    def test_production_proof_validation_failure_prevents_native_call(self):
        production_filter = Mock(side_effect=self.fake_filter)
        with patch.object(subject.layer, "filter_shot", production_filter), \
                patch.object(subject.layer, "validate_result", side_effect=ValueError("synthetic stale proof")):
            subject.install(self.env, filter_fn=production_filter)
            with self.assertRaisesRegex(ValueError, "stale proof"):
                self.env.prepare(self.actions)
        self.assertEqual(self.env.calls, [])

    def test_native_prepare_failure_propagates_without_retry(self):
        native = Mock(side_effect=RuntimeError("synthetic native failure"))
        self.env.prepare = native
        self.attach()
        with self.assertRaisesRegex(RuntimeError, "native failure"):
            self.env.prepare(self.actions)
        native.assert_called_once_with(self.actions)

    def test_original_support_decline_is_not_weakened_by_geometry_acceptance(self):
        self.env.support_missing[0] = True
        self.attach()
        seen = self.env.prepare(self.actions)
        self.assertTrue(seen["support"][0])
        self.assertTrue(self.env.geometry_records[0]["accepted"])

    def test_cleanup_restores_original_method_and_original_gate_state(self):
        original = self.env.prepare.__func__
        cleanup = self.attach()
        self.env.prepare(self.actions)
        cleanup()
        self.assertIs(self.env.prepare.__func__, original)
        self.assertNotIn("prepare", vars(self.env))
        self.assertEqual(self.env.support_missing, [False]*35)
        self.assertIsNone(self.env.proposal_ik_seeds)

    def test_cleanup_restores_preexisting_instance_method_exactly(self):
        original = Mock(return_value="synthetic-instance-prepare")
        self.env.prepare = original
        cleanup = self.attach()
        self.assertEqual(self.env.prepare(self.actions), "synthetic-instance-prepare")
        cleanup()
        self.assertIs(self.env.prepare, original)
        original.assert_called_once_with(self.actions)

    def test_double_install_and_foreign_hook_cleanup_fail_closed(self):
        cleanup = self.attach()
        with self.assertRaisesRegex(ValueError, "already installed"):
            self.attach()
        installed = self.env.prepare
        other = Mock()
        self.env.prepare = other
        with self.assertRaisesRegex(ValueError, "another owner"):
            cleanup()
        self.assertIs(self.env.prepare, other)
        self.env.prepare = installed
        cleanup()

    def test_bad_active_masks_fail_before_filter_or_native_call(self):
        self.attach()
        for mask in ([True]*34, np.ones(35), ["yes"]*35):
            with self.subTest(mask_type=type(mask)):
                self.env.active_mask = mask
                with self.assertRaisesRegex(ValueError, "35 Boolean"):
                    self.env.prepare(self.actions)
        self.assertEqual(self.filter_calls, [])
        self.assertEqual(self.env.calls, [])

    def test_nonfinite_joint_readback_fails_before_filter_or_native_call(self):
        self.attach()
        self.env.arms["Arm10"].data.joint_pos[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "joint readback"):
            self.env.prepare(self.actions)
        self.assertEqual(self.filter_calls, [])
        self.assertEqual(self.env.calls, [])

    def test_filter_error_never_partially_mutates_actions_or_calls_native(self):
        before = copy.deepcopy(self.actions)
        self.classifications[0] = "adjusted"
        def fail_second(*args, **kwargs):
            if args[1]["id"] == "synthetic-1":
                raise RuntimeError("synthetic failed geometry")
            return self.fake_filter(*args, **kwargs)
        self.attach(filter_fn=fail_second)
        with self.assertRaisesRegex(RuntimeError, "failed geometry"):
            self.env.prepare(self.actions)
        self.assertEqual(self.actions, before)
        self.assertEqual(self.env.calls, [])
        self.assertEqual(self.env.geometry_records[0]["classification"], "adjusted")

    def _bad_record(self, edit, *, resign=True):
        def bad(scene, cell, action, links, starts, **kwargs):
            value = result(cell, action)
            edit(value)
            return signed(value) if resign else value
        self.attach(filter_fn=bad)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.env.prepare(self.actions)
        self.assertEqual(self.env.calls, [])

    def test_filter_cannot_claim_release_authority(self):
        self._bad_record(lambda value: value.update(release_authorized=True))

    def test_filter_fingerprint_tampering_fails(self):
        self._bad_record(lambda value: value.update(reason="tampered"), resign=False)

    def test_filter_cannot_change_original_proposal(self):
        self._bad_record(lambda value: value["original_action"].update(height_m=.36))

    def test_filter_cannot_change_frozen_repair_budget(self):
        self._bad_record(lambda value: value["binding"]["protocol"].update(max_nfev=25))

    def test_filter_cannot_exceed_adjustment_limit(self):
        def bad(scene, cell, action, links, starts, **kwargs):
            return result(cell, action, "adjusted", delta=.05)
        self.attach(filter_fn=bad)
        with self.assertRaisesRegex(ValueError, "adjustment exceeds"):
            self.env.prepare(self.actions)
        self.assertEqual(self.env.calls, [])

    def test_filter_cannot_supply_nonfinite_arm_seed(self):
        self._bad_record(lambda value: value["final_preflight"]["qualification"]["ik"]["Arm10"].update(q=[0.]*5))

    def test_filter_input_mutation_is_not_a_new_policy_proposal(self):
        def bad(scene, cell, action, links, starts, **kwargs):
            values = [action[k] for k in subject.geometry.ACTION_NAMES]
            values[2] += .002
            action.update(subject.geometry.decode_action(cell, values, action["id"]))
            return result(cell, action)
        self.attach(filter_fn=bad)
        before = copy.deepcopy(self.actions)
        with self.assertRaisesRegex(ValueError, "different original proposal"):
            self.env.prepare(self.actions)
        self.assertEqual(self.actions, before)
        self.assertEqual(self.env.calls, [])

    def test_actual_state_change_during_filtering_blocks_native_call(self):
        def moving(scene, cell, action, links, starts, **kwargs):
            self.env.arms["Arm10"].data.joint_pos[0, 0] += .001
            return result(cell, action)
        self.attach(filter_fn=moving)
        with self.assertRaisesRegex(ValueError, "state changed"):
            self.env.prepare(self.actions)
        self.assertEqual(self.env.calls, [])

    def test_aliased_action_slots_are_not_executable(self):
        self.attach()
        self.actions[1] = self.actions[0]
        with self.assertRaisesRegex(ValueError, "alias"):
            self.env.prepare(self.actions)
        self.assertEqual(self.env.calls, [])


if __name__ == "__main__":
    unittest.main()
