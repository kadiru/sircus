"""Geometry-union and hybrid-likelihood unit checks; no capture assertions."""

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import continuous_policy as old
import diverse_policy as policy
import diverse_support as support
from continuous_feasibility import HIGH, LOW, canonical_hash, decode_action


def signed(value):
    value = copy.deepcopy(value)
    value.pop("fingerprint", None)
    value["fingerprint"] = canonical_hash(value)
    return value


def fixture():
    cells = [dict(id=f"cell-{i}", bucket_xy_m=[i*.01, -.2]) for i in range(35)]
    rows = []
    for i, cell in enumerate(cells):
        regions = []
        for j in range(0 if i == 0 else 1 if i == 1 else 3):
            center = np.array([j*.2, .4+j*.10, .35+j*.04, .24+j*.05, math.radians(29+j*4)])
            regions.append(dict(id=j, kind="local_geometry", proposal=j, interior_qualified=False,
                anchor_action=decode_action(cell, center, f"{i}-{j}"),
                anchor_ik={n: dict(q=[j*.1]*6) for n in ("Arm10", "Arm14")},
                low=np.maximum(LOW, center-support.HALF_WIDTH).tolist(),
                high=np.minimum(HIGH, center+support.HALF_WIDTH).tolist()))
        local_count, local_coverage = len(regions), support._coverage(regions)
        if regions:
            regions.append(dict(id=local_count, kind="global_exploration", proposal=-1, interior_qualified=False,
                anchor_action=copy.deepcopy(regions[0]["anchor_action"]), anchor_ik=copy.deepcopy(regions[0]["anchor_ik"]),
                low=LOW.tolist(), high=HIGH.tolist(), qualification_scope=support.GLOBAL_QUALIFICATION_SCOPE))
        rows.append(dict(cell_id=cell["id"], cell_index=i, bucket_xy_m=cell["bucket_xy_m"],
            status="supported" if regions else "unresolved", region_count=len(regions),
            local_region_count=local_count, global_region_count=int(bool(local_count)), local_coverage=local_coverage,
            regions=regions, coverage=support._coverage(regions)))
    all_regions = [r for row in rows for r in row["regions"]]
    value = signed(dict(schema=support.SCHEMA, source_fingerprint="a"*64,
        feasibility_fingerprint="b"*64, policy_geometry_only=True,
        action_names=list(support.ACTION_NAMES), global_low=LOW.tolist(), global_high=HIGH.tolist(),
        fixed_half_width=support.HALF_WIDTH.tolist(), cells=rows, region_count=len(all_regions),
        supported_cells=34, local_region_count=sum(r['local_region_count'] for r in rows), global_region_count=34,
        local_coverage=support._coverage([r for r in all_regions if r['kind']=='local_geometry']),
        max_regions_per_cell=4, coverage=support._coverage(all_regions)))
    return cells, value


def legacy_fixture():
    """Original v1 layout stays loadable under its original typed identity."""
    cells, value = fixture()
    value['schema'] = support.SCHEMA_V1
    for row in value['cells']:
        if row['regions']:
            row['regions'].pop()
        for region in row['regions']:
            region.pop('kind'); region.pop('interior_qualified')
        row['region_count'] = len(row['regions'])
        row['coverage'] = support._coverage(row['regions'])
        for key in ('local_region_count','global_region_count','local_coverage'):
            row.pop(key)
    all_regions = [r for row in value['cells'] for r in row['regions']]
    value.update(region_count=len(all_regions),max_regions_per_cell=3,coverage=support._coverage(all_regions))
    for key in ('local_region_count','global_region_count','local_coverage'):
        value.pop(key)
    return cells, signed(value)


class DiverseSupportTests(unittest.TestCase):
    def test_all_regions_are_distinct_continuous_boxes_not_convex_hull(self):
        cells, value = fixture()
        support.validate_support(value, cells)
        chosen = [cells[2]]*3
        idx = [0, 1, 2]
        physical = support.physical_actions(np.zeros((3, 5)), value, chosen, idx)
        for j in idx:
            r = value["cells"][2]["regions"][j]
            np.testing.assert_allclose(physical[j], (np.array(r["low"])+r["high"])/2)
        self.assertTrue(np.all(physical[1] > physical[0]))
        moved = support.physical_actions(np.full((3, 5), 1e-6), value, chosen, idx)
        self.assertTrue(np.all(moved > physical))
        self.assertTrue(np.all(moved-physical < 1e-6))

    def test_unresolved_requires_explicit_accounting_override(self):
        cells, value = fixture()
        with self.assertRaises(ValueError):
            support.physical_actions(np.zeros((1, 5)), value, cells[:1], [0])
        actual = support.physical_actions(np.zeros((1, 5)), value, cells[:1], [0], allow_unresolved=True)
        np.testing.assert_allclose(actual[0], (LOW+HIGH)/2)
        self.assertEqual(support.ik_seeds(value, cells[:1], [0], allow_unresolved=True), [None])
        self.assertEqual(support.support_missing(value, cells[:2]), [True, False])
        with self.assertRaises(ValueError):
            support.physical_actions(np.zeros((1, 5)), value, cells[:1], [1], allow_unresolved=True)

    def test_unknown_context_and_invalid_regions_fail_without_nearest_lookup(self):
        cells, value = fixture()
        changed = dict(cells[2], bucket_xy_m=[.020001, -.2])
        with self.assertRaises(ValueError):
            support.physical_actions(np.zeros((1, 5)), value, [changed], [0])
        for bad in ([4], [-1], [0.2], [float("nan")], [[0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                support.physical_actions(np.zeros((1, 5)), value, cells[2:3], bad)
        for raw in (np.zeros((1, 4)), np.ones((1, 5))*1.01):
            with self.assertRaises(ValueError):
                support.physical_actions(raw, value, cells[2:3], [0])

    def test_all_valid_map_anchors_retained_without_reward_ranking(self):
        cell = dict(id="x", bucket_xy_m=[.67, .3])
        fits = {n: dict(q=[0.]*6) for n in ("Arm10", "Arm14")}
        options = [dict(proposal=i, action=decode_action(cell, [i*.1, .4+i*.1, .35, .24, math.radians(29)], str(i)),
                        ik=fits, captures=999-i, reward=-i) for i in range(4)]
        with patch.object(support, "_links", return_value={}), \
             patch.object(support, "_home", return_value={n: np.zeros(6) for n in fits}), \
             patch.object(support, "approximate_path_check", return_value=dict(accepted=True)):
            row = support._cell_support(({}, cell, 0, options, "unused"))
            altered = copy.deepcopy(options)
            for option in altered:
                option.update(captures=-99999, reward=99999)
                option["action"].update(captures=-88888, reward=88888)
                for fit in option["ik"].values():
                    fit.update(captures=99999, reward=-99999)
            again = support._cell_support(({}, cell, 0, altered, "unused"))
        self.assertEqual(row, again)
        self.assertEqual({r["proposal"] for r in row["regions"] if r['kind']=='local_geometry'}, set(range(4)))
        self.assertEqual(row["region_count"], 5)
        self.assertEqual(row["local_region_count"], 4)
        self.assertEqual(row["global_region_count"], 1)
        self.assertTrue(all("reward" not in r and "captures" not in r for r in row["regions"]))

    def test_rejected_path_is_evidence_not_region_and_no_probe_shrink(self):
        cell = dict(id="x", bucket_xy_m=[.67, .3])
        fits = {n: dict(q=[0.]*6) for n in ("Arm10", "Arm14")}
        options = [dict(proposal=i, action=decode_action(cell, [.1*i, .4, .35, .24, math.radians(29)], str(i)), ik=fits)
                   for i in range(2)]
        with patch.object(support, "_links", return_value={}), \
             patch.object(support, "_home", return_value={n: np.zeros(6) for n in fits}), \
             patch.object(support, "approximate_path_check", side_effect=[dict(accepted=False), dict(accepted=True)]):
            row = support._cell_support(({}, cell, 0, options, "unused"))
        self.assertEqual(len(row["anchor_checks"]), 2)
        self.assertEqual(row["region_count"], 2)
        self.assertEqual(row["local_region_count"], 1)
        self.assertEqual(row["regions"][0]["proposal"], 1)
        center = support._values(options[1]["action"])
        np.testing.assert_array_equal(row["regions"][0]["low"], np.maximum(LOW, center-support.HALF_WIDTH))
        self.assertNotIn("calibration", row)

    def test_tamper_resign_does_not_allow_inconsistent_geometry(self):
        cells, value = fixture()
        changed = copy.deepcopy(value)
        changed["cells"][2]["regions"][0]["low"][2] += .001
        with self.assertRaises(ValueError):
            support.validate_support(changed)
        with self.assertRaises(ValueError):
            support.validate_support(signed(changed))
        for edit in ("coverage", "region_count", "anchor", "status"):
            changed = copy.deepcopy(value)
            if edit == "coverage":
                changed["coverage"]["low"][0] += .1
            elif edit == "region_count":
                changed["region_count"] -= 1
            elif edit == "anchor":
                changed["cells"][2]["regions"][0]["anchor_action"]["board_x_m"] += .001
            else:
                changed["cells"][0]["status"] = "supported"
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                support.validate_support(signed(changed))

    def test_seeds_copy_selected_region_not_first_anchor(self):
        cells, value = fixture()
        seeds = support.ik_seeds(value, cells[2:3], [2])
        self.assertEqual(seeds[0]["Arm10"]["q"], [.2]*6)
        seeds[0]["Arm10"]["q"][0] = 99
        self.assertEqual(value["cells"][2]["regions"][2]["anchor_ik"]["Arm10"]["q"][0], .2)

    def test_global_escape_preserves_entire_original_domain_without_outcome_labels(self):
        cells, value = fixture()
        rng = np.random.default_rng(20261115)
        physical = np.vstack([LOW, HIGH, LOW+rng.random((100, 5))*(HIGH-LOW)])
        normalized = 2*(physical-LOW)/(HIGH-LOW)-1
        contexts = [cells[2]]*len(physical)
        indices = [value['cells'][2]['region_count']-1]*len(physical)
        actual = support.physical_actions(normalized, value, contexts, indices)
        np.testing.assert_allclose(actual, physical, atol=3e-16, rtol=0)
        region = value['cells'][2]['regions'][-1]
        self.assertEqual(region['kind'], 'global_exploration')
        self.assertFalse(region['interior_qualified'])
        self.assertEqual(region['proposal'], -1)
        self.assertEqual(region['anchor_ik'], value['cells'][2]['regions'][0]['anchor_ik'])

    def test_global_escape_cannot_be_omitted_or_claimed_qualified(self):
        cells, value = fixture()
        for key in ('omitted', 'qualified', 'seed', 'bounds'):
            changed = copy.deepcopy(value)
            region = changed['cells'][2]['regions'][-1]
            if key=='omitted': changed['cells'][2]['regions'].pop()
            elif key=='qualified': region['interior_qualified']=True
            elif key=='seed': region['anchor_ik']['Arm10']['q'][0]=9
            else: region['high'][3]-=.01
            with self.subTest(key=key), self.assertRaises(ValueError): support.validate_support(signed(changed))

    def test_v1_support_identity_and_decoder_remain_unchanged(self):
        cells, value = legacy_fixture()
        support.validate_support(value, cells)
        self.assertEqual(value['schema'], 'sircus-diverse-shot-support-v1')
        self.assertEqual(value['region_count'], 100)
        actual = support.physical_actions(np.zeros((1,5)),value,cells[2:3],[2])
        region = value['cells'][2]['regions'][2]
        np.testing.assert_array_equal(actual[0],(np.asarray(region['low'])+region['high'])/2)


@unittest.skipIf(old.torch is None, "Configured CPU Torch unavailable")
class HybridPolicyTests(unittest.TestCase):
    def setUp(self):
        self.torch = old.torch
        self.torch.set_num_threads(1)
        self.torch.manual_seed(20261115)
        self.cells, self.support = fixture()
        self.policy = policy.DiversePolicy(self.support)
        self.obs = self.torch.tensor(policy.build_observations(self.cells))
        self.bindings = dict(study_fingerprint="c"*64, geometry_fingerprint="a"*64,
            domain_fingerprint="d"*64, support_fingerprint=self.support["fingerprint"])

    def test_initial_uniform_mask_zero_residual_deterministic_argmax(self):
        regions, residuals, mask = self.policy.distributions(self.obs)
        self.torch.testing.assert_close(regions.probs[2], self.torch.ones(4)/4)
        self.torch.testing.assert_close(regions.probs[0], self.torch.tensor([1., 0., 0., 0.]))
        self.torch.testing.assert_close(regions.probs[1], self.torch.tensor([.5, .5, 0., 0.]))
        self.torch.testing.assert_close(residuals.loc, self.torch.zeros_like(residuals.loc))
        drawn = policy.sample(self.policy, self.obs, deterministic=True)
        self.assertEqual(drawn["latent"].shape, (35, 6))
        self.assertEqual(drawn["actions_normalized"].shape, (35, 5))
        self.assertEqual(drawn["region_indices"].tolist(), [0]*35)
        self.torch.testing.assert_close(drawn["actions_normalized"], self.torch.zeros((35, 5)))
        self.torch.testing.assert_close(drawn["logp"], drawn["region_logp"]+drawn["residual_logp"])

    def test_joint_likelihood_matches_categorical_plus_tanh_gaussian(self):
        model, obs = self.policy.double(), self.obs.double()
        drawn = policy.sample(model, obs, self.torch.Generator().manual_seed(3))
        categorical, normals, _ = model.distributions(obs)
        idx, z = drawn["region_indices"], drawn["latent"][:, 1:]
        batch = self.torch.arange(35)
        normal = self.torch.distributions.Normal(normals.loc[batch, idx], normals.scale[batch, idx])
        expected = categorical.log_prob(idx)+(normal.log_prob(z)-self.torch.log1p(-z.tanh().square())).sum(-1)
        self.torch.testing.assert_close(model.log_prob(obs, drawn["latent"]), expected, atol=1e-10, rtol=1e-10)
        # Overlapping boxes still carry the selected region identity explicitly.
        alternate = drawn["latent"].clone(); alternate[2, 0] = (idx[2]+1)%3
        self.assertTrue(self.torch.isfinite(model.log_prob(obs, alternate)).all())

    def test_unknown_context_bad_and_unavailable_region_rejected(self):
        changed = self.obs.clone(); changed[2, 0] += 1e-5
        with self.assertRaises(ValueError):
            policy.sample(self.policy, changed)
        drawn = policy.sample(self.policy, self.obs, deterministic=True)
        for region in (-1, 1, .1, 3):
            bad = drawn["latent"].clone(); bad[0, 0] = region
            with self.subTest(region=region), self.assertRaises(ValueError):
                self.policy.log_prob(self.obs, bad)

    def test_generators_keep_assessment_out_of_training_rng(self):
        generator = self.torch.Generator().manual_seed(5)
        saved = generator.get_state().clone()
        expected = policy.sample(self.policy, self.obs, generator)
        generator.set_state(saved)
        policy.sample(self.policy, self.obs, deterministic=True)
        actual = policy.sample(self.policy, self.obs, generator)
        self.torch.testing.assert_close(expected["latent"], actual["latent"])

    def test_joint_entropy_gradients_finite_with_masked_regions(self):
        result = self.policy.entropy_estimate(self.obs, self.torch.Generator().manual_seed(2))
        self.assertTrue(self.torch.isfinite(result).all())
        result.mean().backward()
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                self.assertTrue(self.torch.isfinite(parameter.grad).all())

    def test_ppo_updates_region_and_residual_actors_all_attempts_retained(self):
        obs = self.obs.repeat((4, 1))
        drawn = policy.sample(self.policy, obs, self.torch.Generator().manual_seed(12))
        returns = drawn["region_indices"].float()+drawn["actions_normalized"][:, 3]
        returns[::3] = -1  # mock native rejection penalty remains in buffer
        before_region = self.policy.region_actor[-1].weight.detach().clone()
        before_residual = self.policy.residual_actor[-1].weight.detach().clone()
        optimizer = self.torch.optim.Adam(self.policy.parameters(), lr=.0003)
        result = policy.update(self.policy, optimizer, obs, drawn["latent"], drawn["logp"], returns, old.PPO_DEFAULTS)
        self.assertEqual(result["samples"], 140)
        self.assertEqual(len(result["epochs"]), 4)
        self.assertNotEqual(result["actor_before"], result["actor_after"])
        self.assertFalse(self.torch.equal(before_region, self.policy.region_actor[-1].weight))
        self.assertFalse(self.torch.equal(before_residual, self.policy.residual_actor[-1].weight))

    def test_checkpoint_support_buffers_rng_and_counters_guarded(self):
        payload = policy.checkpoint_payload(self.policy, None, **self.bindings,
            update_count=0, attempted_shots=0, seed=2)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/"policy.pt"; self.torch.save(payload, path)
            rng = self.torch.random.get_rng_state().clone()
            loaded, saved = policy.load_checkpoint(path, support=self.support, **self.bindings,
                expected_updates=0, expected_attempted_shots=0)
            self.torch.testing.assert_close(rng, self.torch.random.get_rng_state())
            self.assertEqual(policy.weight_hash(loaded), policy.weight_hash(self.policy))
            wrong = dict(self.bindings, support_fingerprint="e"*64)
            with self.assertRaises(ValueError):
                policy.load_checkpoint(path, support=self.support, **wrong)
            with self.assertRaises(ValueError):
                policy.load_checkpoint(path, support=self.support, **self.bindings, expected_updates=1)
            changed = copy.deepcopy(payload)
            changed["state_dict"]["region_features"][2, 0, 0] += .1
            self.torch.save(changed, path)
            with self.assertRaisesRegex(ValueError, "geometry buffer"):
                policy.load_checkpoint(path, support=self.support, **self.bindings)
        with self.assertRaises(ValueError):
            policy.checkpoint_payload(old.ContinuousPolicy(), None, **self.bindings,
                update_count=0, attempted_shots=0, seed=2)

    def test_v1_checkpoint_metadata_and_weights_roundtrip_without_relabeling(self):
        cells, value = legacy_fixture()
        model = policy.DiversePolicy(value)
        binding = dict(self.bindings,support_fingerprint=value['fingerprint'])
        payload = policy.checkpoint_payload(model,None,**binding,update_count=0,attempted_shots=0,seed=2)
        self.assertEqual(payload['policy_version'],policy.POLICY_VERSION_V1)
        self.assertEqual(payload['architecture'],policy.ARCHITECTURE_V1)
        self.assertNotEqual(payload['policy_version'],policy.POLICY_VERSION)
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'old.pt';self.torch.save(payload,path)
            loaded,saved=policy.load_checkpoint(path,support=value,**binding)
            self.assertEqual(policy.weight_hash(loaded),policy.weight_hash(model))
            self.assertEqual(saved['policy_version'],'sircus-diverse-hybrid-shot-ppo-v1')


if __name__ == "__main__":
    unittest.main()
