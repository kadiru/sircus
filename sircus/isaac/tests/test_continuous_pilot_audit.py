"""Synthetic audit fixtures exercise corruption detection, never native results."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ISAAC))
import continuous_policy as policy_module
import continuous_pilot_audit as audit
from continuous_pilot import make_study
from expanded_farm import make_spec


def bundle():
    scene = json.loads((ISAAC/"scene.json").read_text())
    return dict(spec=make_spec(scene,10),source=dict(camera_sha256="a"*64))


def write(path,value):
    Path(path).write_text(json.dumps(value,allow_nan=False))


def reseal(study):
    study.pop("fingerprint",None)
    study["fingerprint"] = audit.canonical_hash(study)
    return study


class StudyMembershipTests(unittest.TestCase):
    def test_split_has_twenty_five_train_and_two_disjoint_five_site_holdouts(self):
        source = bundle()
        study = make_study(source,4)
        audit.validate_study(study,source["spec"])
        groups = {label:{c["id"] for c in study["cells"] if c["split"] == label}
                  for label in ("train","validation","test")}
        self.assertEqual({k:len(v) for k,v in groups.items()},dict(train=25,validation=5,test=5))
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["validation"] & groups["test"])
        changed = copy.deepcopy(study)
        changed["cells"][0]["split"] = "train"
        with self.assertRaisesRegex(ValueError,"split changed"):
            audit.validate_study(reseal(changed),source["spec"])

    def test_resigned_domain_or_budget_changes_are_rejected(self):
        source = bundle()
        for field,value in (("training_batches",8),("batches_per_update",2),("assessment_batches",3)):
            changed = copy.deepcopy(make_study(source,4)); changed[field] = value
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.validate_study(reseal(changed),source["spec"])
        changed = copy.deepcopy(make_study(source,4)); changed["domain"]["high"][1] += .1
        changed["domain"].pop("fingerprint"); changed["domain"]["fingerprint"] = audit.canonical_hash(changed["domain"])
        with self.assertRaisesRegex(ValueError,"domain changed"):
            audit.validate_study(reseal(changed),source["spec"])


@unittest.skipIf(policy_module.torch is None,"PyTorch CPU environment is unavailable")
class ContinuousAuditTests(unittest.TestCase):
    def fixture(self,root,coverage=False):
        """Four train batches and two before/two after frozen synthetic checks."""
        from continuous_feasibility import LOW,HIGH,decode_action
        from farm_contract import reward,statistics
        torch = policy_module.torch
        torch.set_num_threads(1); torch.manual_seed(20261112)
        source = bundle(); study = make_study(source,4)
        root.mkdir(exist_ok=True); (root/"trials").mkdir()
        write(root/"contract.json",source["spec"]); write(root/"study.json",study)
        policy = policy_module.ContinuousPolicy()
        optimizer = torch.optim.Adam(policy.parameters(),lr=.0003)
        obs = torch.from_numpy(policy_module.build_observations(study["cells"]))
        train = [i for i,c in enumerate(study["cells"]) if c["split"] == "train"]
        changes,summaries,evals,buffer,coverages = [],[],[],[],[]
        plans = []
        if coverage:
            from continuous_feasibility import geometry_coverage
            records = []
            for cell in study["cells"]:
                for j,height in enumerate((.3,.35)):
                    action = decode_action(cell,[0,.4,height,.24,np.deg2rad(29)],f"geometry-{cell['id']}-{j}")
                    records.append(dict(cell_id=cell["id"],accepted=True,action=action,ik={}))
            feasibility = dict(source_fingerprint=source["spec"]["fingerprint"],
                               cells=[dict(cell_id=c["id"]) for c in study["cells"]],records=records)
            feasibility["fingerprint"] = audit.canonical_hash(feasibility)
            write(root/"feasibility.json",feasibility)
            study["coverage_batches"] = 4
            study["feasibility_sha256"] = audit.digest(root/"feasibility.json")
            reseal(study); write(root/"study.json",study)
            plans = geometry_coverage(feasibility,study["cells"],batches=4)

        def checkpoint(name,batches):
            path = root/name
            payload = policy_module.checkpoint_payload(policy,optimizer,
                study_fingerprint=study["fingerprint"],geometry_fingerprint=study["geometry_fingerprint"],
                domain_fingerprint=study["domain"]["fingerprint"],update_count=len(changes),
                attempted_shots=batches*25,seed=study["seed"])
            torch.save(payload,path)
            return dict(file=name,sha256=audit.digest(path),policy_hash=policy_module.weight_hash(policy),
                        actor_hash=policy_module.actor_weight_hash(policy))

        def trial(mode,batch,cp,plan=None):
            picked = policy_module.sample(policy,obs,deterministic=mode=="eval")
            physical = policy_module.physical_actions(picked["actions_normalized"],LOW,HIGH)
            actions = [decode_action(c,a,f"{mode}-{batch}-{i}") for i,(c,a) in enumerate(zip(study["cells"],physical))]
            if plan is not None: actions = plan["actions"]
            rows = [dict(a,cell_id=c["id"],split=c["split"],invalid="synthetic_admission" if i%3==0 else None,
                         external_contact_steps=0,board_bounce=True,closest_opening_distance_m=.05+i*.01,
                         max_retention_s=0.,route_success=False,route_quality=0,single_bounce_success=False)
                    for i,(c,a) in enumerate(zip(study["cells"],actions))]
            rewards = torch.tensor([reward(r) for r in rows],dtype=torch.float32)
            enabled = ([c["split"] == "train" for c in study["cells"]] if mode=="train"
                       else plan["enabled"] if plan is not None else [True]*35)
            active = [i for i,b in enumerate(enabled) if b]
            name = f"trials/{mode}-{len(summaries)+len(evals)+len(coverages):04d}.json"
            data = dict(schema=study["schema"],mode=mode,batch=batch,study_fingerprint=study["fingerprint"],
                checkpoint=cp,policy_hash=policy_module.weight_hash(policy),update_count=len(changes),
                enabled=enabled,active_cell_ids=[study["cells"][i]["id"] for i in active],
                optimizer_samples=25 if mode=="train" else 0,deterministic=mode=="eval", policy_action_used=mode!="coverage",
                observations=obs.tolist(),latent=picked["latent"].tolist(),
                actions_normalized=picked["actions_normalized"].tolist(),logp=picked["logp"].tolist(),
                actions=actions,rows=rows,rewards=rewards.tolist())
            write(root/name,data)
            summary = dict(file=name,sha256=audit.digest(root/name),mode=mode,batch=batch,
                           update_count=len(changes),**statistics([rows[i] for i in active]))
            return summary,(obs[train],picked["latent"][train],picked["logp"][train],rewards[train])

        initial = checkpoint("policy-initial.pt",0)
        for batch,plan in enumerate(plans):
            summary,_ = trial("coverage",batch,initial,plan); coverages.append(summary)
        for repeat in range(2):
            summary,_ = trial("eval",repeat,initial); evals.append(summary)
        for batch in range(4):
            cp = checkpoint(f"policy-before-{batch:03d}.pt",batch)
            summary,data = trial("train",batch,cp); summaries.append(summary); buffer.append(data)
        change = policy_module.update(policy,optimizer,*[torch.cat(g) for g in zip(*buffer)],study["ppo"])
        change.update(update_count=1,training_batches=[0,1,2,3],sample_count=100)
        changes.append(change)
        final = checkpoint("policy-final.pt",4)
        for repeat in range(2):
            summary,_ = trial("eval",repeat,final); evals.append(summary)
        report = dict(schema=study["schema"],status="completed",study_fingerprint=study["fingerprint"],
            geometry_fingerprint=study["geometry_fingerprint"],workcells=35,articulations=70,
            initial=initial,final=final,initial_hash=initial["policy_hash"],final_hash=final["policy_hash"],
            batches=summaries,assessments=evals,updates=changes,coverage=coverages)
        write(root/"result.json",report)
        return report

    def test_full_synthetic_chain_retains_all_attempts_and_has_no_holdout_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root)
            result = audit.verify_checkpoint_chain(root)
            self.assertEqual(result["status"],"passed")
            self.assertEqual(result["optimizer_samples"],100)
            self.assertEqual(len(result["train_cell_ids"]),25)
            self.assertEqual({k:len(v) for k,v in result["held_out_cell_ids"].items()},dict(validation=5,test=5))
            self.assertGreater(result["train_statistics"]["invalid"],0)
            self.assertTrue(result["actor_changed"])
            self.assertLess(result["max_logp_error"],1e-5)
            self.assertEqual(len(result["checkpoints"]),6)
            self.assertEqual(len(result["trials"]),8)

    def test_geometry_coverage_actions_are_verified_but_never_enter_policy_updates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.fixture(root,coverage=True)
            result = audit.verify_checkpoint_chain(root)
            self.assertEqual(result["geometry_coverage_attempts"],140)
            self.assertTrue(result["geometry_coverage_excluded_from_optimizer"])
            self.assertEqual(result["optimizer_samples"],100)
            self.assertEqual(len(result["trials"]),12)
            report = audit.read(root/"result.json")
            summary = report["coverage"][0]; path=root/summary["file"]; data=audit.read(path)
            data["policy_action_used"] = True
            write(path,data); summary["sha256"] = audit.digest(path); write(root/"result.json",report)
            with self.assertRaisesRegex(ValueError,"protocol changed"):
                audit.verify_checkpoint_chain(root)

    def test_rehashed_logp_or_action_corruption_fails_independent_reconstruction(self):
        for field in ("logp","actions_normalized","rewards","enabled","observations"):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                root = Path(d); report = self.fixture(root)
                summary = report["batches"][0]; path = root/summary["file"]; data = audit.read(path)
                if field == "enabled": data[field][0] = True
                elif field in ("actions_normalized","observations"): data[field][1][0] += .1
                else: data[field][1] += .5
                write(path,data); summary["sha256"] = audit.digest(path); write(root/"result.json",report)
                with self.assertRaises(ValueError): audit.verify_checkpoint_chain(root)

    def test_eval_optimizer_membership_or_update_window_corruption_fails(self):
        for variant in ("eval_membership","drop_invalid_attempts","update_window"):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as d:
                root = Path(d); report = self.fixture(root)
                if variant == "update_window":
                    report["updates"][0]["training_batches"] = [0,1,2,7]
                else:
                    summary = report["assessments"][0] if variant == "eval_membership" else report["batches"][0]
                    path = root/summary["file"]; data = audit.read(path)
                    data["optimizer_samples"] = 35 if variant == "eval_membership" else 17
                    write(path,data); summary["sha256"] = audit.digest(path)
                write(root/"result.json",report)
                with self.assertRaises(ValueError): audit.verify_checkpoint_chain(root)


if __name__ == "__main__":
    unittest.main()
