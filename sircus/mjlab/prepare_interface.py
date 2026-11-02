"""Compose audited studies for the viewer without rewriting experiment evidence."""

import argparse
import json
from pathlib import Path
import numpy as np


def prepare(parent_study, bank_study, output):
    parent_study,bank_study,output=map(Path,(parent_study,bank_study,output))
    for study,name in ((parent_study,"audit.json"),(bank_study,"audit-bank.json")):
        if not json.loads((study/name).read_text()).get("passed"):
            raise ValueError("A completed independent audit is required")
        if output.resolve()==(study/"report.json").resolve():
            raise ValueError("Never overwrite an original experiment report")
    parent=json.loads((parent_study/"report.json").read_text())
    bank=json.loads((bank_study/"report.json").read_text())
    if any(r.get("status")!="completed" or not r.get("frozen_evaluation") for r in (parent,bank)):
        raise ValueError("Both studies must complete before manual testing")
    if bank["parent_checkpoint_sha256"]!=parent["checkpoint_sha256"]:
        raise ValueError("Studies do not share the recorded parent checkpoint")
    selected=[json.loads(line) for line in (bank_study/"shots.jsonl").read_text().splitlines()
              if json.loads(line)["phase"].startswith("test-selected-")]
    geometry=np.load(bank_study/"geometry-test.npz")
    comparison=bank["frozen_evaluation"]["comparison"]
    finding="Learned selection is compared with the same bank chosen uniformly."
    if comparison["selected_bank"]["captures"]<=comparison["uniform_valid_bank"]["captures"]:
        finding+=" This study found no capture advantage over uniform feasible-bank selection."
    coverage=[]
    for i,xy in enumerate(geometry["bucket"]):
        group=[r for r in selected if np.allclose(r["bucket_xy_m"],xy,atol=1e-12,rtol=0)]
        if len(group)!=bank["frozen_evaluation"]["repeats"]:
            raise ValueError("Frozen bank coverage denominator differs from the report")
        coverage.append(dict(bucket_xy_m=xy.tolist(),captures=sum(r["route_success"] for r in group),
                             trials=len(group),eligible_candidates=int(geometry["mask"][i].sum())))
    parent.pop("test_rows",None)
    parent["bank_evaluation"]=dict(status="completed",train_episodes=bank["train_episodes"],updates=bank["updates"],
         test_layouts=50,contract_fingerprint=bank["contract_fingerprint"],source_scope=bank["scope"]+". "+finding,
         comparison={"continuous_v2":comparison["parent_v2"],"bank_uniform":comparison["uniform_valid_bank"],"bank_selected":comparison["selected_bank"]},
         geometry={k:{a:b for a,b in v.items() if a!="chunks"} for k,v in bank["geometry"].items()},
         learning_curve=bank["learning_curve"],coverage=coverage)
    tmp=output.with_suffix(".tmp");tmp.write_text(json.dumps(parent,indent=2,allow_nan=False));tmp.replace(output)
    return parent


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("parent-study","bank-study","output"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();prepare(a.parent_study,a.bank_study,a.output)
