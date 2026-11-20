"""Run accounting with fabricated results only; never load a real PPO model."""

import argparse
import copy
import json
import math
from pathlib import Path
import random

from candidate_continuation_journal import write_new_json
from location_comparison import digest, PARENT_SHA, require
from location_comparison_bounds import Bounds
from location_comparison_runtime import ComparisonRunner, Journal, PREPARED_FINGERPRINT


class FakeClock:
    def __init__(self):
        self.value = 100.

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class MockEngine:
    """Counters and fabricated outcomes; no neural networks or physics."""
    synthetic = True

    def __init__(self, clock, directory=None):
        self.clock = clock
        self.directory = None if directory is None else Path(directory)
        if self.directory is not None:
            self.directory.mkdir(exist_ok=False)
        self.models, self.rngs, self.executed, self.learned = {}, {}, [], []

    def restore(self, name, action_seed, optimizer_seed):
        require(name not in self.models, "Mock restore repeated")
        self.clock.advance(.001)
        self.models[name] = dict(model_id="synthetic-"+name, weight_hash=digest("synthetic-parent-weights"),
            optimizer_hash=digest("synthetic-parent-Adam"), optimizer_steps=292, updates=73, attempts=10000,
            parent_checkpoint_sha256=PARENT_SHA, action_seed=action_seed, optimizer_seed=optimizer_seed)
        self.rngs[name] = random.Random(action_seed)
        return self.state(name)

    def state(self, name):
        return copy.deepcopy(self.models[name])

    def propose(self, name, cells, deterministic):
        self.clock.advance(.001)
        rows = []
        for cell in cells:
            x,y = cell["bucket_xy_m"]
            dx,dy = .67-x,-.10-y
            bearing = math.atan2(dy, dx)
            latent = [.1]*5 if deterministic else [self.rngs[name].uniform(-1.,1.) for _ in range(5)]
            rows.append(dict(request_id=cell["request_id"], target_id=cell["target_id"],
                observation=[(x-.67)/.9,(y+.25)/.625,math.hypot(dx,dy),math.sin(bearing),math.cos(bearing)],
                latent=latent, logp=-2.5, actions_normalized=list(map(math.tanh,latent))))
        return dict(model_id=self.models[name]["model_id"], deterministic=deterministic, rows=rows)

    def execute(self, cells, proposal, film=False):
        self.clock.advance(.01)
        self.executed.append(copy.deepcopy(cells))
        rows = []
        for cell in cells:
            choice = int(digest(cell["request_id"])[:8], 16) % 4
            outcome = ("capture","capture","miss","rejected")[choice]
            rows.append(dict(request_id=cell["request_id"], target_id=cell["target_id"],
                outcome=outcome, clean=choice == 0, rejection_reason="synthetic_geometry_decline" if choice == 3 else None,
                reward=(1.6,1.5,-.5,-1.)[choice]))
        return dict(rows=rows, evidence=dict(synthetic=True, physics_samples=0,
            explanation="Fabricated accounting fixture; no native results", filmed=film),
            costs=dict(synthetic=True, geometry_wall_s=.002,
                       scope="Invented clock; geometry nested in episode; no measured compute"))

    def learn(self, name, examples):
        # Deliberately no optimizer or gradient operations.
        self.clock.advance(.001)
        self.learned.append(dict(model=name, request_ids=[x["proposal"]["request_id"] for x in examples]))
        state = self.models[name]
        state["updates"] += 1
        state["optimizer_steps"] += 4
        state["attempts"] += len(examples)
        state["weight_hash"] = digest(["synthetic-weights",name,state["updates"]])
        state["optimizer_hash"] = digest(["synthetic-Adam",name,state["optimizer_steps"]])
        return self.state(name)

    def checkpoint(self, name, label):
        self.clock.advance(.001)
        state = self.state(name)
        file = f"{name}-{label}.json"
        payload = dict(synthetic=True, state=state)
        raw = json.dumps(payload, sort_keys=True).encode()
        if self.directory is not None:
            with (self.directory/file).open("xb") as stream:
                stream.write(raw)
        return dict(file=file, sha256=__import__("hashlib").sha256(raw).hexdigest(), state=state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    require(plan.get("fingerprint") == PREPARED_FINGERPRINT, "Expected the frozen prepared package")
    require(args.output.is_absolute() and args.output == args.output.resolve()
            and not args.output.exists(), "Mock output must be a new canonical absolute directory")
    args.output.mkdir(parents=False, exist_ok=False)
    clock = FakeClock()
    bounds = Bounds(plan, clock, lambda: dict(free_storage_bytes=10*1024**3))
    engine = MockEngine(clock, args.output/"synthetic-checkpoints")
    runner = ComparisonRunner(plan, engine, bounds, Journal(args.output/"events"), clock=clock)
    trace = runner.run()
    write_new_json(args.output/"synthetic-trace.json", trace)
    from location_comparison_audit import audit_trace
    audit = audit_trace(plan, trace)
    write_new_json(args.output/"synthetic-audit.json", audit)
    result = dict(mode="synthetic", audit=audit, actual_native_requests=0,
                  actual_learning_steps=0, simulator_started=False)
    write_new_json(args.output/"result.json", result)
    print(json.dumps(dict(mode="synthetic", audit_status=audit["status"],
        logical_requests=audit["requested"], completed=audit["completed"],
        actual_native_requests=0, actual_learning_steps=0, simulator_started=False,
        result_file=str(args.output/"result.json"))))


if __name__ == "__main__":
    main()
