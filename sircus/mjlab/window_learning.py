"""Continuous gap learning for a fixed-bucket, terminal shot experiment.

SAC here is the one-step contextual-bandit specialization: Q targets are the
observed terminal rewards, with no next-state bootstrap. The actor optimizes
min(Q1,Q2) plus entropy on normalized [-1,1] actions. Kinematics, admission,
physics, capture scoring and experimental budgets belong to the caller.

Torch is imported lazily. The supervised RBF logistic baseline needs only NumPy.
Neither frozen predictor performs native rollout search at inference.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SAC_VERSION = "sircus-terminal-gap-sac-v1"
MODEL_VERSION = "sircus-window-rbf-logistic-v1"
CONTEXT_BOUNDS = ((.04, .56), (8., 64.))
GAP_BOUNDS = (.02, .65)


def _bounds(context_bounds=CONTEXT_BOUNDS, gap_bounds=GAP_BOUNDS):
    context, gap = np.asarray(context_bounds, float), np.asarray(gap_bounds, float)
    if context.shape != (2, 2) or gap.shape != (2,) or not np.isfinite(context).all() or not np.isfinite(gap).all():
        raise ValueError("Finite context[2,2] and gap[2] bounds required")
    if (context[:, 1] <= context[:, 0]).any() or gap[1] <= gap[0]:
        raise ValueError("Bounds must increase")
    return context, gap


def normalized_contexts(contexts, context_bounds=CONTEXT_BOUNDS):
    bounds, _ = _bounds(context_bounds)
    value = np.asarray(contexts, float)
    if value.ndim == 1 and value.shape == (2,):
        value = value[None]
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("Finite context[N,2] required")
    if ((value < bounds[:, 0] - 1e-9) | (value > bounds[:, 1] + 1e-9)).any():
        raise ValueError("Context outside declared bounds")
    return 2 * (value - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1


def normalized_gaps(gaps, count=None, gap_bounds=GAP_BOUNDS):
    _, bounds = _bounds(gap_bounds=gap_bounds)
    value = np.asarray(gaps, float).reshape(-1)
    if (count is not None and len(value) != count) or not np.isfinite(value).all():
        raise ValueError("One finite gap per context required")
    if ((value < bounds[0] - 1e-9) | (value > bounds[1] + 1e-9)).any():
        raise ValueError("Gap outside declared bounds")
    return 2 * (value - bounds[0]) / (bounds[1] - bounds[0]) - 1


def _train_roles(roles, count):
    roles = ["train"] * count if roles is None else list(roles)
    if len(roles) != count or any(role != "train" for role in roles):
        raise ValueError("Only explicit TRAIN experience may fit a learner")


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def terminal_target(rewards, next_values=None):
    """All shots terminate: an arbitrary next-value can never affect this target."""
    import torch
    value = torch.as_tensor(rewards, device="cpu")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Finite terminal rewards required")
    return value.detach().clone()


def squashed_gaussian_log_prob(pre_tanh, mean, log_std):
    """Stable density for normalized tanh actions, before physical metre scaling."""
    import torch
    normal = -.5 * (((pre_tanh - mean) / log_std.exp()) ** 2 + 2 * log_std + math.log(2 * math.pi))
    jacobian = 2 * (math.log(2) - pre_tanh - torch.nn.functional.softplus(-2 * pre_tanh))
    return (normal - jacobian).sum(dim=-1, keepdim=True)


class SACAgent:
    """CPU terminal SAC with independent replay/actor random generators."""

    def __init__(self, seed=0, context_bounds=CONTEXT_BOUNDS, gap_bounds=GAP_BOUNDS,
                 alpha=.02, learning_rate=3e-4, capacity=100000, device="cpu"):
        import torch
        if device != "cpu":
            raise ValueError("This learner is CPU-only; native shots own the authorized GPU separately")
        if not np.isfinite(alpha) or alpha < 0 or not 0 < learning_rate < 1 or capacity < 1:
            raise ValueError("Invalid bounded SAC settings")
        self.context_bounds, self.gap_bounds = _bounds(context_bounds, gap_bounds)
        self.seed, self.alpha, self.learning_rate, self.capacity = int(seed), float(alpha), float(learning_rate), int(capacity)
        self.device, self.torch = "cpu", torch
        self.rng = np.random.default_rng(self.seed + 17)
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed + 31)
        def net(inputs, outputs):
            def linear(first, second):
                return torch.nn.Linear(first, second, device="cpu", dtype=torch.float32)
            return torch.nn.Sequential(linear(inputs, 64), torch.nn.Tanh(),
                linear(64, 64), torch.nn.Tanh(), linear(64, outputs))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.actor, self.q1, self.q2 = net(2, 2), net(3, 1), net(3, 1)
            torch.nn.init.normal_(self.actor[-1].weight, std=.01)
            torch.nn.init.zeros_(self.actor[-1].bias)
            self.actor[-1].bias.data[1] = -.5
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.learning_rate)
        self.contexts = np.empty((0, 2), np.float32)
        self.actions = np.empty((0, 1), np.float32)
        self.rewards = np.empty((0, 1), np.float32)
        self.total_seen = self.updates = 0
        self.training_provenance = []

    @property
    def replay_size(self):
        return len(self.rewards)

    def weight_hash(self):
        digest = hashlib.sha256()
        for label, model in (("actor", self.actor), ("q1", self.q1), ("q2", self.q2)):
            for name, tensor in sorted(model.state_dict().items()):
                digest.update((label + ":" + name).encode())
                digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def add(self, contexts, gaps, rewards, *, roles=None, provenance=None):
        context = normalized_contexts(contexts, self.context_bounds).astype(np.float32)
        gap = normalized_gaps(gaps, len(context), self.gap_bounds).astype(np.float32)[:, None]
        reward = np.asarray(rewards, float).reshape(-1, 1)
        if reward.shape != gap.shape or not np.isfinite(reward).all() or (np.abs(reward)>np.finfo(np.float32).max).any():
            raise ValueError("One finite reward per experience required")
        _train_roles(roles, len(context))
        # Validate provenance before mutating replay. Never infer a source's split.
        if provenance is not None:
            json.dumps(provenance, allow_nan=False)
        self.contexts = np.concatenate((self.contexts, context))[-self.capacity:]
        self.actions = np.concatenate((self.actions, gap))[-self.capacity:]
        self.rewards = np.concatenate((self.rewards, reward.astype(np.float32)))[-self.capacity:]
        self.total_seen += len(context)
        if provenance is not None:
            self.training_provenance.append(json.loads(json.dumps(provenance, allow_nan=False)))

    def _sample(self, contexts, stochastic=True):
        torch = self.torch
        output = self.actor(contexts)
        mean, log_std = output[:, :1], output[:, 1:].clamp(-5, 1)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(log_std).all()):
            raise RuntimeError("Nonfinite actor output")
        pre = mean + log_std.exp() * torch.randn(mean.shape, generator=self.generator, device="cpu", dtype=mean.dtype) if stochastic else mean
        return pre.tanh(), squashed_gaussian_log_prob(pre, mean, log_std)

    def propose(self, contexts, stochastic=False, *, return_info=False):
        torch = self.torch
        context = torch.tensor(normalized_contexts(contexts, self.context_bounds), dtype=torch.float32, device="cpu")
        with torch.no_grad():
            action, logp = self._sample(context, stochastic)
        normalized = action.numpy().reshape(-1)
        gaps = self.gap_bounds[0] + (normalized.astype(float) + 1) * .5 * np.diff(self.gap_bounds)[0]
        if return_info:
            return dict(gaps=gaps, normalized_actions=normalized, log_probability=logp.numpy().reshape(-1),
                density_scope="Normalized tanh action units, not physical metres")
        return gaps

    def update(self, steps=1, batch_size=64):
        torch = self.torch
        if not isinstance(steps, int) or not 0 <= steps <= 100000 or not isinstance(batch_size, int) or not 1 <= batch_size <= 4096:
            raise ValueError("Bounded integer update and batch settings required")
        before = self.weight_hash()
        logs = []
        if self.replay_size:
            for _ in range(steps):
                indices = self.rng.integers(self.replay_size, size=batch_size)
                context = torch.tensor(self.contexts[indices], device="cpu")
                action = torch.tensor(self.actions[indices], device="cpu")
                target = terminal_target(torch.tensor(self.rewards[indices], device="cpu"))
                inputs = torch.cat((context, action), dim=1)
                critic_loss = ((self.q1(inputs) - target) ** 2).mean() + ((self.q2(inputs) - target) ** 2).mean()
                if not bool(torch.isfinite(critic_loss)):
                    raise RuntimeError("Nonfinite terminal critic loss")
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_norm = torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.)
                if not bool(torch.isfinite(critic_norm)):
                    raise RuntimeError("Nonfinite critic gradient")
                self.critic_optimizer.step()
                critics = list(self.q1.parameters()) + list(self.q2.parameters())
                for parameter in critics:
                    parameter.requires_grad_(False)
                try:
                    sampled, logp = self._sample(context)
                    features = torch.cat((context, sampled), dim=1)
                    actor_loss = (self.alpha * logp - torch.minimum(self.q1(features), self.q2(features))).mean()
                    if not bool(torch.isfinite(actor_loss)):
                        raise RuntimeError("Nonfinite terminal actor loss")
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
                    if not bool(torch.isfinite(actor_norm)):
                        raise RuntimeError("Nonfinite actor gradient")
                    self.actor_optimizer.step()
                finally:
                    for parameter in critics:
                        parameter.requires_grad_(True)
                logs.append((float(critic_loss.detach()), float(actor_loss.detach()), float(-logp.detach().mean())))
                self.updates += 1
        return dict(before=before, after=self.weight_hash(), gradient_steps=len(logs), total_gradient_steps=self.updates,
            replay_rows=self.replay_size, training_rows_seen=self.total_seen,
            critic_loss=float(np.mean([r[0] for r in logs])) if logs else None,
            actor_loss=float(np.mean([r[1] for r in logs])) if logs else None,
            normalized_action_entropy=float(np.mean([r[2] for r in logs])) if logs else None,
            target="Observed terminal reward only; no next-state bootstrap")

    def save(self, path, contract_fingerprint, *, include_replay=False):
        if not isinstance(contract_fingerprint, str) or not contract_fingerprint:
            raise ValueError("A nonempty frozen contract fingerprint is required")
        path = Path(path)
        if path.exists():
            raise ValueError("Preserve prior checkpoints; choose a new path")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch = self.torch
        checkpoint = dict(version=SAC_VERSION, contract_fingerprint=contract_fingerprint,
            context_names=["board_height_m", "tilt_deg"], action="release_gap_m", terminal=True,
            context_bounds=self.context_bounds.tolist(), gap_bounds=self.gap_bounds.tolist(),
            seed=self.seed, alpha=self.alpha, learning_rate=self.learning_rate, capacity=self.capacity,
            actor=self.actor.state_dict(), q1=self.q1.state_dict(), q2=self.q2.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
            rng_state=self.rng.bit_generator.state, generator_state=self.generator.get_state(),
            total_seen=self.total_seen, updates=self.updates, weight_hash=self.weight_hash(),
            data_roles=["train"], training_provenance=self.training_provenance,
            replay_included=bool(include_replay), entropy_scope="Normalized tanh action units")
        if include_replay:
            checkpoint["replay"] = dict(contexts=torch.tensor(self.contexts,device="cpu"), actions=torch.tensor(self.actions,device="cpu"), rewards=torch.tensor(self.rewards,device="cpu"))
        with path.open("xb") as stream:
            torch.save(checkpoint, stream)
        return dict(version=SAC_VERSION, contract_fingerprint=contract_fingerprint, weight_hash=checkpoint["weight_hash"],
            training_rows_seen=self.total_seen, gradient_steps=self.updates, replay_included=bool(include_replay))

    @classmethod
    def load(cls, path, expected_fingerprint, device="cpu"):
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != SAC_VERSION or checkpoint.get("contract_fingerprint") != expected_fingerprint:
            raise ValueError("SAC checkpoint version/contract mismatch")
        if checkpoint.get("terminal") is not True or checkpoint.get("data_roles") != ["train"] or checkpoint.get("context_names") != ["board_height_m", "tilt_deg"] or checkpoint.get("action") != "release_gap_m":
            raise ValueError("Incompatible or contaminated SAC checkpoint")
        obj = cls(checkpoint["seed"], checkpoint["context_bounds"], checkpoint["gap_bounds"],
            checkpoint["alpha"], checkpoint["learning_rate"], checkpoint["capacity"], device)
        for name in ("actor", "q1", "q2"):
            getattr(obj, name).load_state_dict(checkpoint[name], strict=True)
            if not all(bool(torch.isfinite(p).all()) for p in getattr(obj,name).parameters()):
                raise ValueError("Nonfinite SAC checkpoint weights")
        if obj.weight_hash() != checkpoint["weight_hash"]:
            raise ValueError("Changed SAC weight identity")
        obj.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        obj.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        obj.rng.bit_generator.state = checkpoint["rng_state"]
        obj.generator.set_state(checkpoint["generator_state"])
        obj.total_seen, obj.updates = checkpoint["total_seen"], checkpoint["updates"]
        obj.training_provenance = checkpoint["training_provenance"]
        if checkpoint.get("replay_included"):
            replay = checkpoint["replay"]
            obj.contexts = replay["contexts"].numpy().copy()
            obj.actions = replay["actions"].numpy().copy()
            obj.rewards = replay["rewards"].numpy().copy()
            if obj.contexts.shape != (len(obj.rewards), 2) or obj.actions.shape != obj.rewards.shape or obj.rewards.shape[1:] != (1,) or not all(np.isfinite(v).all() for v in (obj.contexts,obj.actions,obj.rewards)):
                raise ValueError("Invalid replay checkpoint")
            if len(obj.rewards)>obj.capacity or obj.total_seen<len(obj.rewards) or (np.abs(obj.contexts)>1+1e-6).any() or (np.abs(obj.actions)>1+1e-6).any():
                raise ValueError("Replay checkpoint violates declared bounds")
        return obj


class SuccessModel:
    """TRAIN-only RBF logistic forward model and fixed-grid inverse proposal.

    This produces a supervised probability estimate, not a calibrated guarantee.
    Its fixed grid evaluates only frozen model predictions, never simulator data.
    """

    def __init__(self, seed=0, context_bounds=CONTEXT_BOUNDS, gap_bounds=GAP_BOUNDS,
                 centres=64, width=.65, regularization=.1, grid_size=129):
        self.context_bounds, self.gap_bounds = _bounds(context_bounds,gap_bounds)
        if not 1 <= centres <= 256 or not np.isfinite(width) or not width > 0 or not np.isfinite(regularization) or not regularization > 0 or not 3 <= grid_size <= 4097:
            raise ValueError("Invalid bounded surrogate settings")
        self.seed, self.centres, self.width, self.regularization, self.grid_size = int(seed), int(centres), float(width), float(regularization), int(grid_size)
        self.fitted = False

    def _features(self, inputs):
        distance = ((inputs[:, None] - self.centers_[None]) ** 2).sum(-1)
        return np.c_[np.ones(len(inputs)), np.exp(-distance / (2 * self.width ** 2))]

    @staticmethod
    def _sigmoid(value):
        return 1 / (1 + np.exp(-np.clip(value, -40, 40)))

    def fit(self, rows):
        rows = list(rows)
        if not rows:
            raise ValueError("A nonempty TRAIN dataset is required")
        # Never let an explicit TRAIN role hide a contradictory held-out split.
        for row in rows:
            for key in ("role", "split"):
                if key in row:
                    _train_roles([row[key]], 1)
            if "reward" in row and (np.asarray(row["reward"]).shape != () or not np.isfinite(float(row["reward"]))):
                raise ValueError("Observed scalar rewards must be finite")
        contexts=[];gaps=[];labels=[]
        for row in rows:
            values = row.get("values")
            context = row.get("context", [row.get("board_height_m"),row.get("tilt_deg")])
            if values is not None:
                if np.asarray(values).shape!=(5,):
                    raise ValueError("A native action must contain five finite values")
                context, gap = values[2:4], values[4]
            else:
                gap = row.get("gap_m")
            label = row.get("route_success", row.get("label"))
            if label not in (False,True,0,1):
                raise ValueError("Observed binary route_success labels required")
            contexts.append(context);gaps.append(gap);labels.append(int(label))
        context = normalized_contexts(contexts,self.context_bounds)
        gap = normalized_gaps(gaps,len(context),self.gap_bounds)
        inputs = np.c_[context,gap];y=np.asarray(labels,float)
        unique = np.unique(inputs,axis=0)
        rng = np.random.default_rng(self.seed); order=rng.permutation(len(unique)); unique=unique[order]
        selected=[0]; nearest=((unique-unique[0])**2).sum(-1)
        while len(selected)<min(self.centres,len(unique)):
            index=int(np.argmax(nearest)); selected.append(index)
            nearest=np.minimum(nearest,((unique-unique[index])**2).sum(-1));nearest[selected]=-1
        self.centers_=unique[selected]
        features=self._features(inputs)
        self.coef_=np.zeros(features.shape[1]);p=(y.sum()+.5)/(len(y)+1)
        self.coef_[0]=math.log(p/(1-p));constant=bool(np.all(y==y[0])); iterations=0
        penalty=np.r_[1e-6,np.full(len(self.coef_)-1,self.regularization)]
        def objective(w):
            z=features@w
            return float(np.sum(np.logaddexp(0,z)-y*z)+.5*np.sum(penalty*w*w))
        if not constant:
            for iterations in range(1,81):
                probabilities=self._sigmoid(features@self.coef_)
                gradient=features.T@(probabilities-y)+penalty*self.coef_
                curvature=probabilities*(1-probabilities)+1e-8
                hessian=(features.T*curvature)@features+np.diag(penalty)
                step=np.linalg.solve(hessian,gradient);scale=1.;current=objective(self.coef_)
                while scale>1e-7 and objective(self.coef_-scale*step)>current:
                    scale*=.5
                self.coef_-=scale*step
                if np.linalg.norm(scale*step)<1e-7:break
        self.fitted=True
        probabilities=self._sigmoid(features@self.coef_)
        self.fit_log=dict(version=MODEL_VERSION,rows=len(rows),positive_labels=int(y.sum()),
            centres=len(self.centers_),iterations=iterations,constant_labels=constant,
            binary_log_loss=float(-np.mean(y*np.log(np.maximum(probabilities,1e-12))+(1-y)*np.log(np.maximum(1-probabilities,1e-12)))),
            training_data_fingerprint=_fingerprint(dict(contexts=np.asarray(contexts).tolist(),gaps=gaps,labels=labels)),
            role="train",probability_scope="Supervised estimate; calibration and held-out performance are separate")
        return self

    def predict(self, contexts, gaps):
        if not self.fitted:
            raise ValueError("Fit or load the TRAIN model before inference")
        context=normalized_contexts(contexts,self.context_bounds)
        gap=normalized_gaps(gaps,len(context),self.gap_bounds)
        return self._sigmoid(self._features(np.c_[context,gap])@self.coef_)

    def curve(self, contexts, grid=None):
        context=np.asarray(contexts,float).reshape(-1,2)
        grid=np.linspace(*self.gap_bounds,self.grid_size) if grid is None else np.asarray(grid,float).reshape(-1)
        repeated=np.repeat(context,len(grid),axis=0);gaps=np.tile(grid,len(context))
        return dict(gaps=grid,probability=self.predict(repeated,gaps).reshape(len(context),len(grid)))

    def propose(self, contexts):
        result=self.curve(contexts);grid=result["gaps"];middle=np.mean(self.gap_bounds)
        # Equal-score priors choose the central grid value, without outcome reads.
        indices=[min(np.flatnonzero(np.isclose(p,p.max(),rtol=0,atol=1e-12)),key=lambda i:abs(grid[i]-middle)) for p in result["probability"]]
        return grid[indices]

    def save(self,path,contract_fingerprint):
        if not self.fitted or not isinstance(contract_fingerprint,str) or not contract_fingerprint:
            raise ValueError("Fitted model and contract fingerprint required")
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        value=dict(version=MODEL_VERSION,contract_fingerprint=contract_fingerprint,seed=self.seed,
            context_bounds=self.context_bounds.tolist(),gap_bounds=self.gap_bounds.tolist(),
            centres=self.centres,width=self.width,regularization=self.regularization,grid_size=self.grid_size,
            centers=self.centers_.tolist(),coef=self.coef_.tolist(),fit_log=self.fit_log,data_roles=["train"])
        value["fingerprint"]=_fingerprint(value)
        with path.open("x") as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
        return value["fingerprint"]

    @classmethod
    def load(cls,path,expected_fingerprint):
        value=json.loads(Path(path).read_text());identity=value.pop("fingerprint",None)
        if identity!=_fingerprint(value) or value.get("version")!=MODEL_VERSION or value.get("contract_fingerprint")!=expected_fingerprint or value.get("data_roles")!=["train"]:
            raise ValueError("Surrogate checkpoint identity/contract mismatch")
        obj=cls(value["seed"],value["context_bounds"],value["gap_bounds"],value["centres"],value["width"],value["regularization"],value["grid_size"])
        obj.centers_,obj.coef_=np.asarray(value["centers"],float),np.asarray(value["coef"],float)
        if obj.centers_.ndim!=2 or obj.centers_.shape[1]!=3 or obj.coef_.shape!=(len(obj.centers_)+1,) or not np.isfinite(obj.centers_).all() or not np.isfinite(obj.coef_).all():
            raise ValueError("Invalid fitted model arrays")
        obj.fit_log=value["fit_log"];obj.fitted=True
        return obj
