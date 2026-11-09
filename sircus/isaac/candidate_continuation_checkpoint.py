"""New, hash-bound continuation checkpoints; model-only loading, never resume.

The original continuous checkpoint schema/loader remain unchanged. These
functions do no learning or native work. Writes require an explicit new path.
"""

import copy
import hashlib
import io
import math
import os
from pathlib import Path
import re
import tempfile

from candidate_continuation_contract import (CHECKPOINT_SCHEMA, PROTOCOL_SHA256,
    IMITATION_MAX_STEPS, DEMONSTRATIONS, DEMO_BATCH, PPO_MAX_UPDATES, PPO_MAX_STEPS,
    PPO_BUFFER, PPO_EPOCHS, SEED)
from candidate_continuation_ledger import PARENT_CHECKPOINT_SHA256
from candidate_continuation_policy import PROBABILITY_LAW
from continuous_policy import ARCHITECTURE, ContinuousPolicy, actor_weight_hash, weight_hash, torch

COUNTERS = {"imitation_steps", "imitation_presentations", "ppo_updates", "ppo_steps", "ppo_samples"}
REQUIRED_BINDINGS = {"parent_checkpoint_sha256", "input_sha256", "support_sha256",
                     "source_sha256", "learning_protocol_sha256"}
INHERITED = dict(seed=20261122, requests=10000, optimizer_updates=73)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value, label):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256 " + label)


def state_hash(value):
    """Stable type-aware hash, including tensor dtype/shape/bytes and integer keys."""
    digest = hashlib.sha256()

    def add(raw):
        digest.update(str(len(raw)).encode()+b":"+raw)

    def visit(item):
        if torch is not None and isinstance(item, torch.Tensor):
            require(item.layout == torch.strided, "Unsupported checkpoint tensor layout")
            tensor = item.detach().cpu().contiguous()
            require(not tensor.is_floating_point() or torch.isfinite(tensor).all().item(),
                    "Nonfinite checkpoint tensor")
            require(not tensor.is_complex(), "Complex checkpoint tensors are unsupported")
            add(b"tensor"); add(str(tensor.dtype).encode()); add(str(tuple(tensor.shape)).encode())
            add(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            require(all(type(k) in (str, int) for k in item), "Unsupported checkpoint mapping key")
            add(b"dict"); add(str(len(item)).encode())
            for key in sorted(item, key=lambda k:(type(k).__name__, str(k))):
                visit(key); visit(item[key])
        elif type(item) in (tuple, list):
            add(type(item).__name__.encode()); add(str(len(item)).encode())
            for element in item:
                visit(element)
        elif type(item) in (str, int, float, bool) or item is None:
            require(type(item) is not float or math.isfinite(item), "Nonfinite checkpoint scalar")
            add(type(item).__name__.encode()); add(repr(item).encode())
        else:
            raise ValueError("Unsupported checkpoint state type: " + type(item).__name__)
    visit(value)
    return digest.hexdigest()


def _cpu(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:_cpu(v) for k,v in value.items()}
    if type(value) is list:
        return [_cpu(v) for v in value]
    if type(value) is tuple:
        return tuple(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _bindings(value):
    require(type(value) is dict and REQUIRED_BINDINGS.issubset(value), "Missing checkpoint bindings")
    for key, fingerprint in value.items():
        require(type(key) is str and bool(key), "Invalid checkpoint binding name")
        sha(fingerprint, key)
    require(value["parent_checkpoint_sha256"] == PARENT_CHECKPOINT_SHA256,
            "Continuation parent checkpoint changed")
    require(value["learning_protocol_sha256"] == PROTOCOL_SHA256, "Learning protocol changed")


def _counters(stage, value):
    require(stage in ("imitation", "ppo", "final"), "Unknown continuation checkpoint stage")
    require(type(value) is dict and set(value) == COUNTERS
            and all(type(v) is int and v >= 0 for v in value.values()), "Invalid continuation counters")
    steps = value["imitation_steps"]
    require(steps <= IMITATION_MAX_STEPS, "Imitation step cap exceeded")
    batches = math.ceil(DEMONSTRATIONS / DEMO_BATCH)
    epochs, partial = divmod(steps, batches)
    require(value["imitation_presentations"] == epochs*DEMONSTRATIONS + partial*DEMO_BATCH,
            "Imitation presentations differ from fixed schedule")
    updates, samples = value["ppo_updates"], value["ppo_samples"]
    require(updates <= PPO_MAX_UPDATES and value["ppo_steps"] == updates*PPO_EPOCHS <= PPO_MAX_STEPS
            and samples <= 1750 and updates == math.ceil(samples/PPO_BUFFER),
            "PPO counters differ from bounded buffer/update schedule")
    if stage == "imitation":
        require(updates == samples == 0, "PPO counters present during imitation")
    else:
        require(steps == IMITATION_MAX_STEPS, "PPO checkpoint precedes complete imitation")
    if stage == "final":
        require(1400 <= samples <= 1750, "Final checkpoint precedes possible TRAIN stopping cap")


def _optimizer(state, parameters, expected_steps, learning_rate, label):
    require(type(state) is dict and set(state) == {"state", "param_groups"}, "Invalid " + label + " optimizer")
    require(type(state["state"]) is dict and type(state["param_groups"]) is list
            and len(state["param_groups"]) == 1, "Invalid " + label + " Adam groups")
    group = state["param_groups"][0]
    require(type(group) is dict and group.get("params") == list(range(len(parameters))),
            "Changed " + label + " optimizer parameter order")
    expected = dict(lr=learning_rate, betas=(.9, .999), eps=1e-8, weight_decay=0.,
                    amsgrad=False, foreach=False, fused=False)
    require(all(group.get(k) == v for k,v in expected.items()), "Changed " + label + " Adam settings")
    require(not any(group.get(k, False) for k in ("maximize", "capturable", "differentiable")),
            "Changed " + label + " Adam behavior")
    require(set(state["state"]) == (set(range(len(parameters))) if expected_steps else set()),
            "Missing or unexpected " + label + " optimizer moments")
    for index, moments in state["state"].items():
        require(type(index) is int and type(moments) is dict
                and set(moments) == {"step", "exp_avg", "exp_avg_sq"}, "Invalid Adam moment schema")
        step = moments["step"]
        require(isinstance(step, torch.Tensor) and step.numel() == 1 and float(step) == expected_steps,
                "Adam step count differs from new continuation counters")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = moments[name]
            require(isinstance(tensor, torch.Tensor) and tensor.dtype == parameters[index].dtype
                    and tensor.shape == parameters[index].shape, "Adam moment parameter shape/dtype changed")
        require((moments["exp_avg_sq"] >= 0).all().item(), "Negative Adam second moment")
    state_hash(state)


def _validate(saved, expected_bindings):
    require(torch is not None, "Continuation checkpoints require configured PyTorch")
    fields = {"schema", "probability_law", "architecture", "learning_protocol_sha256", "seed", "stage",
              "inherited", "counters", "bindings", "state_dict", "weight_hash", "actor_weight_hash",
              "optimizer_states", "optimizer_state_hash", "rng_states", "rng_state_hash",
              "ledger_fingerprint", "journal_fingerprint", "execution_authorized", "release_authorized",
              "fingerprint"}
    require(type(saved) is dict and set(saved) == fields, "Continuation checkpoint schema fields changed")
    require(saved["schema"] == CHECKPOINT_SCHEMA and saved["probability_law"] == PROBABILITY_LAW
            and saved["architecture"] == ARCHITECTURE and saved["learning_protocol_sha256"] == PROTOCOL_SHA256
            and saved["seed"] == SEED and saved["inherited"] == INHERITED, "Continuation identity changed")
    require(saved["execution_authorized"] is False and saved["release_authorized"] is False,
            "Checkpoint cannot authorize execution or release")
    _bindings(saved["bindings"]); _bindings(expected_bindings)
    require(saved["bindings"] == expected_bindings, "Checkpoint source/input/support bindings changed")
    _counters(saved["stage"], saved["counters"])
    for name in ("ledger_fingerprint", "journal_fingerprint", "weight_hash", "actor_weight_hash"):
        sha(saved[name], name)
    unsigned = {k:v for k,v in saved.items() if k != "fingerprint"}
    require(state_hash(unsigned) == saved["fingerprint"], "Continuation checkpoint fingerprint changed")
    require(state_hash(saved["optimizer_states"]) == saved["optimizer_state_hash"]
            and state_hash(saved["rng_states"]) == saved["rng_state_hash"], "Optimizer/RNG hash changed")
    with torch.random.fork_rng(devices=[]):
        policy = ContinuousPolicy()
    expected = policy.state_dict()
    require(type(saved["state_dict"]) is dict and set(saved["state_dict"]) == set(expected),
            "Model tensor names changed")
    for name, tensor in saved["state_dict"].items():
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == expected[name].dtype
                and tensor.shape == expected[name].shape, "Model tensor shape/dtype changed")
    policy.load_state_dict(saved["state_dict"], strict=True)
    require(((policy.logstd >= -3.) & (policy.logstd <= 1.)).all().item(),
            "Saved log scales exceed original clamp bounds")
    require(weight_hash(policy) == saved["weight_hash"]
            and actor_weight_hash(policy) == saved["actor_weight_hash"], "Model tensor hash changed")
    optimizers = saved["optimizer_states"]
    require(type(optimizers) is dict and set(optimizers) == {"imitation", "ppo"}, "Optimizer phases changed")
    _optimizer(optimizers["imitation"], list(policy.actor.parameters()),
               saved["counters"]["imitation_steps"], 1e-4, "imitation")
    if saved["stage"] == "imitation":
        require(optimizers["ppo"] is None, "PPO optimizer exists before stage transition")
    else:
        _optimizer(optimizers["ppo"], list(policy.parameters()), saved["counters"]["ppo_steps"], 3e-4, "PPO")
    rng = saved["rng_states"]
    require(type(rng) is dict and set(rng) == {"shuffle", "sample"}, "RNG phases changed")
    for tensor in rng.values():
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.uint8
                and tensor.ndim == 1 and tensor.device.type == "cpu", "Invalid CPU generator state")
        generator = torch.Generator(device="cpu")
        try:
            generator.set_state(tensor)
        except RuntimeError as error:
            raise ValueError("Invalid dedicated RNG state") from error
    return policy


def checkpoint_payload(policy, *, stage, counters, bindings, rng_states, optimizer_states,
                       ledger_fingerprint, journal_fingerprint):
    """Snapshot detached state under the new schema, without changing any state."""
    require(torch is not None, "Continuation checkpoints require configured PyTorch")
    saved = dict(schema=CHECKPOINT_SCHEMA, probability_law=PROBABILITY_LAW,
        architecture=copy.deepcopy(ARCHITECTURE), learning_protocol_sha256=PROTOCOL_SHA256,
        seed=SEED, stage=stage, inherited=copy.deepcopy(INHERITED), counters=_cpu(counters),
        bindings=_cpu(bindings), state_dict=_cpu(dict(policy.state_dict())), weight_hash=weight_hash(policy),
        actor_weight_hash=actor_weight_hash(policy), optimizer_states=_cpu(optimizer_states),
        optimizer_state_hash=state_hash(optimizer_states), rng_states=_cpu(rng_states),
        rng_state_hash=state_hash(rng_states), ledger_fingerprint=ledger_fingerprint,
        journal_fingerprint=journal_fingerprint, execution_authorized=False, release_authorized=False)
    saved["fingerprint"] = state_hash(saved)
    _validate(saved, bindings)
    return saved


def save_checkpoint_new(path, payload):
    """Fsync a new complete file, publish without replacement, then fsync directory."""
    _validate(payload, payload["bindings"])
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".candidate-checkpoint-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_continuation(path, *, expected_sha256, expected_bindings, device="cpu"):
    """Load only the model; never restore optimizer or global/dedicated RNG state."""
    require(torch is not None, "Continuation checkpoints require configured PyTorch")
    sha(expected_sha256, "external checkpoint bytes")
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Continuation checkpoint file SHA changed")
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    policy = _validate(saved, expected_bindings)
    return policy.to(device), saved
