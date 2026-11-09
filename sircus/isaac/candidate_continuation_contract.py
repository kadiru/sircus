"""Fixed learning settings for the seed-22 continuation implementation.

This is protocol identity, never execution permission. Importing it performs no
learning, file writes, native simulation or physical work.
"""

from candidate_continuation_ledger import canonical_hash, PROTOCOL_SHA256 as ACCOUNTING_SHA256
from candidate_continuation_policy import PROBABILITY_LAW

SCHEMA = "sircus-candidate-continuation-learning-protocol-v1"
CHECKPOINT_SCHEMA = "sircus-candidate-continuation-checkpoint-v1"
JOURNAL_SCHEMA = "sircus-candidate-continuation-journal-v1"
RUN_SCHEMA = "sircus-candidate-continuation-run-v1"
AUDIT_SCHEMA = "sircus-candidate-continuation-learning-audit-v1"
IMITATION_EPOCHS = 20
DEMONSTRATIONS = 1534
DEMO_BATCH = 128
IMITATION_MAX_STEPS = 240
PPO_BUFFER = 140
PPO_EPOCHS = 4
PPO_MAX_UPDATES = 13
PPO_MAX_STEPS = 52
SEED = 20261125


def protocol():
    return dict(schema=SCHEMA, probability_law=PROBABILITY_LAW,
        accounting_protocol_sha256=ACCOUNTING_SHA256, seed=SEED,
        inherited=dict(seed=20261122, requests=10000, optimizer_updates=73),
        imitation=dict(epochs=IMITATION_EPOCHS, demonstrations=DEMONSTRATIONS,
            batch_size=DEMO_BATCH, max_steps=IMITATION_MAX_STEPS,
            presentations=IMITATION_EPOCHS*DEMONSTRATIONS, learning_rate=1e-4,
            optimizer="Adam", parameters="actor-only", critic_frozen=True,
            log_scales_frozen=True, kl_coefficient=.05, wall_limit_s=600),
        ppo=dict(buffer_requests=PPO_BUFFER, epochs=PPO_EPOCHS,
            max_updates=PPO_MAX_UPDATES, max_steps=PPO_MAX_STEPS,
            learning_rate=3e-4, optimizer="Adam", fresh_state=True,
            clip=.2, value_coefficient=.5, entropy_coefficient=.005,
            gradient_norm=1., log_scale_limits=[-3., 1.],
            terminal_advantage="return-minus-detached-old-value",
            population_std_epsilon=1e-6, singleton_advantage="raw",
            flush_final_partial=True, all_rejections_included=True,
            wall_limit_s=7200),
        adam=dict(betas=[.9, .999], eps=1e-8, weight_decay=0., amsgrad=False,
            foreach=False, fused=False),
        assessment=dict(optimizer_steps=0, fixed_final_artifact=True),
        execution_authorized=False, release_authorized=False)


PROTOCOL_SHA256 = canonical_hash(protocol())
