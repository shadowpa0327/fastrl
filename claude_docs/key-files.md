# Key Files Reference

This document provides a quick reference to the most important files for understanding and modifying FastRL. Files are organized by functionality with hackability notes.

## Quick Navigation

| Area | Primary File | Lines | Purpose |
|------|--------------|-------|---------|
| Training Entry | `verl/trainer/main_fastrl.py` | ~185 | FastRL training entry point |
| RL Orchestration | `verl/trainer/ppo/ray_trainer.py` | ~1423 | Main training loop, worker coordination |
| FSDP Workers | `verl/workers/fsdp_workers.py` | ~2500 | Actor, Critic, Rollout workers |
| Drafter Co-training | `verl/workers/drafter/eagle_background_trainer.py` | ~708 | **Novel: EAGLE co-training** |
| SGLang Rollout | `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | ~1693 | Inference engine integration |
| EAGLE Worker | `third-party/sglang/.../eagle_worker.py` | ~500+ | Speculative decoding execution |
| MAB Strategy | `third-party/sglang/.../eagle_mab.py` | ~200 | **Novel: Adaptive SD strategy** |
| Config | `verl/trainer/config/fastrl_trainer.yaml` | ~200 | All hyperparameters |

## Entry Points

### `verl/trainer/main_fastrl.py`
**Purpose**: Main entry point for FastRL training

```python
# Key sections:
@hydra.main(config_path="config", config_name="fastrl_trainer")
def main(config):
    # Initialize Ray cluster
    ray.init(...)

    # Create dataset and sampler
    dataset = create_rl_dataset(config.data)
    sampler = create_rl_sampler(dataset, config.data)

    # Create trainer with speculative decoding config
    trainer = RayPPOTrainer(
        config=config,
        speculative_config=config.speculative,  # Key for SD
    )

    # Start training
    trainer.fit()
```

**Hack here for**: Custom dataset loading, cluster initialization, high-level config changes

---

## Training Orchestration

### `verl/trainer/ppo/ray_trainer.py`
**Purpose**: Distributed RL training orchestrator (most important file)

```python
class RayPPOTrainer:
    """
    Key methods to understand:
    """

    def init_workers(self):
        """Creates Ray actors for all workers."""
        # Lines ~200-400
        # Hack: Add new worker types, modify parallelism

    def fit(self):
        """Main training loop."""
        # Lines ~600-800
        # Hack: Modify training schedule, add callbacks

    def _training_step(self):
        """Single RL step: rollout → reward → update."""
        # Lines ~800-1200
        # Hack: Modify RL algorithm, add metrics

        # IMPORTANT: Drafter training trigger location
        if self._should_train_drafter(step):
            self.actor_rollout_worker.train_drafter(...)

    def generate_rollouts(self):
        """Generate rollout data with speculative decoding."""
        # Lines ~400-600
        # Hack: Modify rollout generation, sampling
```

**Hack here for**: Training loop modifications, new RL algorithms, drafter training schedule

---

### `verl/trainer/ppo/core_algos.py`
**Purpose**: RL algorithm implementations

```python
# Advantage estimators (line ~50-200)
ADVANTAGE_ESTIMATORS = {
    'gae': compute_gae_advantage,
    'grpo': compute_grpo_advantage,      # Group Relative Policy Optimization
    'reinforce++': compute_reinforce_pp,
    'rloo': compute_rloo_advantage,
    'opo': compute_opo_advantage,
}

# Policy loss functions (line ~200-350)
def compute_policy_loss(old_log_probs, new_log_probs, advantages, clip_ratio):
    """PPO clipped objective."""
    ...
```

**Hack here for**: New advantage estimators, loss function modifications

---

## FSDP Workers

### `verl/workers/fsdp_workers.py`
**Purpose**: FSDP-based distributed workers (~2500 lines, major file)

```python
class ActorRolloutRefWorker:
    """Combined Actor + Rollout + Reference worker."""

    # Key methods:
    def generate_sequences(self):
        """Generate rollout data via SGLang."""
        # Lines ~500-700
        # Calls into sglang_rollout

    def compute_log_probs(self):
        """Compute log probs for PPO."""
        # Lines ~700-900

    def update_policy(self):
        """Policy gradient update."""
        # Lines ~900-1100

    def train_drafter(self):
        """Trigger EAGLE drafter training."""
        # Lines ~1100-1300
        # KEY: This is where co-training is triggered

class CriticWorker:
    """Value function worker."""
    # Lines ~1500-1900

class RewardWorker:
    """Reward computation worker."""
    # Lines ~1900-2200
```

**Hack here for**: Worker behavior, model sharding, gradient computation

---

## EAGLE Co-training (Novel)

### `verl/workers/drafter/eagle_background_trainer.py`
**Purpose**: Background EAGLE drafter training (the core novelty)

```python
class EagleBackgroundTrainer:
    """
    FSDP2-compatible background trainer for EAGLE.

    This is the CORE INNOVATION of FastRL.
    """

    def __init__(self, config, device_mesh, model_cls):
        # Lines ~50-150
        # Hack: Model initialization, FSDP config

    def _train_step(self, batch):
        # Lines ~200-350
        """
        Single training step:
        1. Prepare shifted hidden states
        2. Forward through drafter
        3. Compute SmoothL1 loss
        4. Backward + optimizer step
        """
        # Hack: Loss function, training dynamics

    def get_drafter_weights(self):
        # Lines ~400-450
        """Extract weights for SGLang sync."""
        # Hack: Weight filtering, export format

    def sync_weights_to_sglang(self):
        # Lines ~450-500
        """Push weights to inference engine."""
        # Hack: Sync mechanism, timing

    def _save_checkpoint_async(self):
        # Lines ~500-600
        """Async DCP checkpointing."""
        # Hack: Checkpoint format, frequency
```

**Hack here for**: Training algorithm changes, loss modifications, sync mechanism

---

### `verl/workers/drafter/model/qwen2_eagle.py`
**Purpose**: EAGLE model architecture for Qwen2

```python
class Qwen2EagleModel(nn.Module):
    """
    Lightweight drafter model.

    Architecture:
    - Frozen embeddings (from base model)
    - 1-2 trainable transformer layers
    - Frozen LM head (for inference only)
    """

    def forward(self, hidden_states, input_ids):
        # Predict next hidden state from current
        ...
```

**Hack here for**: Architecture changes, new model families

---

## Speculative Decoding

### `verl/workers/rollout/sglang_rollout/sglang_rollout.py`
**Purpose**: SGLang integration for inference (~1693 lines)

```python
class SGLangRollout:
    """Manages SGLang inference engine."""

    def __init__(self, config):
        # Lines ~50-200
        # Hack: Engine initialization, memory config

    def generate(self, prompts, sampling_params):
        # Lines ~300-600
        """
        Generate sequences with speculative decoding.

        Key: Hidden state collection happens here.
        """
        # Hack: Sampling modifications, output processing

    def update_drafter_weights(self, weights):
        # Lines ~700-800
        """Update EAGLE weights in running engine."""
        # Hack: Hot-reload mechanism
```

**Hack here for**: Inference behavior, hidden state collection, engine config

---

### `third-party/sglang/python/sglang/srt/speculative/eagle_worker.py`
**Purpose**: EAGLE speculative decoding executor

```python
class EAGLEWorker:
    """Executes speculative decoding with EAGLE."""

    def forward_draft(self, hidden_states):
        # Lines ~100-200
        """Generate draft tokens from drafter."""
        # Hack: Draft generation strategy

    def forward_verify(self, draft_tree):
        # Lines ~200-350
        """Verify drafts with target model."""
        # Hack: Verification algorithm

    def build_draft_tree(self):
        # Lines ~350-450
        """Construct tree-based speculation."""
        # Hack: Tree structure, depth/width
```

**Hack here for**: Speculative decoding algorithm, tree structure

---

### `third-party/sglang/python/sglang/srt/speculative/eagle_mab.py`
**Purpose**: Multi-Armed Bandit for adaptive SD (novel)

```python
class EpsilonGreedyMAB:
    """Epsilon-greedy strategy selection."""
    # Lines ~20-100

class UCB1MAB:
    """Upper Confidence Bound variant."""
    # Lines ~100-150

class MABGroupManager:
    """Manages MABs per batch-size group."""
    # Lines ~150-200

    def select_strategy(self, batch_size):
        """Select SD config for batch size."""
        # Hack: Strategy selection logic

    def update(self, batch_size, strategy, reward):
        """Update with observed acceptance length."""
        # Hack: Reward signal, learning rate
```

**Hack here for**: Adaptive strategy algorithms, new reward signals

---

## Configuration

### `verl/trainer/config/fastrl_trainer.yaml`
**Purpose**: All hyperparameters

```yaml
# Key sections:

speculative:          # Lines ~10-50
  enable: true
  train:
    enable_drafter_training: true
    training_interval_steps: 10
    # ...

actor_rollout_ref:    # Lines ~50-100
  rollout:
    name: sglang
    # ...

algorithm:            # Lines ~100-150
  adv_estimator: grpo
  # ...

data:                 # Lines ~150-200
  # Dataset config
```

**Hack here for**: All hyperparameters, quick experiments

---

## Standalone EAGLE Training

### `eagle-train/eagle_trainer.py`
**Purpose**: Offline EAGLE drafter warm-up training

```python
# For initial drafter training before RL
# Uses DeepSpeed for efficiency

def main():
    # Data generation
    generate_training_data()

    # Train drafter
    train_eagle_drafter()
```

**Hack here for**: Pre-training drafter, data generation

---

## File Dependency Graph

```
main_fastrl.py
    │
    └──▶ ray_trainer.py
            │
            ├──▶ fsdp_workers.py
            │       │
            │       ├──▶ sglang_rollout.py
            │       │       │
            │       │       └──▶ eagle_worker.py
            │       │               │
            │       │               └──▶ eagle_mab.py
            │       │
            │       └──▶ eagle_background_trainer.py
            │               │
            │               └──▶ qwen2_eagle.py (model)
            │
            └──▶ core_algos.py
```

## SGLang Hidden States Collection Files

For understanding how hidden states flow through SGLang:

| File | Key Lines | Purpose |
|------|-----------|---------|
| `server_args.py` | 460 | `enable_return_hidden_states` server flag |
| `io_struct.py` | 126, 600 | Request hidden_states field |
| `schedule_batch.py` | 430, 591, 987 | Req class storage, batch flag |
| `forward_batch_info.py` | 139-157 | `CaptureHiddenMode` enum |
| `cuda_graph_runner.py` | 281-282, 704-708 | CUDA graph capture setup |
| `logits_processor.py` | 65, 474-505 | `LogitsProcessorOutput.hidden_states` |
| `scheduler_output_processor_mixin.py` | 358-360, 882-885 | Per-request accumulation |

See [Hidden States Collection](./hidden-states-collection.md) for detailed flow.

---

## Quick Modification Guide

| Goal | Files to Modify |
|------|-----------------|
| Change RL algorithm | `core_algos.py` |
| Modify drafter training | `eagle_background_trainer.py` |
| Change SD strategy | `eagle_mab.py` |
| Add new model architecture | `qwen2_eagle.py` + register |
| Modify inference | `sglang_rollout.py`, `eagle_worker.py` |
| Change training schedule | `ray_trainer.py` |
| Quick hyperparameter tuning | `fastrl_trainer.yaml` |
| Modify hidden states collection | `sglang_rollout.py`, `logits_processor.py` |

## See Also

- [Architecture Overview](./architecture.md)
- [Co-training Pipeline](./co-training-pipeline.md)
- [Speculative Decoding](./speculative-decoding.md)
