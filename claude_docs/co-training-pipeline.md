# EAGLE Co-training Pipeline

This document details FastRL's novel approach to co-training EAGLE draft models alongside the main target model during RL training.

## Core Innovation

Traditional approaches train the drafter separately (offline), leading to:
- Drafter becoming stale as target model evolves
- Decreasing acceptance rates over RL training
- Need for periodic retraining

FastRL's co-training approach:
- Trains drafter continuously during RL training
- Uses FSDP2 data-parallel training across TP groups
- Maintains high acceptance rates throughout training

## Implementation Overview

Drafter training runs as a **synchronous foreground phase** in the RL loop, orchestrated by `RayPPOTrainer`:

```
RayPPOTrainer._training_step()
        │
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Phase 1: Rollout (generate_sequences)                      │
  │    - SGLang generates with speculative decoding              │
  │    - Hidden states collected into data buffer                │
  └────────────────────────────┬────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Phase 2: Drafter Training (train_drafter)                  │
  │    if (rl_step % training_interval == 0):                   │
  │                                                              │
  │    fsdp_workers.train_drafter()                             │
  │      └── drafter_trainer.train(num_steps=200)               │
  │            ├── activate_training_model()  # CPU → GPU        │
  │            ├── for step in range(num_steps):                 │
  │            │     training_step(step)      # fwd/bwd/optim   │
  │            └── cleanup_training()         # GPU → CPU        │
  └────────────────────────────┬────────────────────────────────┘
        │
        ▼
  Phase 3: Reward → Actor Update → Critic Update → next step
```

## Key File: `eagle_background_trainer.py`

**Location**: `verl/workers/drafter/eagle_background_trainer.py`

### Class Structure

```python
class EagleBackgroundTrainer:
    """
    FSDP2-compatible trainer for EAGLE draft model.

    Responsibilities:
    1. Manage drafter model GPU offloading (CPU ↔ GPU)
    2. Collect hidden states from rollouts into data buffer
    3. Run training loop with FSDP2 gradient sync
    4. Handle async checkpointing
    """

    def __init__(self, model, optimizer, scheduler, config, device_mesh, model_config):
        self.model = model                  # FSDP2-wrapped drafter
        self.optimizer = optimizer          # AdamW
        self.training_device_mesh = device_mesh  # Per-DP-group mesh
        self.data_buffer = DataBuffer(...)  # Cross-step data storage
        self.collected_data = deque(...)    # Per-step data collection
```

### Top-Level Entry: `train()`

```python
def train(self, num_steps: int = 200):
    """Synchronous training: activate, train N steps, cleanup."""
    self.activate_training_model()     # Load model+optimizer to GPU
    for step in range(num_steps):
        if not self.training_step(step):  # Returns False to stop early
            break
    self.cleanup_training()             # Offload back to CPU
```

### Training Step Implementation

```python
def _training_step_impl(self, step, batch):
    # Unpack batch
    input_ids = batch['input_ids']           # (1, total_seq_len)
    hidden_states = batch['hidden_states']   # (1, total_seq_len, D)
    loss_mask = batch['loss_mask']           # (1, total_seq_len)

    # Forward through drafter (with FSDP2 all-gather)
    outputs = self.model(
        input_ids=input_ids,
        hidden_states=hidden_states,
        output_hidden_states=True,
    )

    # Compute loss (SmoothL1 for stability)
    loss = smooth_l1_loss(pred_hidden, target_hidden, reduction='none')
    loss = (loss * loss_mask).sum() / loss_mask.sum()

    # Backward (FSDP2 reduce-scatter handles gradient sync)
    loss.backward()

    # Optimizer step (each GPU updates its local shard)
    self.optimizer.step()
    self.optimizer.zero_grad()
```

## FSDP2 Data Parallel Training (TP=4 Example)

With 8 GPUs and TP=4, we get 2 independent DP groups:

```
DP Group 0: GPU [0, 1, 2, 3]  ← drafter FSDP mesh
DP Group 1: GPU [4, 5, 6, 7]  ← drafter FSDP mesh

Each group trains independently (no cross-group communication).
```

### Within a DP Group

```
Step 1: Data (identical buffer on all 4 GPUs from broadcast)
───────────────────────────────────────────────────────────
Pool: [sample_A, sample_B, ..., sample_N]  (same on all GPUs)

GPU0: random.sample() → [A, D]     ← likely different
GPU1: random.sample() → [B, F]     ← likely different
GPU2: random.sample() → [C, A]     ← can overlap
GPU3: random.sample() → [E, B]     ← can overlap

Step 2: Forward (FSDP2 all-gathers full params)
───────────────────────────────────────────────
GPU0        GPU1        GPU2        GPU3
  │           │           │           │
  └──── all-gather (collect full params) ────┘
  │           │           │           │
  ▼           ▼           ▼           ▼
forward     forward     forward     forward
(batch_0)   (batch_1)   (batch_2)   (batch_3)

Step 3: Backward (FSDP2 reduce-scatter averages grads)
──────────────────────────────────────────────────────
grad_0      grad_1      grad_2      grad_3
  │           │           │           │
  └──── reduce-scatter (average + re-shard) ─────┘
  │           │           │           │
  ▼           ▼           ▼           ▼
grad_shard  grad_shard  grad_shard  grad_shard
   _0          _1          _2          _3

Step 4: optimizer.step() — each GPU updates local shard only
```

## Weight Synchronization to SGLang

Drafter weights are **not explicitly synced after training**. Instead, they sync automatically at the **start of the next rollout** via a shared FSDP module reference.

**The shared reference** (`fsdp_workers.py`):

```python
# Same drafter_module_fsdp object is given to both:
self.drafter_trainer = EagleBackgroundTrainer(
    drafter_module_fsdp, ...       # ← Training updates this in-place
)
rollout_sharding_manager.drafter_module = drafter_module_fsdp  # ← wake_up() reads from this
```

**Sync happens in `wake_up()`** (`fsdp_sglang.py`), triggered by `with self.rollout_sharding_manager:` at the start of each rollout:

```python
async def wake_up(self):
    # 1. Sync actor weights to SGLang
    params = self.module.state_dict()
    await self.update_weights(params)

    # 2. Sync drafter weights to SGLang
    if self.drafter_module is not None:
        drafter_params = self.drafter_module.state_dict()  # reads trained weights
        drafter_params = convert_weight_keys(drafter_params, ...)
        await self.update_drafter_weights(drafter_params)  # pushes to SGLang
```

**Timeline**:

```
Step N:  Train drafter (updates FSDP module in-place) → offload to CPU
                                                            │
Step N+1: with sharding_manager: ───────────────────────────┘
              └── wake_up()
                   ├── sync actor weights
                   ├── sync drafter weights  ◀── reads latest trained weights
                   └── flush KV cache
```

## Hidden State Collection

Hidden states are collected during rollout from SGLang's speculative decoding verification step:

```python
# In sglang_rollout.py
def _batch_level_generate_sequences(self, prompts, ...):
    output = self._engine.async_generate(
        ...,
        return_hidden_states=self._should_collect_hidden_states(),
    )

    # Extract hidden states from output
    if should_collect:
        for sample in output:
            hidden_states = torch.tensor(sample["meta_info"]["hidden_states"])
            engine_hidden_states.append(hidden_states)

        self.drafter_trainer.collect_online_data(filtered_batch, engine_hidden_states)
```

This is efficient because hidden states are already computed during the verification step — no extra forward passes needed.

## Training Schedule

```
Configuration Parameters (fastrl_trainer.yaml):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

speculative.train:
  enable_drafter_training: true    # Master switch
  training_interval_steps: 10      # Train every N RL steps
  max_training_steps_per_round: 200  # Max gradient steps per round
  collect_hidden_states_from_sgl: true  # Collect from SGLang engine
  batch_size_per_gpu: 2            # Per-GPU batch size
  max_seq_len: 8192                # Max sequence length
  checkpoint_path: null            # Optional checkpoint save path

  optim:
    lr: 1e-6                       # Learning rate (conservative)
    lr_warmup_steps: 1000          # Warmup steps
    weight_decay: 0.0              # No weight decay by default
```

Training is gated by two checks in `fsdp_workers.train_drafter()`:

```python
def train_drafter(self):
    if not self._enable_drafter_training:
        return
    if self._drafter_rl_step % self._drafter_training_interval != 0:
        return
    self.drafter_trainer.train(num_steps=self._drafter_max_steps)
```

## FSDP2 Integration Details

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FSDP2 Sharding for Drafter                          │
│                                                                              │
│   Device Mesh: per-DP-group (e.g. [0,1,2,3] for TP=4)                     │
│   Same mesh used for both inference TP and drafter FSDP                    │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Drafter Model Structure                         │  │
│   │                                                                      │  │
│   │   ┌───────────────────────────────────────────────────────────────┐ │  │
│   │   │  Embedding Layer (FROZEN, from target model)                  │ │  │
│   │   │  - No gradient computation                                     │ │  │
│   │   │  - Shared with target model weights                           │ │  │
│   │   └───────────────────────────────────────────────────────────────┘ │  │
│   │                              │                                       │  │
│   │                              ▼                                       │  │
│   │   ┌───────────────────────────────────────────────────────────────┐ │  │
│   │   │  EAGLE Layers (TRAINABLE, FSDP sharded)                       │ │  │
│   │   │  - 1-2 transformer layers                                      │ │  │
│   │   │  - Lightweight compared to target                              │ │  │
│   │   │  - Full backward pass enabled                                  │ │  │
│   │   └───────────────────────────────────────────────────────────────┘ │  │
│   │                              │                                       │  │
│   │                              ▼                                       │  │
│   │   ┌───────────────────────────────────────────────────────────────┐ │  │
│   │   │  LM Head (FROZEN, from target model)                          │ │  │
│   │   │  - Used only for inference, not training loss                  │ │  │
│   │   └───────────────────────────────────────────────────────────────┘ │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Async Checkpointing

```python
def _save_checkpoint_async(self, step):
    """Non-blocking checkpoint save using DCP async_save."""
    if self.checkpoint_path is None:
        return

    state_dict = {
        'model': self._get_trainable_state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'step': step,
    }

    # Async save (returns immediately)
    checkpoint_dir = f"{self.checkpoint_path}/step_{step}"
    dcp.async_save(
        state_dict,
        checkpoint_dir,
        process_group=self.training_device_mesh.get_group(),
    )
```

## Tuning Guidelines

| Scenario | Recommendation |
|----------|----------------|
| High acceptance rate (>0.7) | Reduce training frequency (interval=20) |
| Low acceptance rate (<0.4) | Increase training frequency (interval=5) |
| Memory pressure | Reduce batch_size_per_gpu |
| Fast target evolution | Decrease interval, increase max_steps |
| Stable target model | Increase interval, save compute |

## See Also

- [Architecture Overview](./architecture.md)
- [Drafter Training Activation](./drafter-training-activation.md) - GPU memory management, activation call chain
- [Hidden States Collection](./hidden-states-collection.md) - Deep dive into hidden state data flow
- [Speculative Decoding](./speculative-decoding.md)
- [Key Files Reference](./key-files.md)
