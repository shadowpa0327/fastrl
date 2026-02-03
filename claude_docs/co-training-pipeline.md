# EAGLE Co-training Pipeline

This document details FastRL's novel approach to co-training EAGLE draft models alongside the main target model during RL training.

## Core Innovation

Traditional approaches train the drafter separately (offline), leading to:
- Drafter becoming stale as target model evolves
- Decreasing acceptance rates over RL training
- Need for periodic retraining

FastRL's co-training approach:
- Trains drafter continuously during RL training
- Uses spare GPU resources (no additional cost)
- Maintains high acceptance rates throughout training

## Implementation Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Co-training Decision Flow                            │
│                                                                              │
│   RayPPOTrainer._training_step()                                            │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │  if (step % training_interval_steps == 0) and enable_drafter_training │ │
│   │     and (len(data_buffer) >= min_samples):                             │ │
│   │                                                                        │ │
│   │     trigger_drafter_training()                                         │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│           │                                                                  │
│           ▼                                                                  │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │  EagleBackgroundTrainer.train_step(                                   │ │
│   │      data_buffer=collected_hidden_states,                              │ │
│   │      actor_model=current_actor_weights,                                │ │
│   │  )                                                                     │ │
│   └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key File: `eagle_background_trainer.py`

**Location**: `verl/workers/drafter/eagle_background_trainer.py` (708 lines)

### Class Structure

```python
class EagleBackgroundTrainer:
    """
    FSDP2-compatible background trainer for EAGLE draft model.

    Responsibilities:
    1. Initialize drafter model with frozen embeddings
    2. Manage training loop with data buffer
    3. Handle async checkpointing
    4. Synchronize weights to SGLang inference engine
    """

    def __init__(self, config, device_mesh, model_cls, tokenizer):
        # Initialize drafter model
        self.drafter_model = self._create_drafter_model()

        # Apply FSDP2 sharding
        self.drafter_model = self._apply_fsdp2(self.drafter_model)

        # Setup optimizer (AdamW default)
        self.optimizer = self._create_optimizer()

        # Data buffer for hidden states
        self.data_buffer = deque(maxlen=config.buffer_size)  # default: 2000
```

### Training Step Implementation

```python
def _train_step(self, batch):
    """Single training iteration on drafter model."""

    # Unpack batch
    input_ids = batch['input_ids']           # (B, L)
    hidden_states = batch['hidden_states']   # (B, L, D)
    loss_mask = batch['loss_mask']           # (B, L)

    # Prepare inputs: shift hidden states by 1 position
    # Input: hidden_states[:, :-1, :]  (predict next from current)
    # Target: hidden_states[:, 1:, :]  (what we want to predict)
    input_hidden = hidden_states[:, :-1, :]
    target_hidden = hidden_states[:, 1:, :]
    input_tokens = input_ids[:, :-1]

    # Forward through drafter
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        pred_hidden = self.drafter_model(
            input_hidden=input_hidden,
            input_ids=input_tokens,
        )

    # Compute loss (SmoothL1 for stability)
    loss = F.smooth_l1_loss(
        pred_hidden,
        target_hidden,
        reduction='none'
    )

    # Apply mask and reduce
    loss = (loss * loss_mask[:, 1:, None]).sum() / loss_mask[:, 1:].sum()

    # Backward pass (FSDP2 handles gradient sync)
    loss.backward()

    # Optimizer step
    self.optimizer.step()
    self.optimizer.zero_grad()

    return loss.item()
```

### Weight Synchronization

```python
def get_drafter_weights(self):
    """Extract drafter weights for SGLang update."""

    # Gather full state dict from FSDP shards
    with FSDP.state_dict_type(
        self.drafter_model,
        StateDictType.FULL_STATE_DICT
    ):
        state_dict = self.drafter_model.state_dict()

    # Filter to only trainable parameters
    trainable_dict = self._get_trainable_state_dict(state_dict)

    return trainable_dict

def sync_weights_to_sglang(self, sglang_client):
    """Push updated weights to running SGLang engine."""

    weights = self.get_drafter_weights()

    # Send via SGLang's weight update API
    sglang_client.update_weights(
        UpdateWeightsFromTensorReqInput(
            model_name="eagle_drafter",
            weights=weights,
        )
    )
```

## Hidden State Collection

Hidden states are collected from SGLang during speculative decoding verification:

```python
# In sglang_rollout.py
def generate_sequences(self, prompts, ...):
    # Enable hidden state collection
    sampling_params = SamplingParams(
        ...,
        return_hidden_states=True,  # Key flag
    )

    outputs = self.engine.generate(prompts, sampling_params)

    # Extract hidden states from outputs
    for output in outputs:
        hidden_states = output.hidden_states  # (L, D)
        self.data_buffer.append({
            'input_ids': output.token_ids,
            'hidden_states': hidden_states,
            'loss_mask': output.loss_mask,
        })
```

This is efficient because hidden states are already computed during the verification step of speculative decoding - no extra forward passes needed.

## Training Schedule

```
Configuration Parameters (fastrl_trainer.yaml):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

speculative.train:
  enable_drafter_training: true    # Master switch
  training_interval_steps: 10      # Train every N RL steps
  batch_size_per_gpu: 2            # Per-GPU batch size
  max_seq_len: 8192                # Max sequence length
  max_epochs: 10                   # Epochs per training trigger
  checkpoint_path: null            # Optional checkpoint save path
  min_workers_for_training: 1      # Min workers required

  optim:
    lr: 1e-6                       # Learning rate (conservative)
    lr_warmup_steps: 1000          # Warmup steps
    weight_decay: 0.0              # No weight decay by default
```

## Training Loop in RayPPOTrainer

```python
# In ray_trainer.py, _training_step()

def _training_step(self, step):
    # ... RL training logic ...

    # Check if drafter training should trigger
    if self._should_train_drafter(step):
        # Collect hidden states from recent rollouts
        data_buffer = self._get_drafter_training_data()

        if len(data_buffer) >= self.config.min_samples:
            # Trigger background training
            self.actor_rollout_worker.train_drafter(
                data_buffer=data_buffer,
                max_epochs=self.config.max_epochs,
            )

            # Weight sync happens automatically after training

def _should_train_drafter(self, step):
    return (
        self.config.enable_drafter_training and
        step > 0 and
        step % self.config.training_interval_steps == 0
    )
```

## FSDP2 Integration Details

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FSDP2 Sharding for Drafter                          │
│                                                                              │
│   Device Mesh: (dp=N, tp=1) for drafter (simpler than target model)        │
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

    # Prepare state dict (FSDP aware)
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
        process_group=self.process_group,
    )

    logger.info(f"Initiated async checkpoint save to {checkpoint_dir}")
```

## Monitoring and Metrics

Key metrics to track:
- `drafter/loss`: Training loss (should decrease)
- `drafter/acceptance_rate`: Acceptance rate during SD
- `drafter/training_time`: Time spent in drafter training
- `drafter/buffer_size`: Current data buffer size

```python
# Logging in train_step
metrics = {
    'drafter/loss': avg_loss,
    'drafter/epochs': num_epochs,
    'drafter/samples': len(data_buffer),
    'drafter/lr': current_lr,
}
wandb.log(metrics, step=global_step)
```

## Tuning Guidelines

| Scenario | Recommendation |
|----------|----------------|
| High acceptance rate (>0.7) | Reduce training frequency (interval=20) |
| Low acceptance rate (<0.4) | Increase training frequency (interval=5) |
| Memory pressure | Reduce batch_size_per_gpu |
| Fast target evolution | Decrease interval, increase epochs |
| Stable target model | Increase interval, save compute |

## See Also

- [Architecture Overview](./architecture.md)
- [Drafter Training Activation](./drafter-training-activation.md) - GPU memory management, activation call chain
- [Hidden States Collection](./hidden-states-collection.md) - Deep dive into hidden state data flow
- [Speculative Decoding](./speculative-decoding.md)
- [Key Files Reference](./key-files.md)
