# Drafter Training Activation & GPU Memory Management

This document details how FastRL manages GPU memory for drafter training and the foreground training activation flow.

## Overview

1. **Memory Allocation Strategy**: CPU offloading with on-demand GPU loading
2. **Foreground Scheduling**: Synchronous training phase in the RL loop
3. **Activation Flow**: Complete call chain from `ray_trainer` to training steps

## GPU Memory Allocation Strategy

### Separate Model Instances

The drafter is **not sharing memory** with the target model. It's a separate FSDP2 module:

```python
# fsdp_workers.py
if enable_drafter_training:
    self.drafter_trainer = EagleBackgroundTrainer(
        drafter_module_fsdp,      # Separate FSDP model instance
        drafter_optimizer,         # Separate optimizer
        drafter_lr_scheduler,
        drafter_train_config,
        self.drafter_device_mesh,  # Per-DP-group device mesh
        model_config=self.actor_model_config,
    )
    rollout.drafter_trainer = self.drafter_trainer  # Shared ref for data collection
```

### CPU Offloading by Default

The drafter model and optimizer live on CPU by default, only loading to GPU during the training phase:

```python
# verl/utils/fsdp_utils.py

@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    """Move FSDP model parameters from GPU to CPU."""
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()

@torch.no_grad()
def load_fsdp_model_to_gpu(model: FSDP):
    """Move FSDP model parameters from CPU to GPU."""
    device = get_device_id()
    model.to(device)

@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    """Move optimizer state tensors to CPU."""
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)
```

### Lifecycle: Offloaded Until Training Phase

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Drafter Model Memory Lifecycle                             │
│                                                                                  │
│   INITIALIZATION          ROLLOUT TIME             TRAINING PHASE               │
│   ──────────────          ────────────             ───────────────               │
│                                                                                  │
│   ┌──────────────┐       ┌──────────────┐        ┌──────────────┐              │
│   │   Create     │       │   Model on   │        │   Model on   │              │
│   │   FSDP Model │──────▶│     CPU      │───────▶│     GPU      │              │
│   │              │       │  (offloaded) │        │  (activated) │              │
│   └──────────────┘       └──────────────┘        └──────────────┘              │
│                                 ▲                        │                      │
│                                 │                        │                      │
│                                 └────────────────────────┘                      │
│                                   (cleanup_training)                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Complete Call Chain

### Initialization Phase

```
fsdp_workers._build_rollout()
─────────────────────────────────────────────────────────────────────────────────
1. enable_drafter_training = config check
2. _build_drafter_model()
   ├── Create drafter_device_mesh (per-DP-group)
   ├── Apply FSDP2 to drafter module
   └── Create optimizer + scheduler
3. self.drafter_trainer = EagleBackgroundTrainer(drafter_module_fsdp, ...)
4. rollout.drafter_trainer = self.drafter_trainer   # shared ref for data collection
5. self._enable_drafter_training = True
6. self._drafter_training_interval = config value
7. self._drafter_max_steps = config value
8. self._drafter_rl_step = 0

• Drafter model offloaded to CPU immediately after build
```

### Runtime Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RayPPOTrainer._training_step()                                                  │
│  ray_trainer.py                                                                  │
│                                                                                  │
│      # After rollout completes:                                                 │
│      with marked_timer("train_drafter"):                                        │
│          self.actor_rollout_wg.train_drafter()                                  │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ Dispatch.ONE_TO_ALL (sent to all workers)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ActorRolloutRefWorker.train_drafter()                                           │
│  fsdp_workers.py                                                                 │
│                                                                                  │
│      if not self._enable_drafter_training:                                      │
│          return                                                                  │
│      if self._drafter_rl_step % self._drafter_training_interval != 0:           │
│          return                                                                  │
│      self.drafter_trainer.train(num_steps=self._drafter_max_steps)              │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  EagleBackgroundTrainer.train()                                                  │
│  eagle_background_trainer.py                                                     │
│                                                                                  │
│      self.activate_training_model()                                             │
│      for step in range(num_steps):                                              │
│          if not self.training_step(step):                                       │
│              break                                                               │
│      self.cleanup_training()                                                    │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────────┐   ┌───────────────────┐
│ activate_training │   │    training_step()     │   │ cleanup_training  │
│ _model()          │   │                        │   │ ()                │
│                   │   │ try:                   │   │                   │
│ Load model to GPU │   │   _training_step_impl  │   │ Save checkpoint   │
│ Load optimizer    │   │   (fwd/bwd/optim)      │   │ Offload model/opt │
│ to GPU            │   │ except:                │   │ to CPU            │
│                   │   │   log + return False   │   │ Empty CUDA cache  │
└───────────────────┘   └───────────────────────┘   └───────────────────┘
```

### RL Step Tracking

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  ActorRolloutRefWorker.increment_rl_step()                                       │
│  fsdp_workers.py (called at end of each RL step)                                │
│                                                                                  │
│      if self._enable_drafter_training:                                          │
│          self._drafter_rl_step += 1                                             │
│          self.drafter_trainer.increment_rl_step()                               │
│              └── self.data_buffer.increment_step()  # mark step boundary        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Class Relationships

```
RayPPOTrainer (ray_trainer.py)
    │
    └── actor_rollout_wg: WorkerGroup
            │
            └── ActorRolloutRefWorker (fsdp_workers.py)
                    │
                    ├── drafter_trainer: EagleBackgroundTrainer
                    │       ├── model (FSDP2-wrapped drafter)
                    │       ├── optimizer (AdamW)
                    │       ├── data_buffer (DataBuffer)
                    │       └── training_device_mesh (per-DP-group)
                    │
                    └── rollout: SGLangRollout (sglang_rollout.py)
                            │
                            └── drafter_trainer ──▶ same object (shared ref)
                                 (used for collect_online_data during rollout)
```

## Timeline View

```
Time ──────────────────────────────────────────────────────────────────────────────▶

One RL Step:

 ┌────────────────────┐ ┌──────────────────┐ ┌──────┐ ┌──────────┐ ┌──────────┐
 │     Rollout        │ │  Train Drafter   │ │Reward│ │  Actor   │ │  Critic  │
 │  (SGLang generate  │ │ (CPU→GPU→train   │ │      │ │  Update  │ │  Update  │
 │   + collect data)  │ │  →cleanup→CPU)   │ │      │ │  (FSDP)  │ │  (FSDP)  │
 └────────────────────┘ └──────────────────┘ └──────┘ └──────────┘ └──────────┘
         │                       │
         │ hidden states         │ GPU memory freed for
         │ → data_buffer         │ actor/critic updates
         │                       │
         └── rollout.drafter_trainer.collect_online_data()
```

The drafter trains in a dedicated phase **after rollout and before RL updates**. This is important: by the time actor/critic updates need GPU memory, the drafter model has already been offloaded back to CPU.

## Key Implementation Details

### Training Interval Control

Training doesn't happen every RL step — it's controlled by `training_interval_steps`:

```python
# fsdp_workers.py
def train_drafter(self):
    if not self._enable_drafter_training:
        return
    if self._drafter_rl_step % self._drafter_training_interval != 0:
        return
    self.drafter_trainer.train(num_steps=self._drafter_max_steps)
```

### Early Stop on Empty Data

If the data buffer is empty or the model is missing, training stops early:

```python
# eagle_background_trainer.py
def _training_step_impl(self, step):
    if not self.model:
        return False
    batch = self._prepare_training_batch()
    if batch is None:
        return False   # → train() loop breaks
    ...
```

### Cleanup After Training

When training completes, models are offloaded back to CPU:

```python
# eagle_background_trainer.py
def cleanup_training(self):
    # Wait for any pending async checkpoint
    if self._pending_checkpoint_future is not None:
        self._pending_checkpoint_future.result()

    # Save final checkpoint
    if self.checkpoint_dir and self.model is not None:
        self._save_checkpoint_async(self.training_steps, is_final=True)

    # Offload to CPU
    offload_fsdp_model_to_cpu(self.model)
    offload_fsdp_optimizer(self.optimizer)
    torch.cuda.empty_cache()
    self.training_steps = 0
```

### Weight Sync to SGLang (After Training)

Trained weights are **not synced immediately**. At the start of the next rollout, `FSDPSGLangShardingManager.wake_up()` reads the latest `drafter_module.state_dict()` and pushes it to SGLang. This works because `drafter_module_fsdp` is the same Python object shared between the trainer and the sharding manager — training updates it in-place. See [Co-training Pipeline](./co-training-pipeline.md#weight-synchronization-to-sglang) for full details.

## Key Files Reference

| File | Purpose |
|------|---------|
| `ray_trainer.py` | Orchestrates `train_drafter()` call in RL loop |
| `fsdp_workers.py` | `train_drafter()`, `increment_rl_step()`, drafter model build |
| `eagle_background_trainer.py` | `train()`, `activate_training_model()`, `training_step()`, `cleanup_training()` |
| `sglang_rollout.py` | Hidden states collection via `collect_online_data()` |
| `fsdp_utils.py` | CPU/GPU offload utilities |

## See Also

- [Worker Hierarchy & GPU Allocation](./worker-hierarchy-gpu-allocation.md) - Ownership chain above SGLangRollout
- [Co-training Pipeline](./co-training-pipeline.md) - Training logic details
- [Hidden States Collection](./hidden-states-collection.md) - Data flow for training
- [Architecture Overview](./architecture.md) - System-level view
