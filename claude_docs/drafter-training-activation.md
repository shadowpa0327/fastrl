# Drafter Training Activation & GPU Memory Management

This document details how FastRL manages GPU memory for drafter training and implements the "bubble resource" utilization mechanism.

## Overview

FastRL claims to use "spare GPU resources" for drafter training at no additional cost. This document explains the actual implementation:

1. **Memory Allocation Strategy**: CPU offloading with on-demand GPU loading
2. **Bubble Detection**: Event-driven scheduling via ZMQ coordination
3. **Activation Flow**: Complete call chain from inference completion to training start

## GPU Memory Allocation Strategy

### Separate Model Instances

The drafter is **not sharing memory** with the target model. Instead, it's instantiated as a completely separate FSDP module:

```python
# fsdp_workers.py:788-796
if enable_drafter_training:
    rollout.drafter_manager.background_trainer = EagleBackgroundTrainer(
        drafter_module_fsdp,      # Separate FSDP model instance
        drafter_optimizer,         # Separate optimizer
        drafter_lr_scheduler,
        drafter_train_config,
        self.drafter_device_mesh,  # Separate device mesh
        model_config=self.actor_model_config,
    )
```

### CPU Offloading by Default

The key memory management strategy uses **dynamic GPU/CPU offloading**. The drafter model and optimizer states live on CPU by default, not consuming GPU memory during inference:

```python
# verl/utils/fsdp_utils.py:138-219

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

### Lifecycle: Offloaded Until Training Activates

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Drafter Model Memory Lifecycle                             │
│                                                                                  │
│   INITIALIZATION          INFERENCE TIME           TRAINING TIME                │
│   ──────────────          ──────────────           ─────────────                │
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

## "Bubble Resources" Implementation

The "bubble resource" claim is implemented through **event-driven scheduling**, not continuous GPU utilization monitoring.

### Worker State Machine

Workers transition through states managed by `CentralCoordinator`:

```python
# worker_manager.py:91-283
class WorkerState(Enum):
    IDLE = "idle"
    GENERATING = "generating"    # GPU occupied by inference
    RELEASED = "released"        # GPU memory freed - training can start!
    TRAINING = "training"        # Now training drafter
    COMPLETED = "completed"
```

### Coordination Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ZMQ-based Coordination System                             │
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                      CentralCoordinator (rank 0)                          │  │
│   │                                                                           │  │
│   │   • Runs in dedicated thread                                              │  │
│   │   • Maintains worker_states dict                                          │  │
│   │   • ZMQ REP socket for state updates                                     │  │
│   │   • ZMQ PUB socket for broadcasts                                         │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                    ▲                              │                              │
│                    │ REQ-REP                      │ PUB-SUB                      │
│                    │ (state updates)              │ (broadcasts)                 │
│                    │                              ▼                              │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                    RolloutDrafterManager (all ranks)                      │  │
│   │                                                                           │  │
│   │   • WorkerClient for communication                                        │  │
│   │   • Event listener thread (receives broadcasts)                           │  │
│   │   • Holds reference to EagleBackgroundTrainer                             │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Event Flow

```
Worker completes inference
         │
         ▼
release_worker_memory()  ──────▶  WorkerClient.release_worker()
                                           │
                                           │ ZMQ REQ
                                           ▼
                                  CentralCoordinator._handle_request()
                                           │
                                           │ worker.state = RELEASED
                                           ▼
                                  _check_and_start_training()
                                           │
                                           │ if enough workers released
                                           ▼
                                  _broadcast_event(START_TRAINING)
                                           │
                                           │ ZMQ PUB (broadcast)
                                           ▼
                                  Event listener thread receives
                                           │
                                           ▼
                                  _start_training(training_ranks)
                                           │
                                           ▼
                                  activate_training_model()
```

## Complete Call Chain for `activate_training_model()`

### Initialization Phase

```
fsdp_workers.py:788-796
─────────────────────────────────────────────────────────────────────────────────
rollout.drafter_manager.background_trainer = EagleBackgroundTrainer(...)

• RolloutDrafterManager holds reference to EagleBackgroundTrainer
• EagleBackgroundTrainer model starts on CPU (offloaded)
```

### Runtime Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  SGLangRollout.generate_sequences()                                              │
│  sglang_rollout.py:591-614                                                       │
│                                                                                  │
│      return self._batch_level_generate_sequences(prompts)                       │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  SGLangRollout._batch_level_generate_sequences()                                 │
│  sglang_rollout.py:618-996                                                       │
│                                                                                  │
│      if self._tp_rank == 0:                                                     │
│          output = loop.run_until_complete(                                      │
│              self._generate_with_drafter(idx_list, image_list, ...)             │
│          )                                                                       │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  SGLangRollout._generate_with_drafter()         ◀── PIVOT POINT                 │
│  sglang_rollout.py:998-1031                                                      │
│                                                                                  │
│      # 1. Run inference                                                          │
│      output = await self._engine.async_generate(...)                            │
│                                                                                  │
│      # 2. Release inference GPU memory                                          │
│      await self.sharding_manager.release_memory()                               │
│      torch.cuda.empty_cache()                                                   │
│                                                                                  │
│      # 3. Notify coordinator                                                    │
│      await self.drafter_manager.release_worker_memory(worker_id)                │
│                                                                                  │
│      return output                                                               │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RolloutDrafterManager.release_worker_memory()                                   │
│  worker_manager.py:635-657                                                       │
│                                                                                  │
│      if not self.should_train_this_step():                                      │
│          return True  # Skip training this RL step                              │
│                                                                                  │
│      response = await self.worker_client.release_worker(worker_id)              │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ ZMQ REQ-REP
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  CentralCoordinator._handle_request("release")                                   │
│  worker_manager.py:194-201                                                       │
│                                                                                  │
│      self.worker_states[worker_id].state = WorkerState.RELEASED                 │
│      await self._check_and_start_training()                                     │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  CentralCoordinator._check_and_start_training()                                  │
│  worker_manager.py:245-283                                                       │
│                                                                                  │
│      # Count RELEASED workers                                                   │
│      released_dp_ranks = {w.dp_rank for w in worker_states.values()             │
│                           if w.state == WorkerState.RELEASED}                   │
│                                                                                  │
│      if len(released_dp_ranks) >= min_workers_for_training:                     │
│          await self._broadcast_event(CoordinatorEvent.START_TRAINING, ranks)   │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ ZMQ PUB-SUB (broadcast)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RolloutDrafterManager._event_listener_thread_func()                             │
│  worker_manager.py:562-609  (runs in separate thread)                            │
│                                                                                  │
│      command = await self.worker_client.wait_for_event()                        │
│                                                                                  │
│      if command.event == CoordinatorEvent.START_TRAINING:                       │
│          if self.rank in command.training_ranks:                                │
│              await self._start_training(command.training_ranks)                 │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RolloutDrafterManager._start_training()                                         │
│  worker_manager.py:667-689                                                       │
│                                                                                  │
│      self._training_task = asyncio.create_task(                                 │
│          self._run_training_loop_with_init(training_ranks)                      │
│      )                                                                           │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RolloutDrafterManager._run_training_loop_with_init()                            │
│  worker_manager.py:691-711                                                       │
│                                                                                  │
│      success = await asyncio.wait_for(                                          │
│          self.background_trainer.activate_training_model(...),                  │
│          timeout=30.0                                                            │
│      )                                                                           │
│                                                                                  │
│      if success:                                                                 │
│          await self._run_training_loop()                                        │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  EagleBackgroundTrainer.activate_training_model()                                │
│  eagle_background_trainer.py:154-210                                             │
│                                                                                  │
│      # Load model from CPU to GPU                                               │
│      load_fsdp_model_to_gpu(self.model)                                         │
│                                                                                  │
│      # Load optimizer states from CPU to GPU                                    │
│      load_fsdp_optimizer(self.optimizer, device_id)                             │
│                                                                                  │
│      self._training_active = True                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Class Relationships

```
SGLangRollout (sglang_rollout.py)
    │
    └── drafter_manager: RolloutDrafterManager (worker_manager.py:411)
            │
            ├── worker_client: WorkerClient (worker_manager.py:314)
            │       └── Communicates with CentralCoordinator via ZMQ
            │
            ├── coordinator: CentralCoordinator (worker_manager.py:91)  # Only on rank 0
            │       └── Manages worker states, broadcasts events
            │
            └── background_trainer: EagleBackgroundTrainer (eagle_background_trainer.py)
                    └── Actually loads model to GPU and runs training steps
```

## `_generate_with_drafter`: The Pivot Point

The function `_generate_with_drafter` (sglang_rollout.py:998-1031) is the critical junction that bridges inference and training:

```python
async def _generate_with_drafter(self, idx_list, image_list, request_sampling_params):
    """Generate sequences with early memory release using global coordination."""
    current_worker_id = dist.get_rank()

    # Step 1: Run inference
    output = await self._engine.async_generate(
        prompt=None,
        input_ids=idx_list,
        return_hidden_states=bool(...),  # Collect for drafter training
    )

    # Step 2: Free inference GPU memory
    if self.sharding_manager is not None:
        if self._tp_rank == 0:
            await self.sharding_manager.release_memory()
    torch.cuda.empty_cache()

    # Step 3: Notify coordinator → triggers training on freed GPU
    if self.drafter_manager:
        await self.drafter_manager.release_worker_memory(current_worker_id)

    return output
```

### Three Responsibilities

1. **Performs inference** via `self._engine.async_generate()`
2. **Releases inference memory** via `sharding_manager.release_memory()` + `torch.cuda.empty_cache()`
3. **Triggers training** via `drafter_manager.release_worker_memory()`

## Timeline View

```
Time ──────────────────────────────────────────────────────────────────────────────▶

Worker GPU:
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│   Inference Running     │ │   GPU Memory Released   │ │   Drafter Training      │
│   (_engine.async_gen)   │ │   (release_memory +     │ │   (activate_training    │
│                         │ │    empty_cache)         │ │    _model → training    │
│                         │ │                         │ │    _step loop)          │
└─────────────────────────┘ └─────────────────────────┘ └─────────────────────────┘
          ▲                           ▲                           ▲
          │                           │                           │
          └── _generate_with_drafter ─┴─ release_worker_memory ───┘
              (lines 1003-1015)          (line 1029)
```

The "bubble" is the gap between when inference completes on a worker and when the next batch arrives. `_generate_with_drafter` explicitly frees GPU memory and signals the coordinator, allowing drafter training to occupy that freed GPU.

## Key Implementation Details

### Not Real "Bubbles" - Sequential Execution

The implementation is **sequential, not truly concurrent**:
- Training happens **after** rollout completes on a worker
- It uses the **time gap** between rollout completion and next batch arrival
- The "free" claim means training doesn't add wall-clock time if it finishes before next rollout

### 30-Second Timeout

`activate_training_model()` is wrapped with a timeout to prevent blocking:

```python
# worker_manager.py:697-700
success = await asyncio.wait_for(
    self.background_trainer.activate_training_model(...),
    timeout=30.0,
)
```

### Training Interval Control

Training doesn't happen every RL step - it's controlled by configuration:

```python
# worker_manager.py:484-487
def should_train_this_step(self) -> bool:
    if not self.train_drafter:
        return False
    return self.current_rl_step % self.training_interval_steps == 0
```

### Cleanup After Training

When training completes or is stopped, models are offloaded back to CPU:

```python
# eagle_background_trainer.py:678-690
async def cleanup_training(self):
    self._training_active = False

    if self.model is not None:
        offload_fsdp_model_to_cpu(self.model)

    if self.optimizer is not None:
        offload_fsdp_optimizer(self.optimizer)
```

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `sglang_rollout.py` | 998-1031 | `_generate_with_drafter()` - pivot point |
| `worker_manager.py` | 411-800 | `RolloutDrafterManager` - coordination |
| `worker_manager.py` | 91-308 | `CentralCoordinator` - state management |
| `worker_manager.py` | 314-408 | `WorkerClient` - ZMQ communication |
| `eagle_background_trainer.py` | 154-210 | `activate_training_model()` - GPU loading |
| `fsdp_utils.py` | 138-219 | CPU/GPU offload utilities |

## See Also

- [Co-training Pipeline](./co-training-pipeline.md) - Training logic details
- [Hidden States Collection](./hidden-states-collection.md) - Data flow for training
- [Architecture Overview](./architecture.md) - System-level view
