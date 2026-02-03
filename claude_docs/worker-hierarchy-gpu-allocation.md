# Worker Hierarchy & GPU Allocation

This document explains who owns SGLangRollout, how GPUs are allocated, and how data flows across workers.

## Concrete Example: 8 GPUs with TP=1

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RayPPOTrainer                                       │
│                         (Single controller process)                              │
│                                                                                  │
│   actor_rollout_wg: RayWorkerGroup                                              │
│       └── _workers: [8 Ray Actor handles]                                       │
│       └── _world_size: 8                                                         │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │               Data dispatch: batch.chunk(8)                   │
        │                                                               │
        ▼               ▼               ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐     ┌─────────────┐ ┌─────────────┐
│   GPU 0     │ │   GPU 1     │ │   GPU 2     │ ... │   GPU 6     │ │   GPU 7     │
│             │ │             │ │             │     │             │ │             │
│ ActorRollout│ │ ActorRollout│ │ ActorRollout│     │ ActorRollout│ │ ActorRollout│
│ RefWorker   │ │ RefWorker   │ │ RefWorker   │     │ RefWorker   │ │ RefWorker   │
│ (rank=0)    │ │ (rank=1)    │ │ (rank=2)    │     │ (rank=6)    │ │ (rank=7)    │
│             │ │             │ │             │     │             │ │             │
│ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │     │ ┌─────────┐ │ │ ┌─────────┐ │
│ │ SGLang  │ │ │ │ SGLang  │ │ │ │ SGLang  │ │     │ │ SGLang  │ │ │ │ SGLang  │ │
│ │ Rollout │ │ │ │ Rollout │ │ │ │ Rollout │ │     │ │ Rollout │ │ │ │ Rollout │ │
│ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │     │ └─────────┘ │ │ └─────────┘ │
│             │ │             │ │             │     │             │ │             │
│ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │     │ ┌─────────┐ │ │ ┌─────────┐ │
│ │ Actor   │ │ │ │ Actor   │ │ │ │ Actor   │ │     │ │ Actor   │ │ │ │ Actor   │ │
│ │ (FSDP   │ │ │ │ (FSDP   │ │ │ │ (FSDP   │ │     │ │ (FSDP   │ │ │ │ (FSDP   │ │
│ │ shard)  │ │ │ │ shard)  │ │ │ │ shard)  │ │     │ │ shard)  │ │ │ │ shard)  │ │
│ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │     │ └─────────┘ │ │ └─────────┘ │
└─────────────┘ └─────────────┘ └─────────────┘     └─────────────┘ └─────────────┘

     DP=0           DP=1           DP=2               DP=6           DP=7
```

### Component Counts (8 GPUs, TP=1)

| Component | Count | Explanation |
|-----------|-------|-------------|
| **RayPPOTrainer** | 1 | Single controller orchestrating everything |
| **RayWorkerGroup** | 1 | Manages all 8 workers collectively |
| **ActorRolloutRefWorker** | **8** | One per GPU (one Ray actor per GPU) |
| **SGLangRollout** | **8** | One per worker (each has its own inference engine) |
| **Actor Model (FSDP shard)** | **8** | Model sharded across all 8 GPUs |

### Key Points

- **TP=1 means no tensor parallelism** - each model replica fits on a single GPU
- **DP=8** - we have 8 data parallel replicas
- **Each ActorRolloutRefWorker is independent** - they don't communicate during inference
- **FSDP shards the model for training** - during `update_actor()`, gradients are synchronized
- **SGLang engines are independent** - each has its own KV cache

## Ownership Hierarchy

```
main_fastrl.py
    └── RayPPOTrainer (ray_trainer.py)
            │
            ├── actor_rollout_wg: RayWorkerGroup
            │       └── _workers: [ActorRolloutRefWorker × N]
            │
            ├── critic_wg: RayWorkerGroup
            │       └── _workers: [CriticWorker × N]
            │
            └── ref_policy_wg: RayWorkerGroup (optional)

ActorRolloutRefWorker (fsdp_workers.py)
    ├── actor_model: FSDP wrapped model (for training)
    ├── rollout: SGLangRollout (for inference)
    └── rollout_sharding_manager: handles weight sync

SGLangRollout (sglang_rollout.py)
    ├── _engine: SGLang Engine
    └── drafter_manager: RolloutDrafterManager
```

## Data Flow Example

With batch size = 1024 and 8 GPUs:

```
RayPPOTrainer: batch = DataProto(1024 samples)
        │
        │ actor_rollout_wg.generate_sequences(batch)
        │
        ▼ dispatch: chunk(8)
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  GPU 0: 128 samples ──▶ SGLangRollout.generate_sequences() ──▶ 128 outputs   │
│  GPU 1: 128 samples ──▶ SGLangRollout.generate_sequences() ──▶ 128 outputs   │
│  GPU 2: 128 samples ──▶ SGLangRollout.generate_sequences() ──▶ 128 outputs   │
│  GPU 3: 128 samples ──▶ SGLangRollout.generate_sequences() ──▶ 128 outputs   │
│  GPU 4: 128 samples ──▶ SGLangRollout.generate_sequences() ──▶ 128 outputs   │
│  GPU 5: 128 samples ──▶ SGLangRollout.generate_sequences() ──▶ 128 outputs   │
│  GPU 6: 128 samples ──▶ SGLangRollout.generate_sequences() ──▶ 128 outputs   │
│  GPU 7: 128 samples ──▶ SGLangRollout.generate_sequences() ──▶ 128 outputs   │
│                                                                               │
│                         (parallel execution via Ray)                          │
└───────────────────────────────────────────────────────────────────────────────┘
        │
        ▼ collect: concat()
RayPPOTrainer: output = DataProto(1024 outputs)
```

## GPU Allocation

### How Workers Get GPUs

1. **Config specifies resources**:
   ```python
   resource_pool_spec = {
       global_pool_id: [n_gpus_per_node] * nnodes,
   }
   # Example: 8 GPUs, 1 node → [8]
   ```

2. **Ray placement groups reserve GPUs**:
   ```python
   bundle = {"CPU": 1, "GPU": 1}  # 1 GPU per worker
   pg = placement_group(bundles=[bundle] * 8)
   ```

3. **Workers spawn on assigned GPUs**:
   ```python
   worker = ActorRolloutRefWorker.options(
       scheduling_strategy=PlacementGroupSchedulingStrategy(
           placement_group=pg,
           placement_group_bundle_index=rank,
       ),
   ).remote(config, role="actor_rollout")
   ```

### FSDP Device Mesh

Each worker participates in a global device mesh for gradient synchronization:

```python
# With 8 GPUs, fsdp_size=-1 (full shard):
device_mesh = init_device_mesh("cuda", mesh_shape=(8,), mesh_dim_names=["fsdp"])
```

## Dispatch Mechanism

Methods use `@register` decorator to control data distribution:

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def generate_sequences(self, prompts: DataProto):
    # Each worker receives 1/N of the batch
    ...
```

| Dispatch Mode | Behavior |
|--------------|----------|
| `ONE_TO_ALL` | Same data to all workers (e.g., `init_model`) |
| `DP_COMPUTE_PROTO` | Split batch across workers, concat results |

## Key Files

| File | Purpose |
|------|---------|
| `verl/trainer/main_fastrl.py` | Entry point |
| `verl/trainer/ppo/ray_trainer.py` | RayPPOTrainer, orchestration |
| `verl/single_controller/ray/base.py` | RayResourcePool, RayWorkerGroup |
| `verl/workers/fsdp_workers.py` | ActorRolloutRefWorker |
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | SGLangRollout |

## Configuration

```yaml
trainer:
  n_gpus_per_node: 8
  nnodes: 1

actor_rollout_ref:
  actor:
    strategy: "fsdp2"
    fsdp_config:
      fsdp_size: -1  # -1 = shard across all GPUs

  rollout:
    name: "sglang"
    mode: "sync"
```

## See Also

- [Drafter Training Activation](./drafter-training-activation.md) - GPU memory for drafter training
- [Architecture Overview](./architecture.md) - System-level view
- [Co-training Pipeline](./co-training-pipeline.md) - EAGLE training details
