# Worker Hierarchy & GPU Allocation

This document details the ownership hierarchy above SGLangRollout, how GPUs are allocated to workers, and how data is sharded/dispatched across distributed workers.

## Ownership Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              main_fastrl.py                                      │
│                                                                                  │
│   ray.init() → TaskRunner.run(config)                                           │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RayPPOTrainer                                       │
│                         (ray_trainer.py:152-782)                                 │
│                                                                                  │
│   • Orchestrates the entire RL training loop                                    │
│   • Creates ResourcePoolManager for GPU allocation                              │
│   • Spawns worker groups via RayWorkerGroup                                     │
│   • Dispatches data to workers via worker groups                                │
│                                                                                  │
│   Key attributes:                                                                │
│   ├── resource_pool_manager: ResourcePoolManager                                │
│   ├── actor_rollout_wg: RayWorkerGroup (for ActorRolloutRefWorker)             │
│   ├── critic_wg: RayWorkerGroup (for CriticWorker)                             │
│   └── ref_policy_wg: RayWorkerGroup (for reference policy)                     │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RayWorkerGroup                                      │
│                           (ray/base.py:255-500)                                  │
│                                                                                  │
│   • Manages a group of Ray actor workers                                        │
│   • Creates workers on specific GPUs via placement groups                       │
│   • Handles data dispatch (chunking) and collection (concatenation)            │
│   • Binds worker methods with dispatch decorators                               │
│                                                                                  │
│   Key attributes:                                                                │
│   ├── _workers: list[ray.actor.ActorHandle]  # The actual Ray actors           │
│   ├── resource_pool: RayResourcePool                                           │
│   └── _world_size: int                                                          │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │ (Each worker is a Ray actor)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ActorRolloutRefWorker                                   │
│                         (fsdp_workers.py:123-1200)                               │
│                                                                                  │
│   • A single worker process running on 1 GPU                                    │
│   • Can play multiple roles: actor, rollout, ref (hybrid engine)               │
│   • Creates device mesh for FSDP sharding                                       │
│   • Owns the actual model and rollout engine                                    │
│                                                                                  │
│   Key attributes:                                                                │
│   ├── device_mesh: DeviceMesh  # For FSDP/tensor parallelism                   │
│   ├── actor_model: FSDP  # The policy model (for training)                     │
│   ├── rollout: SGLangRollout  # The inference engine                           │
│   ├── rollout_sharding_manager: FSDPUlyssesShardingManager                     │
│   └── role: str  # "actor", "rollout", "ref", "actor_rollout", etc.            │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SGLangRollout                                       │
│                      (sglang_rollout/sglang_rollout.py)                          │
│                                                                                  │
│   • The actual inference engine for generating sequences                        │
│   • Uses SGLang for efficient batched inference                                 │
│   • Manages drafter training coordination                                       │
│                                                                                  │
│   Key attributes:                                                                │
│   ├── _engine: SGLang Engine                                                    │
│   ├── drafter_manager: RolloutDrafterManager                                   │
│   └── sharding_manager: Reference to parent's sharding manager                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## GPU Allocation Mechanism

### 1. Resource Pool Specification

GPU allocation starts with specifying resources in the config:

```python
# main_fastrl.py:120-127
resource_pool_spec = {
    global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
}
# Example: 8 GPUs per node, 4 nodes → [8, 8, 8, 8]

mapping = {
    Role.ActorRollout: global_pool_id,
    Role.Critic: global_pool_id,
}
```

### 2. Ray Placement Groups

`RayResourcePool` creates Ray placement groups to reserve GPUs:

```python
# ray/base.py:102-132
class RayResourcePool:
    def get_placement_groups(self, strategy="STRICT_PACK"):
        # Bundle: 1 GPU per worker process
        bundle = {"CPU": self.max_colocate_count, "GPU": 1}

        # Create placement group per node
        pg_scheme = [[bundle.copy() for _ in range(process_count)]
                     for process_count in self._store]

        pgs = [placement_group(bundles=bundles, strategy=strategy, ...)
               for bundles in pg_scheme]

        ray.get([pg.ready() for pg in pgs])  # Wait for GPU reservation
        return pgs
```

### 3. Worker Spawning with GPU Assignment

`RayWorkerGroup` spawns workers on specific GPUs:

```python
# ray/base.py:345-430
def _init_with_resource_pool(self, resource_pool, ...):
    pgs = resource_pool.get_placement_groups()

    for pg_idx, pg in enumerate(pgs):
        for local_rank in range(local_world_size):
            rank += 1

            # Environment variables for distributed training
            env_vars = {
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
                "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
                "RAY_LOCAL_RANK": str(local_rank),
            }

            # Create Ray actor on specific GPU via placement group
            worker = ray_cls.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=local_rank,
                ),
                num_gpus=1 / max_colocate_count,  # Fractional GPU for colocation
            ).remote(...)

            self._workers.append(worker)
```

### 4. Device Mesh Creation (Within Worker)

Each worker creates a device mesh for FSDP:

```python
# fsdp_workers.py:101-108, 147-150
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        # Full sharding across all GPUs
        device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,),
                                       mesh_dim_names=["fsdp"])
    else:
        # Hybrid sharding: DDP across groups, FSDP within groups
        device_mesh = init_device_mesh("cuda",
            mesh_shape=(world_size // fsdp_size, fsdp_size),
            mesh_dim_names=["ddp", "fsdp"])
    return device_mesh

# In ActorRolloutRefWorker.__init__():
self.device_mesh = create_device_mesh(world_size, self.config.actor.fsdp_config.fsdp_size)
```

## Data Sharding & Dispatch Mechanism

### 1. Dispatch Decorator

Methods are decorated with dispatch modes that control data distribution:

```python
# fsdp_workers.py:958-960
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
def generate_sequences(self, prompts: DataProto):
    ...
```

### 2. Dispatch Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     RayPPOTrainer._training_step()                               │
│                                                                                  │
│   batch = DataProto.from_single_dict(batch_dict)   # Full batch on controller  │
│   gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)        │
│                                      │                                           │
└──────────────────────────────────────┼──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          RayWorkerGroup.generate_sequences()                     │
│                       (Bound method via _bind_worker_method)                     │
│                                                                                  │
│   1. DISPATCH PHASE (dispatch_fn):                                              │
│      ┌────────────────────────────────────────────────────────────────────────┐ │
│      │  # Split DataProto into chunks for each worker                         │ │
│      │  splitted_args = data_proto.chunk(chunks=world_size)                   │ │
│      │                                                                         │ │
│      │  Full batch: [sample_0, sample_1, ..., sample_N]                       │ │
│      │              ↓                                                          │ │
│      │  Worker 0: [sample_0, sample_4, sample_8, ...]                         │ │
│      │  Worker 1: [sample_1, sample_5, sample_9, ...]                         │ │
│      │  Worker 2: [sample_2, sample_6, sample_10, ...]                        │ │
│      │  Worker 3: [sample_3, sample_7, sample_11, ...]                        │ │
│      └────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│   2. EXECUTION PHASE:                                                           │
│      ┌────────────────────────────────────────────────────────────────────────┐ │
│      │  # Call worker methods in parallel via Ray                             │ │
│      │  futures = [worker.generate_sequences.remote(chunk)                    │ │
│      │             for worker, chunk in zip(workers, chunks)]                 │ │
│      │  results = ray.get(futures)                                            │ │
│      └────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│   3. COLLECT PHASE (collect_fn):                                                │
│      ┌────────────────────────────────────────────────────────────────────────┐ │
│      │  # Concatenate results from all workers                                │ │
│      │  output = DataProto.concat(results)                                    │ │
│      └────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│   return output  # Full batch back to controller                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3. Dispatch Mode Types

```python
# decorator.py:311-334
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL: {
        # Same data to all workers (e.g., init_model, update config)
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ALL_TO_ALL: {
        # Each worker gets different pre-split data
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.DP_COMPUTE_PROTO: {
        # Split DataProto by batch dimension (data parallelism)
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
}

# Custom dispatch for n-dimensional meshes
make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout")
# → Uses worker's registered dispatch info for that mesh
```

### 4. DataProto Chunking

```python
# decorator.py:71-84
def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    splitted_args = []
    for arg in args:
        assert isinstance(arg, DataProto | DataProtoFuture)
        splitted_args.append(arg.chunk(chunks=chunks))

    splitted_kwargs = {}
    for key, val in kwargs.items():
        assert isinstance(val, DataProto | DataProtoFuture)
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    return splitted_args, splitted_kwargs
```

## GPU Memory Sharing Between Roles

When using hybrid engine (`role="actor_rollout"`), the same GPU runs both training and inference:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Single GPU - Hybrid Engine                                │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    ActorRolloutRefWorker (role="actor_rollout")          │   │
│   │                                                                          │   │
│   │   ┌─────────────────────┐     ┌─────────────────────────────────────┐  │   │
│   │   │    Actor Model      │     │         SGLangRollout               │  │   │
│   │   │    (FSDP wrapped)   │     │                                     │  │   │
│   │   │                     │     │   ┌───────────────────────────────┐ │  │   │
│   │   │  • Training forward │     │   │   SGLang Engine               │ │  │   │
│   │   │  • Backward pass    │     │   │   • Inference forward          │ │  │   │
│   │   │  • Optimizer step   │     │   │   • KV cache management        │ │  │   │
│   │   │                     │     │   │   • Speculative decoding       │ │  │   │
│   │   └─────────────────────┘     │   └───────────────────────────────┘ │  │   │
│   │             ▲                 │                                     │  │   │
│   │             │                 │   ┌───────────────────────────────┐ │  │   │
│   │             │ Weight Sync    │   │   RolloutDrafterManager       │ │  │   │
│   │             └─────────────────│───│   • Background drafter train   │ │  │   │
│   │                               │   └───────────────────────────────┘ │  │   │
│   │                               └─────────────────────────────────────┘  │   │
│   │                                                                          │   │
│   │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│   │   │              FSDPUlyssesShardingManager                          │   │   │
│   │   │   • Manages weight sync between actor and rollout               │   │   │
│   │   │   • Handles memory offloading during role transitions           │   │   │
│   │   └─────────────────────────────────────────────────────────────────┘   │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Weight Synchronization Flow

```python
# fsdp_workers.py:976-980
def generate_sequences(self, prompts: DataProto):
    with self.rollout_sharding_manager:
        # 1. Sync weights from FSDP actor to SGLang engine
        # 2. Run inference
        output = self.rollout.generate_sequences(prompts=prompts)
    # 3. Optionally offload weights after inference
    return output
```

## Complete Data Flow Example

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Training Step Data Flow                             │
│                                                                                  │
│   1. DataLoader yields batch_dict                                               │
│                    │                                                             │
│                    ▼                                                             │
│   2. batch = DataProto.from_single_dict(batch_dict)                             │
│      [BS=1024 samples, on CPU]                                                  │
│                    │                                                             │
│                    ▼                                                             │
│   3. actor_rollout_wg.generate_sequences(batch)                                 │
│      ┌────────────────────────────────────────────────────────────────────────┐ │
│      │  Dispatch: chunk(8) → 128 samples per worker                          │ │
│      │                                                                         │ │
│      │  Worker 0 (GPU 0): 128 samples → generate → 128 outputs                │ │
│      │  Worker 1 (GPU 1): 128 samples → generate → 128 outputs                │ │
│      │  ...                                                                    │ │
│      │  Worker 7 (GPU 7): 128 samples → generate → 128 outputs                │ │
│      │                                                                         │ │
│      │  Collect: concat() → 1024 outputs                                      │ │
│      └────────────────────────────────────────────────────────────────────────┘ │
│                    │                                                             │
│                    ▼                                                             │
│   4. Compute rewards, advantages                                                │
│                    │                                                             │
│                    ▼                                                             │
│   5. actor_rollout_wg.update_actor(batch)                                       │
│      ┌────────────────────────────────────────────────────────────────────────┐ │
│      │  Dispatch: chunk(8) → 128 samples per worker                          │ │
│      │                                                                         │ │
│      │  Each worker:                                                           │ │
│      │  • Forward pass on local samples                                        │ │
│      │  • Backward pass (FSDP syncs gradients)                                │ │
│      │  • Optimizer step                                                       │ │
│      └────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `verl/trainer/main_fastrl.py` | Entry point, creates RayPPOTrainer |
| `verl/trainer/ppo/ray_trainer.py` | RayPPOTrainer, ResourcePoolManager |
| `verl/single_controller/ray/base.py` | RayResourcePool, RayWorkerGroup |
| `verl/single_controller/base/decorator.py` | Dispatch modes, @register decorator |
| `verl/workers/fsdp_workers.py` | ActorRolloutRefWorker, create_device_mesh |
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | SGLangRollout |

## Configuration Reference

```yaml
# Key config fields for GPU allocation
trainer:
  n_gpus_per_node: 8          # GPUs per node
  nnodes: 1                    # Number of nodes

actor_rollout_ref:
  actor:
    strategy: "fsdp2"          # FSDP or Megatron
    fsdp_config:
      fsdp_size: -1            # -1 = shard across all GPUs
      param_offload: false     # CPU offload for params
      optimizer_offload: false # CPU offload for optimizer

  rollout:
    name: "sglang"             # Inference engine
    mode: "sync"               # sync or async
```

## See Also

- [Drafter Training Activation](./drafter-training-activation.md) - GPU memory for drafter training
- [Architecture Overview](./architecture.md) - System-level view
- [Co-training Pipeline](./co-training-pipeline.md) - EAGLE training details
