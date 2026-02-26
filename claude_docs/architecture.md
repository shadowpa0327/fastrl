# FastRL Architecture

## System Overview

FastRL is an RL training system that co-trains EAGLE draft models alongside the main target model. The key insight is that spare GPU resources during RL training can be harvested for continuous drafter alignment.

## Unified Overview (8 GPUs, TP=4)

This diagram shows the complete system end-to-end: how the RL loop, speculative decoding,
hidden state collection, foreground drafter co-training, and weight synchronization all fit together.

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  RayPPOTrainer                                                                      │
│  RL Loop: Rollout → Train Drafter → Reward → Actor Upd → Critic Upd → ↺            │
└──────────────────────────────┬──────────────────────────────────────────────────────┘
                               │
   DP Group 0            DP Group 1          Per-Worker Lifecycle
  GPU [0,1,2,3]         GPU [4,5,6,7]       ┌────────────────────────────────────┐
  ┌──────────┐          ┌──────────┐        │ PHASE 1: Rollout                   │
  │ SGLang   │          │ SGLang   │        │                                    │
  │ TP=4     │          │ TP=4     │        │ wake_up() → sync weights → SGLang  │
  │ +Drafter │          │ +Drafter │        │ EAGLE Draft → Verify → MAB Select  │
  │ (FSDP2)  │          │ (FSDP2)  │        │     └─▶ hidden states → buffer     │
  └──────────┘          └──────────┘        └──────────────────┬─────────────────┘
       │                     │                                  │
       │   Independent       │                                  ▼
       │   (no cross-group   │              ┌────────────────────────────────────┐
       │    communication)   │              │ PHASE 2: Foreground Drafter Train  │
       │                     │              │                                    │
       │                     │              │ train_drafter() dispatched to ALL   │
       │                     │              │   → activate (CPU→GPU)             │
       │                     │              │   → N training steps (FSDP2)       │
       │                     │              │   → cleanup (GPU→CPU)              │
       │                     │              └──────────────────┬─────────────────┘
       │                     │                                  │
       └─────────────────────┘                                  ▼
                                            ┌────────────────────────────────────┐
                                            │ PHASE 3: RL Update                 │
                                            │                                    │
                                            │ FSDP all-reduce → optimizer.step   │
                                            └──────────────────┬─────────────────┘
                                                               │
                                            ┌──────────────────┘
                                            ▼
                                     Back to PHASE 1
                               (wake_up syncs trained drafter)
```

### FSDP2 Training Within a DP Group (TP=4)

During Phase 2, each DP group trains its drafter independently using FSDP2:

```
DP Group 0: GPU [0, 1, 2, 3]
  Each GPU holds 1/4 of drafter params (FSDP2 sharded)
  All GPUs have identical data buffer (from broadcast)
  Each GPU samples its own mini-batch (data parallelism)

  Forward:  all-gather params → each GPU computes on its batch
  Backward: reduce-scatter gradients (averaged across 4 GPUs)
  Optimize: each GPU updates its local 1/4 shard
```

## Multi-Armed Bandit Strategy Selection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Adaptive Speculative Decoding (MAB)                      │
│                  (third-party/sglang/.../eagle_mab.py)                       │
│                                                                              │
│   Batch Size Groups            Strategy Space                                │
│   ┌─────────────────┐         ┌────────────────────────────────────────┐   │
│   │ Group 0: bs=1   │         │  Config: "{spec_steps}_{topk}_{draft}" │   │
│   │ Group 1: bs=2-4 │◀───────▶│                                        │   │
│   │ Group 2: bs=5-20│         │  • "8_4_32" = 8 steps, top-4, 32 draft │   │
│   │ Group 3: bs≥21  │         │  • "8_4_16" = 8 steps, top-4, 16 draft │   │
│   └─────────────────┘         │  • "8_4_8"  = 8 steps, top-4, 8 draft  │   │
│          │                    │  • "disable"= no speculative decoding  │   │
│          │                    └────────────────────────────────────────┘   │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    MABGroupManager                                   │   │
│   │                                                                      │   │
│   │  For each batch_size_group:                                         │   │
│   │    ┌────────────────────────────────────────────────────────────┐   │   │
│   │    │              EpsilonGreedyMAB / UCB1MAB                     │   │   │
│   │    │                                                             │   │   │
│   │    │  state = {                                                  │   │   │
│   │    │    strategy_rewards: Dict[str, List[float]],  # sliding win │   │   │
│   │    │    strategy_counts:  Dict[str, int],                        │   │   │
│   │    │    total_pulls:      int,                                   │   │   │
│   │    │  }                                                          │   │   │
│   │    │                                                             │   │   │
│   │    │  select_strategy():                                         │   │   │
│   │    │    if random() < epsilon:                                   │   │   │
│   │    │      return random_strategy()  # explore                    │   │   │
│   │    │    else:                                                    │   │   │
│   │    │      return argmax(avg_reward)  # exploit                   │   │   │
│   │    │                                                             │   │   │
│   │    │  update(strategy, acceptance_length):                       │   │   │
│   │    │    strategy_rewards[strategy].append(acceptance_length)     │   │   │
│   │    │    # Maintain sliding window of 1000 samples                │   │   │
│   │    └────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GPU Memory Layout (per GPU)                         │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     SGLang Inference Engine                             │ │
│  │                  gpu_memory_utilization: 0.4                           │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────┐  ┌─────────────────────────────────┐ │ │
│  │  │     Target Model Weights    │  │      KV Cache                   │ │ │
│  │  │     (tensor parallel)       │  │   (dynamic allocation)          │ │ │
│  │  └─────────────────────────────┘  └─────────────────────────────────┘ │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────┐                                       │ │
│  │  │    EAGLE Drafter Weights    │                                       │ │
│  │  │    (small, ~100M params)    │                                       │ │
│  │  └─────────────────────────────┘                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        FSDP Training                                    │ │
│  │                  Remaining GPU memory (~60%)                           │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────┐  ┌─────────────────────────────────┐ │ │
│  │  │   Actor Model (sharded)     │  │    Optimizer States             │ │ │
│  │  │   FSDP2 + activation ckpt   │  │    (Adam, momentum, etc.)       │ │ │
│  │  └─────────────────────────────┘  └─────────────────────────────────┘ │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────┐  ┌─────────────────────────────────┐ │ │
│  │  │    Gradients (transient)    │  │   Drafter Training (foreground) │ │ │
│  │  │                             │  │   Runs between rollout & update │ │ │
│  │  └─────────────────────────────┘  └─────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## See Also

- [Worker Hierarchy & GPU Allocation](./worker-hierarchy-gpu-allocation.md) - Ownership chain, Ray resource pools, data dispatch
- [Co-training Pipeline Details](./co-training-pipeline.md)
- [Drafter Training Activation](./drafter-training-activation.md) - GPU memory management, activation call chain
- [Speculative Decoding Deep Dive](./speculative-decoding.md)
- [Key Files Reference](./key-files.md)
