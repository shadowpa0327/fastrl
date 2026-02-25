# FastRL Architecture

## System Overview

FastRL is an RL training system that co-trains EAGLE draft models alongside the main target model. The key insight is that spare GPU resources during RL training can be harvested for continuous drafter alignment.

## Unified Overview (8 GPUs, TP=1)

This diagram shows the complete system end-to-end: how the RL loop, speculative decoding,
hidden state collection, drafter co-training, and weight synchronization all fit together.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  RayPPOTrainer          RL Loop: Rollout → Reward → Actor Upd → Critic Upd → ↺ │
│  batch.chunk(8) ──┬──                                                           │
└───────────────────┼─────────────────────────────────────────────────────────────┘
   GPU 0    GPU 1   │      GPU 7           Per-Worker Lifecycle
  ┌─────┐ ┌─────┐  │    ┌─────┐          ┌─────────────────────────────────────┐
  │SGLng│ │SGLng│  │    │SGLng│          │ PHASE 1: Rollout                    │
  │Actor│ │Actor│◀─┘    │Actor│          │                                     │
  │Draft│ │Draft│  ...  │Draft│          │ wake_up() ──▶ sync weights ──▶ SGLang│
  └─────┘ └─────┘       └─────┘          │                                     │
     │  FSDP sync   │                    │ EAGLE Draft ─▶ Verify ─▶ MAB Select │
     └──────────────┘                    │       └─▶ hidden states ─▶ buffer   │
                                          └──────────────────┬──────────────────┘
                                                             │
  Workers finish at different times (long-tail):              ▼
  ┌──────────────────────────────────┐   ┌─────────────────────────────────────┐
  │ GPU 0: ██████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │   │ PHASE 2: "Bubble" Co-training      │
  │ GPU 1: ██████████████░░░░░░░░░░ │   │                                     │
  │ GPU 2: ██████████████████░░░░░░ │   │ 1st worker done → "release"         │
  │ GPU 7: ████████████████████████ │   │   → CentralCoordinator ZMQ START    │
  │                              ▲  │   │   → activate (CPU→GPU) → train      │
  │  █ rollout                   │  │   │ only min_workers_for_training (=1)  │
  │  ▓ drafter train (1st done)  │  │   │   workers train; rest idle (░)      │
  │  ░ idle (not selected)  done─┘  │   │ all done → ZMQ STOP → cleanup      │
  └──────────────────────────────────┘   └──────────────────┬──────────────────┘
                                                             │
                                                             ▼
                                          ┌─────────────────────────────────────┐
                                          │ PHASE 3: RL Update                  │
                                          │                                     │
                                          │ FSDP all-reduce ──▶ optimizer.step  │
                                          └──────────────────┬──────────────────┘
                                                             │
                                          ┌──────────────────┘
                                          ▼
                                   Back to PHASE 1
                             (wake_up syncs trained drafter)
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
│  │  │    Gradients (transient)    │  │   Drafter Training (background) │ │ │
│  │  │                             │  │   Uses spare compute cycles     │ │ │
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
