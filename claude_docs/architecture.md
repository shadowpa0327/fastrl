# FastRL Architecture

## System Overview

FastRL is an RL training system that co-trains EAGLE draft models alongside the main target model. The key insight is that spare GPU resources during RL training can be harvested for continuous drafter alignment.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FastRL Training System                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Ray Distributed Coordinator                        │    │
│  │                  (verl/trainer/ppo/ray_trainer.py)                   │    │
│  └───────────────────────────┬───────────────────────────────────────────┘    │
│                              │                                               │
│          ┌───────────────────┼───────────────────┐                          │
│          │                   │                   │                          │
│          ▼                   ▼                   ▼                          │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                 │
│  │ Actor Worker  │   │ Critic Worker │   │ Reward Worker │                 │
│  │   (FSDP)      │   │   (FSDP)      │   │   (Model)     │                 │
│  └───────┬───────┘   └───────────────┘   └───────────────┘                 │
│          │                                                                   │
│          │ Weight Sync                                                       │
│          ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Rollout Engine (SGLang)                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │                  Speculative Decoding Pipeline                   │  │  │
│  │  │  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │  │  │
│  │  │  │   EAGLE     │─────▶│   Target    │─────▶│    MAB      │     │  │  │
│  │  │  │   Drafter   │      │   Verify    │      │  Strategy   │     │  │  │
│  │  │  └─────────────┘      └─────────────┘      └─────────────┘     │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│          │                                                                   │
│          │ Hidden States                                                     │
│          ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │              EAGLE Background Trainer (Co-training)                    │  │
│  │           (verl/workers/drafter/eagle_background_trainer.py)           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │ Data Buffer │─▶│  Forward    │─▶│  Backward   │─▶│ Checkpoint  │  │  │
│  │  │  (Deque)    │  │  (FSDP2)    │  │  + Optim    │  │   (Async)   │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Interaction Diagram

```
                                    ┌──────────────┐
                                    │   Hydra      │
                                    │   Config     │
                                    └──────┬───────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              RayPPOTrainer                                    │
│                         (verl/trainer/ppo/ray_trainer.py)                    │
│                                                                               │
│   init_workers()              fit()                    _training_step()       │
│   ┌────────────┐      ┌────────────────────┐      ┌────────────────────┐     │
│   │ Create Ray │      │   Main RL Loop     │      │ Per-Step Actions   │     │
│   │ Actors for │─────▶│   ┌──────────────┐ │─────▶│ • generate_rollout │     │
│   │ all workers│      │   │ fit() entry  │ │      │ • compute_rewards  │     │
│   └────────────┘      │   └──────┬───────┘ │      │ • update_policy    │     │
│                       │          │         │      │ • update_critic    │     │
│                       │          ▼         │      │ • train_drafter    │     │
│                       │   for step in      │      └────────────────────┘     │
│                       │   range(total):    │                                  │
│                       │     _training_step │                                  │
│                       └────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                    │                    │                    │
                    │                    │                    │
          ┌─────────▼────────┐ ┌─────────▼────────┐ ┌─────────▼────────┐
          │  Actor/Rollout   │ │    Critic        │ │    Reward        │
          │  Ref Worker      │ │    Worker        │ │    Worker        │
          └─────────┬────────┘ └──────────────────┘ └──────────────────┘
                    │
                    │ uses
                    ▼
          ┌─────────────────────────────────────────────────────────────┐
          │                    SGLang Rollout                           │
          │              (verl/workers/rollout/sglang_rollout/)         │
          │                                                              │
          │  ┌──────────────────────────────────────────────────────┐  │
          │  │              EAGLEWorker                              │  │
          │  │       (third-party/sglang/.../eagle_worker.py)       │  │
          │  │                                                       │  │
          │  │   forward_draft() ──▶ forward_verify() ──▶ accept()  │  │
          │  └──────────────────────────────────────────────────────┘  │
          │                          │                                  │
          │                          │ metrics                         │
          │                          ▼                                  │
          │  ┌──────────────────────────────────────────────────────┐  │
          │  │              MABGroupManager                          │  │
          │  │       (third-party/sglang/.../eagle_mab.py)          │  │
          │  │                                                       │  │
          │  │   select_strategy() ◀── update() ◀── acceptance_len  │  │
          │  └──────────────────────────────────────────────────────┘  │
          └─────────────────────────────────────────────────────────────┘
```

## Data Flow: Rollout Generation

```
Prompts (from dataset)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Speculative Decoding Pipeline                         │
│                                                                              │
│   Step 1: Draft                    Step 2: Verify                           │
│   ┌─────────────────────┐         ┌─────────────────────┐                   │
│   │   EAGLE Drafter     │         │   Target Model      │                   │
│   │                     │         │                     │                   │
│   │ Input: last hidden  │    ┌───▶│ Input: draft tree   │                   │
│   │ Output: draft tree  │────┘    │ Output: verified    │                   │
│   │   (spec_steps=8,    │         │   tokens + probs    │                   │
│   │    spec_topk=4)     │         │                     │                   │
│   └─────────────────────┘         └──────────┬──────────┘                   │
│                                              │                               │
│                                              ▼                               │
│                                   ┌─────────────────────┐                   │
│   Step 3: MAB Update              │  Acceptance Check   │                   │
│   ┌─────────────────────┐         │                     │                   │
│   │ Track accept_length │◀────────│ Compare draft vs    │                   │
│   │ Update strategy     │         │ verified tokens     │                   │
│   │ Select next config  │         └─────────────────────┘                   │
│   └─────────────────────┘                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Output Tensors                                    │
│                                                                              │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌──────────────┐ │
│  │   input_ids    │ │  log_probs     │ │ hidden_states  │ │  loss_mask   │ │
│  │   (B, L)       │ │    (B, L)      │ │  (B, L, D)     │ │    (B, L)    │ │
│  └────────────────┘ └────────────────┘ └────────────────┘ └──────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow: Drafter Co-training

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Drafter Co-training Pipeline                          │
│                     (verl/workers/drafter/eagle_background_trainer.py)       │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                     Hidden State Collection                           │  │
│   │                        (from SGLang)                                  │  │
│   │                                                                       │  │
│   │           ┌─────────────────────┐                                    │  │
│   │           │ Collect during      │                                    │  │
│   │           │ speculative decoding│                                    │  │
│   │           │ verification step   │                                    │  │
│   │           │ (efficient: no      │                                    │  │
│   │           │  extra forward)     │                                    │  │
│   │           └────────┬────────────┘                                    │  │
│   │                    ▼                                                  │  │
│   │           ┌─────────────────────┐                                    │  │
│   │           │    Data Buffer      │                                    │  │
│   │           │   (deque, max=2000) │                                    │  │
│   │           │                     │                                    │  │
│   │           │ {input_ids,         │                                    │  │
│   │           │  hidden_states,     │                                    │  │
│   │           │  loss_mask}         │                                    │  │
│   │           └──────────┬──────────┘                                    │  │
│   └──────────────────────┼───────────────────────────────────────────────┘  │
│                          │                                                   │
│   ┌──────────────────────┼───────────────────────────────────────────────┐  │
│   │                      │  Training Loop                                 │  │
│   │                      ▼                                                │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│   │  │ for epoch in range(max_epochs):  # default: 10                  │ │  │
│   │  │   for batch in data_buffer.sample(batch_size):                  │ │  │
│   │  │                                                                  │ │  │
│   │  │     # Prepare inputs (shift by 1)                               │ │  │
│   │  │     input_hidden  = hidden_states[:, :-1, :]                    │ │  │
│   │  │     target_hidden = hidden_states[:, 1:, :]                     │ │  │
│   │  │                                                                  │ │  │
│   │  │     # Forward through EAGLE drafter                             │ │  │
│   │  │     pred_hidden = drafter(input_hidden, input_ids)              │ │  │
│   │  │                                                                  │ │  │
│   │  │     # Compute loss                                              │ │  │
│   │  │     loss = SmoothL1Loss(pred_hidden, target_hidden)             │ │  │
│   │  │                                                                  │ │  │
│   │  │     # Backward + optimize (FSDP2)                               │ │  │
│   │  │     loss.backward()                                             │ │  │
│   │  │     optimizer.step()                                            │ │  │
│   │  └─────────────────────────────────────────────────────────────────┘ │  │
│   │                          │                                            │  │
│   │                          ▼                                            │  │
│   │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│   │  │ Checkpoint & Weight Sync                                        │ │  │
│   │  │                                                                  │ │  │
│   │  │  • Async DCP save (non-blocking)                                │ │  │
│   │  │  • Extract weights: get_drafter_weights()                       │ │  │
│   │  │  • Send to SGLang: UpdateWeightsFromTensorReqInput              │ │  │
│   │  └─────────────────────────────────────────────────────────────────┘ │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## FSDP Worker Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FSDP Worker Hierarchy                                 │
│                        (verl/workers/fsdp_workers.py)                        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    ActorRolloutRefWorker                                │ │
│  │                                                                         │ │
│  │  Combines: Actor + Rollout + Reference Policy in single worker         │ │
│  │                                                                         │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │ │
│  │  │  Actor Model    │  │ Rollout Engine  │  │   Ref Policy    │        │ │
│  │  │  (trainable)    │  │   (SGLang)      │  │   (frozen)      │        │ │
│  │  │                 │  │                 │  │                 │        │ │
│  │  │ • Policy head   │  │ • Spec decode   │  │ • KL reference  │        │ │
│  │  │ • FSDP shard    │  │ • Batch gen     │  │ • Log probs     │        │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │ │
│  │                                                                         │ │
│  │  Key Methods:                                                           │ │
│  │  • generate_sequences() -> rollout with SD                              │ │
│  │  • compute_log_probs()  -> for PPO loss                                │ │
│  │  • update_policy()      -> gradient step                                │ │
│  │  • train_drafter()      -> trigger co-training                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐  │
│  │        CriticWorker             │  │       RewardWorker              │  │
│  │                                 │  │                                 │  │
│  │  • Value function estimation    │  │  • Compute rewards              │  │
│  │  • FSDP sharded                 │  │  • Rule-based + Model-based     │  │
│  │  • GAE advantage computation    │  │  • Sandbox execution            │  │
│  └─────────────────────────────────┘  └─────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
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

## Training Timeline

```
Time ─────────────────────────────────────────────────────────────────────────▶

RL Step 0        RL Step 10       RL Step 20       RL Step 30
    │                │                │                │
    ▼                ▼                ▼                ▼
┌───────┐        ┌───────┐        ┌───────┐        ┌───────┐
│Rollout│        │Rollout│        │Rollout│        │Rollout│
│  +SD  │        │  +SD  │        │  +SD  │        │  +SD  │
└───┬───┘        └───┬───┘        └───┬───┘        └───┬───┘
    │                │                │                │
    │ collect        │ collect        │ collect        │ collect
    │ hidden         │ hidden         │ hidden         │ hidden
    │ states         │ states         │ states         │ states
    ▼                ▼                ▼                ▼
┌───────┐        ┌───────┐        ┌───────┐        ┌───────┐
│ Data  │───────▶│ Data  │───────▶│ Data  │───────▶│ Data  │
│Buffer │        │Buffer │        │Buffer │        │Buffer │
└───────┘        └───┬───┘        └───┬───┘        └───┬───┘
                     │                │                │
                     │                │                │
                     ▼                ▼                ▼
              ┌───────────┐    ┌───────────┐    ┌───────────┐
              │  Drafter  │    │  Drafter  │    │  Drafter  │
              │  Training │    │  Training │    │  Training │
              │ (10 epochs│    │ (10 epochs│    │ (10 epochs│
              │  async)   │    │  async)   │    │  async)   │
              └───────────┘    └───────────┘    └───────────┘
                     │                │                │
                     ▼                ▼                ▼
              ┌───────────┐    ┌───────────┐    ┌───────────┐
              │  Weight   │    │  Weight   │    │  Weight   │
              │   Sync    │    │   Sync    │    │   Sync    │
              │to SGLang  │    │to SGLang  │    │to SGLang  │
              └───────────┘    └───────────┘    └───────────┘

Legend:
  Rollout+SD = Generate sequences with speculative decoding
  Data Buffer = Accumulate hidden states for drafter training
  Drafter Training = Background FSDP2 training on collected data
  Weight Sync = Update SGLang's drafter weights with newly trained weights
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
