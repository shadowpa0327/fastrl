# FastRL - EAGLE Co-training for Efficient RL

## Project Overview

FastRL is a **high-efficiency RL training framework** that implements novel **EAGLE draft model co-training**. The system eliminates long-tail rollout bottlenecks through adaptive speculative decoding and opportunistic drafter training.

**Paper**: [Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter](https://arxiv.org/abs/2511.16665) (ASPLOS 2026)

## Core Innovation

The key novelty is **co-training the EAGLE drafter alongside the target model during RL training**:

```
Traditional Approach:
  Train Drafter → Freeze → RL Training → Drafter becomes stale

FastRL Approach:
  RL Training ←──┬──→ Drafter continuously updated
                 │
                 └── Uses spare GPU resources (free!)
```

This maintains high acceptance rates throughout RL training without additional compute cost.

## Architecture Quick View

```
┌─────────────────────────────────────────────────────────────────┐
│                      RayPPOTrainer                               │
│                 (verl/trainer/ppo/ray_trainer.py)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  ┌──────────┐       ┌──────────┐       ┌──────────┐
  │  Actor   │       │  Critic  │       │  Reward  │
  │  Worker  │       │  Worker  │       │  Worker  │
  └────┬─────┘       └──────────┘       └──────────┘
       │
       │ Weight Sync
       ▼
  ┌─────────────────────────────────────────────────────────────┐
  │               SGLang + Speculative Decoding                  │
  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
  │  │ EAGLE Draft  │───▶│ Target Verify│───▶│ MAB Strategy │  │
  │  └──────────────┘    └──────────────┘    └──────────────┘  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ Hidden States
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │          EAGLE Background Trainer (Co-training)             │
  │              (verl/workers/drafter/eagle_background_trainer.py)                │
  └─────────────────────────────────────────────────────────────┘
```

## Three Novel Components

### 1. Lossless Speculative Decoding
- Mathematically preserves exact rollout distribution
- Critical for RL correctness (cannot alter policy gradient)
- **File**: `third-party/sglang/.../eagle_worker.py`

### 2. Opportunistic Drafter Training
- Trains EAGLE drafter in background during RL
- Collects hidden states during rollouts
- Updates drafter every N RL steps (default: 10)
- **File**: `verl/workers/drafter/eagle_background_trainer.py`

### 3. Multi-Armed Bandit (MAB) Adaptive SD
- Learns optimal SD configuration per batch-size group
- Dynamically enables/disables speculation
- **File**: `third-party/sglang/.../eagle_mab.py`

## Project Structure

```
fastrl/
├── verl/                              # Core RL framework
│   ├── trainer/
│   │   ├── main_fastrl.py            # Entry point
│   │   ├── ppo/ray_trainer.py        # Main orchestrator (1423 lines)
│   │   └── config/fastrl_trainer.yaml # Configuration
│   └── workers/
│       ├── fsdp_workers.py           # FSDP workers
│       ├── drafter/                  # ★ EAGLE co-training
│       │   └── eagle_background_trainer.py
│       └── rollout/sglang_rollout/   # Inference
│
├── eagle-train/                      # Standalone drafter training
├── third-party/sglang/              # Enhanced SGLang with SD
│   └── .../speculative/
│       ├── eagle_worker.py          # SD executor
│       └── eagle_mab.py             # ★ MAB strategy
│
└── examples/                         # Training scripts
    ├── grpo_7B.sh
    └── grpo_32B_multi_nodes.sh
```

## Key Files to Hack

| Goal | Primary File |
|------|--------------|
| Modify RL algorithm | `verl/trainer/ppo/core_algos.py` |
| Change drafter training | `verl/workers/drafter/eagle_background_trainer.py` |
| Adjust SD strategy | `third-party/sglang/.../eagle_mab.py` |
| Tune hyperparameters | `verl/trainer/config/fastrl_trainer.yaml` |
| Add model architecture | `verl/workers/drafter/model/qwen2_eagle.py` |

## Configuration Quick Reference

```yaml
# Enable/disable co-training
speculative.train.enable_drafter_training: true

# Training frequency
speculative.train.training_interval_steps: 10  # Every N RL steps

# SD parameters
speculative.eagle.spec_steps: 8     # Draft depth
speculative.eagle.spec_topk: 4      # Candidates per level

# MAB configs
speculative.eagle.mab_configs: ["8_4_32", "8_4_16", "8_4_8"]
```

## Common Commands

```bash
# Single-node training (7B model)
bash examples/grpo_7B.sh

# Multi-node training (32B model)
sbatch examples/grpo_32B_multi_nodes.sh

# Benchmark speculative decoding
bash examples/bench_sd.sh

# Standalone EAGLE training (warm-up)
python eagle-train/eagle_trainer.py --base_model_path <path>
```

## Detailed Documentation

See `claude_docs/` for comprehensive documentation:

- **[Architecture](./claude_docs/architecture.md)**: System diagrams, component interactions, data flow
- **[Co-training Pipeline](./claude_docs/co-training-pipeline.md)**: EAGLE background training details
- **[Drafter Training Activation](./claude_docs/drafter-training-activation.md)**: GPU memory management, bubble resource usage, activation call chain
- **[Hidden States Collection](./claude_docs/hidden-states-collection.md)**: How hidden states flow from rollouts to drafter training
- **[Speculative Decoding](./claude_docs/speculative-decoding.md)**: SD algorithm, MAB strategy selection
- **[Key Files Reference](./claude_docs/key-files.md)**: File-by-file guide for hacking

## Development Notes

### Code Style
- Uses Hydra for configuration management
- Ray for distributed training
- FSDP2 for model sharding
- SGLang for inference engine

### Testing
- Benchmark SD speedup with `examples/bench_sd.sh`
- Verify lossless property by comparing distributions with/without SD

### Debugging
- Check SD enabled: Look for "Speculative decoding enabled" in logs
- Monitor acceptance rate: `drafter/acceptance_rate` metric
- MAB behavior: Debug logs in `eagle_mab.py`

## Dependencies

- Python 3.12+
- PyTorch 2.x with FSDP2
- Ray for distributed coordination
- SGLang (included in `third-party/`)
- flash-attention 2.8+
