# Speculative Decoding in FastRL

This document covers the speculative decoding implementation, including the EAGLE drafter and the novel Multi-Armed Bandit (MAB) adaptive strategy selection.

## Why Speculative Decoding for RL?

```
Problem: Long-tail rollouts in reasoning RL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rollout Length Distribution:
│
│    ▓▓                              Long-tail: Few samples
│    ▓▓▓▓                            take 10-100x longer
│    ▓▓▓▓▓▓                          than median
│    ▓▓▓▓▓▓▓▓
│    ▓▓▓▓▓▓▓▓▓▓▓▓                              ░░░
│    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░
└────────────────────────────────────────────────────────────▶
                       Sequence Length

Impact:
- Batch completes when LONGEST sequence finishes
- GPUs idle while waiting for stragglers
- Significant throughput loss (30-50%)

Solution: Speculative Decoding
- Draft multiple tokens at once
- Verify in parallel
- Accelerates long sequences most (exactly where needed!)
```

## Speculative Decoding Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EAGLE Speculative Decoding Flow                          │
│                   (third-party/sglang/.../eagle_worker.py)                   │
│                                                                              │
│   Input: Prompt tokens                                                       │
│          │                                                                   │
│          ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Step 1: Get Initial Hidden State                                    │   │
│   │                                                                      │   │
│   │  hidden = target_model.get_last_hidden_state(prompt)                │   │
│   │  # Shape: (batch, 1, hidden_dim)                                     │   │
│   └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Step 2: EAGLE Draft Loop (spec_steps iterations)                   │   │
│   │                                                                      │   │
│   │  for i in range(spec_steps):  # default: 8                          │   │
│   │      # Draft next tokens                                             │   │
│   │      logits = drafter(hidden, prev_tokens)                          │   │
│   │                                                                      │   │
│   │      # Sample top-k candidates                                       │   │
│   │      candidates = top_k_sample(logits, k=spec_topk)  # default: 4   │   │
│   │                                                                      │   │
│   │      # Build draft tree (tree-based speculation)                     │   │
│   │      draft_tree.add_level(candidates)                                │   │
│   │                                                                      │   │
│   │      # Update hidden state for next iteration                        │   │
│   │      hidden = drafter.get_hidden(candidates)                        │   │
│   └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Step 3: Target Model Verification                                   │   │
│   │                                                                      │   │
│   │  # Verify all draft tokens in ONE forward pass                       │   │
│   │  verify_logits = target_model(draft_tree.all_tokens)                │   │
│   │                                                                      │   │
│   │  # Check which drafts match target distribution                      │   │
│   │  for path in draft_tree.paths:                                       │   │
│   │      accepted = verify_tokens(                                       │   │
│   │          draft_probs=drafter_probs[path],                           │   │
│   │          target_probs=target_probs[path],                           │   │
│   │      )                                                               │   │
│   │      if len(accepted) > best_path:                                   │   │
│   │          best_path = path                                            │   │
│   └──────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Step 4: Accept Verified Tokens                                      │   │
│   │                                                                      │   │
│   │  # Lossless acceptance (preserves exact distribution)                │   │
│   │  accepted_tokens = draft_tree[best_path][:num_accepted]             │   │
│   │  output_tokens.extend(accepted_tokens)                               │   │
│   │                                                                      │   │
│   │  # Sample correction token if needed                                 │   │
│   │  if rejected_at_position:                                            │   │
│   │      correction = sample_from_target(verify_logits[reject_pos])     │   │
│   │      output_tokens.append(correction)                                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Output: Verified tokens (mathematically equivalent to autoregressive)      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tree-Based Speculation

```
Draft Tree Structure (spec_steps=3, spec_topk=2):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    [root]
                   /      \
                  A        B          Level 1: top-2 candidates
                /   \    /   \
               C     D  E     F       Level 2: top-2 from each
              /|\   /|  |\   /|\
             G H I J K L M N O P      Level 3: top-2 from each

Total draft tokens: 2 + 4 + 8 = 14 tokens
Verified in single forward pass!

Paths to verify: [A→C→G, A→C→H, A→C→I, A→D→J, A→D→K, ...]
Best path selected based on acceptance length
```

## Lossless Verification Algorithm

```python
def verify_tokens(draft_probs, target_probs, draft_tokens):
    """
    Lossless speculative decoding verification.

    Key Property: Output distribution is IDENTICAL to autoregressive sampling.
    This is crucial for RL - we cannot alter the rollout distribution!
    """
    accepted = []

    for i, (d_prob, t_prob, token) in enumerate(
        zip(draft_probs, target_probs, draft_tokens)
    ):
        # Acceptance probability
        accept_prob = min(1.0, t_prob[token] / d_prob[token])

        if random.random() < accept_prob:
            accepted.append(token)
        else:
            # Rejection: sample from residual distribution
            # residual = max(0, target - draft) / Z
            residual = torch.clamp(t_prob - d_prob, min=0)
            residual = residual / residual.sum()
            correction = torch.multinomial(residual, 1)
            accepted.append(correction)
            break  # Stop at first rejection

    return accepted
```

## Multi-Armed Bandit (MAB) Adaptive Selection

**Location**: `third-party/sglang/python/sglang/srt/speculative/eagle_mab.py`

### Problem Statement

```
Different batch sizes benefit from different SD configurations:

Batch Size 1-2:   Aggressive speculation beneficial (high latency tolerance)
Batch Size 5-20:  Moderate speculation
Batch Size 20+:   Conservative or disabled (memory/compute trade-off)

Solution: Learn optimal strategy per batch-size group using MAB
```

### Strategy Space

```python
# Configuration format: "{spec_steps}_{spec_topk}_{draft_tokens}"
STRATEGY_SPACE = [
    "8_4_32",   # Aggressive: 8 steps, top-4, 32 draft tokens
    "8_4_16",   # Moderate: 8 steps, top-4, 16 draft tokens
    "8_4_8",    # Conservative: 8 steps, top-4, 8 draft tokens
    "disable",  # No speculation
]

# Batch size thresholds for grouping
BS_THRESHOLDS = [1, 2, 5, 21]  # Groups: [1], [2-4], [5-20], [21+]
```

### MAB Algorithm

```python
class EpsilonGreedyMAB:
    """
    Epsilon-greedy multi-armed bandit for SD strategy selection.

    Reward signal: Acceptance length (higher = better)
    """

    def __init__(self, strategies, epsilon=0.1, window_size=1000):
        self.strategies = strategies
        self.epsilon = epsilon
        self.window_size = window_size

        # Track rewards per strategy (sliding window)
        self.rewards = {s: deque(maxlen=window_size) for s in strategies}
        self.counts = {s: 0 for s in strategies}

    def select_strategy(self):
        """Select strategy using epsilon-greedy."""
        if random.random() < self.epsilon:
            # Explore: random strategy
            return random.choice(self.strategies)
        else:
            # Exploit: best average reward
            avg_rewards = {
                s: sum(self.rewards[s]) / max(len(self.rewards[s]), 1)
                for s in self.strategies
            }
            return max(avg_rewards, key=avg_rewards.get)

    def update(self, strategy, reward):
        """Update with observed reward (acceptance length)."""
        self.rewards[strategy].append(reward)
        self.counts[strategy] += 1
```

### MAB Group Manager

```python
class MABGroupManager:
    """
    Manages separate MAB instances for different batch-size groups.
    """

    def __init__(self, bs_thresholds, strategies):
        self.thresholds = bs_thresholds  # [1, 2, 5, 21]
        self.mabs = {}

        # Create MAB for each batch-size group
        for i in range(len(bs_thresholds)):
            group_id = f"bs_group_{i}"
            self.mabs[group_id] = EpsilonGreedyMAB(strategies)

    def get_group(self, batch_size):
        """Map batch size to group ID."""
        for i, threshold in enumerate(self.thresholds):
            if batch_size < threshold:
                return f"bs_group_{i-1}" if i > 0 else f"bs_group_0"
        return f"bs_group_{len(self.thresholds)-1}"

    def select_strategy(self, batch_size):
        """Select strategy for given batch size."""
        group = self.get_group(batch_size)
        return self.mabs[group].select_strategy()

    def update(self, batch_size, strategy, acceptance_length):
        """Update MAB with observed acceptance length."""
        group = self.get_group(batch_size)
        self.mabs[group].update(strategy, acceptance_length)
```

### Integration with EAGLE Worker

```python
# In eagle_worker.py
class EAGLEWorker:
    def __init__(self, config):
        self.mab_manager = MABGroupManager(
            bs_thresholds=config.mab_bs_threshold,
            strategies=config.mab_configs,
        )

    def generate(self, batch):
        batch_size = len(batch.prompts)

        # Select SD strategy via MAB
        strategy = self.mab_manager.select_strategy(batch_size)

        if strategy == "disable":
            # Fallback to autoregressive
            return self.autoregressive_generate(batch)

        # Parse strategy
        spec_steps, spec_topk, draft_tokens = self._parse_strategy(strategy)

        # Generate with selected SD config
        outputs, acceptance_lengths = self.speculative_generate(
            batch,
            spec_steps=spec_steps,
            spec_topk=spec_topk,
            max_draft_tokens=draft_tokens,
        )

        # Update MAB with observed acceptance
        avg_acceptance = sum(acceptance_lengths) / len(acceptance_lengths)
        self.mab_manager.update(batch_size, strategy, avg_acceptance)

        return outputs
```

## Configuration Parameters

```yaml
# In fastrl_trainer.yaml
speculative:
  enable: true
  spec_strategy: "eagle"

  # Batch size threshold for enabling SD
  bs_threshold: 32  # Only enable SD when batch_size >= 32

  eagle:
    spec_model_path: null          # Path to EAGLE drafter checkpoint
    spec_steps: 8                  # Max speculation depth
    spec_topk: 4                   # Top-k candidates per level
    spec_verify_tokens: 48         # Max tokens to verify per iteration

    # MAB configuration
    tune_algorithm: "BEG"          # Algorithm: BEG (Batch-size-Explicit Gaussian)
    mab_configs:                   # Strategy space
      - "8_4_32"
      - "8_4_16"
      - "8_4_8"
    mab_bs_threshold:              # Batch size group boundaries
      - 1
      - 2
      - 5
      - 21
```

## Performance Characteristics

```
Speedup Analysis:
━━━━━━━━━━━━━━━━━

Without SD (autoregressive):
  Time = N_tokens × T_forward

With SD (EAGLE):
  Time ≈ (N_tokens / acceptance_length) × (T_draft + T_verify)

Where:
  T_draft << T_forward (drafter is much smaller)
  T_verify ≈ T_forward (but verifies multiple tokens)
  acceptance_length typically 3-8 tokens

Expected Speedup:
  - Best case (high acceptance): 3-5x
  - Typical case: 1.5-2.5x
  - Long sequences benefit most (tail acceleration)
```

## Debugging Tips

### Check SD is Enabled

```python
# In logs, look for:
logger.info(f"Speculative decoding enabled: {config.speculative.enable}")
logger.info(f"Using strategy: {current_strategy}")
logger.info(f"Acceptance length: {avg_acceptance:.2f}")
```

### Monitor MAB Behavior

```python
# Add logging in eagle_mab.py
def select_strategy(self):
    selected = ...
    logger.debug(f"MAB selected: {selected}, "
                 f"avg_rewards: {self._get_avg_rewards()}")
    return selected
```

### Verify Losslessness

```python
# Compare output distributions with/without SD
# They should be statistically identical
def test_lossless():
    outputs_sd = generate_with_sd(prompts, seed=42)
    outputs_ar = generate_autoregressive(prompts, seed=42)

    # Token sequences should match (same seed)
    assert outputs_sd == outputs_ar
```

## See Also

- [Architecture Overview](./architecture.md)
- [Co-training Pipeline](./co-training-pipeline.md)
- [Key Files Reference](./key-files.md)
