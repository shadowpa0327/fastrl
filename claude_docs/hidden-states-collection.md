# Hidden States Collection & Consumption for EAGLE Co-training

This document details how hidden states are collected during RL rollouts and consumed for EAGLE drafter training, including exact tensor shapes at each stage.

## Overview

Hidden states from the target model's last layer serve as training data for the EAGLE drafter. The drafter learns to predict the next hidden state given the current one, enabling it to draft tokens that the target model is likely to accept.

> **⚠️ IMPORTANT: Hidden States Source Clarification**
>
> There are two collection paths in the codebase, but **only SGLang is the effective source**:
>
> | Path | Code Location | Status |
> |------|---------------|--------|
> | **Source A: SGLang** | `sglang_rollout.py:890-940` | ✅ **Active** - Used for training |
> | **Source B: Actor Forward** | `ray_trainer.py:1202-1227` | ⚠️ **Collects but unused** |
>
> The Actor path collects hidden states to `data_buffer`, but the trainer **blocks** (refuses to run)
> unless `collect_hidden_states_from_sgl=True`. See [Trainer Blocking Behavior](#trainer-blocking-behavior) below.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Hidden States Collection Flow                            │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Two Collection Sources                            │   │
│   │                                                                      │   │
│   │  Source A: SGLang Engine              Source B: Actor Forward        │   │
│   │  ┌─────────────────────────┐         ┌─────────────────────────┐    │   │
│   │  │ Collected during        │         │ Explicit forward pass   │    │   │
│   │  │ speculative decoding    │         │ on rollout data         │    │   │
│   │  │ verification step       │         │ (fallback method)       │    │   │
│   │  │                         │         │                         │    │   │
│   │  │ Efficient: no extra     │         │ More compute but        │    │   │
│   │  │ forward passes needed   │         │ always available        │    │   │
│   │  └───────────┬─────────────┘         └───────────┬─────────────┘    │   │
│   │              │                                   │                   │   │
│   │              └─────────────┬─────────────────────┘                   │   │
│   │                            ▼                                         │   │
│   │              ┌─────────────────────────┐                            │   │
│   │              │     Normalization       │                            │   │
│   │              │  Shape: [seq_len, D]    │                            │   │
│   │              │  Dtype: bfloat16        │                            │   │
│   │              │  Device: CPU            │                            │   │
│   │              └───────────┬─────────────┘                            │   │
│   └──────────────────────────┼──────────────────────────────────────────┘   │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Dual Storage System                             │   │
│   │                                                                      │   │
│   │  ┌─────────────────────────┐    ┌─────────────────────────────────┐ │   │
│   │  │   collected_data        │    │        DataBuffer               │ │   │
│   │  │   (deque, max=2000)     │    │   (deque, max=10000)            │ │   │
│   │  │                         │    │                                 │ │   │
│   │  │   Current step samples  │    │   Cross-step accumulation      │ │   │
│   │  │   Fast access           │    │   Richer training data         │ │   │
│   │  └───────────┬─────────────┘    └───────────┬─────────────────────┘ │   │
│   │              │                              │                        │   │
│   │              └──────────────┬───────────────┘                        │   │
│   └─────────────────────────────┼────────────────────────────────────────┘   │
│                                 ▼                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                   Training Batch Preparation                         │   │
│   │                                                                      │   │
│   │  1. Filter samples with valid hidden_states                         │   │
│   │  2. Apply windowing (max 512 tokens per sample)                     │   │
│   │  3. Concatenate sequences (remove padding)                          │   │
│   │  4. Create input/target pairs (shift by 1 position)                 │   │
│   │                                                                      │   │
│   │  Output: {input_ids, hidden_states, target, loss_mask, attn_mask}   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Source A: SGLang Engine Collection (Preferred)

**File**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py`

### Enabling Collection

```python
# Line 480-485: Engine initialization
enable_return_hidden_states=bool(
    self.use_spec
    and self.config.speculative.train.enable_drafter_training
    and self.config.speculative.train.get("collect_hidden_states_from_sgl", False)
),
```

### Extraction During Generation

```python
# Lines 890-940: Main collection logic
if should_collect:
    logger.info(f"[Rank {self._rank}] Collecting hidden states for drafter training...")

    engine_hidden_states = []
    valid_batch_indices = []

    for idx, sample in enumerate(output):
        # Extract hidden states from SGLang output
        hidden_states_list = []
        for i in range(len(sample["meta_info"]["hidden_states"])):
            # Convert to tensor with bfloat16 precision
            h_state = torch.tensor(
                sample["meta_info"]["hidden_states"][i],
                dtype=torch.bfloat16
            )

            if h_state.numel() == 0:
                continue

            # Normalize shape to [seq_len, hidden_dim]
            if h_state.dim() == 1:
                h_state = h_state.unsqueeze(0)  # [D] → [1, D]
            elif h_state.dim() > 2:
                h_state = h_state.view(-1, h_state.size(-1))  # Flatten to 2D

            hidden_states_list.append(h_state)

        if hidden_states_list:
            # Concatenate all hidden states for this sample
            hidden_states = torch.cat(hidden_states_list, dim=0)
            engine_hidden_states.append(hidden_states)  # List[Tensor[seq_len, D]]
            valid_batch_indices.append(idx)

    # Filter batch to only valid samples
    filtered_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.size(0) == len(output):
            filtered_batch[key] = value[valid_batch_indices]
        else:
            filtered_batch[key] = value

    # Pass to background trainer
    self.drafter_manager.background_trainer.collect_online_data(
        filtered_batch,
        engine_hidden_states
    )
```

**Advantages of SGLang Collection**:
- No extra forward passes needed
- Hidden states already computed during verification
- Efficient: reuses existing computation

## Source B: Actor Model Collection (Fallback)

**Files**:
- `verl/workers/fsdp_workers.py` (lines 1014-1036) - Entry point
- `verl/workers/actor/dp_actor.py` (lines 354-456) - Core implementation

### Entry Point: FSDP Worker

**File**: `verl/workers/fsdp_workers.py:1014-1036`

```python
# Line 1014: Extract return_hidden_states flag from meta_info
return_hidden_states = data.meta_info.pop("return_hidden_states", False)

# Line 1025: Call compute_log_prob with hidden states flag
result = self.actor.compute_log_prob(
    data=data,
    calculate_entropy=True,
    return_hidden_states=return_hidden_states
)

# Lines 1026-1029: Unpack results
if return_hidden_states:
    output, entropys, hidden_states = result
else:
    output, entropys = result

# Lines 1034-1036: Store hidden states in meta_info for downstream use
if return_hidden_states:
    meta_info_dict["hidden_states"] = hidden_states
```

### Core Implementation: DPActor.compute_log_prob

**File**: `verl/workers/actor/dp_actor.py:354-456`

```python
# Line 354: Method signature
def compute_log_prob(self, data: DataProto, calculate_entropy=False, return_hidden_states=False):
    """
    Args:
        return_hidden_states (bool): Whether to return hidden states from the last layer
    Returns:
        hidden_states: list of torch.Tensor or None (if return_hidden_states=True)
    """

    # Line 397: Initialize hidden states list
    hidden_states_lst = [] if return_hidden_states else None

    # Lines 399-405: Forward through micro-batches
    for micro_batch in micro_batches:
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        with torch.no_grad():
            result = self._forward_micro_batch(
                model_inputs,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
                return_hidden_states=return_hidden_states
            )
```

### Micro-Batch Forward: _forward_micro_batch

**File**: `verl/workers/actor/dp_actor.py:92-232`

```python
# Line 93: Method signature
def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, return_hidden_states=False):
    """
    Returns:
        hidden_states: # (bs, seqlen, hidden_dim) - optional, if return_hidden_states=True
    """

    # Lines 180-182: Request hidden states from model
    if return_hidden_states:
        extra_args["output_hidden_states"] = True

    # Lines 184-191: Model forward pass
    output = self.actor_module(
        input_ids=input_ids_rmpad,
        attention_mask=None,
        position_ids=position_ids_rmpad,
        use_cache=False,
        **extra_args,
    )

    # Lines 220-231: Extract hidden states from model output
    hidden_states_rmpad = None
    hidden_states_indices = None
    hidden_states_shape_info = None
    if return_hidden_states:
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            # Get last layer hidden states (in rmpad format)
            # Shape: ((total_nnz / sp) + pad, hidden_dim)
            hidden_states_rmpad = output.hidden_states[-1].squeeze(0)
            # Store reconstruction info
            hidden_states_shape_info = (batch_size, seqlen)
            hidden_states_indices = indices
```

### Hidden States Reconstruction

**File**: `verl/workers/actor/dp_actor.py:407-432`

```python
# Lines 407-432: Handle rmpad format from flash-attention
if return_hidden_states:
    entropy, log_probs, hidden_states = result

    if hidden_states is not None:
        if isinstance(hidden_states, tuple):
            # rmpad format: (rmpad_hidden_states, indices, (batch_size, seqlen))
            rmpad_hs, indices, (bsz, seqlen) = hidden_states

            # Line 416-423: Reconstruct full tensor using flash-attn padding utility
            from flash_attn.bert_padding import pad_input as pad_input_cpu

            full_hs = pad_input_cpu(
                hidden_states=rmpad_hs,  # already on CPU
                indices=indices,
                batch=bsz,
                seqlen=seqlen,
            )  # (batch_size, seqlen, hidden_dim) on CPU

            # Lines 425-426: Extract per-sample hidden states
            for sample_idx in range(bsz):
                hidden_states_lst.append(full_hs[sample_idx])  # [seq_len, hidden_dim]
        else:
            # Lines 428-432: Regular tensor format (non-rmpad path)
            for sample_idx in range(hidden_states.size(0)):
                sample_hidden = hidden_states[sample_idx]  # [seq_len, hidden_dim]
                hidden_states_lst.append(sample_hidden.detach().cpu())
```

## Data Buffer Implementation

**File**: `verl/utils/data_buffer.py`

### Buffer Class

```python
class DataBuffer:
    """Buffer for accumulating training data across RL steps.

    Enables drafter to train on data from multiple RL steps,
    not just the current step.

    Args:
        max_size: Maximum samples to store (default: 10000)
        store_hidden_states: Whether to store hidden_states (default: True)
    """

    def __init__(self, max_size: int = 10000, store_hidden_states: bool = True):
        self.max_size = max_size
        self.store_hidden_states = store_hidden_states
        self.buffer = deque(maxlen=max_size)  # Auto-evicts oldest
        self._current_step = 0
```

### Adding Samples

```python
def add_batch(self, batch: dict, hidden_states: Optional[list] = None):
    """Add a batch of samples to the buffer.

    Args:
        batch: Dictionary with input_ids, responses, prompts
        hidden_states: List of tensors [seq_len, hidden_dim] per sample
    """
    input_ids = batch.get("input_ids")
    responses = batch.get("responses")
    prompts = batch.get("prompts")

    if input_ids is None:
        logger.warning("Cannot add batch without input_ids")
        return

    batch_size = input_ids.size(0) if input_ids.dim() > 1 else 1

    for i in range(batch_size):
        sample = {
            "input_ids": input_ids[i].detach().cpu(),
            "step": self._current_step,
        }

        if responses is not None:
            sample["responses"] = responses[i].detach().cpu()

        if prompts is not None:
            sample["prompts"] = prompts[i].detach().cpu()

        # Store hidden states if enabled
        if self.store_hidden_states and hidden_states is not None:
            if i < len(hidden_states):
                h_state = hidden_states[i]

                # Normalize shape to [seq_len, hidden_dim]
                if h_state.dim() == 3:
                    h_state = h_state.squeeze(0)
                elif h_state.dim() == 1:
                    h_state = h_state.unsqueeze(0)

                sample["hidden_states"] = h_state.detach().cpu()

        self.buffer.append(sample)
```

### Retrieval Methods

```python
def get_all_data(self) -> list:
    """Return all samples in buffer."""
    return list(self.buffer)

def get_data_from_last_n_steps(self, n: int) -> list:
    """Get samples from the last N RL steps only."""
    if n <= 0:
        return []
    min_step = max(0, self._current_step - n + 1)
    return [s for s in self.buffer if s.get("step", 0) >= min_step]

def get_data_count(self) -> int:
    """Return number of samples in buffer."""
    return len(self.buffer)

def increment_step(self):
    """Mark boundary between RL steps."""
    self._current_step += 1
```

## Training Batch Preparation

**File**: `verl/workers/drafter/eagle_background_trainer.py`

### Batch Preparation Flow

```python
def _prepare_training_batch(self, step):
    """Prepare batch for drafter training.

    Steps:
    1. Retrieve data from buffer (multi-step) or collected_data (current)
    2. Filter samples with valid hidden_states
    3. Apply windowing (max 512 tokens)
    4. Concatenate sequences
    5. Create input/target pairs
    """

    # Retrieve samples (lines 290-311)
    if self.data_buffer is not None:
        samples = self.data_buffer.get_all_data()
    else:
        samples = list(self.collected_data)

    # Filter valid samples (lines 313-323)
    valid_samples = []
    for sample in samples:
        if "hidden_states" in sample and sample["hidden_states"] is not None:
            h = sample["hidden_states"]
            if h.numel() > 0 and h.size(0) > 1:  # Need at least 2 tokens
                valid_samples.append(sample)

    if len(valid_samples) == 0:
        return None
```

### Windowing and Concatenation

```python
    # Apply windowing (lines 355-402)
    max_window = 512  # Maximum tokens per sample

    for sample in valid_samples[:batch_size]:
        input_ids = sample["input_ids"]
        hidden_states = sample["hidden_states"].to(torch.bfloat16)
        loss_mask = sample["loss_mask"]

        full_len = input_ids.size(0)
        max_len = min(full_len, 512)

        # IMPORTANT: Window positioning prioritizes the END of response
        # Find response token positions (where loss_mask = 1)
        nonzero = torch.nonzero(loss_mask).flatten()
        if nonzero.numel() > 0:
            resp_start_idx = nonzero[0].item()
            resp_end_idx = nonzero[-1].item() + 1
            window_span = max_len

            # Try to start at response beginning
            start = max(0, min(resp_start_idx, full_len - window_span))

            # KEY: If response END doesn't fit, adjust to capture END
            if resp_end_idx - start > window_span:
                start = resp_end_idx - window_span  # Prioritize response END

            end = min(full_len, start + window_span)
        else:
            # No response tokens: take last max_len tokens
            start = max(0, full_len - max_len)
            end = full_len

        # Extract window (captures LAST 512 tokens including response end)
        seq_input_ids = input_ids[start:end]
        seq_hidden_states = hidden_states[start:end]
        seq_loss_mask = loss_mask[start:end]

    # Concatenate all sequences (lines 410-413)
    concat_input_ids = torch.cat(all_input_ids, dim=0)      # [total_len]
    concat_hidden = torch.cat(all_hidden_states, dim=0)     # [total_len, D]
    concat_loss_mask = torch.cat(all_loss_masks, dim=0)     # [total_len]
```

**Window Selection Behavior:**
- If response ≤ 512 tokens: captures entire response
- If response > 512 tokens: captures **LAST 512 tokens** (truncates beginning)
- Prioritizes response **END** over beginning (for better end-of-generation prediction)

```
Example: 2048 token sequence with 548 response tokens

Full sequence:
┌──────────────────────────────────────────────────────────────────────────┐
│ Prompt (1500 tokens)                │ Response (548 tokens)              │
│ [positions 0-1499]                  │ [positions 1500-2047]              │
└──────────────────────────────────────────────────────────────────────────┘
                                      ↑                                   ↑
                                resp_start=1500                    resp_end=2048

Window positioning (max 512):
  Initial: start = 1500 (try to start at response)
  Check: resp_end - start = 2048 - 1500 = 548 > 512? YES
  Adjust: start = resp_end - 512 = 2048 - 512 = 1536

Result:
                                 ┌────────────────────────────────────────┐
                                 │     Window [1536:2048] = 512 tokens    │
                                 │     (LAST 512 tokens captured)         │
                                 └────────────────────────────────────────┘
                                 ↑
                          36 response tokens TRUNCATED (positions 1500-1535)
```

### Input/Target Pair Creation

```python
    # Create input/target pairs (shift by 1)
    # Input: tokens 0 to N-1, predict hidden state at position 1 to N

    batch = {
        "input_ids": concat_input_ids[:-1].unsqueeze(0),           # [1, L-1]
        "hidden_states": concat_hidden[:-1].unsqueeze(0),          # [1, L-1, D]
        "target": concat_hidden[1:].unsqueeze(0),                  # [1, L-1, D]
        "loss_mask": concat_loss_mask[1:].unsqueeze(0),            # [1, L-1]
        "attention_mask": torch.ones(1, concat_input_ids.size(0)-1),
    }

    return batch
```

## Data Shapes Summary

```
Collection Stage:
━━━━━━━━━━━━━━━━━
  SGLang output:    sample["meta_info"]["hidden_states"][i]  (numpy/list)
  After conversion: Tensor[seq_len, hidden_dim]              (bfloat16)

Storage Stage:
━━━━━━━━━━━━━━
  DataBuffer sample: {
      "input_ids":     Tensor[seq_len],
      "hidden_states": Tensor[seq_len, hidden_dim],
      "responses":     Tensor[response_len],     (optional)
      "prompts":       Tensor[prompt_len],       (optional)
      "step":          int,
  }

Training Batch:
━━━━━━━━━━━━━━━
  After preparation: {
      "input_ids":     Tensor[1, total_seq_len],
      "hidden_states": Tensor[1, total_seq_len, hidden_dim],
      "target":        Tensor[1, total_seq_len, hidden_dim],
      "loss_mask":     Tensor[1, total_seq_len],
      "attention_mask": Tensor[1, total_seq_len],
  }

Typical dimensions:
  seq_len:    256 - 2048 tokens
  hidden_dim: 768 - 4096 (model dependent)
  total_seq_len after concat: 512 - 2048 tokens
```

## Configuration Options

```yaml
speculative:
  enable: true
  train:
    # Master switch for drafter training
    enable_drafter_training: true

    # Enable SGLang hidden state collection (preferred)
    collect_hidden_states_from_sgl: true

    # Buffer sizes
    buffer_max_samples: 2000        # collected_data deque size
    data_buffer_max_size: 10000     # DataBuffer cross-step size

    # Training parameters
    batch_size_per_gpu: 32
    training_interval_steps: 10     # Train every N RL steps

    # Loss weights
    vloss_weight: 0.5               # Velocity (hidden state) loss
    ploss_weight: 0.5               # Probability (logit) loss
```

## Debugging Hidden State Collection

### Check Collection is Enabled

```python
# Look for this log message
logger.info(f"[Rank {self._rank}] Collecting hidden states for drafter training...")
```

### Verify Hidden State Shapes

```python
# Add debug logging in collect_online_data
def collect_online_data(self, batch, hidden_states):
    for i, h in enumerate(hidden_states):
        logger.debug(f"Sample {i}: hidden_states shape = {h.shape}")
```

### Monitor Buffer Size

```python
# Check buffer statistics
logger.info(f"DataBuffer size: {self.data_buffer.get_data_count()}")
logger.info(f"collected_data size: {len(self.collected_data)}")
```

### Validate Training Data

```python
# In _prepare_training_batch
if batch is not None:
    logger.debug(f"Training batch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logger.debug(f"  {k}: {v.shape}")
```

## SGLang Internal Flow (Deep Dive)

The hidden states collection within SGLang follows this detailed flow:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SGLang Hidden States Internal Flow                        │
│                                                                              │
│  1. Server Initialization (server_args.py:460)                              │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │  enable_return_hidden_states: bool = False                          │ │
│     │  # Set to True to enable hidden state collection at server level    │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  2. Request Handling (io_struct.py:126, schedule_batch.py:430)             │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │  GenerateReqInput:                                                  │ │
│     │    return_hidden_states: Union[List[bool], bool] = False           │ │
│     │                                                                     │ │
│     │  Req.__init__():                                                    │ │
│     │    self.return_hidden_states = return_hidden_states                │ │
│     │    self.hidden_states: List[List[float]] = []  # Storage           │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  3. Batch Creation (schedule_batch.py:1031)                                │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │  ScheduleBatch.init_new():                                          │ │
│     │    return_hidden_states=any(req.return_hidden_states for req in...)│ │
│     │                                                                     │ │
│     │  # Determines capture mode based on batch requirements             │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  4. Forward Batch Setup (schedule_batch.py:1755-1761)                      │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │  get_model_worker_batch():                                          │ │
│     │    capture_hidden_mode=(                                            │ │
│     │      CaptureHiddenMode.FULL if self.return_hidden_states           │ │
│     │      else spec_info.capture_hidden_mode or CaptureHiddenMode.NULL  │ │
│     │    )                                                                │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  5. CUDA Graph Runner (cuda_graph_runner.py:281-282, 704-708)              │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │  # Initial capture mode setup                                       │ │
│     │  if model_runner.server_args.enable_return_hidden_states:           │ │
│     │      self.capture_hidden_mode = CaptureHiddenMode.FULL             │ │
│     │                                                                     │ │
│     │  # During forward, determine required capture mode                  │ │
│     │  capture_hidden_mode_required_for_returning_hidden_states = (       │ │
│     │      CaptureHiddenMode.FULL                                         │ │
│     │      if self.model_runner.server_args.enable_return_hidden_states  │ │
│     │      else CaptureHiddenMode.NULL                                    │ │
│     │  )                                                                  │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  6. Model Forward (llama.py:483-489)                                       │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │  def forward(self, input_ids, positions, forward_batch, ...):       │ │
│     │      hidden_states = self.model(input_ids, positions, ...)         │ │
│     │                                                                     │ │
│     │      return self.logits_processor(                                  │ │
│     │          input_ids,                                                 │ │
│     │          hidden_states,        # <-- Hidden states passed here     │ │
│     │          self.lm_head,                                              │ │
│     │          forward_batch,                                             │ │
│     │          aux_hidden_states,                                         │ │
│     │      )                                                              │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  7. Logits Processor (logits_processor.py:474-505)                         │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │  def forward(self, input_ids, hidden_states, lm_head, ...):         │ │
│     │                                                                     │ │
│     │      hidden_states_to_store: Optional[torch.Tensor] = None         │ │
│     │      if logits_metadata.capture_hidden_mode.need_capture():        │ │
│     │          if logits_metadata.capture_hidden_mode.is_full():         │ │
│     │              hidden_states_to_store = hidden_states  # Store ALL   │ │
│     │          elif logits_metadata.capture_hidden_mode.is_last():       │ │
│     │              hidden_states_to_store = pruned_states  # LAST only   │ │
│     │                                                                     │ │
│     │      return LogitsProcessorOutput(                                  │ │
│     │          next_token_logits=sampled_logits,                         │ │
│     │          hidden_states=hidden_states_to_store,  # <-- Stored here  │ │
│     │      )                                                              │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  8. Output Processing (scheduler_output_processor_mixin.py:358-360)        │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │  # During decode                                                    │ │
│     │  if req.return_hidden_states and logits_output.hidden_states:      │ │
│     │      req.hidden_states.append(                                      │ │
│     │          logits_output.hidden_states[i].cpu().clone().tolist()     │ │
│     │      )                                                              │ │
│     │                                                                     │ │
│     │  # During extend/prefill (lines 123-132)                           │ │
│     │  if req.return_hidden_states and logits_output.hidden_states:      │ │
│     │      req.hidden_states.append(                                      │ │
│     │          logits_output.hidden_states[offset:offset+len(input_ids)] │ │
│     │      )                                                              │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  9. Response Assembly (scheduler_output_processor_mixin.py:882-885)        │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │  if req.return_hidden_states:                                       │ │
│     │      if output_hidden_states is None:                               │ │
│     │          output_hidden_states = []                                  │ │
│     │      output_hidden_states.append(req.hidden_states)                │ │
│     │                                                                     │ │
│     │  # Included in BatchTokenIDOut (line 927)                          │ │
│     │  output_hidden_states=output_hidden_states,                        │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  10. FastRL Collection (sglang_rollout.py:890-940)                         │
│     ┌─────────────────────────────────────────────────────────────────────┐ │
│     │  # Extract from SGLang output                                       │ │
│     │  for idx, sample in enumerate(output):                              │ │
│     │      for i in range(len(sample["meta_info"]["hidden_states"])):    │ │
│     │          h_state = torch.tensor(                                    │ │
│     │              sample["meta_info"]["hidden_states"][i],              │ │
│     │              dtype=torch.bfloat16                                   │ │
│     │          )                                                          │ │
│     │          hidden_states_list.append(h_state)                        │ │
│     └─────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CaptureHiddenMode Enum

**File**: `forward_batch_info.py:139-157`

```python
class CaptureHiddenMode(IntEnum):
    # Do not capture anything
    NULL = 0

    # Capture hidden state of the LAST token only
    # Used by speculative decoding (EAGLE) for draft prediction
    LAST = 1

    # Capture ALL hidden states for the entire sequence
    # Used when return_hidden_states=True for full sequence collection
    FULL = 2

    def need_capture(self):
        return self != CaptureHiddenMode.NULL

    def is_full(self):
        return self == CaptureHiddenMode.FULL

    def is_last(self):
        return self == CaptureHiddenMode.LAST
```

### SGLang Key Files for Hidden States

| File | Lines | Purpose |
|------|-------|---------|
| `server_args.py` | 460 | `enable_return_hidden_states` server flag |
| `io_struct.py` | 126, 600, 844, 918 | Request/Response hidden_states fields |
| `schedule_batch.py` | 430, 472, 591, 987, 1031, 1697 | Req class hidden_states storage, batch flag |
| `forward_batch_info.py` | 139-157, 294 | CaptureHiddenMode enum, ForwardBatch flag |
| `cuda_graph_runner.py` | 281-282, 704-708 | CUDA graph capture mode setup |
| `logits_processor.py` | 65, 474-505 | LogitsProcessorOutput, hidden_states extraction |
| `scheduler_output_processor_mixin.py` | 123-132, 358-360, 882-885 | Per-request hidden_states accumulation |
| `eagle_worker.py` | 652, 678, 829, 1020, 1090 | SD sets `return_hidden_states=False` (SD handles internally) |

---

## Hidden States Consumption in EAGLE Training

### Overview: Training Objective

The EAGLE drafter learns to **predict the next token's hidden state** given the current hidden state. This enables the drafter to generate draft tokens that the target model is likely to accept.

```
Training Target:
  Given: hidden_states[t], input_ids[t]
  Predict: hidden_states[t+1]

Loss:
  1. Value Loss (L1): ||predicted_hidden - target_hidden||
  2. Policy Loss (KL): KL(softmax(lm_head(target)), softmax(lm_head(pred)))
```

### Consumption in Background Trainer

**File**: `verl/workers/drafter/eagle_background_trainer.py`

#### Batch Preparation (Lines 275-456)

```python
def _prepare_training_batch(self, use_buffer_data=True, buffer_steps=2):
    """Prepare batch for training using Ulysses SP to remove padding."""

    # 1. Retrieve samples from buffer
    items = self.data_buffer.get_data_from_last_n_steps(buffer_steps)

    # 2. Filter samples with valid hidden_states
    items = [item for item in items if "hidden_states" in item]

    # 3. Window extraction (max 512 tokens)
    for item in items:
        max_len = min(full_len, 512)
        # Select window around response tokens
        seq_hidden_states = item["hidden_states"][start:end]

    # 4. Concatenate sequences (remove padding between samples)
    input_ids_concat = torch.cat(input_ids_list, dim=0).unsqueeze(0)      # [1, total_seq_len]
    hidden_states_concat = torch.cat(hidden_states_list, dim=0).unsqueeze(0)  # [1, total_seq_len, D]

    # 5. Shift for next-token prediction
    target = hidden_states_concat[:, 1:].contiguous()      # [1, total_seq_len-1, D]
    base_h = hidden_states_concat[:, :-1].contiguous()     # [1, total_seq_len-1, D]
    input_ids = input_ids_concat[:, :-1].contiguous()      # [1, total_seq_len-1]
    loss_mask = loss_mask_concat[:, 1:].contiguous()       # [1, total_seq_len-1]

    return {
        "input_ids": input_ids,
        "hidden_states": base_h,     # Input to model
        "target": target,            # Training target
        "loss_mask": loss_mask,
        "attention_mask": attn_mask,
    }
```

#### Training Step (Lines 458-609)

```python
async def _training_step_impl(self, step):
    batch = self._prepare_training_batch()

    # Forward pass
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = self.model(
            input_ids=batch["input_ids"],               # [1, L]
            attention_mask=batch["attention_mask"],     # [1, L]
            base_model_hidden_states=batch["hidden_states"],  # [1, L, D]
            output_hidden_states=True,
        )

    # Extract outputs
    hidden_states = outputs.hidden_states[-1]  # [1, L, D] predicted
    target = batch["target"]                    # [1, L, D] target

    # Compute logits from hidden states
    logits = self.model.lm_head(hidden_states)  # [1, L, vocab_size]

    # Loss computation
    # Policy Loss (KL divergence)
    out_logp = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        target_logits = self.model.lm_head(target)
        target_p = F.softmax(target_logits, dim=-1)

    plogp = target_p * out_logp
    ploss = -sum(loss_mask * sum(plogp, dim=-1)) / num_valid_tokens

    # Value Loss (L1)
    vloss = self.criterion(hidden_states, target)  # SmoothL1
    vloss = sum(loss_mask * mean(vloss, dim=-1)) / num_valid_tokens

    # Combined loss
    loss = vloss_weight * vloss + ploss_weight * ploss

    loss.backward()
    self.optimizer.step()
```

---

### Consumption in Standalone EAGLE Training

**File**: `eagle-train/eagle_trainer.py`

#### Dataset Loading (Lines 47-84)

```python
class EagleDataset(Dataset):
    def __getitem__(self, idx):
        data = torch.load(self.datapath[idx])

        return {
            "input_ids": data["input_ids"][1:],      # Shift right
            "hidden_states": data["hidden_state"],   # [seq_len, 4096]
            "target": data["hidden_state"][1:],      # Shift right for target
            "loss_mask": data["loss_mask"],
        }
```

#### Batch Collation (Lines 98-147)

```python
class EagleDataCollator:
    def __call__(self, features):
        max_length = features[0]["max_seq_len"] or max(len(f["input_ids"]) for f in features)

        # Pad and batch
        batch = {
            "input_ids": pad_and_stack(...),          # [B, max_len]
            "hidden_states": pad_and_stack(...),      # [B, max_len, 4096]
            "target": pad_and_stack(...),             # [B, max_len, 4096]
            "loss_mask": pad_and_stack(...),          # [B, max_len]
            "attention_mask": create_mask(...),       # [B, max_len]
        }
        return batch
```

#### Loss Computation (Lines 357-392)

```python
def _compute_loss(self, batch):
    # Model forward
    outputs = self.model_engine(
        batch["hidden_states"],      # [B, L, 4096] base model hidden states
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )

    predict = outputs.hidden_states[-1]  # [B, L, 4096] predicted
    target = batch["target"]              # [B, L, 4096] target

    # LM Head projections
    out_head = self.model.lm_head(predict)      # [B, L, vocab_size]
    target_head = self.model.lm_head(target)    # [B, L, vocab_size]

    # Policy Loss
    target_p = nn.Softmax(dim=2)(target_head)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -sum(loss_mask * sum(plogp, dim=2)) / total_tokens

    # Value Loss
    vloss = self.criterion(predict, target)  # SmoothL1: [B, L, 4096]
    vloss = sum(loss_mask * mean(vloss, dim=2)) / total_tokens

    # Combined
    loss = value_weight * vloss + prob_weight * ploss
    return loss
```

---

### EAGLE Model Architecture (How Hidden States Are Used)

**File**: `eagle-train/model/llama_eagle.py`

```python
class LlamaModel(LlamaModelTF):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(...)])  # 1 layer
        self.fc = nn.Linear(hidden_size * 2, hidden_size)      # Key: concat projection

    def forward(self, base_model_hidden_states, input_ids, ...):
        # Step 1: Get token embeddings
        inputs_embeds = self.embed_tokens(input_ids)  # [B, L, 4096]

        # Step 2: Concatenate with base model hidden states
        concat = torch.cat([inputs_embeds, base_model_hidden_states], dim=-1)
        # concat shape: [B, L, 8192]

        # Step 3: Project back to hidden_size
        hidden_states = self.fc(concat)  # [B, L, 4096]

        # Step 4: Pass through transformer layer(s)
        for layer in self.layers:
            hidden_states = layer(hidden_states, ...)  # [B, L, 4096]

        return hidden_states  # Predicted next hidden states
```

---

### Complete Tensor Shape Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HIDDEN STATES CONSUMPTION FLOW                            │
│                                                                              │
│  COLLECTION STAGE                                                            │
│  ════════════════                                                            │
│  SGLang output: sample["meta_info"]["hidden_states"]                        │
│    → List of numpy arrays                                                    │
│    → torch.tensor(dtype=bfloat16)                                           │
│    → Normalize to [seq_len, hidden_dim]                                     │
│                                                                              │
│  STORAGE STAGE                                                               │
│  ═════════════                                                               │
│  DataBuffer item:                                                            │
│    input_ids:     [seq_len]          (int64)                                │
│    hidden_states: [seq_len, 4096]    (bfloat16, CPU)                        │
│    loss_mask:     [seq_len]          (float32)                              │
│                                                                              │
│  BATCH PREPARATION                                                           │
│  ═════════════════                                                           │
│  Window selection (max 512 tokens, prioritizes response END):               │
│    - If response > 512: captures LAST 512 tokens (truncates beginning)      │
│    - If response ≤ 512: captures entire response                            │
│    hidden_states: [window_len, 4096]                                        │
│                                                                              │
│  Concatenate N samples:                                                      │
│    input_ids_concat:     [1, total_len]                                     │
│    hidden_states_concat: [1, total_len, 4096]                               │
│                                                                              │
│  Shift for next-token prediction:                                           │
│    input_hidden (input):  [1, total_len-1, 4096]  ← hidden[:-1]            │
│    target_hidden (target): [1, total_len-1, 4096] ← hidden[1:]             │
│    input_ids:             [1, total_len-1]        ← ids[:-1]               │
│    loss_mask:             [1, total_len-1]        ← mask[1:]               │
│                                                                              │
│  MODEL FORWARD                                                               │
│  ═════════════                                                               │
│  EAGLE Model:                                                                │
│    Input:                                                                    │
│      input_ids:             [1, L]      (token IDs)                         │
│      base_model_hidden:     [1, L, 4096] (base model hidden states)         │
│                                                                              │
│    Internal:                                                                 │
│      embed_tokens(input_ids): [1, L, 4096]                                  │
│      concat([embed, hidden]): [1, L, 8192]                                  │
│      fc projection:           [1, L, 4096]                                  │
│      transformer layer:       [1, L, 4096]                                  │
│                                                                              │
│    Output:                                                                   │
│      predicted_hidden:        [1, L, 4096]                                  │
│                                                                              │
│  LOSS COMPUTATION                                                            │
│  ════════════════                                                            │
│  Value Loss (SmoothL1):                                                      │
│    predicted_hidden: [1, L, 4096]                                           │
│    target_hidden:    [1, L, 4096]                                           │
│    vloss = SmoothL1(pred, target)  → [1, L, 4096]                          │
│    vloss = mean(vloss, dim=-1)     → [1, L]                                │
│    vloss = sum(loss_mask * vloss) / num_valid                               │
│                                                                              │
│  Policy Loss (KL):                                                           │
│    pred_logits = lm_head(predicted_hidden)  → [1, L, vocab_size]           │
│    tgt_logits = lm_head(target_hidden)      → [1, L, vocab_size]           │
│    pred_logp = log_softmax(pred_logits)     → [1, L, vocab_size]           │
│    tgt_p = softmax(tgt_logits)              → [1, L, vocab_size]           │
│    plogp = tgt_p * pred_logp                → [1, L, vocab_size]           │
│    ploss = -sum(loss_mask * sum(plogp, dim=-1)) / num_valid                │
│                                                                              │
│  Combined Loss:                                                              │
│    loss = vloss_weight * vloss + ploss_weight * ploss                       │
│         = 0.5 * vloss + 0.5 * ploss  (default weights)                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Key Dimensions Reference

| Stage | Tensor | Shape | Notes |
|-------|--------|-------|-------|
| **Collection** | Raw from SGLang | `List[numpy]` | Variable per token |
| | After conversion | `[seq_len, D]` | D=4096 for Llama-7B |
| **Storage** | In DataBuffer | `[seq_len, D]` | CPU, bfloat16 |
| **Batch Prep** | After window | `[window, D]` | window ≤ 512, captures response END |
| | After concat | `[1, total, D]` | total = sum of windows |
| | Input hidden | `[1, L-1, D]` | L = total_len |
| | Target hidden | `[1, L-1, D]` | Shifted by 1 |
| **Model** | Embedding | `[1, L-1, D]` | From input_ids |
| | Concat | `[1, L-1, 2D]` | embed + base_hidden |
| | After FC | `[1, L-1, D]` | Projected back |
| | Output | `[1, L-1, D]` | Predicted hidden |
| **Loss** | Pred logits | `[1, L-1, V]` | V = vocab_size |
| | Value loss | `[1, L-1, D]` → scalar | SmoothL1 |
| | Policy loss | `[1, L-1, V]` → scalar | KL divergence |

---

### EAGLE2 vs EAGLE3 Hidden State Differences

| Aspect | EAGLE2 | EAGLE3 |
|--------|--------|--------|
| **Input hidden states** | Last layer only | Concatenated from 4 layers |
| **Input shape** | `[B, L, 4096]` | `[B, L, 16384]` (4×4096) |
| **FC projection** | 8192 → 4096 | 16384 → 4096 (or 12288 → 4096) |
| **Target** | Hidden states | Logits (vocab distribution) |
| **Loss** | Value + Policy | Policy only (multi-step) |
| **Multi-step** | No | Yes (prediction_length steps) |

---

### Consumption Files Reference

| Component | File | Key Lines | Purpose |
|-----------|------|-----------|---------|
| Background Trainer | `eagle_background_trainer.py` | 275-456, 458-609 | Online training during RL |
| Standalone Dataset | `eagle-train/eagle_trainer.py` | 47-84, 98-147 | Offline data loading |
| Loss Computation | `eagle-train/eagle_trainer.py` | 357-392 | EAGLE2 loss |
| EAGLE Model | `eagle-train/model/llama_eagle.py` | 34-99 | Model forward pass |
| EAGLE3 Model | `eagle-train/model/llama_eagle3.py` | 297-468 | Multi-step prediction |
| Data Generation | `eagle-train/eagle_datagen.py` | 371-407 | Extract hidden states |

### Collection Files Reference (Source B: Actor Forward)

| Component | File | Key Lines | Purpose |
|-----------|------|-----------|---------|
| Entry Point | `verl/workers/fsdp_workers.py` | 1014-1036 | Trigger hidden states collection |
| Core Logic | `verl/workers/actor/dp_actor.py` | 354-456 | `compute_log_prob` with hidden states |
| Micro-batch Forward | `verl/workers/actor/dp_actor.py` | 92-232 | `_forward_micro_batch` extraction |
| Reconstruction | `verl/workers/actor/dp_actor.py` | 407-432 | Handle rmpad format, per-sample extraction |

---

## Trainer Blocking Behavior

### What "Trainer Blocks" Means

The EAGLE background trainer has a guard condition that **refuses to execute training** if `collect_hidden_states_from_sgl=False`:

**File**: `verl/workers/drafter/eagle_background_trainer.py:472-479`

```python
async def _training_step_impl(self, step: int) -> bool:
    """Execute a single training step."""

    # THIS IS THE BLOCKING CHECK
    collect_hidden_states_from_sgl = bool(self.config.get("collect_hidden_states_from_sgl", False))
    if not collect_hidden_states_from_sgl:
        logger.debug(
            f"[EagleTrainer rank {self.rank}] Skipping training step {step} "
            f"because collect_hidden_states_from_sgl=False"
        )
        return False  # ← EARLY RETURN - training does not proceed

    # Training code below never executes if above returns False
    batch = self._prepare_training_batch()
    ...
```

### Why This Creates a Problem

The Actor path (`ray_trainer.py:1202-1227`) collects hidden states **independently** of this flag:

```python
# ray_trainer.py - Actor collection trigger
should_collect_for_drafter = (
    self.config.speculative.get("train", {}).get("enable_drafter_training", False)
    # Note: Does NOT check collect_hidden_states_from_sgl!
)

if should_collect_for_drafter:
    batch.meta_info["return_hidden_states"] = True
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
    hidden_states = old_log_prob.meta_info.pop("hidden_states")
    # Data IS added to buffer...
    self.actor_rollout_wg.apply("add_drafter_data_to_buffer", batch, hidden_states)
```

### The Result

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Data Flow When collect_hidden_states_from_sgl=False    │
│                                                                              │
│   Actor Forward                                                              │
│   (ray_trainer.py:1202-1227)                                                │
│         │                                                                    │
│         │ Collects hidden_states                                            │
│         ▼                                                                    │
│   add_drafter_data_to_buffer()                                              │
│   (fsdp_workers.py:1163-1191)                                               │
│         │                                                                    │
│         │ Adds to data_buffer                                               │
│         ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     data_buffer                                      │   │
│   │                  (contains hidden states!)                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                    │
│         │ Training step called                                              │
│         ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  _training_step_impl()                                               │   │
│   │                                                                      │   │
│   │  if not collect_hidden_states_from_sgl:                             │   │
│   │      return False  ← BLOCKED! Data in buffer is IGNORED            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Result: Hidden states collected but NEVER USED for training               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Summary Table

| `collect_hidden_states_from_sgl` | SGLang Collects | Actor Collects | Training Runs | Effective Behavior |
|----------------------------------|-----------------|----------------|---------------|-------------------|
| `True` | ✅ Yes | ✅ Yes | ✅ Yes | Normal training with SGLang data |
| `False` | ❌ No | ✅ Yes | ❌ **Blocked** | Actor collects but data is wasted |

### Potential Fix (Not Implemented)

To enable Actor-based collection, the blocking check should be modified:

```python
# Current (blocks Actor path):
if not collect_hidden_states_from_sgl:
    return False

# Potential fix (allow training if data_buffer has data):
if len(self.data_buffer) == 0 and len(self.collected_data) == 0:
    return False  # Only block if NO data available
```

## See Also

- [Co-training Pipeline](./co-training-pipeline.md)
- [Drafter Training Activation](./drafter-training-activation.md) - GPU memory management, activation call chain
- [Architecture Overview](./architecture.md)
- [Key Files Reference](./key-files.md)
