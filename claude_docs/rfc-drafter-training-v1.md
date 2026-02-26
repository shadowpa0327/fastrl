# RFC V1: EAGLE Drafter Co-training — Train-Only MVP

## Status: Draft
## Author: brian1009
## Date: 2026-02-18
## Supersedes: Subset of `rfc-drafter-training.md` (full design reference)

---

## 1. Scope

**In scope (V1)**:
- Drafter `TrainingWorker` + `FSDPDrafterEngine` initialization inside `ActorRolloutRefWorker`
- Hidden state collection during `compute_log_prob` (actor forward pass)
- Bounded hidden state buffer on each worker
- EAGLE dual loss (SmoothL1 + cross-entropy)
- Frozen layer refresh (`lm_head`, `embed_tokens`) from actor via FSDP2 local shard copy
- Synchronous `update_drafter()` step in the RL loop
- Drafter metrics logging
- Config-driven enablement (no Role enum changes)
- `use_legacy_worker_impl == "disable"` path only (engine_workers.py)
- FSDP/FSDP2 backend only

**Deferred (V2)**:
- [ ] SGLang/vLLM drafter weight sync (`is_draft_model` routing — requires SGLang/vLLM patch, see Section 10)
- [ ] Drafter checkpoint save/load
- [ ] MAB adaptive speculative decoding
- [ ] Megatron backend
- [ ] Async/background training mode
- [ ] EAGLE-3 model support

**V1 validates**: Loss convergence, training overhead, memory budget, architecture correctness. Drafter weights are trained but not pushed to the rollout engine at runtime.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      RayPPOTrainer.fit()                         │
│                                                                  │
│  ① generate_sequences()         ← rollout engine (SGLang)        │
│  ② compute_reward()             ← reward workers                 │
│  ③ compute_log_prob()           ← actor engine (eval mode)       │
│     └─ extract hidden_states    ← NEW: piggyback on forward pass │
│        └─ buffer on worker      ← NEW: bounded deque             │
│  ④ compute_advantage()          ← driver                         │
│  ⑤ update_critic()              ← critic engine (train)          │
│  ⑥ update_actor()               ← actor engine (train)           │
│  ⑦ update_drafter()             ← NEW: drafter engine (train)    │
│     └─ refresh lm_head/embed    ← NEW: local shard copy          │
│  ⑧ update_weights()             ← sync actor → rollout (no drafter in V1) │
└──────────────────────────────────────────────────────────────────┘
```

### Worker Composition

```python
ActorRolloutRefWorker  (drafter enabled via config)
├── self.actor:           TrainingWorker  → FSDPEngineWithLMHead   # policy gradient
├── self.ref:             TrainingWorker  → FSDPEngineWithLMHead   # frozen reference
├── self.drafter:         TrainingWorker  → FSDPDrafterEngine      # EAGLE training (NEW)
├── self.rollout:         BaseRollout                              # token generation
└── self._drafter_buffer: deque[dict]                              # hidden state buffer (NEW)
```

---

## 3. Data Shape Reference

This section documents the exact tensor shapes at each stage of the drafter training pipeline.
Shapes are verified against FastRL's `eagle_background_trainer.py` and `qwen2_eagle.py`.

**Notation**: B = batch size, L = sequence length (prompt + response), D = hidden_dim, V = vocab_size.

**Assumption**: `ulysses_sequence_parallel_size == 1` (no Ulysses SP). This eliminates all SP gather/scatter logic from the hidden state extraction path.

### 3.0.1 Per-Sample Buffer Storage

Each sample in `_drafter_buffer` stores tensors at full (unpadded) sequence length:

| Key | Shape | Dtype | Source | Notes |
|-----|-------|-------|--------|-------|
| `input_ids` | `(L_actual,)` | int64 | nested tensor unbind | Variable-length per sample |
| `hidden_states` | `(L_actual, D)` | bf16/fp32 | nested tensor unbind | Last-layer hidden states from actor forward |
| `loss_mask` | `(L_actual,)` | float32 | Reconstructed: `zeros(prompt_actual) + ones(response_actual)` | 1.0 on response tokens, 0.0 on prompt tokens |

All three tensors have the **same** first dimension `L_actual` (the actual non-padded token count for that sample). Stored on CPU.

**loss_mask reconstruction**: `batch["loss_mask"]` = `response_mask` is `(bs, response_max_len)` — covering ONLY the response portion, NOT full sequence (padding.py:71, ray_trainer.py:124-127). We reconstruct the full-sequence mask from `response_actual = sum(response_mask[i])` and `prompt_actual = L_actual - response_actual`, then concatenate `zeros(prompt_actual) + ones(response_actual)`. This matches FastRL's `collect_online_data` logic (eagle_background_trainer.py:247-253).

**FastRL comparison**: FastRL stores the same three keys with identical shapes (eagle_background_trainer.py:260-266). Both FastRL and the RFC construct `loss_mask` from prompt/response lengths.

### 3.0.2 Training Batch (output of `_sample_drafter_batch`)

After sampling, padding, and shift alignment:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `input_ids` | `(B, max_len-1)` | int64 | Token IDs, dropped last position |
| `attention_mask` | `(B, max_len-1)` | int64 | All 1s (computed fresh, not from rollout) |
| `base_hidden_states` | `(B, max_len-1, D)` | bf16 | Base model hidden states at positions `[0..T-2]` |
| `target_hidden_states` | `(B, max_len-1, D)` | bf16 | Base model hidden states at positions `[1..T-1]` |
| `loss_mask` | `(B, max_len-1)` | float32 | Shifted by 1: `mask[:, 1:]` — aligns with target |
| `micro_batch_size_per_gpu` | scalar tensor | int | Required by `prepare_micro_batches` when `use_dynamic_bsz=False` |

**Shift alignment rule** (from eagle_background_trainer.py:443-456):
```
Position:    0    1    2   ...  T-2   T-1
input_ids:   t_0  t_1  t_2 ... t_{T-2}        ← ids[:, :-1]
base_h:      h_0  h_1  h_2 ... h_{T-2}        ← hidden[:, :-1]
target_h:    h_1  h_2  h_3 ... h_{T-1}        ← hidden[:, 1:]
loss_mask:   m_1  m_2  m_3 ... m_{T-1}        ← mask[:, 1:]
```
The EAGLE model takes `(t_i, h_i)` and predicts `h_{i+1}`. Loss is masked to response-only positions.

**FastRL comparison**: FastRL uses B=1 with N<=4 samples **concatenated** along the sequence dimension: `(1, T, D)` where `T = sum(W_i)`. Each sample is windowed to max 512 tokens. The RFC instead uses standard padded batching with `B = train_batch_size`, which is simpler and compatible with FSDP micro-batching. The tradeoff is slightly more padding waste but avoids cross-sample attention leakage.

### 3.0.3 EAGLE Model Forward Signature

```python
# qwen2_eagle.py:204 / llama_eagle.py:204
def forward(
    self,
    base_model_hidden_states: torch.Tensor,  # (B, L, D)
    input_ids: torch.LongTensor,             # (B, L)
    attention_mask: torch.Tensor,            # (B, L)
    output_hidden_states: bool = True,
) -> CausalLMOutputWithPast:
    # Internal flow:
    #   embed_tokens(input_ids)              → (B, L, D)
    #   fc(cat(embeds, base_h), dim=-1)      → (B, L, 2D) → Linear(2D, D) → (B, L, D)
    #   1x decoder layer                     → (B, L, D)
    #   lm_head(hidden_states)               → (B, L, V)
```

| Output field | Shape | Description |
|-------------|-------|-------------|
| `logits` | `(B, L, V)` | Predicted next-token distribution |
| `hidden_states` | tuple of `(B, L, D)` | Per-layer hidden states (2 entries: input + output of the single decoder layer) |

### 3.0.4 Loss Function Expected Shapes

```python
eagle_dual_loss(model_output, data, dp_group, lm_head, w_v, w_p)
```

| Input | Key | Shape | Source |
|-------|-----|-------|--------|
| `model_output` | `logits` | `(B, L-1, V)` | EAGLE model output |
| `model_output` | `hidden_states` | `(B, L-1, D)` | EAGLE model last-layer hidden |
| `data` | `target_hidden_states` | `(B, L-1, D)` | Shifted base model hidden states |
| `data` | `loss_mask` | `(B, L-1)` | Response-only mask (shifted) |

| Intermediate | Shape | Operation |
|-------------|-------|-----------|
| `target_logits` | `(B, L-1, V)` | `lm_head(target_hidden_states)` |
| `target_p` | `(B, L-1, V)` | `softmax(target_logits)` (detached) |
| `out_logp` | `(B, L-1, V)` | `log_softmax(logits)` |
| `vloss` (per-element) | `(B, L-1, D)` | `SmoothL1(pred_hidden, target_hidden)` |
| `vloss` (reduced) | scalar | `(mask * mean(vloss, dim=-1)).sum() / num_valid` |
| `ploss` | scalar | `-(mask * (target_p * out_logp).sum(dim=-1)).sum() / num_valid` |

**FastRL loss weight defaults**: `w_v=1.0, w_p=0.1` (fsdp_workers.py:602-603). V1 now uses the same defaults (`w_v=1.0, w_p=0.1`) to match FastRL's stable starting point.

### 3.0.5 End-to-End Shape Trace

```
Actor Forward Pass (compute_log_prob, engine forward_step)
  output.hidden_states[-1]                            (assumes SP=1, no gather)
    rmpad path: (1, total_nnz, D) → squeeze(0) → (total_nnz, D)
             → nested_tensor_from_jagged(hs_rmpad, cu_seqlens) → nested tensor
    postprocess_batch_func: nested → unbind → reassemble → nested (B, L_i, D)
                                                              │
Worker-level extraction (_collect_drafter_sample)              │
  hidden_states.unbind() → per sample: (L_actual, D)          │
  input_ids.unbind()     → per sample: (L_actual,)            │
  loss_mask: zeros(prompt) + ones(resp) → (L_actual,)  ←── reconstructed from response_mask (bs, resp_max_len)
                                                              │
                                          ┌───────────────────┘
                                          ▼
_drafter_buffer (bounded deque, per sample on CPU)
  { input_ids: (L_actual,), hidden_states: (L_actual, D), loss_mask: (L_actual,) }
                                          │
                                          ▼
_sample_drafter_batch (pad + shift)
  1. Sample B items from buffer
  2. Pad/truncate to max_len
  3. Shift: base_h=h[:,:-1], target=h[:,1:], ids=ids[:,:-1], mask=mask[:,1:]
                                          │
                                          ▼
Training batch (to FSDPDrafterEngine.train_batch)
  input_ids:            (B, max_len-1)      int64
  attention_mask:       (B, max_len-1)      int64, all 1s
  base_hidden_states:   (B, max_len-1, D)   bf16
  target_hidden_states: (B, max_len-1, D)   bf16
  loss_mask:            (B, max_len-1)      float32
                                          │
                                          ▼
FSDPDrafterEngine.forward_step
  → EAGLE model forward:
    embed(input_ids) + fc(cat(embed, base_h)) → 1 decoder layer → lm_head
  → outputs:
    logits:        (B, max_len-1, V)
    hidden_states: (B, max_len-1, D)
                                          │
                                          ▼
eagle_dual_loss
  vloss: SmoothL1(pred_hidden, target_h), masked → scalar
  ploss: CE(softmax(lm_head(target_h)), log_softmax(logits)), masked → scalar
  loss = w_v * vloss + w_p * ploss → scalar
```

---

## 4. Detailed Design

### 4.1 DrafterConfig

```python
# verl/workers/config/drafter.py

@dataclass
class DrafterConfig:
    """Configuration for EAGLE drafter co-training (V1: train-only)."""

    # Master switch
    enable: bool = False

    # Model
    model_path: Optional[str] = None      # pretrained EAGLE checkpoint (None = init from scratch)
    model_arch: str = "eagle"             # "eagle" only in V1
    base_model_arch: str = "qwen2"        # must match target model

    # Training schedule
    training_interval_steps: int = 10     # train every N RL steps
    train_batch_size: int = 4             # FastRL-aligned default effective sample count
    min_samples: int = 2                  # FastRL-aligned minimum before training

    # Micro-batching: drafter is tiny, single micro-batch is fine
    # When use_dynamic_bsz=False, prepare_micro_batches() reads
    # data["micro_batch_size_per_gpu"] and divides (engine/utils.py:87-88).
    # Must be injected into batch metadata before train_batch().
    # Runtime should clamp this value to <= actual sampled batch size.
    micro_batch_size: int = 4             # micro-batch size upper bound

    # Engine (reuses FSDPEngineConfig)
    engine: FSDPEngineConfig = field(default_factory=lambda: FSDPEngineConfig(
        param_offload=True,
        optimizer_offload=True,
        strategy="fsdp2",
        use_remove_padding=False,         # drafter uses padded inputs (simpler)
        use_dynamic_bsz=False,            # requires micro_batch_size_per_gpu in batch metadata
    ))

    # Optimizer
    optim: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(
        lr=1e-5,
        weight_decay=0.0,
    ))

    # Loss weights
    vloss_weight: float = 1.0             # hidden state prediction (SmoothL1)
    ploss_weight: float = 0.1             # FastRL-aligned distribution matching weight

    # Hidden state buffer
    buffer_size: int = 2000               # max samples in bounded deque
    max_seq_len: int = 512                # FastRL-aligned response-window training length

    # Checkpoint (V1: disabled)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
```

### 4.2 FSDPDrafterEngine (~50 LOC)

Registered in `EngineRegistry`. Inherits from `FSDPEngine` (not `FSDPEngineWithLMHead`).
Overrides exactly 4 methods.

```python
# verl/workers/drafter/engine.py

@EngineRegistry.register(model_type="drafter_model", backend=["fsdp", "fsdp2"])
class FSDPDrafterEngine(FSDPEngine):
    """FSDP engine for EAGLE drafter model.

    Overrides:
      _build_module     — instantiate EAGLE model (not AutoModel)
      prepare_model_inputs  — extract base_hidden_states from batch
      prepare_model_outputs — extract logits + predicted hidden states
      forward_step      — compose the above (same template as FSDPEngineWithLMHead)
    """

    def _build_module(self):
        """Build EAGLE model directly (not via AutoModel.from_pretrained)."""
        arch = self.model_config.hf_config.model_type  # "qwen2" or "llama"
        if arch == "qwen2":
            from verl.workers.drafter.model.qwen2_eagle import Qwen2ForCausalLMEagle
            module = Qwen2ForCausalLMEagle(self.model_config.hf_config)
        elif arch == "llama":
            from verl.workers.drafter.model.llama_eagle import LlamaForCausalLMEagle
            module = LlamaForCausalLMEagle(self.model_config.hf_config)
        else:
            raise ValueError(f"Unsupported EAGLE base arch: {arch}")

        # Optional: load pretrained EAGLE weights
        if self.model_config.local_path:
            state = torch.load(self.model_config.local_path, map_location="cpu")
            module.load_state_dict(state, strict=False)

        return module

    def prepare_model_inputs(self, micro_batch: TensorDict):
        """Extract EAGLE model inputs from the training batch.

        EAGLE forward takes: input_ids, attention_mask, base_model_hidden_states.
        No temperature scaling, no remove-padding gymnastics.
        """
        model_inputs = {
            "input_ids": micro_batch["input_ids"],
            "attention_mask": micro_batch["attention_mask"],
            "base_model_hidden_states": micro_batch["base_hidden_states"],
        }
        output_args = {}  # no post-processing args needed
        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        """Extract drafter predictions for the loss function.

        Returns logits and predicted hidden states (last layer).
        """
        model_output = {
            "logits": output.logits,
        }
        if output.hidden_states is not None:
            model_output["hidden_states"] = output.hidden_states[-1]
        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        """Same template as FSDPEngineWithLMHead.forward_step (transformer_impl.py:991-1022)."""
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            raw_output = self.module(
                **model_inputs,
                use_cache=False,
                output_hidden_states=True,  # always needed for vloss
            )

            model_output = self.prepare_model_outputs(
                output=raw_output, output_args=output_args, micro_batch=micro_batch
            )

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output,
                    data=micro_batch,
                    dp_group=self.get_data_parallel_group(),
                )
            else:
                assert forward_only
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss.detach().item(),
                "metrics": metrics,
            }
            return loss, output
```

**What FSDPDrafterEngine inherits for free from FSDPEngine**:
`_build_fsdp_module`, `_build_optimizer`, `_build_lr_scheduler`, `_build_model_optimizer`,
`initialize`, `train_mode`/`eval_mode` (CPU/GPU offload), `to()`, `optimizer_step` (gradient clipping),
`forward_backward_batch` (micro-batching + backward), `save_checkpoint`/`load_checkpoint`,
`get_per_tensor_param` (weight extraction for future rollout sync).

### 4.3 Hidden State Extraction from Actor Forward Pass

Hidden states are collected during `compute_log_prob()`. The approach threads an `output_hidden_states`
flag through the data TensorDict, read in `forward_step` to pass to the HF model. Hidden states are
then intercepted locally inside the worker before returning output to the trainer.

**Engine layer** — modify `FSDPEngineWithLMHead.forward_step()` (transformer_impl.py:991):

```python
# Read flag from micro_batch (default False — no perf impact when drafter disabled)
output_hidden_states = tu.get_non_tensor_data(
    data=micro_batch, key="output_hidden_states", default=False
)

raw_output = self.module(
    **model_inputs,
    use_cache=False,
    output_hidden_states=output_hidden_states,   # NEW: conditionally request
)

model_output = self.prepare_model_outputs(
    output=raw_output, output_args=output_args, micro_batch=micro_batch
)

# NEW: attach hidden states to model_output if requested
# Assumes ulysses_sequence_parallel_size == 1 (no SP gather needed).
if output_hidden_states and raw_output.hidden_states is not None:
    hs = raw_output.hidden_states[-1]               # (1, total_nnz, D) in rmpad path
    use_remove_padding = tu.get_non_tensor_data(
        data=micro_batch, key="use_remove_padding", default=True
    )
    if use_remove_padding:
        hs_rmpad = hs.squeeze(0)                     # (total_nnz, D)
        cu_seqlens = micro_batch["input_ids"].offsets()
        hs = torch.nested.nested_tensor_from_jagged(hs_rmpad, cu_seqlens)
    model_output["hidden_states"] = hs.detach()
```

**Why this works**: `postprocess_batch_func` (engine/utils.py:118-135) iterates ALL keys in
`model_output`, unbinds nested tensors per micro-batch, and reassembles via
`torch.nested.as_nested_tensor(..., layout=torch.jagged)`. Hidden states in jagged format
survive this pipeline transparently. Verified by subagent analysis of the full call chain.

**Worker layer** — `ActorRolloutRefWorker.compute_log_prob()` (engine_workers.py:590):

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
def compute_log_prob(self, data: TensorDict) -> TensorDict:
    # Signal hidden state extraction
    if self._is_drafter:
        tu.assign_non_tensor(data, output_hidden_states=True)

    output = self.actor.infer_batch(data)

    # Extract and buffer hidden states locally (never returned to driver)
    if self._is_drafter and output is not None:
        hidden_states = tu.pop(output, "hidden_states", default=None)
        if hidden_states is not None:
            self._collect_drafter_sample(data, hidden_states)

    return output.cpu() if output is not None else None
```

### 4.4 Hidden State Buffering

Two private methods on `ActorRolloutRefWorker` plus a bounded `deque`.

```python
def _collect_drafter_sample(self, batch: TensorDict, hidden_states):
    """Buffer per-sample hidden states from actor forward pass.

    Shape alignment:
      - `batch["input_ids"]` and `hidden_states` are jagged nested tensors
        (variable-length per sample, remove-padded). Per-sample shape: (L_actual,)
        where L_actual = prompt_actual + response_actual.
      - `batch["loss_mask"]` = `response_mask` — a padded (bs, response_max_len)
        tensor covering ONLY the response portion (NOT full sequence!).
        Set by padding.py:71: `data["loss_mask"] = data["response_mask"]`.
        `response_mask` is computed as `attention_mask[:, -response_length:]`
        (ray_trainer.py:124-127).

    We reconstruct a full-sequence loss_mask by prepending prompt-length zeros
    to the response_mask values, matching FastRL's collect_online_data logic
    (eagle_background_trainer.py:247-253).
    """
    input_ids_nested = batch["input_ids"]          # nested tensor
    hidden_list = hidden_states.unbind()           # list of (L_actual, D)
    input_ids_list = input_ids_nested.unbind()     # list of (L_actual,)

    # loss_mask = response_mask: (bs, response_max_len) — response portion ONLY
    # NOT (bs, full_seq_len)! Cannot index by total actual length.
    response_mask_padded = batch["loss_mask"]       # (bs, response_max_len)

    # Get per-sample actual lengths from nested offsets
    offsets = input_ids_nested.offsets()
    lengths = offsets.diff().tolist()

    for i, (ids_i, hidden_i, length_i) in enumerate(
        zip(input_ids_list, hidden_list, lengths)
    ):
        # response_mask[i] is (response_max_len,): [1,..,1, 0,..,0]
        # Number of 1s = actual response tokens (including EOS)
        resp_mask_i = response_mask_padded[i]           # (response_max_len,)
        response_actual_i = int(resp_mask_i.sum().item())
        prompt_actual_i = length_i - response_actual_i

        # Reconstruct full-sequence mask: 0s for prompt, 1s for response
        full_mask_i = torch.cat([
            torch.zeros(prompt_actual_i, dtype=torch.float32),
            torch.ones(response_actual_i, dtype=torch.float32),
        ])  # (L_actual,)

        assert full_mask_i.shape[0] == ids_i.shape[0] == hidden_i.shape[0], (
            f"Shape mismatch: mask={full_mask_i.shape[0]}, "
            f"ids={ids_i.shape[0]}, hidden={hidden_i.shape[0]}"
        )

        self._drafter_buffer.append({
            "input_ids": ids_i.detach().cpu(),
            "hidden_states": hidden_i.detach().cpu(),
            "loss_mask": full_mask_i.cpu(),
        })


def _sample_drafter_batch(self, batch_size: int) -> TensorDict:
    """Sample from buffer, apply EAGLE shift alignment, return padded batch.

    Shift rule (from eagle_background_trainer.py:443-456):
        base_hidden    = hidden_states[:, :-1]
        target_hidden  = hidden_states[:, 1:]
        input_ids      = input_ids[:, :-1]
        loss_mask      = loss_mask[:, 1:]
    """
    samples = random.sample(
        list(self._drafter_buffer),
        min(batch_size, len(self._drafter_buffer)),
    )

    max_len = min(
        max(s["input_ids"].shape[0] for s in samples),
        self.config.drafter.max_seq_len,
    )

    def select_response_window(sample):
        """FastRL-aligned windowing: keep response tokens in the selected span."""
        ids = sample["input_ids"]
        hidden = sample["hidden_states"]
        mask = sample["loss_mask"]

        full_len = ids.shape[0]
        window_len = min(full_len, max_len)

        # response-aware window (same idea as eagle_background_trainer.py:359-370)
        nonzero = torch.nonzero(mask > 0).flatten()
        if nonzero.numel() > 0:
            resp_start = int(nonzero[0].item())
            resp_end = int(nonzero[-1].item()) + 1
            start = max(0, min(resp_start, full_len - window_len))
            if resp_end - start > window_len:
                start = resp_end - window_len
            end = min(full_len, start + window_len)
        else:
            # fallback: use tail window when no response tokens are marked
            start = max(0, full_len - window_len)
            end = full_len

        return ids[start:end], hidden[start:end], mask[start:end]

    def pad_to_max(t, pad_value=0):
        if t.shape[0] == max_len:
            return t
        pad_size = max_len - t.shape[0]
        if t.dim() == 1:
            return F.pad(t, (0, pad_size), value=pad_value)
        else:
            return F.pad(t, (0, 0, 0, pad_size), value=pad_value)

    windowed = [select_response_window(s) for s in samples]

    # Track actual lengths BEFORE padding (needed for attention_mask)
    actual_lens = [ids_i.shape[0] for ids_i, _, _ in windowed]

    ids = torch.stack([pad_to_max(ids_i, pad_value=0) for ids_i, _, _ in windowed])
    hidden = torch.stack([pad_to_max(hidden_i, pad_value=0) for _, hidden_i, _ in windowed])
    mask = torch.stack([pad_to_max(mask_i, pad_value=0) for _, _, mask_i in windowed])

    # Build attention_mask: 1 for real tokens, 0 for zero-padded positions.
    # Using all-1s would let the EAGLE decoder attend to padding tokens,
    # corrupting representations of real tokens via attention.
    attn_mask = torch.zeros(len(samples), max_len, dtype=torch.int64)
    for i, alen in enumerate(actual_lens):
        attn_mask[i, :alen] = 1

    # Apply shift alignment
    return tu.get_tensordict(
        tensor_dict={
            "input_ids":            ids[:, :-1],
            "attention_mask":       attn_mask[:, :-1],
            "base_hidden_states":   hidden[:, :-1],
            "target_hidden_states": hidden[:, 1:],
            "loss_mask":            mask[:, 1:],
        },
        non_tensor_dict={},
    )
```

### 4.5 Frozen Layer Refresh (FSDP2 Local Shard Copy)

The drafter holds its own copies of `lm_head` and `embed_tokens` (separate FSDP2 ownership).
Before each drafter training step, copy weights from actor using `_local_tensor.copy_()`.

**Precondition**: Actor and drafter must use the same FSDP device mesh for these layers.

```python
@torch.no_grad()
def _refresh_drafter_frozen_layers(self):
    """Copy lm_head and embed_tokens from actor to drafter (both on CPU).

    Uses DTensor local shard copy — zero NCCL communication.
    Proven pattern: fsdp2_utils.py:117 in the codebase.
    """
    _copy_fsdp2_submodule(
        self.actor.engine.module.lm_head,
        self.drafter.engine.module.lm_head,
    )
    _copy_fsdp2_submodule(
        self.actor.engine.module.model.embed_tokens,
        self.drafter.engine.module.model.embed_tokens,
    )


@torch.no_grad()
def _copy_fsdp2_submodule(src_mod: nn.Module, dst_mod: nn.Module):
    """Copy parameters between identically-sharded FSDP2 sub-modules."""
    for (s_name, s_param), (d_name, d_param) in zip(
        src_mod.named_parameters(),
        dst_mod.named_parameters(),
    ):
        assert s_name == d_name, f"Name mismatch: {s_name} vs {d_name}"
        assert s_param.shape == d_param.shape, f"Shape mismatch for {s_name}"

        if isinstance(s_param, DTensor) and isinstance(d_param, DTensor):
            assert s_param._spec.placements == d_param._spec.placements, (
                f"Sharding mismatch for {s_name}"
            )
            d_param._local_tensor.copy_(s_param._local_tensor)
        elif not isinstance(s_param, DTensor) and not isinstance(d_param, DTensor):
            d_param.data.copy_(s_param.data)
        else:
            raise RuntimeError(f"DTensor type mismatch for {s_name}")
```

### 4.6 EAGLE Loss Function

```python
# verl/workers/drafter/losses.py

def eagle_dual_loss(model_output, data, dp_group, lm_head, w_v=1.0, w_p=1.0):
    """EAGLE drafter training loss.

    Combines:
      - vloss: SmoothL1 between drafter-predicted and target hidden states
      - ploss: Cross-entropy between drafter logits and target distribution
               (target_logits computed on-the-fly via lm_head)

    Matches eagle_background_trainer.py:550-578.
    """
    logits = model_output["logits"]              # (B, L-1, V)
    pred_hidden = model_output["hidden_states"]  # (B, L-1, D)

    target_hidden = data["target_hidden_states"] # (B, L-1, D)
    loss_mask = data["loss_mask"]                 # (B, L-1)
    num_valid = loss_mask.sum().clamp(min=1)

    # vloss: hidden state prediction (SmoothL1)
    vloss = F.smooth_l1_loss(pred_hidden, target_hidden, reduction="none")
    vloss = torch.mean(vloss, dim=-1)            # (B, L-1)
    vloss = (loss_mask * vloss).sum() / num_valid

    # ploss: distribution matching (cross-entropy H(target_p, drafter_p))
    with torch.no_grad():
        target_logits = lm_head(target_hidden)   # (B, L-1, V)
        target_p = F.softmax(target_logits, dim=-1)

    out_logp = F.log_softmax(logits, dim=-1)
    ploss = -(target_p * out_logp).sum(dim=-1)   # (B, L-1)
    ploss = (loss_mask * ploss).sum() / num_valid

    loss = w_v * vloss + w_p * ploss

    metrics = {
        "drafter/loss": loss.detach().item(),
        "drafter/vloss": vloss.detach().item(),
        "drafter/ploss": ploss.detach().item(),
    }
    return loss, metrics
```

### 4.7 Worker-Level `update_drafter` Method

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def update_drafter(self) -> TensorDict:
    """Train the drafter model on collected hidden states.

    Uses Dispatch.ONE_TO_ALL (no data dispatch needed — trains from local buffer).
    """
    if len(self._drafter_buffer) < self.config.drafter.min_samples:
        return tu.get_tensordict(
            tensor_dict={},
            non_tensor_dict={"metrics": {
                "drafter/skipped": 1.0,
                "drafter/buffer_size": float(len(self._drafter_buffer)),
            }},
        )

    # Refresh frozen layers from actor (both on CPU at this point)
    self._refresh_drafter_frozen_layers()

    # Build drafter training batch from buffer
    drafter_batch = self._sample_drafter_batch(
        batch_size=self.config.drafter.train_batch_size
    )

    # Metadata required by prepare_micro_batches() when use_dynamic_bsz=False
    # (engine/utils.py:87): reads data["micro_batch_size_per_gpu"] to chunk batch.
    actual_batch_size = len(drafter_batch)
    micro_bsz = min(self.config.drafter.micro_batch_size, actual_batch_size)
    drafter_batch["micro_batch_size_per_gpu"] = torch.tensor(micro_bsz)

    # Metadata for MFU accounting
    global_token_num = drafter_batch["attention_mask"].sum(dim=-1).tolist()
    tu.assign_non_tensor(drafter_batch, global_token_num=global_token_num)
    tu.assign_non_tensor(drafter_batch, use_dynamic_bsz=False)

    # Train (BaseEngineCtx handles GPU load/offload automatically)
    output = self.drafter.train_batch(data=drafter_batch)

    if output is not None:
        return output.cpu()
    else:
        return tu.get_tensordict(
            tensor_dict={},
            non_tensor_dict={"metrics": {"drafter/empty_output": 1.0}},
        )
```

### 4.8 Orchestrator Integration

**In `RayPPOTrainer.fit()`** — inject at ray_trainer.py between line 1614 and 1617:

```python
# After update_actor, before update_weights:

if (self.enable_drafter_training
    and self.global_steps % self.config.drafter.training_interval_steps == 0):
    with marked_timer("update_drafter", timing_raw, color="purple"):
        # Dispatch.ONE_TO_ALL with blocking=True (default):
        # func_generator (ray/base.py:54-55) already calls ray.get() before
        # returning. No .get() needed — result is already materialized.
        drafter_outputs = self.actor_rollout_wg.update_drafter()

        # Average metrics across all DP workers (each trains independently
        # on its local buffer — picking only worker 0 is non-representative).
        if isinstance(drafter_outputs, list) and len(drafter_outputs) > 0:
            all_metrics = [
                tu.get(o, "metrics", default={}) for o in drafter_outputs
            ]
            drafter_metrics = {}
            if all_metrics and all_metrics[0]:
                for key in all_metrics[0]:
                    vals = [m.get(key, 0.0) for m in all_metrics]
                    drafter_metrics[key] = sum(vals) / len(vals)
        else:
            drafter_metrics = tu.get(drafter_outputs, "metrics", default={})
        metrics.update(drafter_metrics)
```

**In `RayPPOTrainer.__init__()`** — add config flag:

```python
drafter_config = self.config.actor_rollout_ref.get("drafter", {})
self.enable_drafter_training = drafter_config.get("enable", False)
```

### 4.9 Drafter Initialization in `ActorRolloutRefWorker.init_model()`

After building actor/ref/rollout (engine_workers.py, after line 581):

```python
# 5. Build drafter training engine (NEW)
self._is_drafter = bool(self.config.get("drafter", {}).get("enable", False))
if self._is_drafter:
    # Codex fix #5: omega_conf_to_dataclass without dataclass_type asserts
    # "_target_" in config (config.py:42-47). Pass dataclass_type explicitly.
    from verl.workers.config.drafter import DrafterConfig
    drafter_cfg = omega_conf_to_dataclass(
        self.config.drafter, dataclass_type=DrafterConfig
    )

    # Codex fix #4: HFModelConfig.__post_init__ (model.py:144-175) does:
    #   1. self.local_path = copy_to_local(self.path) — requires path != MISSING
    #   2. self.hf_config = AutoConfig.from_pretrained(...) — overwrites any
    #      pre-built hf_config we pass in
    # Solution: construct with actor's path (so copy_to_local/AutoConfig work),
    # then mutate hf_config AFTER __post_init__ completes.
    import copy
    drafter_model_config = HFModelConfig(
        path=model_config.path,            # actor model path (required by __post_init__)
        load_tokenizer=False,              # drafter doesn't need tokenizer
        use_remove_padding=False,          # drafter uses padded inputs
    )
    # Override hf_config for EAGLE (1 decoder layer, untied embeddings)
    drafter_model_config.hf_config = copy.deepcopy(model_config.hf_config)
    drafter_model_config.hf_config.num_hidden_layers = 1
    drafter_model_config.hf_config.tie_word_embeddings = False
    # Override local_path if pretrained EAGLE weights exist
    if drafter_cfg.model_path:
        drafter_model_config.local_path = drafter_cfg.model_path

    drafter_training_config = TrainingWorkerConfig(
        model_type="drafter_model",            # registered FSDPDrafterEngine
        model_config=drafter_model_config,
        engine_config=drafter_cfg.engine,
        optimizer_config=drafter_cfg.optim,
        checkpoint_config=drafter_cfg.checkpoint,
    )

    self.drafter = TrainingWorker(config=drafter_training_config)
    self.drafter.reset()

    # Bind loss function with frozen lm_head reference
    lm_head = self.drafter.engine.module.lm_head
    self.drafter.set_loss_fn(partial(
        eagle_dual_loss,
        lm_head=lm_head,
        w_v=drafter_cfg.vloss_weight,
        w_p=drafter_cfg.ploss_weight,
    ))

    # Initialize hidden state buffer
    self._drafter_buffer = deque(maxlen=drafter_cfg.buffer_size)
else:
    self.drafter = None
    self._drafter_buffer = None
```

---

## 5. GPU Memory Timeline

```
Time ─────────────────────────────────────────────────────────────────────────►

│← RL step N ───────────────────────────────────────────────────→│

┌────────────┐
│  Rollout   │  generate_sequences()
│  (on GPU)  │
└────────────┘
              ┌──────────────┐
              │    Actor     │  compute_log_prob()
              │   (on GPU)   │  + extract hidden states → _drafter_buffer (CPU)
              └──────────────┘
                              ┌──────────┐
                              │  Actor   │  update_actor()
                              │ (on GPU) │
                              └──────────┘
                                          ┌──────────┐
                                          │ Drafter  │  update_drafter()
                                          │ (on GPU) │  (~0.5% of actor size)
                                          └──────────┘
                                                      ┌────────────┐
                                                      │ Weight Sync│  actor → rollout
                                                      │ (no drafter│  (drafter sync deferred)
                                                      │  in V1)    │
                                                      └────────────┘
```

`BaseEngineCtx` automatically loads drafter to GPU on `train_mode().__enter__` and offloads on `__exit__`.

---

## 6. File Layout

```
migration_targets/verl/verl/
├── workers/
│   ├── engine_workers.py              # MODIFIED: _is_drafter, init drafter, compute_log_prob,
│   │                                  #   update_drafter, _collect_drafter_sample,
│   │                                  #   _sample_drafter_batch, _refresh_drafter_frozen_layers
│   ├── config/
│   │   └── drafter.py                 # NEW: DrafterConfig
│   ├── drafter/                       # NEW directory
│   │   ├── __init__.py
│   │   ├── engine.py                  # FSDPDrafterEngine (EngineRegistry, ~50 LOC)
│   │   ├── losses.py                  # eagle_dual_loss (~40 LOC)
│   │   └── model/                     # EAGLE model architectures (copied from fastrl)
│   │       ├── __init__.py
│   │       ├── qwen2_eagle.py
│   │       └── llama_eagle.py
│   └── engine/fsdp/
│       └── transformer_impl.py        # MODIFIED: output_hidden_states flag in forward_step
│                                      #   (~15 lines added to FSDPEngineWithLMHead.forward_step)
├── trainer/
│   └── ppo/
│       └── ray_trainer.py             # MODIFIED: update_drafter hook in fit() (~10 lines)
└── config/
    └── ppo_trainer.yaml               # MODIFIED: drafter section added
```

**Estimated total**: ~250 LOC new code + ~25 LOC modifications + model files (copied).

---

## 7. YAML Configuration

```yaml
actor_rollout_ref:
  actor: ...
  rollout: ...
  ref: ...

  drafter:                               # NEW section
    enable: false                        # master switch
    model_path: null                     # pretrained EAGLE ckpt path
    model_arch: eagle
    base_model_arch: qwen2
    training_interval_steps: 10
    train_batch_size: 4
    micro_batch_size: 4                  # upper bound; clamped to actual sampled batch size at runtime
    min_samples: 2
    vloss_weight: 1.0
    ploss_weight: 0.1
    buffer_size: 2000
    max_seq_len: 512
    engine:
      strategy: fsdp2
      param_offload: true
      optimizer_offload: true
      use_remove_padding: false
      use_dynamic_bsz: false             # requires micro_batch_size_per_gpu in batch metadata
    optim:
      lr: 1e-5
```

---

## 8. Implementation Phases

### Phase 1: Foundation (~100 LOC)
1. Add `DrafterConfig` to `verl/workers/config/drafter.py`
2. Copy EAGLE model files (`qwen2_eagle.py`, `llama_eagle.py`) to `verl/workers/drafter/model/`
3. Implement `FSDPDrafterEngine` in `verl/workers/drafter/engine.py`
4. Implement `eagle_dual_loss` in `verl/workers/drafter/losses.py`

### Phase 2: Worker Integration (~120 LOC)
5. Add `output_hidden_states` flag to `FSDPEngineWithLMHead.forward_step()` (~15 lines)
6. Add `_is_drafter` + drafter init to `ActorRolloutRefWorker.init_model()`
7. Add `_collect_drafter_sample()`, `_sample_drafter_batch()` to worker
8. Add `_refresh_drafter_frozen_layers()` + `_copy_fsdp2_submodule()` to worker
9. Add `update_drafter()` method to worker

### Phase 3: Orchestrator (~30 LOC)
10. Add `enable_drafter_training` flag to `RayPPOTrainer.__init__()`
11. Wire `update_drafter()` into `fit()` loop
12. Add drafter metrics logging

### Phase 4: Validation
13. Unit test `eagle_dual_loss`
14. Unit test `_sample_drafter_batch()` shift alignment
15. Integration test: run RL loop with drafter enabled, verify loss decreases
16. Profile: drafter training overhead per RL step + CPU memory usage

---

## 9. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `Dispatch.ONE_TO_ALL` for `update_drafter` | Drafter trains from local buffer, not dispatched data. Avoids wasteful data transfer. No `set_dispatch_collect` needed. |
| `FSDPDrafterEngine(FSDPEngine)` with 4 overrides | EAGLE model has non-standard forward signature (`base_model_hidden_states`). Cannot reuse `FSDPEngineWithLMHead`. Override `_build_module`, `prepare_model_inputs`, `prepare_model_outputs`, `forward_step`. |
| `_local_tensor.copy_()` for frozen layer refresh | FSDP2 `load_state_dict` on sub-modules returns DTensors — cross-engine `load_state_dict` is undefined. Local shard copy is zero-communication, proven in `fsdp2_utils.py:117`. |
| `use_remove_padding=False` for drafter | Drafter batch is pre-padded in `_sample_drafter_batch`. Avoids nested tensor complexity in the tiny EAGLE model's engine path. |
| Hidden states as jagged nested tensors | Required by `postprocess_batch_func` (engine/utils.py:128) which unbinds and reassembles all `model_output` keys. Jagged format survives the micro-batch aggregation pipeline. |
| Padded batching (B>1) instead of FastRL's concat (B=1) | FastRL concatenates N<=4 samples along seq dim into `(1, T, D)` with max 512-token windows. RFC uses standard padded batching `(B, max_len, D)` — simpler, compatible with FSDP micro-batching, avoids cross-sample attention leakage. Tradeoff is slightly more padding waste. See Section 3.0.2. |
| V1: no SGLang weight sync | `is_draft_model` routing is a FastRL-specific SGLang patch (3 files, ~20 LOC). Upstream SGLang lacks it. Defer to V2. |

---

## 10. Deferred: SGLang Drafter Weight Sync (V2 TODO)

**Blocker**: The `sgl_update_weights()` function accepts `is_draft_model: bool` only in FastRL's vendored SGLang. Upstream SGLang (0.5.2 through main) lacks:
1. `is_draft_model` parameter in `sgl_update_weights()` (`weight_sync/utils.py`)
2. `is_draft_model` field in `UpdateWeightsFromTensorReqInput` (`managers/io_struct.py`)
3. Per-request draft/target routing in `scheduler_update_weights_mixin.py`
4. `is_draft_model` forwarding in the HTTP adapter

**V2 plan**:
1. Contribute `is_draft_model` routing upstream to SGLang (3 files, ~20 LOC patch)
2. Add `update_drafter_weights()` to `BaseRollout` (default no-op)
3. Implement in SGLang `ServerAdapter`
4. Inject into `ActorRolloutRefWorker.update_weights()` between actor CPU offload (line 668) and KV cache resume (line 670)

---

## 11. Risk Register

| Risk | Mitigation |
|------|------------|
| `_local_tensor` is a private PyTorch API | Already used in `fsdp2_utils.py:60,113,117`. Add assertions on placements. |
| CPU memory for hidden state buffer (~33 GB at 2000 samples) | Start with `buffer_size=500`, profile, increase if needed. Consider response-only windowed storage in V2. |
| FSDP2 device mesh mismatch between actor and drafter | Actor and drafter share the same worker process → same NCCL world → use same mesh. Verify at init time. |

---

## 12. Resolved Issues (from Codex review)

| # | Severity | Issue | Fix (section) |
|---|----------|-------|---------------|
| 1 | Critical | `micro_batch_size_per_gpu` never set — `prepare_micro_batches` (utils.py:87) reads it from data when `use_dynamic_bsz=False`, crashes on `None` | Added `micro_batch_size` to `DrafterConfig` (4.1), injected into batch metadata via `data["micro_batch_size_per_gpu"]` in `update_drafter` (4.7), and clamped to `<= actual_batch_size` |
| 2 | Critical | `loss_mask` alignment — `response_mask` is `(bs, response_max_len)` (response portion ONLY, not full sequence), but `hidden_states` is a nested tensor of full-sequence length `(L_actual, D)`. Slicing `response_mask[i, :L_actual]` causes out-of-bounds access since `L_actual > response_max_len`. | Rewrote `_collect_drafter_sample` (4.4) to reconstruct full-sequence mask via `zeros(prompt_actual) + ones(response_actual)`, matching FastRL's `collect_online_data` (eagle_background_trainer.py:247-253) |
| 3 | Critical | `.get()` on blocking `Dispatch.ONE_TO_ALL` output — `func_generator` (ray/base.py:54-55) already calls `ray.get()` for blocking dispatches | Removed `.get()` call in orchestrator integration (4.8) |
| 4 | High | `HFModelConfig.__post_init__` (model.py:144-175) overwrites `local_path` via `copy_to_local(self.path)` and rebuilds `hf_config` from `AutoConfig.from_pretrained` | Changed to construct with actor `path`, then mutate `hf_config` after `__post_init__` completes (4.9) |
| 5 | High | `omega_conf_to_dataclass` (config.py:42-47) without `dataclass_type` asserts `_target_` in config | Added `dataclass_type=DrafterConfig` parameter (4.9) |
| 6 | Medium | Metrics from worker 0 non-representative — each worker trains independently on local buffer | Changed to average metrics across all DP workers (4.8) |
