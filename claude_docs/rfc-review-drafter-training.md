# RFC Review: Drafter Co-training Integration for verl Migration Target

**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-18
**RFC Under Review**: `claude_docs/rfc-drafter-training.md`
**Related Document**: `claude_docs/migration-plan.md`

---

## Overall Assessment

This is a **well-structured RFC** that correctly identifies the key architectural patterns in the migration target and leverages them effectively. The core insight — making drafter training a first-class `TrainingWorker` in the sequential RL loop instead of a background thread — is sound and dramatically simplifies the system (~200 LOC new vs ~1500 LOC replaced).

However, there are **several technical gaps** that would lead to bugs or architectural friction during implementation. The most critical involve: hidden state flow through the micro-batch pipeline, the drafter engine's custom forward path, and FSDP2 sub-module weight copying.

---

## 1. Critical Issues (P0 — would cause bugs or failures)

### 1.1 Missing drafter `forward_step` / `prepare_model_inputs` / `prepare_model_outputs`

**RFC Section**: 6 (EngineRegistry Integration)

The RFC registers `FSDPDrafterEngine(FSDPEngine)` and mentions overriding `_build_model`. But the EAGLE model has a fundamentally different forward signature from a standard LLM:

```python
# Standard LLM (FSDPEngineWithLMHead.forward_step at transformer_impl.py:991):
raw_output = self.module(input_ids=..., attention_mask=..., position_ids=..., use_cache=False)

# EAGLE model (from qwen2_eagle.py forward):
outputs = self.model(input_ids=..., attention_mask=..., base_model_hidden_states=..., output_hidden_states=True)
```

The EAGLE model takes `base_model_hidden_states` as input and returns both `logits` and `hidden_states`. The existing `FSDPEngineWithLMHead.prepare_model_inputs()` (`transformer_impl.py:741-879`) doesn't know about `base_model_hidden_states`, and `prepare_model_outputs()` (`transformer_impl.py:881-989`) only extracts `log_probs` / `entropy`.

The `FSDPDrafterEngine` needs its own:
- **`prepare_model_inputs()`** — pass `base_hidden_states` and `input_ids` from the drafter batch to the EAGLE model
- **`prepare_model_outputs()`** — return `{"logits": raw_output.logits, "hidden_states": raw_output.hidden_states[-1]}` so the loss function can consume them
- **`forward_step()`** — or at least ensure the inherited one works with the custom prepare methods

Without these, the engine will try to use the standard LLM forward path and fail.

**Recommendation**: Add the full `FSDPDrafterEngine` implementation (prepare_model_inputs, prepare_model_outputs, and optionally forward_step) to the RFC, not just `_build_model`.

---

### ~~1.2 Hidden states lost in micro-batch aggregation pipeline~~ [RETRACTED — Downgraded to P2]

**RFC Section**: 3.4 (Hidden State Extraction from Actor Forward Pass)

**Original claim**: Hidden states would be "lost" because `postprocess_batch_func` can't aggregate them.

**Correction**: This claim was overstated. `postprocess_batch_func` (`engine/utils.py:118-129`) iterates **all keys** in `model_output`, not a hardcoded set. Any key added to `model_output` in `forward_step()` — including `hidden_states` — is automatically aggregated across micro-batches via the same `unbind()` → `as_nested_tensor(layout=torch.jagged)` path used by `log_probs` and `entropy`. No custom aggregation is needed.

**Actual remaining risks (P2)**:
1. **Shape/type consistency**: Hidden states are 2D `(seq_len, D)` per sample vs 1D `(seq_len,)` for `log_probs`. Both are handled correctly by `unbind()` + `as_nested_tensor()`, but worth a unit test.
2. **Ulysses SP gather**: If SP > 1, hidden states need `gather_outputs_and_unpad()` before jagged conversion (the RFC already handles this).
3. **Transient GPU memory**: Keeping `(total_nnz, D)` hidden states through the micro-batch pipeline until consumed in `_collect_drafter_sample` adds transient GPU pressure during `compute_log_prob`.

---

### 1.3 FSDP2 sub-module `load_state_dict` in `_refresh_drafter_frozen_layers`

**RFC Section**: 3.5 (Worker-Level Methods) / 10 (Open Questions — Resolved)

```python
def _refresh_drafter_frozen_layers(self):
    actor_module = self.actor.engine.module       # FSDPModule
    drafter_module = self.drafter.engine.module    # FSDPModule
    drafter_module.lm_head.load_state_dict(actor_module.lm_head.state_dict())
    drafter_module.model.embed_tokens.load_state_dict(
        actor_module.model.embed_tokens.state_dict()
    )
```

Under FSDP2, `self.actor.engine.module` is an `FSDPModule`. Calling `.state_dict()` on a sub-module (e.g., `module.lm_head`) may return `DTensor` objects (sharded), not regular tensors. The `load_state_dict` between two FSDP2-wrapped sub-modules with DTensor parameters has subtle behavior — it depends on whether both modules are on the same device mesh and whether the FSDP sharding is compatible.

The RFC says "both engines are on CPU" at this point, but FSDP2 DTensors on CPU still carry their mesh/placement metadata. A direct `load_state_dict` between DTensors from different FSDP2 modules may fail silently or produce incorrect results.

For reference, the original FastRL implementation avoids this entirely by sharing via direct Python reference assignment **before** FSDP wrapping (`fsdp_workers.py:558-571`):
```python
drafter_module.lm_head = base_module.lm_head  # direct reference, pre-FSDP
```

**Recommendation**: Prototype this early. Possible approaches:
1. Materialize full tensors via `full_tensor()` before copying
2. Use `fsdp2_load_full_state_dict()` (which the codebase already uses)
3. Do the copy at the raw parameter level: iterate `lm_head.parameters()` and do `drafter_param.data.copy_(actor_param.data)` directly
4. If all else fails, share by reference during init (pre-FSDP), which constrains the actor and drafter to use the same device mesh for those layers

---

### 1.4 `update_drafter` dispatch mode wastes data transfer

**RFC Section**: 3.5 / 3.7

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter"))
def update_drafter(self, data: TensorDict) -> TensorDict:
    # ... ignores `data`, samples from self._drafter_buffer instead ...
```

And in the trainer (Section 3.7):
```python
output = self.actor_rollout_wg.update_drafter(batch_td)  # sends entire PPO batch
```

`make_nd_compute_dataproto_dispatch_fn` splits `batch_td` across DP ranks and sends chunks to each worker. But `update_drafter` ignores `data` entirely — it samples from the local `_drafter_buffer`. This sends the entire PPO batch to each worker for no reason. With large batches (hundreds of MB per worker), this wastes network bandwidth and memory.

**Recommendation**: Use `Dispatch.ONE_TO_ALL` instead, since no input data needs to be dispatched. Change the trainer wrapper to pass an empty trigger:
```python
def _update_drafter(self, batch: DataProto) -> DataProto:
    output = self.actor_rollout_wg.update_drafter()  # no data arg needed
    ...
```

---

## 2. Design Concerns (P1 — correctness at risk, need clarification)

### 2.1 Padded vs. no-padding format mismatch for drafter training data

**RFC Section**: 3.3 / 3.4

The actor `compute_log_prob` path operates in **no-padding mode** (jagged nested tensors). Hidden states extracted here will be in jagged format. But `_sample_drafter_batch` (Section 3.3) produces standard **padded** tensors:

```python
hidden_states = torch.stack([pad_or_truncate(s["hidden_states"]) for s in samples])  # (B, L, D)
```

The drafter engine will receive this padded batch. But `FSDPEngine.forward_backward_batch()` calls `prepare_micro_batches()` which may expect specific data formats depending on `use_remove_padding`. If the drafter engine is configured with `use_remove_padding=False`, the standard FSDP path expects padded inputs with `attention_mask` — which the RFC provides.

But this is never stated explicitly. Without it, the drafter engine could inherit the actor's `use_remove_padding=True` setting and break.

**Recommendation**: Explicitly document in the RFC and `DrafterConfig` that the drafter engine uses `use_remove_padding=False` and padded inputs. This is different from the actor engine's no-padding path but simpler for the tiny drafter model.

---

### 2.2 CPU memory budget for hidden state buffer

**RFC Section**: 3.3

A `buffer_size=2000` with average 2K-token sequences at `hidden_dim=4096` in bf16:

```
2000 samples * 2048 tokens * 4096 dim * 2 bytes/element = ~33 GB per worker
```

With 8 GPUs per node each holding their own buffer: **~264 GB of CPU RAM**.

The original FastRL implementation (`eagle_background_trainer.py:275-456`) uses **windowed selection** — extracting only a ~512-token window centered on response tokens — which reduces storage by ~4x.

The RFC's `_collect_drafter_sample` stores full-length sequences. Although `_sample_drafter_batch` truncates to `max_seq_len` at sample time, the buffer itself holds full sequences.

**Recommendation**: Either:
1. Adopt the windowed extraction from the original implementation (only store response-portion hidden states + small context window), or
2. Add an explicit note about memory budget and recommend smaller `buffer_size` for large models, or
3. Truncate/window at collection time in `_collect_drafter_sample` rather than at sample time

Since `loss_mask` zeros out prompt tokens anyway, storing prompt hidden states is wasteful.

---

### 2.3 Nested tensor indexing in `_collect_drafter_sample`

**RFC Section**: 3.3

```python
for i in range(batch.shape[0]):
    input_ids_i = batch["input_ids"][i].detach().cpu()
    hidden_i = hidden_states[i].detach().cpu()
```

If `batch["input_ids"]` is a jagged nested tensor (from the no-padding path after Section 3.4's conversion), `[i]` indexing returns the i-th sequence as a 1D tensor of variable length. This should work with PyTorch's nested tensor API, but:

1. `batch.shape[0]` on a TensorDict with nested tensors may behave differently than expected
2. The `hidden_states` output from Section 3.4 is converted to `torch.nested.nested_tensor_from_jagged(hs_rmpad, cu_seqlens)` — `[i]` indexing on this returns a variable-length tensor

**Recommendation**: Add a short comment confirming the expected tensor types and that `[i]` indexing returns a 1D tensor of the sample's actual length. Consider using `unbind()` for clearer semantics:
```python
for input_ids_i, hidden_i in zip(batch["input_ids"].unbind(), hidden_states.unbind()):
    ...
```

---

### 2.4 `response_mask` field availability

**RFC Section**: 3.3

```python
response_mask_i = batch["response_mask"][i].detach().cpu().to(torch.float32)
```

`compute_log_prob` receives the batch after `left_right_2_no_padding()` conversion (`ray_trainer.py:1204`). You should verify that `response_mask` survives this conversion and is available under that exact name. In the no-padding path, the masking may use different field names (e.g., `loss_mask`).

Looking at `ray_trainer.py:1206`: `tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)` — only these two non-tensor fields are added. The actual tensor fields come from `batch.to_tensordict()` which depends on what `generate_sequences()` produced. The field might be `loss_mask`, `response_mask`, or something else.

**Recommendation**: Trace the exact field names available in the TensorDict at the `compute_log_prob` call site. Document which field provides the response-portion mask. This is a correctness-critical detail.

---

### 2.5 Drafter device mesh configuration

**RFC Section**: 3.2

The RFC doesn't detail how the drafter's FSDP device mesh is set up. `TrainingWorker.__init__` creates the engine via `EngineRegistry.new()`, which calls `FSDPEngine.__init__()`, which expects a device mesh. The `DrafterConfig.engine: FSDPEngineConfig` would need the device mesh to be configured.

In the original implementation, the drafter gets a per-DP-group device mesh (`fsdp_workers.py:484-489`):
```python
dp_size = self.world_size // infer_tp
self.global_device_mesh_list = [
    DeviceMesh("cuda", list(range(i * infer_tp, (i + 1) * infer_tp)))
    for i in range(dp_size)
]
self.drafter_device_mesh = self.global_device_mesh_list[self.rollout_dp_rank]
```

**Recommendation**: Add a note about device mesh derivation. The drafter likely reuses the actor's engine device mesh or creates a compatible one from the same world topology. This affects FSDP sharding correctness.

---

## 3. Minor Issues (P2 — nitpicks and clarifications)

### 3.1 Inconsistency between RFC and migration-plan.md

Two documents exist with conflicts:

| Aspect | `rfc-drafter-training.md` | `migration-plan.md` |
|--------|---------------------------|---------------------|
| Model file location | `verl/workers/drafter/model/` | `verl/models/eagle/` |
| Trainer class | Private methods on `ActorRolloutRefWorker` | Mentions `EagleDrafterTrainer` class |
| EAGLE variants | Mentions `eagle3` in `_build_model` switch | "EAGLE-2 only (no EAGLE-3)" |
| Shared layers | Separate ownership + explicit refresh | "direct memory references, no cross-process transfer" |

**Recommendation**: Consolidate into one canonical document. Mark `migration-plan.md` as superseded by the RFC, or update it to match.

---

### 3.2 `get_per_tensor_param` double-offload in drafter sync

**RFC Section**: 3.5

```python
async def _sync_drafter_weights_to_rollout(self):
    per_tensor_param, _ = self.drafter.engine.get_per_tensor_param()
    await self.rollout.update_drafter_weights(per_tensor_param)
    self.drafter.engine.to("cpu", model=True, optimizer=False, grad=False)  # ← redundant?
```

`get_per_tensor_param()` (`transformer_impl.py:639-691`) already does:
```python
load_fsdp_model_to_gpu(self.module)  # loads to GPU
# ... extracts state dict ...
if self._is_offload_param:
    offload_fsdp_model_to_cpu(self.module)  # auto-offloads if configured
```

If `DrafterConfig.engine.param_offload=True` (which the RFC recommends), `get_per_tensor_param` already offloads to CPU. The subsequent `to("cpu")` is redundant.

**Recommendation**: Remove the explicit `to("cpu")` call or add a guard:
```python
if not self.drafter.engine._is_offload_param:
    self.drafter.engine.to("cpu", model=True, optimizer=False, grad=False)
```

---

### 3.3 Missing `train_epochs` handling

**RFC Section**: 3.5 / 4.1

`DrafterConfig` has `train_epochs: int = 1`. But `update_drafter` calls `self.drafter.train_batch()` once, which runs a single forward-backward pass. There is no loop for `train_epochs > 1`. The `train_mini_batch` method supports epochs, but the RFC explicitly chose `train_batch` to avoid mini-batch complexity.

**Recommendation**: Either:
- Remove `train_epochs` from `DrafterConfig` (always 1 for V1), or
- Add a simple loop in `update_drafter`:
  ```python
  for epoch in range(self.config.drafter.train_epochs):
      output = self.drafter.train_batch(data=drafter_batch)
  ```

---

### 3.4 Loss normalization across DP ranks

**RFC Section**: 5

The `eagle_dual_loss` uses local normalization:
```python
num_valid = loss_mask.sum().clamp(min=1)
```

But `FSDPEngine.forward_backward_batch()` (`transformer_impl.py:493-497`) computes a global `batch_num_tokens` via all-reduce across DP ranks:
```python
batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
torch.distributed.all_reduce(batch_num_tokens, op=ReduceOp.SUM, group=self.get_data_parallel_group())
```

This global count is passed to the loss function via `tu.assign_non_tensor(data, batch_num_tokens=...)`. The drafter loss doesn't use it — it uses its own `loss_mask.sum()` which is local.

**Recommendation**: Clarify whether the drafter loss should use local or global normalization. For V1 with the same batch on all DP ranks (sampled from local buffers), local normalization produces different loss scales than global. This affects gradient magnitude. Document the choice explicitly.

---

### 3.5 `ploss` description: cross-entropy, not KL divergence

**RFC Section**: 5

The RFC says "ploss: KL divergence between drafter logits and target logits" but the implementation is:
```python
ploss = -(target_p * out_logp).sum(dim=-1)  # This is cross-entropy H(target_p, drafter_p)
```

KL divergence would be `H(target_p, drafter_p) - H(target_p)`. Since `H(target_p)` is constant w.r.t. drafter parameters, the gradients are identical, so training is unaffected. But the reported metric value differs.

This matches the original implementation (`eagle_background_trainer.py:560-570`) which also uses cross-entropy. Just the description should be corrected.

**Recommendation**: Change "KL divergence" to "cross-entropy" in the loss function docstring and Section 5 description.

---

### 3.6 Missing learning rate scheduler configuration

**RFC Section**: 4.1

The original implementation (`fsdp_workers.py:608-650`) supports constant, cosine, and linear warmup LR schedulers. The RFC's `DrafterConfig` only has `OptimizerConfig(lr=1e-5)` — no scheduler config.

`TrainingWorker.train_batch()` supports `update_lr_scheduler` via non-tensor data (`engine_workers.py:333-338`), but there's no scheduler to step if none is configured.

**Recommendation**: Either add an LR scheduler config to `DrafterConfig`, or document that V1 uses a constant learning rate (no scheduler). The original uses `lr=1e-6` with warmup — consider whether that's needed.

---

## 4. What's Done Well

1. **Leveraging `TrainingWorker` + `BaseEngineCtx`**: The insight that the drafter gets FSDP sharding, CPU/GPU offloading, and checkpointing "for free" by being a standard `TrainingWorker` is the key architectural win.

2. **Config-driven enablement**: Not touching `Role` enum, `_get_role_string()`, or `main_ppo.py` minimizes blast radius. The `_is_drafter = bool(self.config.get("drafter", {}).get("enable", False))` pattern is clean.

3. **Buffer-raw, shift-at-sample**: Storing raw (unshifted) hidden states and applying the EAGLE shift alignment only at training time is clean and avoids data corruption from premature alignment.

4. **Target logits computed on-the-fly**: Not buffering `(B, L, V)` tensors saves ~4.9 GB per batch at V=150k. This is a deliberate and well-documented design choice (Section 3.3, 5).

5. **Explicit `update_weights` injection over override**: The RFC correctly identifies that overriding `update_weights()` would silently drop LoRA base sync, `set_expandable_segments`, and rollout resume lifecycle (`engine_workers.py:615-676`). Injecting drafter sync is safer.

6. **Separate ownership + refresh for frozen layers**: Correct decision for the migration target's clean engine boundaries, even though it's more work than FastRL's shared-reference approach.

7. **Thorough comparison table** (Section 8): Honest about what's lost (background training during rollout bubble) and what's gained (simplicity, correctness, reuse).

8. **Decision log** (Section 11): Captures the "why" behind each design choice with code references. Excellent for future maintainers.

---

## 5. Summary Table

| Priority | Issue | RFC Section | Action |
|----------|-------|-------------|--------|
| **P0** | Missing drafter `forward_step`/`prepare_model_inputs`/`prepare_model_outputs` | 6 | Add full `FSDPDrafterEngine` implementation |
| ~~P0~~ **P2** | ~~Hidden states lost in micro-batch aggregation~~ RETRACTED: `postprocess_batch_func` handles arbitrary keys. Remaining risk: shape consistency, SP gather, transient GPU memory | 3.4 | Unit test 2D nested tensor aggregation |
| **P0** | FSDP2 DTensor `load_state_dict` for frozen layer refresh | 3.5, 10 | Prototype and document cross-engine weight copy mechanism |
| **P0** | Wasteful dispatch in `update_drafter` | 3.5, 3.7 | Switch to `Dispatch.ONE_TO_ALL` |
| **P1** | Padded vs. no-padding format mismatch | 3.3, 3.4 | Explicitly document drafter uses padded inputs |
| **P1** | CPU memory budget for hidden state buffer | 3.3 | Add windowed storage or document memory budget |
| **P1** | Nested tensor indexing semantics | 3.3 | Verify and document `[i]` behavior on jagged tensors |
| **P1** | `response_mask` field availability | 3.3 | Trace field names through `left_right_2_no_padding` |
| **P1** | Drafter device mesh configuration | 3.2 | Document how device mesh is derived |
| **P2** | Consolidate RFC and migration-plan.md | — | Mark one as canonical |
| **P2** | `get_per_tensor_param` double-offload | 3.5 | Remove redundant `to("cpu")` |
| **P2** | `train_epochs` not implemented | 3.5, 4.1 | Remove config field or add loop |
| **P2** | Loss normalization (local vs global) | 5 | Document the choice |
| **P2** | "KL divergence" → "cross-entropy" | 5 | Fix terminology |
| **P2** | Missing LR scheduler config | 4.1 | Add scheduler or document constant LR |
