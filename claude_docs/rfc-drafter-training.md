# RFC: Drafter Co-training Integration for verl Migration Target

## Status: Draft
## Author: brian1009
## Date: 2026-02-16

---

## 1. Motivation

FastRL's key innovation is **co-training an EAGLE draft model alongside the target model during RL**. The current implementation (`eagle_background_trainer.py`) is tightly coupled to the legacy FSDP worker architecture with bespoke ZMQ coordination, CPU↔GPU offloading, and a custom training loop.

The migration target (`migration_targets/verl/`) introduces a clean 3-layer architecture:

```
Layer 3: ActorRolloutRefWorker     — RL-aware, composes workers + rollout
Layer 2: TrainingWorker            — Batch orchestration, metrics, loss injection
Layer 1: BaseEngine / BaseRollout  — Pure computation
```

This RFC proposes integrating drafter training into this layered architecture by:
1. Extending `ActorRolloutRefWorker` to own a drafter `TrainingWorker`
2. Buffering hidden states in the worker for drafter training data
3. Adding first-class drafter training + weight sync to the orchestrator loop

---

## 2. Design Overview

### 2.1 Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                      RayPPOTrainer.fit()                         │
│                                                                  │
│  ① generate_sequences()         ← rollout engine (SGLang)        │
│  ② compute_reward()             ← reward workers                 │
│  ③ compute_log_prob()           ← actor engine (eval)            │
│     └─ collect_hidden_states()  ← NEW: piggyback on forward pass │
│  ④ compute_advantage()          ← driver                         │
│  ⑤ update_critic()              ← critic engine (train)          │
│  ⑥ update_actor()               ← actor engine (train)           │
│  ⑦ update_drafter()             ← NEW: drafter engine (train)    │
│  ⑧ update_weights()             ← sync actor + drafter → rollout │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Worker Composition (Extended)

```
ActorRolloutRefWorker  (with drafter enabled via config)
├── self.actor:          TrainingWorker  → BaseEngine   # policy gradient
├── self.ref:            TrainingWorker  → BaseEngine   # frozen reference (optional)
├── self.drafter:        TrainingWorker  → BaseEngine   # EAGLE drafter training (NEW)
├── self.rollout:        BaseRollout                    # token generation
└── self._drafter_buffer: deque[dict]                   # hidden state buffer (NEW)
```

The drafter `TrainingWorker` is a **first-class citizen** — same `TrainingWorker` + `BaseEngine` infrastructure as actor and critic. No special background thread, no ZMQ coordination. It simply takes a turn in the sequential RL loop.

### 2.3 V1 Scope

- Backend scope for drafter training in V1 is **FSDP/FSDP2 only**.
- Megatron/other backends are explicitly out of scope for this RFC and can be added in follow-up RFCs.
- `update_drafter()` must always return a collectable object (`TensorDict`), never `None`.
- Drafter rollout weight sync in V1 requires `actor_rollout_ref.rollout.checkpoint_engine.backend == "naive"`.

---

## 3. Detailed Design

### 3.1 Config-Driven Drafter Enablement

Do not add new role strings. Keep existing role wiring and gate drafter behavior by config:

```python
assert self.role in [
    "actor", "rollout", "ref",
    "actor_rollout", "actor_rollout_ref",
]
self._is_drafter = bool(self.config.get("drafter", {}).get("enable", False))
```

**Rationale**: The drafter is still colocated with rollout and actor, but no trainer/Role wiring changes are required. This avoids touching `Role` enum selection paths while keeping GPU time-multiplexing in the same Ray actor.

### 3.2 Drafter TrainingWorker Initialization

Inside `ActorRolloutRefWorker.init_model()`, after building actor/ref/rollout:

```python
# 5. Build drafter training engine (NEW)
if self._is_drafter:
    drafter_config = omega_conf_to_dataclass(self.config.drafter)
    drafter_model_config = build_drafter_model_config(
        base_model_config=omega_conf_to_dataclass(self.config.model),
        drafter_config=drafter_config,
    )
    drafter_training_config = TrainingWorkerConfig(
        model_type="drafter_model",              # new registry key
        model_config=drafter_model_config,       # EAGLE model config (derived)
        engine_config=drafter_config.engine,
        optimizer_config=drafter_config.optim,
        checkpoint_config=drafter_config.checkpoint,
    )
    self.drafter = TrainingWorker(config=drafter_training_config)
    self.drafter.reset()

    # Bind lm_head + loss weights via closure (drafter holds its own copy; see Section 10)
    lm_head = self.drafter.engine.module.lm_head
    self.drafter.set_loss_fn(partial(
        eagle_dual_loss,
        lm_head=lm_head,
        w_v=drafter_config.vloss_weight,
        w_p=drafter_config.ploss_weight,
    ))

    self.set_dispatch_collect(mesh_name="drafter", **self.drafter.get_dispatch_collect())

    # 6. Initialize hidden state buffer
    self._drafter_buffer: deque = deque(maxlen=drafter_config.buffer_size)
```

`DrafterConfig` intentionally stays lightweight (`model_arch`, `base_model_arch`, `model_path`, ...). `TrainingWorkerConfig.model_config` is built in `init_model()` via `build_drafter_model_config(...)` so there is no dataclass field mismatch.

**Key point**: The EAGLE model gets registered in `EngineRegistry` under `model_type="drafter_model"`:

```python
@EngineRegistry.register(model_type="drafter_model", backend=["fsdp", "fsdp2"])
class FSDPDrafterEngine(FSDPEngine):
    """Thin wrapper that handles EAGLE-specific model loading."""
    # Uses the same FSDPEngine infrastructure, but loads EAGLE architecture
    # (single transformer layer + fusion linear) instead of full LLM
```

This means the drafter automatically gets FSDP sharding, CPU↔GPU offloading via `BaseEngineCtx`, checkpointing — all for free.

### 3.3 Hidden State Buffering (Private Methods)

Hidden state collection and batch preparation live directly on the worker as two private methods plus a `deque` buffer. No standalone class needed for V1.

**Storage schema**: The buffer stores raw per-sample dicts. The **shift alignment** (base vs. target) is applied at sample time, not at collection time:

```python
# In ActorRolloutRefWorker:

def _collect_drafter_sample(self, batch: TensorDict, hidden_states: torch.Tensor):
    """Buffer raw (unshifted) hidden states from actor forward pass.

    Called inside compute_log_prob() after extracting hidden states.
    `hidden_states` is expected to be a nested tensor aligned with `batch["input_ids"]`
    (see Section 3.4 engine-side conversion for no-padding FSDP path).
    Tensors are detached and moved to CPU to avoid holding GPU memory
    and autograd graph references across RL steps.
    """
    for i in range(batch.shape[0]):
        input_ids_i = batch["input_ids"][i].detach().cpu()
        hidden_i = hidden_states[i].detach().cpu()

        # response_mask is response-only length; expand to full-sequence loss mask
        response_mask_i = batch["response_mask"][i].detach().cpu().to(torch.float32)
        response_len = int(response_mask_i.sum().item())
        seq_loss_mask = torch.zeros(input_ids_i.shape[0], dtype=torch.float32)
        if response_len > 0:
            seq_loss_mask[-response_len:] = response_mask_i[:response_len]

        self._drafter_buffer.append({
            "input_ids": input_ids_i,
            "hidden_states": hidden_i,
            "loss_mask": seq_loss_mask,
            "attention_mask": torch.ones(input_ids_i.shape[0], dtype=torch.int64),
        })

def _sample_drafter_batch(self, batch_size: int) -> TensorDict:
    """Sample from buffer and build a training-ready batch with shift alignment.

    Applies the EAGLE shift rule (see eagle_background_trainer.py:443-456):
        base_hidden    = hidden_states[:, :-1]   (input to drafter)
        target_hidden  = hidden_states[:, 1:]    (prediction target)
        input_ids      = input_ids[:, :-1]
        loss_mask      = loss_mask[:, 1:]
        attention_mask = attention_mask[:, :-1]

    Returns TensorDict with keys:
        input_ids, attention_mask, base_hidden_states,
        target_hidden_states, loss_mask
    """
    samples = random.sample(list(self._drafter_buffer), min(batch_size, len(self._drafter_buffer)))

    # Pad to max length in this batch (right-pad; loss_mask zeros out padding)
    max_len = min(max(s["input_ids"].shape[0] for s in samples), self.config.drafter.max_seq_len)

    def pad_or_truncate(t, pad_value=0):
        """Right-pad or truncate tensor to max_len along dim 0."""
        if t.shape[0] >= max_len:
            return t[:max_len]
        pad_size = max_len - t.shape[0]
        if t.dim() == 1:
            return F.pad(t, (0, pad_size), value=pad_value)
        else:  # (L, D)
            return F.pad(t, (0, 0, 0, pad_size), value=pad_value)

    hidden_states  = torch.stack([pad_or_truncate(s["hidden_states"]) for s in samples])  # (B, L, D)
    input_ids      = torch.stack([pad_or_truncate(s["input_ids"]) for s in samples])      # (B, L)
    loss_mask      = torch.stack([pad_or_truncate(s["loss_mask"]) for s in samples])      # (B, L)
    attention_mask = torch.stack([pad_or_truncate(s["attention_mask"]) for s in samples])  # (B, L)

    # Apply shift alignment
    return TensorDict({
        "input_ids":            input_ids[:, :-1],
        "attention_mask":       attention_mask[:, :-1],
        "base_hidden_states":   hidden_states[:, :-1],
        "target_hidden_states": hidden_states[:, 1:],
        "loss_mask":            loss_mask[:, 1:],
    })
```

**Key design points**:
- `target_logits` are NOT stored. They are computed on-the-fly during the training step via `lm_head(target_hidden_states)` (see Section 5). This avoids buffering `(seq_len, vocab_size)` tensors (~4.9 GB per 8-sample batch at V=150k).
- Tensors are `.detach().cpu()` on collection to avoid GPU memory leaks and autograd graph retention across RL steps.
- If a standalone `HiddenStateCollector` class is needed later (for separate unit tests or reuse), extracting one from these two methods is a trivial refactor.

### 3.4 Hidden State Extraction from Actor Forward Pass

Hidden states must be collected during `compute_log_prob()` without disrupting the existing `compute_loss=False` path. The approach follows the existing FastRL pattern (`fsdp_workers.py:1014-1029`):

**Constraints** (why the original Approach A fails):
- `ray_trainer.py:1206` sets `compute_loss=False` for old-log-prob
- `engine_workers.py:379` maps that to `loss_function=None`
- `FSDPEngine.forward_step` (transformer_impl.py:1007) skips the loss callback when `loss_function is None`
- Even if forced, the PPO loss expects `old_log_probs`/`advantages` fields that don't exist yet
- Putting large tensors into `metrics` is wrong — metrics are DP all-gathered as objects

**Approach (revised)**: Thread an `output_hidden_states` flag through the data TensorDict, read it in the engine forward to call the HF model with `output_hidden_states=True`, then intercept and consume hidden states locally inside the worker before returning output to the trainer.

**Engine layer** — read the flag in `forward_step()`:

```python
# In FSDPEngine.forward_step() (V1 scope: fsdp/fsdp2 only):
output_hidden_states = tu.get(micro_batch, "output_hidden_states", default=False)
use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
raw_output = self.module(
    **model_inputs,
    use_cache=False,
    output_hidden_states=output_hidden_states,  # NEW: conditionally request
)
# If requested, attach to model_output dict (NOT metrics):
if output_hidden_states and raw_output.hidden_states is not None:
    hs = raw_output.hidden_states[-1]  # [(1, total_nnz, D)] in no-padding path
    if use_remove_padding:
        hs_rmpad = hs.squeeze(0)  # (total_nnz, D)
        if self.use_ulysses_sp:
            hs_rmpad = gather_outputs_and_unpad(
                hs_rmpad,
                gather_dim=0,
                unpad_dim=0,
                padding_size=output_args["pad_size"],
            )
        cu_seqlens = micro_batch["input_ids"].offsets()
        hs = torch.nested.nested_tensor_from_jagged(hs_rmpad, cu_seqlens)
    model_output["hidden_states"] = hs.detach()
```

This is independent of `loss_function` — works whether `compute_loss` is True or False.

**Worker layer** — collect locally, strip before returning. The full `compute_log_prob` implementation with the `@register` decorator is in Section 3.5. The key pattern: hidden states are produced inside the worker, consumed locally via `_collect_drafter_sample`, and never returned to the driver. This mirrors the existing `return_hidden_states` pattern in `fsdp_workers.py:1014-1029`.

### 3.5 Worker-Level Methods

New methods on `ActorRolloutRefWorker` (drafter support is config-driven, not a subclass).

`_should_collect_drafter_data` is `True` when `self._is_drafter` and the current step is a collection step (V1: always `True` when drafter is enabled; could be gated by `training_interval_steps` in a future optimization to skip collection on non-training steps).

```python
# In ActorRolloutRefWorker:

@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
def compute_log_prob(self, data: TensorDict) -> TensorDict:
    """Extended: collect hidden states alongside log prob computation."""
    # Signal that we want hidden states
    if self._is_drafter and self._should_collect_drafter_data:
        tu.assign_non_tensor(data, output_hidden_states=True)

    output = self.actor.infer_batch(data)

    # Extract and buffer hidden states locally
    if self._is_drafter and output is not None:
        hidden_states = tu.pop(output, "hidden_states", default=None)
        if hidden_states is not None:
            self._collect_drafter_sample(data, hidden_states)

    return output.cpu() if output is not None else None

@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="drafter"))
def update_drafter(self, data: TensorDict) -> TensorDict:
    """Train the drafter model on collected hidden states.

    Called by the orchestrator every N RL steps.
    GPU timeline: actor offloaded → refresh frozen layers → drafter loaded → train → drafter offloaded
    """
    if len(self._drafter_buffer) < self.config.drafter.min_samples:
        return tu.get_tensordict(
            tensor_dict={},
            non_tensor_dict={"metrics": {
                "drafter/skipped": 1.0,
                "drafter/buffer_size": float(len(self._drafter_buffer)),
            }},
        )

    # Refresh frozen layers from actor (both on CPU at this point; see Section 10)
    self._refresh_drafter_frozen_layers()

    # Prepare drafter training batch from buffered hidden states
    drafter_batch = self._sample_drafter_batch(
        batch_size=self.config.drafter.train_batch_size
    )

    # train_batch metadata for MFU accounting
    global_token_num = drafter_batch["attention_mask"].sum(dim=-1).tolist()
    tu.assign_non_tensor(drafter_batch, global_token_num=global_token_num)

    # Train (BaseEngineCtx handles GPU load/offload automatically)
    output = self.drafter.train_batch(data=drafter_batch)
    return output.cpu() if output is not None else tu.get_tensordict(
        tensor_dict={},
        non_tensor_dict={"metrics": {"drafter/empty_output": 1.0}},
    )

async def _sync_drafter_weights_to_rollout(self):
    """Push trained drafter weights to inference server."""
    per_tensor_param, _ = self.drafter.engine.get_per_tensor_param()
    await self.rollout.update_drafter_weights(per_tensor_param)
    self.drafter.engine.to("cpu", model=True, optimizer=False, grad=False)

def _refresh_drafter_frozen_layers(self):
    """Copy lm_head and embed_tokens from actor to drafter (both on CPU).

    Called before each drafter training step. See Section 10 for rationale.
    """
    actor_module = self.actor.engine.module
    drafter_module = self.drafter.engine.module
    drafter_module.lm_head.load_state_dict(actor_module.lm_head.state_dict())
    drafter_module.model.embed_tokens.load_state_dict(
        actor_module.model.embed_tokens.state_dict()
    )
```

**Note on `update_weights()`**: Do NOT override `update_weights()`. The existing method (`engine_workers.py:616-676`) has complex branching: async backend early return, LoRA base sync, `set_expandable_segments`, and rollout resume lifecycle. Instead, **inject** `_sync_drafter_weights_to_rollout()` into the existing method body **after step 3 (actor CPU offload) and before step 4 (`rollout.resume(tags=["kv_cache"])`)**. This avoids transient actor+drafter GPU overlap when actor param offload is disabled.

V1 drafter rollout sync is supported only when `checkpoint_engine.backend == "naive"`. In non-`naive` mode, fail fast (or disable drafter sync explicitly) instead of silently skipping `_sync_drafter_weights_to_rollout()`.

### 3.6 Rollout Interface Extension

`BaseRollout` gains one new method for drafter weight sync:

```python
class BaseRollout(ABC):
    # ... existing methods ...

    async def update_drafter_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """Update the weights of the drafter model in the inference server.

        Default: no-op (for rollout engines that don't support SD).
        Concrete implementation in SGLang ServerAdapter.
        """
        pass  # Optional — non-SD rollouts simply ignore this
```

SGLang's `ServerAdapter` implements this by posting drafter weights to the SD engine endpoint.

### 3.7 Orchestrator Integration (RayPPOTrainer.fit)

Add a trainer helper, mirroring `_update_actor()` / `_update_critic()`, so worker `TensorDict` output is normalized to `DataProto` metrics:

```python
def _update_drafter(self, batch: DataProto) -> DataProto:
    if self.use_legacy_worker_impl == "disable":
        batch_td = batch.to_tensordict()
        output = self.actor_rollout_wg.update_drafter(batch_td)
        output = tu.get(output, "metrics")
        output = rename_dict(output, "drafter/")
        drafter_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
    else:
        drafter_output = self.actor_rollout_wg.update_drafter(batch)
    return drafter_output
```

Then the training loop gains two new steps:

```python
# In RayPPOTrainer.fit(), after update_actor():

# ⑦ update drafter (NEW) — every N RL steps
if (self.use_drafter_training
    and self.global_steps % self.config.drafter.training_interval_steps == 0):
    with marked_timer("update_drafter", timing_raw, color="purple"):
        drafter_output = self._update_drafter(batch)
    drafter_metrics = reduce_metrics(drafter_output.meta_info["metrics"])
    metrics.update(drafter_metrics)

# ⑧ update weights (MODIFIED) — now syncs actor + drafter
with marked_timer("update_weights", timing_raw, color="red"):
    self.checkpoint_manager.update_weights()
```

### 3.8 GPU Memory Timeline

The key insight: actor, drafter, and rollout **time-share** the same GPU. `BaseEngineCtx` already handles the CPU↔GPU offloading:

```
Time ──────────────────────────────────────────────────────────────────────────────►

│← RL step N ─────────────────────────────────────────────────────→│← step N+1 ──→│

┌────────────┐
│  Rollout   │  generate_sequences()
│  (on GPU)  │
└────────────┘
              ┌──────────────┐
              │    Actor     │  compute_log_prob()
              │   (on GPU)   │  + collect hidden states → _drafter_buffer
              └──────────────┘
                              ┌──────────┐
                              │  Actor   │  update_actor()
                              │ (on GPU) │  (PPO gradient step)
                              └──────────┘
                                          ┌──────────┐
                                          │ Drafter  │  update_drafter()
                                          │ (on GPU) │  (EAGLE training)
                                          └──────────┘
                                                      ┌────────────┐
                                                      │  Weight    │  sync actor + drafter → rollout
                                                      │  Sync      │
                                                      └────────────┘
                                                                    ┌────────────┐
                                                                    │  Rollout   │  generate_sequences()
                                                                    │  (on GPU)  │  (next iteration)
                                                                    └────────────┘
```

The `BaseEngineCtx` context manager (in `base.py:229-263`) automatically:
- Loads model/optimizer to GPU on `__enter__`
- Offloads to CPU on `__exit__`

So the drafter training slot is simply another `with engine.train_mode():` block — no custom offloading code needed.

For predictable memory behavior in this sequence, V1 should use CPU offload for both actor and drafter params (`actor.engine.param_offload=true`, `drafter.engine.param_offload=true`).

---

## 4. Configuration

### 4.1 New Config Dataclass

```python
# verl/workers/config/drafter.py

@dataclass
class DrafterConfig(BaseConfig):
    """Configuration for EAGLE drafter co-training."""

    # Model
    model_path: Optional[str] = None      # pretrained EAGLE checkpoint (None = init from scratch)
    model_arch: str = "eagle"             # "eagle" | "eagle3"
    base_model_arch: str = "qwen2"        # must match target model

    # Training schedule
    enable: bool = False                  # master switch
    training_interval_steps: int = 10     # train every N RL steps
    train_batch_size: int = 8             # samples per drafter training step
    train_epochs: int = 1                 # epochs per trigger
    min_samples: int = 4                  # skip training if buffer has fewer samples

    # Engine (reuses existing EngineConfig)
    engine: FSDPEngineConfig = field(default_factory=lambda: FSDPEngineConfig(
        param_offload=True,               # CPU offload when not training
        optimizer_offload=True,
        strategy="fsdp2",
    ))

    # Optimizer
    optim: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(
        lr=1e-5,
        weight_decay=0.0,
    ))

    # Loss weights
    vloss_weight: float = 1.0             # hidden state prediction weight
    ploss_weight: float = 1.0             # distribution matching weight

    # Hidden state buffer
    buffer_size: int = 2000               # max samples in bounded worker queue
    max_seq_len: int = 8192               # truncate sequences longer than this

    # Checkpoint
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
```

### 4.2 YAML Configuration

```yaml
# In ppo_trainer.yaml (or fastrl_trainer.yaml)
actor_rollout_ref:
  actor: ...
  rollout: ...
  ref: ...

  drafter:                               # NEW section
    enable: false
    model_path: null                     # path to pretrained EAGLE ckpt
    model_arch: eagle
    base_model_arch: qwen2
    training_interval_steps: 10
    train_batch_size: 8
    train_epochs: 1
    min_samples: 4
    vloss_weight: 1.0
    ploss_weight: 1.0
    buffer_size: 2000
    max_seq_len: 8192
    engine:
      strategy: fsdp2
      param_offload: true
      optimizer_offload: true
    optim:
      lr: 1e-5
```

---

## 5. EAGLE Loss Function

Injected via `drafter.set_loss_fn()`. Follows the pattern from `eagle_background_trainer.py:550-578`.

**Key design**: `target_logits` are computed **on-the-fly** from `target_hidden_states` via `lm_head`, not stored in the buffer. This avoids buffering `(B, L, V)` tensors (~4.9 GB at V=150k).

The data contract between `_sample_drafter_batch()` and this loss:

```
_sample_drafter_batch() output:      Loss function reads:
  input_ids       (B, L-1)            data["input_ids"]
  attention_mask  (B, L-1)            data["attention_mask"]
  base_hidden     (B, L-1, D)  →      data["base_hidden_states"]   (drafter input)
  target_hidden   (B, L-1, D)  →      data["target_hidden_states"] (vloss target)
  loss_mask       (B, L-1)            data["loss_mask"]
```

```python
# verl/workers/drafter/losses.py

def eagle_dual_loss(model_output, data, dp_group, lm_head, w_v=1.0, w_p=1.0):
    """EAGLE drafter training loss.

    Combines:
      - vloss: SmoothL1 between drafter-predicted hidden states and target
      - ploss: KL divergence between drafter logits and target logits
               (target_logits computed on-the-fly via lm_head)

    Matches eagle_background_trainer.py:550-578.

    Args:
        model_output: dict with "logits", "hidden_states" (drafter predictions)
        data: TensorDict with base_hidden_states, target_hidden_states, loss_mask
        dp_group: data parallel group (for distributed reduction)
        lm_head: nn.Module — bound via partial at init (Section 3.2)
        w_v: vloss weight — bound via partial at init from DrafterConfig
        w_p: ploss weight — bound via partial at init from DrafterConfig
    """
    logits = model_output["logits"]                  # (B, L-1, V) drafter output
    pred_hidden = model_output["hidden_states"]      # (B, L-1, D) drafter predicted hidden

    target_hidden = data["target_hidden_states"]     # (B, L-1, D) from buffer
    loss_mask = data["loss_mask"]                     # (B, L-1)

    num_valid = loss_mask.sum().clamp(min=1)

    # --- vloss: hidden state prediction (SmoothL1) ---
    vloss = F.smooth_l1_loss(pred_hidden, target_hidden, reduction="none")  # (B, L-1, D)
    vloss = torch.mean(vloss, dim=-1)                                       # (B, L-1)
    vloss = (loss_mask * vloss).sum() / num_valid

    # --- ploss: distribution matching (KL divergence) ---
    # Compute target_logits on-the-fly via lm_head (not stored in buffer)
    # See eagle_background_trainer.py:560
    with torch.no_grad():
        target_logits = lm_head(target_hidden)       # (B, L-1, V)
        target_p = F.softmax(target_logits, dim=-1)

    out_logp = F.log_softmax(logits, dim=-1)
    ploss = -(target_p * out_logp).sum(dim=-1)        # (B, L-1)
    ploss = (loss_mask * ploss).sum() / num_valid

    # --- combined ---
    # Weights are bound via partial at init time (Section 3.2), not from data dict
    loss = w_v * vloss + w_p * ploss

    metrics = {
        "drafter/loss": loss.detach().item(),
        "drafter/vloss": vloss.detach().item(),
        "drafter/ploss": ploss.detach().item(),
    }

    return loss, metrics
```

**Note on `lm_head` access**: The drafter engine holds its own copy of `lm_head` (separate ownership — see Section 10). It is bound into the loss function via `partial` at init time (Section 3.2):
```python
lm_head = self.drafter.engine.module.lm_head
self.drafter.set_loss_fn(partial(eagle_dual_loss, lm_head=lm_head, w_v=..., w_p=...))
```
Because the `partial` captures the `nn.Module` reference (not a snapshot of weights), refreshing the drafter's `lm_head` weights from the actor (Section 10) is automatically reflected in subsequent loss computations.

---

## 6. EngineRegistry Integration

Register the EAGLE model architecture:

```python
# verl/workers/drafter/engine.py

@EngineRegistry.register(model_type="drafter_model", backend=["fsdp", "fsdp2"])
class FSDPDrafterEngine(FSDPEngine):
    """FSDP engine for EAGLE drafter model.

    The EAGLE model is tiny (single transformer layer + fusion linear),
    so FSDP overhead is minimal. The main benefit is getting automatic
    CPU↔GPU offloading via BaseEngineCtx.
    """

    def _build_model(self, model_config):
        """Load EAGLE architecture instead of full LLM."""
        if model_config.arch == "eagle":
            if model_config.base_arch == "qwen2":
                from verl.workers.drafter.model.qwen2_eagle import Qwen2ForCausalLMEagle
                return Qwen2ForCausalLMEagle(model_config.hf_config)
            elif model_config.base_arch == "llama":
                from verl.workers.drafter.model.llama_eagle import LlamaForCausalLMEagle
                return LlamaForCausalLMEagle(model_config.hf_config)
        # ... eagle3 variants
```

---

## 7. File Layout

```
migration_targets/verl/verl/workers/
├── engine_workers.py              # MODIFIED: config-gated drafter integration
│                                  #   + _collect_drafter_sample(), _sample_drafter_batch()
├── config/
│   ├── __init__.py                # MODIFIED: export drafter config
│   └── drafter.py                 # NEW: DrafterConfig
├── drafter/                       # NEW directory
│   ├── __init__.py
│   ├── engine.py                  # FSDPDrafterEngine (EngineRegistry)
│   ├── losses.py                  # eagle_dual_loss
│   └── model/                     # EAGLE model architectures
│       ├── qwen2_eagle.py         # (migrated from fastrl)
│       ├── llama_eagle.py
│       ├── qwen2_eagle3.py
│       └── llama_eagle3.py
└── rollout/
    └── base.py                    # MODIFIED: add update_drafter_weights()
```

---

## 8. Comparison with Current FastRL Implementation

| Aspect | Current (FastRL) | Proposed (Migration) |
|--------|-----------------|---------------------|
| **Training loop** | Background thread with ZMQ coordination | Sequential step in RL loop |
| **GPU management** | Manual `offload_fsdp_model_to_cpu` / `load_fsdp_model_to_gpu` | Automatic via `BaseEngineCtx` |
| **Model wrapping** | Bespoke FSDP2 setup in `_build_model()` | Standard `EngineRegistry` + `FSDPEngine` |
| **Data collection** | Custom `collect_online_data()` + `DataBuffer` | Private methods + bounded `deque` on worker |
| **Weight sync** | Implicit via shared FSDP object reference | Explicit `update_drafter_weights()` on `BaseRollout` |
| **Worker coordination** | ZMQ pub-sub `CentralCoordinator` | Single controller dispatch (Ray RPC) |
| **Checkpointing** | Custom async DCP | Reuse `BaseEngine.save_checkpoint()` |
| **Loss function** | Hardcoded in training loop | Injected via `set_loss_fn()` |
| **Complexity** | ~700 LOC `eagle_background_trainer.py` + ~800 LOC `worker_manager.py` | ~200 LOC new code, rest reused from framework |

### What We Lose

- **Background training during rollout bubble**: The current design trains the drafter in the gap between fast and slow workers finishing rollout. The proposed design makes drafter training a synchronous step, which adds latency to each RL iteration.

### What We Gain

- **Simplicity**: No ZMQ, no background threads, no manual GPU offload
- **Correctness**: Deterministic execution order, no race conditions
- **Reuse**: `TrainingWorker`, `BaseEngine`, `BaseEngineCtx`, checkpointing — all free
- **Extensibility**: New drafter architectures just register in `EngineRegistry`

### Mitigations for Lost Background Training

If the synchronous drafter training step becomes a bottleneck (the EAGLE model is very small, so this is unlikely), we can:

1. **Overlap with critic training**: The drafter doesn't depend on critic output, so the orchestrator can dispatch `update_drafter()` and `update_critic()` in parallel (both are `blocking=False`).
2. **Skip-step training**: Only train every N steps (already supported via `training_interval_steps`).
3. **Future: async mode**: The Ray single-controller already supports `blocking=False` dispatch. A future PR could dispatch drafter training asynchronously.

---

## 9. Implementation Plan

### Phase 1: Core Infrastructure
1. Add `DrafterConfig` to `verl/workers/config/drafter.py`
2. Register EAGLE model in `EngineRegistry` via `FSDPDrafterEngine`
3. Implement `eagle_dual_loss` in `verl/workers/drafter/losses.py`

### Phase 2: Worker Integration
4. Extend `ActorRolloutRefWorker` with drafter role, `_collect_drafter_sample()`, `_sample_drafter_batch()`
5. Add `update_drafter()` method to worker, including frozen layer refresh (`lm_head`/`embed_tokens` copy from actor before training — see Section 10)
6. Add `update_drafter_weights()` to `BaseRollout` interface
7. Implement drafter weight sync in SGLang `ServerAdapter`
8. Inject drafter weight sync into existing `update_weights()` flow

### Phase 3: Orchestrator Integration
9. Wire `update_drafter()` into `RayPPOTrainer.fit()` loop
10. Add hidden state collection trigger to `compute_log_prob()` path
11. Add drafter metrics logging

### Phase 4: Testing & Migration
12. Unit test `eagle_dual_loss`
13. Unit test `_sample_drafter_batch()` shift alignment
14. Integration test: full RL loop with drafter co-training enabled
15. Benchmark: drafter training overhead per RL step

---

## 10. Open Questions

### Resolved Assumptions (2026-02-17)

- V1 backend scope is FSDP/FSDP2 only for drafter training.
- `update_drafter()` returns an empty-metrics `TensorDict` on skip, not `None`.
- V1 drafter rollout sync supports `checkpoint_engine.backend == "naive"` only.

1. **Parallel drafter + critic training**: If both critic and drafter use the same GPU, they cannot run simultaneously. If they are on different GPUs (different resource pools), they can. The orchestrator should detect colocation and serialize if needed. Separate resource pools allows true parallelism.

### Resolved: Separate ownership for `lm_head` / `embed_tokens`

**Decision**: V1 uses separate ownership + explicit refresh policy, instead of cross-engine shared references.

**Context**: The EAGLE drafter uses `lm_head` and `embed_tokens` from the target model in two places:
- **Forward pass**: `self.lm_head(hidden_states)` produces drafter output logits (`qwen2_eagle.py:244`, `llama_eagle.py:241`).
- **Loss computation**: `self.model.lm_head(target_hidden_states)` computes the target distribution for ploss (`eagle_background_trainer.py:560`).

In FastRL, these are shared by direct reference assignment (`fsdp_workers.py:560,566`), set to `requires_grad=False` (`fsdp_workers.py:562,568`), and excluded from the drafter's trainable checkpoint (`eagle_background_trainer.py:83,105`). Because they are shared references, the drafter automatically sees actor weight updates — this matters for full fine-tuning where `lm_head`/`embed_tokens` change every PPO step.

**Why not shared references in migration target**: Cross-engine reference sharing is fragile under the migration target's clean engine boundaries. Two `TrainingWorker`/`BaseEngine` lifecycles with independent FSDP ownership, CPU↔GPU offloading, and checkpoint semantics make aliased parameters error-prone.

**V1 policy**:
- Drafter engine holds its own copies of `lm_head` and `embed_tokens` (separate ownership).
- Both are excluded from drafter optimizer (`requires_grad=False`) and drafter checkpoints (matching `_frozen_param_names` pattern at `eagle_background_trainer.py:83`).
- Explicit refresh: `_refresh_drafter_frozen_layers()` (Section 3.5) copies both `lm_head` and `embed_tokens` from actor to drafter before each drafter training step. Both layers change at the same rate under full fine-tuning, so they refresh together.
- **Timing**: The refresh is called at the top of `update_drafter()`, while both engines are on CPU — after actor offload (`update_actor()` completes), before drafter loads to GPU (`BaseEngineCtx.__enter__`). This is a CPU-to-CPU `load_state_dict` between FSDP-sharded parameters on the same device mesh.

---

## 11. Key Design Decisions

Decisions made during design review to avoid pitfalls in the migration target:

| Decision | Rationale |
|----------|-----------|
| Config-driven `_is_drafter` instead of new `Role` enum entries | `main_ppo.py` and `Role` enum only produce `ActorRollout`/`ActorRolloutRef`. Adding drafter role variants would require changes to trainer selection, resource pool wiring, and `_get_role_string()`. Config flag avoids all of this. (Section 3.1) |
| `output_hidden_states` flag in TensorDict instead of loss hook | `compute_log_prob` runs with `compute_loss=False` → `loss_function=None` → loss callback never fires. Threading the flag through data and reading it in `forward_step()` is independent of the loss path. Mirrors existing `return_hidden_states` in `fsdp_workers.py:1014-1029`. (Section 3.4) |
| Buffer raw hidden states, compute `target_logits` on-the-fly via `lm_head` | Avoids buffering `(B, L, V)` tensors (~4.9 GB at V=150k). Follows `eagle_background_trainer.py:560`. (Sections 3.3, 5) |
| V1 backend scope: fsdp/fsdp2 only | Hidden-state extraction and jagged conversion are specified against FSDP no-padding path; defers Megatron complexity to follow-up RFC. (Sections 2.3, 3.4) |
| Use `_update_drafter()` trainer wrapper | New engine workers return `TensorDict`; wrapper normalizes to `DataProto(meta_info["metrics"])` so the fit loop can use existing `reduce_metrics(...)` flow. (Section 3.7) |
| `train_batch` instead of `train_mini_batch` | `train_mini_batch` asserts `mini_batch_size` or `num_mini_batch` metadata. EAGLE model is tiny — no need for mini-batching. `train_batch` only needs `global_token_num`. (Section 3.5) |
| Inject drafter sync into existing `update_weights()` instead of overriding | Existing method has async backend early return, LoRA base sync, `set_expandable_segments`, rollout resume lifecycle. Override would silently drop all of these. (Section 3.5) |
| V1 drafter rollout sync requires `checkpoint_engine.backend == "naive"` | Existing `update_weights()` has early return in non-`naive` branch; constraining V1 avoids silent sync omission and keeps behavior explicit. (Sections 2.3, 3.5, 10) |
| Separate `lm_head`/`embed_tokens` ownership + explicit refresh | Cross-engine shared references are fragile under FSDP2 ownership, CPU↔GPU offloading, and checkpoint semantics. Drafter holds own copies, refreshed from actor before each training step via `_refresh_drafter_frozen_layers()`. (Section 10) |
| `update_drafter()` never returns `None` | ND collect path expects DataProto/TensorDict-compatible outputs; empty-metrics TensorDict avoids collect-time type assertions. (Sections 2.3, 3.5) |
