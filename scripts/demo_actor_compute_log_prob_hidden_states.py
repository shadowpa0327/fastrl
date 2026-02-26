#!/usr/bin/env python3
"""Standalone actor compute_log_prob hidden-state collection script.

This script replicates the fsdp_workers.compute_log_prob → drafter_trainer.collect_online_data
flow that collects hidden states from the actor forward pass for EAGLE drafter co-training.

The production flow is:
  1. fsdp_workers.compute_log_prob captures drafter_batch = {input_ids, responses} on CPU
  2. dp_actor.compute_log_prob runs forward with output_hidden_states=True
  3. Hidden states are returned as List[Tensor[seq_len, hidden_dim]] on CPU (per-sample)
  4. fsdp_workers feeds drafter_batch + hidden_states to drafter_trainer.collect_online_data()
  5. Hidden states never enter DataProto — they stay local on the worker

This demo replicates steps 1-3 standalone, and optionally exercises step 4 with
--test-drafter-collect to validate the full pipeline end-to-end.

Two input modes:
- `--input-batch-pt <path>`: Load a saved .pt batch from a prior SGLang run (comparison mode).
- `--train-file <path>`: Self-contained mode -- load prompts from parquet, generate responses.

Example (self-contained):
  python3 scripts/demo_actor_compute_log_prob_hidden_states.py \
    --model-path /path/to/Qwen2.5-0.5B \
    --train-file /path/to/dapo.parquet \
    --num-prompts 2 \
    --max-prompt-length 512 \
    --max-response-length 64

Example (with drafter collection test):
  python3 scripts/demo_actor_compute_log_prob_hidden_states.py \
    --model-path /path/to/Qwen2.5-0.5B \
    --train-file /path/to/dapo.parquet \
    --num-prompts 2 \
    --test-drafter-collect

Example (comparison with SGLang):
  # Step 1: Run SGLang demo and save batch
  python3 scripts/demo_sglang_rollout_hidden_states.py \
    --model-path /path/to/model --train-file data.parquet \
    --save-batch-pt /tmp/sglang_batch.pt --save-tensors /tmp/sglang_hs.pt

  # Step 2: Run this script with saved batch
  python3 scripts/demo_actor_compute_log_prob_hidden_states.py \
    --model-path /path/to/model --input-batch-pt /tmp/sglang_batch.pt \
    --save-tensors /tmp/actor_hs.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.utils.torch_functional import logprobs_from_logits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Actor forward pass hidden-state collection (standalone, no FSDP/Ray)."
    )

    # Model
    parser.add_argument("--model-path", type=str, required=True, help="HF model id or local path.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype.")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trust remote code in HF model loading.",
    )

    # Input mode 1: saved batch from SGLang run
    parser.add_argument(
        "--input-batch-pt",
        type=str,
        default=None,
        help="Path to .pt file with saved batch (keys: input_ids, attention_mask, position_ids, responses).",
    )

    # Input mode 2: self-contained from parquet
    parser.add_argument("--train-file", type=str, default=None, help="RLHF parquet file path.")
    parser.add_argument("--num-prompts", type=int, default=2, help="Number of prompts to load.")
    parser.add_argument("--max-prompt-length", type=int, default=1024, help="Max prompt length for tokenization.")
    parser.add_argument("--max-response-length", type=int, default=64, help="Max response length for generation.")
    parser.add_argument("--prompt-key", type=str, default="prompt", help="Prompt column key in dataset.")

    # Forward pass
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for log_prob computation.")
    parser.add_argument("--micro-batch-size", type=int, default=0, help="Micro-batch size (0 = full batch).")

    # Drafter collection test
    parser.add_argument(
        "--test-drafter-collect",
        action="store_true",
        default=False,
        help="Test collect_online_data with the collected hidden states (exercises full pipeline).",
    )

    # Output
    parser.add_argument("--summary-json", type=str, default=None, help="Path to write JSON summary.")
    parser.add_argument("--save-tensors", type=str, default=None, help="Path to save .pt with hidden_states etc.")
    parser.add_argument("--save-batch-pt", type=str, default=None, help="Path to save input batch for reuse.")

    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu).")
    parser.add_argument("--cuda-visible-devices", type=str, default=None, help="CUDA_VISIBLE_DEVICES override.")

    return parser.parse_args()


def load_model(args: argparse.Namespace):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

    print(f"[actor-hs] Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[actor-hs] Set pad_token_id = eos_token_id = {tokenizer.eos_token_id}")

    print(f"[actor-hs] Loading model from: {args.model_path} (dtype={args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device).eval()
    print(f"[actor-hs] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, tokenizer


def load_batch_from_pt(path: str, device: str) -> dict[str, torch.Tensor]:
    """Mode 1: Load saved batch .pt from prior run."""
    print(f"[actor-hs] Loading batch from: {path}")
    batch = torch.load(path, map_location=device, weights_only=True)
    required = {"input_ids", "attention_mask", "position_ids", "responses"}
    missing = required - set(batch.keys())
    if missing:
        raise ValueError(f"Saved batch missing keys: {missing}. Available: {set(batch.keys())}")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
    return batch


def load_batch_from_parquet(args: argparse.Namespace, model, tokenizer, device: str) -> dict[str, torch.Tensor]:
    """Mode 2: Load prompts from parquet, generate responses, build batch."""
    from omegaconf import OmegaConf

    from torch.utils.data import DataLoader

    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

    data_cfg = OmegaConf.create({
        "prompt_key": args.prompt_key,
        "max_prompt_length": args.max_prompt_length,
        "return_raw_chat": False,
        "return_full_prompt": False,
        "filter_overlong_prompts": False,
        "truncation": "left",
    })

    dataset = RLHFDataset(data_files=args.train_file, tokenizer=tokenizer, config=data_cfg, processor=None)
    if len(dataset) == 0:
        raise RuntimeError(f"RLHFDataset is empty: {args.train_file}")

    loader = DataLoader(dataset, batch_size=args.num_prompts, shuffle=False, collate_fn=collate_fn)
    raw = next(iter(loader))

    prompt_ids = raw["input_ids"]  # (B, prompt_len), left-padded
    attention_mask = raw["attention_mask"]  # (B, prompt_len)

    print(f"[actor-hs] Loaded {prompt_ids.shape[0]} prompts, prompt_len={prompt_ids.shape[1]}")
    print(f"[actor-hs] Generating responses (greedy, max_new_tokens={args.max_response_length})...")

    prompt_ids_dev = prompt_ids.to(device)
    attention_mask_dev = attention_mask.to(device)

    with torch.no_grad():
        gen_output = model.generate(
            input_ids=prompt_ids_dev,
            attention_mask=attention_mask_dev,
            max_new_tokens=args.max_response_length,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # gen_output is (B, prompt_len + response_len)
    prompt_len = prompt_ids.shape[1]
    responses = gen_output[:, prompt_len:]  # (B, actual_response_len)
    print(f"[actor-hs] Generated responses: shape={list(responses.shape)}")

    # Build full input_ids = cat(prompt, response)
    full_ids = gen_output  # already concatenated
    full_mask = torch.ones_like(full_ids)
    # Preserve prompt padding in attention mask
    full_mask[:, :prompt_len] = attention_mask_dev
    full_pos = full_mask.cumsum(dim=-1) - 1
    full_pos = full_pos.clamp(min=0)

    return {
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "position_ids": full_pos,
        "responses": responses,
    }


def forward_with_hidden_states(
    model, input_ids, attention_mask, position_ids, responses, temperature=1.0
):
    """Replicate dp_actor._forward_micro_batch non-rmpad path.

    Returns hidden states per-sample as List[Tensor[seq_len, hidden_dim]] on CPU,
    matching dp_actor.compute_log_prob return format exactly.

    Reference: verl/workers/actor/dp_actor.py:286-331
    """
    response_length = responses.size(-1)
    device_type = "cuda" if input_ids.is_cuda else "cpu"

    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )

    # Log prob computation (same as dp_actor non-fused-kernel path)
    logits = output.logits / temperature
    logits_resp = logits[:, -response_length - 1 : -1, :]  # (B, response_length, vocab_size)
    log_probs = logprobs_from_logits(logits_resp, responses)

    # Last layer hidden states: (B, seq_len, hidden_dim)
    hidden_states_batched = output.hidden_states[-1]

    # Per-sample extraction to CPU — matches dp_actor.compute_log_prob non-rmpad path
    # (verl/workers/actor/dp_actor.py:427-432)
    hidden_states_lst = []
    for sample_idx in range(hidden_states_batched.size(0)):
        hidden_states_lst.append(hidden_states_batched[sample_idx].detach().cpu())

    return log_probs, hidden_states_lst


def build_summary(
    hidden_states_list: list[torch.Tensor],
    log_probs: torch.Tensor,
    input_ids: torch.Tensor,
    responses: torch.Tensor,
    response_hidden_states_list: list[torch.Tensor],
) -> dict:
    """Build JSON-serializable summary matching SGLang demo format."""
    summary = {
        "source": "actor_compute_log_prob",
        "num_samples": len(hidden_states_list),
        "input_ids_shape": list(input_ids.shape),
        "log_probs_shape": list(log_probs.shape),
        "hidden_state_shapes": [list(h.shape) for h in hidden_states_list],
        "hidden_state_dtypes": [str(h.dtype) for h in hidden_states_list],
        "response_hidden_state_shapes": [list(h.shape) for h in response_hidden_states_list],
        "log_probs_first_8": log_probs[0, :8].float().tolist() if log_probs.numel() > 0 else [],
    }

    # Per-sample stats (for numerical comparison without loading full tensors)
    per_sample_stats = []
    for i, h in enumerate(hidden_states_list):
        h_float = h.float()
        resp_h = response_hidden_states_list[i].float()
        per_sample_stats.append({
            "sample_idx": i,
            "full_sequence": {
                "shape": list(h.shape),
                "mean": h_float.mean().item(),
                "std": h_float.std().item(),
                "min": h_float.min().item(),
                "max": h_float.max().item(),
            },
            "response_only": {
                "shape": list(resp_h.shape),
                "mean": resp_h.mean().item(),
                "std": resp_h.std().item(),
                "min": resp_h.min().item(),
                "max": resp_h.max().item(),
            },
        })
    summary["hidden_state_stats"] = per_sample_stats

    return summary


def test_drafter_collect(drafter_batch: dict[str, torch.Tensor], hidden_states_lst: list[torch.Tensor], model_config):
    """Exercise eagle_background_trainer.collect_online_data with the collected data.

    This validates the full fsdp_workers.compute_log_prob → collect_online_data pipeline
    without requiring Ray, FSDP, or a real drafter model.
    """
    from verl.workers.drafter.eagle_background_trainer import EagleBackgroundTrainer

    print("\n[actor-hs] === Testing collect_online_data ===")
    print(f"  drafter_batch keys: {list(drafter_batch.keys())}")
    print(f"  drafter_batch['input_ids']: shape={list(drafter_batch['input_ids'].shape)}")
    print(f"  drafter_batch['responses']: shape={list(drafter_batch['responses'].shape)}")
    print(f"  hidden_states_lst: {len(hidden_states_lst)} samples")
    for i, hs in enumerate(hidden_states_lst):
        print(f"    sample {i}: shape={list(hs.shape)}, dtype={hs.dtype}, device={hs.device}")

    # Validate format compatibility with collect_online_data expectations
    input_ids = drafter_batch["input_ids"]
    assert input_ids.dim() == 2, f"input_ids should be 2D, got {input_ids.dim()}D"
    assert len(hidden_states_lst) == input_ids.size(0), (
        f"hidden_states_lst length ({len(hidden_states_lst)}) != batch_size ({input_ids.size(0)})"
    )
    for i, hs in enumerate(hidden_states_lst):
        assert hs.dim() == 2, f"hidden_states[{i}] should be 2D [seq_len, hidden_dim], got {hs.dim()}D"
        assert hs.size(0) == input_ids.size(1), (
            f"hidden_states[{i}] seq_len ({hs.size(0)}) != input_ids seq_len ({input_ids.size(1)})"
        )

    print("[actor-hs] Format validation passed.")

    # Actually call collect_online_data with a minimal EagleBackgroundTrainer mock
    # We only need the data collection path, not the training path
    class _MinimalTrainer:
        """Minimal mock that only exercises the collect_online_data code path."""
        def __init__(self, model_config):
            self.rank = 0
            self.model_config = model_config
            # Minimal data_buffer mock
            self.data_buffer = type("MockBuffer", (), {"add_batch": lambda self, b, h: None})()
            self.collected_data = __import__("collections").deque(maxlen=100)

    trainer = _MinimalTrainer(model_config)
    # Bind the real method to our mock
    trainer.collect_online_data = EagleBackgroundTrainer.collect_online_data.__get__(trainer)

    pre_count = len(trainer.collected_data)
    trainer.collect_online_data(drafter_batch, hidden_states_lst)
    post_count = len(trainer.collected_data)

    print(f"[actor-hs] collect_online_data added {post_count - pre_count} samples to deque (batch_size={input_ids.size(0)})")

    # Inspect collected samples
    for i, sample in enumerate(trainer.collected_data):
        print(f"  collected[{i}]: keys={list(sample.keys())}")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={list(v.shape)}, dtype={v.dtype}")

    print("[actor-hs] collect_online_data test passed.")


def main():
    args = _parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[actor-hs] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    # Validate input mode
    if args.input_batch_pt is None and args.train_file is None:
        print("ERROR: Must specify either --input-batch-pt or --train-file.", file=sys.stderr)
        sys.exit(1)

    # --- Load model (once) and data ---
    model, tokenizer = load_model(args)

    if args.input_batch_pt is not None:
        # Mode 1: load saved batch
        batch = load_batch_from_pt(args.input_batch_pt, args.device)
    else:
        # Mode 2: load prompts from parquet, generate responses
        batch = load_batch_from_parquet(args, model, tokenizer, args.device)

    # Optionally save the batch for reuse
    if args.save_batch_pt:
        out_path = Path(args.save_batch_pt).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {k: v.cpu() for k, v in batch.items()}
        torch.save(save_data, out_path)
        print(f"[actor-hs] Batch saved to: {out_path}")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    position_ids = batch["position_ids"]
    responses = batch["responses"]
    response_length = responses.size(-1)

    print(f"[actor-hs] input_ids: {list(input_ids.shape)}, responses: {list(responses.shape)}")

    # --- Simulate fsdp_workers.compute_log_prob flow ---
    # Step 1: Capture drafter_batch on CPU BEFORE data goes to GPU
    # (matches fsdp_workers.py: drafter_batch = {input_ids, responses} before data.to(device))
    drafter_batch = {
        "input_ids": input_ids.detach().cpu().clone(),
        "responses": responses.detach().cpu().clone(),
    }
    print(f"[actor-hs] Captured drafter_batch: input_ids={list(drafter_batch['input_ids'].shape)}, "
          f"responses={list(drafter_batch['responses'].shape)}")

    # Step 2: Forward pass with hidden states (micro-batched like dp_actor)
    micro_bs = args.micro_batch_size if args.micro_batch_size > 0 else input_ids.size(0)
    all_log_probs = []
    # Build hidden_states_lst per-sample per micro-batch (matches dp_actor.compute_log_prob)
    hidden_states_lst: list[torch.Tensor] = []

    num_micro = (input_ids.size(0) + micro_bs - 1) // micro_bs
    for i in range(num_micro):
        start = i * micro_bs
        end = min(start + micro_bs, input_ids.size(0))
        print(f"[actor-hs] Forward micro-batch {i + 1}/{num_micro} (samples {start}-{end - 1})...")

        lp, hs_lst = forward_with_hidden_states(
            model,
            input_ids[start:end],
            attention_mask[start:end],
            position_ids[start:end],
            responses[start:end],
            temperature=args.temperature,
        )
        all_log_probs.append(lp.cpu())
        hidden_states_lst.extend(hs_lst)  # extend, not append — already per-sample

    log_probs = torch.cat(all_log_probs, dim=0)  # (B, response_length)

    # Response-only hidden states (for comparison with SGLang which collects response tokens only)
    response_hidden_states_list = [
        hs[-response_length:, :] for hs in hidden_states_lst
    ]

    print(f"[actor-hs] Done. hidden_states: {len(hidden_states_lst)} samples, "
          f"each [{hidden_states_lst[0].shape[0]}, {hidden_states_lst[0].shape[1]}], "
          f"log_probs: {list(log_probs.shape)}")

    # --- Step 3: Test drafter collection if requested ---
    if args.test_drafter_collect:
        test_drafter_collect(drafter_batch, hidden_states_lst, model.config)

    # --- Summary ---
    summary = build_summary(
        hidden_states_lst, log_probs, input_ids, responses, response_hidden_states_list
    )

    print("\n[actor-hs] Summary:")
    print(json.dumps(summary, indent=2))

    if args.summary_json:
        out_path = Path(args.summary_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[actor-hs] Summary written to: {out_path}")

    # --- Save tensors ---
    if args.save_tensors:
        out_path = Path(args.save_tensors).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "hidden_states": hidden_states_lst,
            "response_hidden_states": response_hidden_states_list,
            "log_probs": log_probs,
            "input_ids": input_ids.cpu(),
            "responses": responses.cpu(),
        }, out_path)
        print(f"[actor-hs] Tensors saved to: {out_path}")

    print("[actor-hs] Done.")


if __name__ == "__main__":
    main()
