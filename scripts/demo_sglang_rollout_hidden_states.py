#!/usr/bin/env python3
"""Standalone SGLangRollout hidden-state debugging script.

This script mirrors VerL's single-turn `SGLangRollout.generate_sequences(...)`
path closely enough to debug hidden-state collection without launching the full
trainer stack.

Two execution modes are supported:
- `--mode local`: run everything in the current process.
- `--mode ray`: run the rollout call in one Ray actor (controller-like boundary).

Example:
  python3 scripts/demo_sglang_rollout_hidden_states.py \
    --model-path /path/to/Qwen2.5-7B \
    --train-file /path/to/dapo.parquet \
    --num-prompts 2 \
    --max-prompt-length 1024 \
    --max-response-length 64 \
    --mode local
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import socket
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Ensure the repo root (containing `verl`) and vendored sglang are importable
# when running this script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Prefer sglang from the active env. Fallback to vendored copy if unavailable.
try:  # noqa: SIM105
    import sglang  # type: ignore  # noqa: F401
except Exception:
    sglang_py_root = REPO_ROOT / "third-party" / "sglang" / "python"
    if sglang_py_root.exists() and str(sglang_py_root) not in sys.path:
        sys.path.insert(0, str(sglang_py_root))

from verl.protocol import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one SGLangRollout.generate_sequences step and inspect hidden "
            "states passed to drafter training collector."
        )
    )
    parser.add_argument("--model-path", type=str, required=True, help="HF model id or local model path.")
    parser.add_argument("--train-file", type=str, required=True, help="RLHF parquet file path.")
    parser.add_argument("--num-prompts", type=int, default=2, help="Batch size for this debug run.")
    parser.add_argument("--max-prompt-length", type=int, default=1024, help="Dataset tokenization max prompt length.")
    parser.add_argument("--max-response-length", type=int, default=64, help="Rollout response length.")
    parser.add_argument(
        "--prompt-key",
        type=str,
        default="prompt",
        help="Prompt column key expected by RLHFDataset.",
    )
    parser.add_argument(
        "--truncation",
        choices=("left", "right", "middle", "error"),
        default="error",
        help="Prompt truncation strategy in RLHFDataset.",
    )
    parser.add_argument(
        "--filter-overlong-prompts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether RLHFDataset filters overlong prompts before sampling.",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set prompts.meta_info['do_sample'].",
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set prompts.meta_info['validate'] like validation path.",
    )
    parser.add_argument(
        "--rollout-n",
        type=int,
        default=1,
        help="Repeat generation batch N times (trainer-style rollout.n).",
    )
    parser.add_argument(
        "--global-steps",
        type=int,
        default=1,
        help="global_steps meta value passed into generation batch.",
    )
    parser.add_argument(
        "--pad-to-divisor",
        type=int,
        default=1,
        help="If >1, pad DataProto to divisor before generate and unpad output.",
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top_p.")
    parser.add_argument("--top-k", type=int, default=1, help="Sampling top_k.")
    parser.add_argument(
        "--disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set rollout.disable_cuda_graph.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.35,
        help="SGLang static memory fraction.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Rollout dtype.")
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="SGLang attention backend in engine_kwargs.sglang.attention_backend (e.g. fa3, flashinfer).",
    )
    parser.add_argument(
        "--collect-hidden-states-from-sgl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable speculative.train.collect_hidden_states_from_sgl.",
    )
    parser.add_argument(
        "--enable-drafter-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable speculative.train.enable_drafter_training gate.",
    )
    parser.add_argument(
        "--force-collect-on-first-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Override drafter_manager.should_collect_data_this_step to True.",
    )
    parser.add_argument(
        "--require-hidden-states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit with error if no hidden states were collected.",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail early if CUDA is unavailable.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Explicit CUDA_VISIBLE_DEVICES value for this run.",
    )
    parser.add_argument(
        "--mode",
        choices=("local", "ray"),
        default="local",
        help="Execution mode. ray mode runs rollout in a single Ray actor.",
    )
    parser.add_argument("--ray-address", type=str, default=None, help="Optional Ray address (for --mode ray).")
    parser.add_argument("--ray-num-cpus", type=float, default=2, help="Ray actor CPU reservation.")
    parser.add_argument("--ray-num-gpus", type=float, default=1, help="Ray actor GPU reservation.")
    parser.add_argument(
        "--shutdown-ray",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shutdown Ray after the run in --mode ray.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to HF and rollout model loading.",
    )
    parser.add_argument("--master-port", type=int, default=None, help="torch.distributed master port.")
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help="Optional output path to write JSON summary.",
    )
    parser.add_argument(
        "--print-traceback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print full traceback on error.",
    )
    return parser.parse_args()


def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        return int(s.getsockname()[1])


def _init_dist_single_process(master_port: int | None = None, cuda_visible_devices: str | None = None) -> None:
    """Initialize torch.distributed for a one-process debug run."""
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port if master_port is not None else _get_free_port())
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=1,
            rank=0,
        )


def _build_rollout_config(args: argparse.Namespace) -> DictConfig:
    """Build minimal rollout config for direct SGLangRollout usage."""
    cfg = {
        "name": "sglang",
        "mode": "sync",
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_model_parallel_size": 1,
        "prompt_length": args.max_prompt_length,
        "response_length": args.max_response_length,
        "max_model_len": args.max_prompt_length + args.max_response_length,
        "ignore_eos": False,
        "free_cache_engine": True,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "disable_cuda_graph": args.disable_cuda_graph,
        "calculate_log_probs": False,
        "multi_stage_wake_up": False,
        "load_format": "dummy_hf",
        "engine_kwargs": {"sglang": {"attention_backend": args.attention_backend}},
        "val_kwargs": {"top_k": -1, "top_p": 1.0, "temperature": 0.0, "n": 1, "do_sample": False},
        "multi_turn": {
            "enable": False,
            "max_assistant_turns": None,
            "max_user_turns": None,
            "tool_config_path": None,
            "interaction_config_path": None,
            "max_parallel_calls": 1,
            "max_tool_response_length": 256,
            "tool_response_truncate_side": "middle",
            "use_inference_chat_template": False,
            "tokenization_sanity_check_mode": "strict",
            "format": "hermes",
        },
        # Hidden-state return in SGLangRollout is gated by speculative + train flags.
        "speculative": {
            "enable": True,
            "spec_strategy": "eagle",
            "bs_threshold": 32,
            "eagle": {
                "spec_model_path": None,
                "spec_steps": 8,
                "spec_topk": 4,
                "spec_verify_tokens": 48,
                "tune_algorithm": "BEG",
                "mab_configs": ["8_4_32", "8_4_16", "8_4_8"],
                "mab_bs_threshold": [1, 2, 5, 21],
            },
            "train": {
                "enable_drafter_training": args.enable_drafter_training,
                "training_interval_steps": 1,
                "batch_size_per_gpu": 2,
                "max_seq_len": args.max_prompt_length + args.max_response_length,
                "max_epochs": 1,
                "checkpoint_path": None,
                "min_workers_for_training": 1,
                "collect_hidden_states_from_sgl": args.collect_hidden_states_from_sgl,
            },
        },
    }
    return OmegaConf.create(cfg)


class DummyBackgroundTrainer:
    """Minimal collector matching `collect_online_data(batch, hidden_states)`."""

    def __init__(self):
        self.collected: list[dict[str, Any]] = []

    def collect_online_data(self, batch: dict[str, Any], hidden_states: list[torch.Tensor]):
        self.collected.append(
            {
                "batch": {
                    k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
                },
                "hidden_states": [h.detach().cpu().clone() for h in hidden_states],
            }
        )


class DummyShardingManager:
    """No-op sharding manager that satisfies release hook used by rollout."""

    async def release_memory(self):
        return True


def _load_raw_batch(args: argparse.Namespace, tokenizer) -> DataProto:
    data_cfg = OmegaConf.create(
        {
            "prompt_key": args.prompt_key,
            "max_prompt_length": args.max_prompt_length,
            "return_raw_chat": True,
            "return_full_prompt": True,
            "filter_overlong_prompts": args.filter_overlong_prompts,
            "truncation": args.truncation,
        }
    )

    dataset = RLHFDataset(
        data_files=args.train_file,
        tokenizer=tokenizer,
        config=data_cfg,
        processor=None,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"RLHFDataset is empty after filtering: {args.train_file}")

    loader = DataLoader(dataset, batch_size=args.num_prompts, shuffle=False, collate_fn=collate_fn)
    batch_dict = next(iter(loader))
    return DataProto.from_single_dict(batch_dict)


def _prepare_gen_batch_like_trainer(batch: DataProto, args: argparse.Namespace, tokenizer) -> DataProto:
    """Mirror RayPPOTrainer preprocessing before worker.generate_sequences."""
    # Add uid like trainer does.
    if "uid" not in batch.non_tensor_batch:
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
    for key in (
        "multi_modal_data",
        "raw_prompt",
        "tools_kwargs",
        "interaction_kwargs",
        "index",
        "agent_name",
    ):
        if key in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append(key)

    gen_batch = batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
    )

    # Worker wrapper sets eos/pad token ids. Add them here since we call rollout directly.
    gen_batch.meta_info.update(
        {
            "global_steps": args.global_steps,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": args.do_sample,
            "validate": args.validate,
        }
    )
    if args.validate:
        gen_batch.meta_info["recompute_log_prob"] = False

    if args.rollout_n > 1:
        gen_batch = gen_batch.repeat(repeat_times=args.rollout_n, interleave=True)

    # Worker wrapper moves prompts to device before rollout call.
    gen_batch = gen_batch.to("cuda" if torch.cuda.is_available() else "cpu")
    return gen_batch


def _build_summary(collected: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_collected_samples": len(collected),
        "sample": None,
    }
    if not collected:
        return summary

    sample = collected[0]
    batch = sample["batch"]
    hidden_states_list = sample["hidden_states"]
    sample_summary: dict[str, Any] = {
        "batch_keys": sorted(batch.keys()),
        "num_hidden_state_tensors": len(hidden_states_list),
        "hidden_state_shapes": [list(t.shape) for t in hidden_states_list],
        "hidden_state_dtypes": [str(t.dtype) for t in hidden_states_list],
    }
    if "input_ids" in batch:
        sample_summary["input_ids_shape"] = list(batch["input_ids"].shape)
        sample_summary["input_ids_dtype"] = str(batch["input_ids"].dtype)
        sample_summary["first_input_ids_16"] = batch["input_ids"][0][:16].tolist()
    if "loss_mask" in batch:
        sample_summary["loss_mask_shape"] = list(batch["loss_mask"].shape)
        sample_summary["loss_mask_nonzero_first_32"] = (
            batch["loss_mask"][0].nonzero(as_tuple=False).view(-1)[:32].tolist()
        )

    summary["sample"] = sample_summary
    return summary


def _run_rollout_once(args: argparse.Namespace) -> dict[str, Any]:
    rollout = None
    _ensure_event_loop()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable but SGLangRollout usually requires CUDA. "
            "Either run on a CUDA machine or pass --no-require-cuda for quick dry checks."
        )

    _init_dist_single_process(master_port=args.master_port, cuda_visible_devices=args.cuda_visible_devices)

    from transformers import AutoConfig, AutoTokenizer
    from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

    rollout_cfg = _build_rollout_config(args)

    print(f"[demo] Loading model config/tokenizer from: {args.model_path}")
    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id; cannot build prompt padding.")
        tokenizer.pad_token = tokenizer.eos_token
        print(f"[demo] tokenizer.pad_token_id was None. Reusing eos_token_id={tokenizer.eos_token_id} as pad token.")

    raw_batch = _load_raw_batch(args, tokenizer)
    gen_batch = _prepare_gen_batch_like_trainer(raw_batch, args, tokenizer)

    try:
        print("[demo] Initializing SGLangRollout...")
        rollout = SGLangRollout(
            actor_module=args.model_path,
            config=rollout_cfg,
            processing_class=tokenizer,
            model_hf_config=hf_config,
            trust_remote_code=args.trust_remote_code,
        )

        rollout.sharding_manager = DummyShardingManager()
        dummy_trainer = DummyBackgroundTrainer()
        rollout.drafter_manager.background_trainer = dummy_trainer

        # No training in this debug script; only collect hidden states.
        rollout.drafter_manager.should_train_this_step = lambda: False
        if args.force_collect_on_first_step:
            rollout.drafter_manager.should_collect_data_this_step = lambda: True

        print("[demo] Running generate_sequences...")
        if args.pad_to_divisor > 1:
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor=args.pad_to_divisor)
            output_padded = rollout.generate_sequences(gen_batch_padded)
            _ = unpad_dataproto(output_padded, pad_size=pad_size)
        else:
            _ = rollout.generate_sequences(gen_batch)
        summary = _build_summary(dummy_trainer.collected)

        if args.require_hidden_states and summary["num_collected_samples"] == 0:
            raise RuntimeError(
                "No hidden states were collected. Check: "
                "speculative.train.collect_hidden_states_from_sgl=True, "
                "enable_drafter_training=True, and collection gating hooks."
            )

        return summary
    finally:
        # Cleanup best-effort to avoid orphaned SGLang subprocesses.
        if rollout is not None and hasattr(rollout, "_engine") and rollout._engine is not None:
            with contextlib.suppress(Exception):
                rollout._engine.shutdown()
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                dist.destroy_process_group()


def _run_rollout_with_ray(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import ray
    except Exception as exc:
        raise RuntimeError("Ray mode requested but `ray` is not importable in this environment.") from exc

    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    if args.ray_address:
        init_kwargs["address"] = args.ray_address
    ray.init(**init_kwargs)

    @ray.remote(num_cpus=args.ray_num_cpus, num_gpus=args.ray_num_gpus)
    class _RayRolloutActor:
        def run_once(self, args_dict: dict[str, Any]) -> dict[str, Any]:
            local_args = argparse.Namespace(**args_dict)
            local_args.mode = "local"
            return _run_rollout_once(local_args)

    actor = _RayRolloutActor.remote()
    args_dict = vars(args).copy()
    args_dict["mode"] = "local"
    try:
        return ray.get(actor.run_once.remote(args_dict))
    finally:
        if args.shutdown_ray:
            ray.shutdown()


def _print_error_hints(error: Exception) -> None:
    msg = str(error)
    hints: list[str] = []
    if "No hidden states were collected" in msg:
        hints.append(
            "- Force collection gate for first step: use --force-collect-on-first-step "
            "(enabled by default in this script)."
        )
        hints.append(
            "- Verify speculative flags: --collect-hidden-states-from-sgl and --enable-drafter-training must be true."
        )
    if "Prompt length" in msg and "longer than" in msg:
        hints.append("- Reduce --max-prompt-length or set --truncation left/right/middle.")
    if "max_position_embeddings" in msg or "model context length" in msg:
        hints.append("- Lower max lengths: --max-prompt-length / --max-response-length.")
    if "CUDA is unavailable" in msg:
        hints.append("- Run on GPU machine or pass --no-require-cuda for dry checks.")
    if "ModuleNotFoundError" in msg:
        hints.append("- Ensure env has torch/transformers/omegaconf/sglang installed.")

    if hints:
        print("[demo] Hints:", file=sys.stderr)
        for hint in hints:
            print(hint, file=sys.stderr)


def main() -> None:
    args = _parse_args()
    try:
        if args.mode == "ray":
            summary = _run_rollout_with_ray(args)
        else:
            summary = _run_rollout_once(args)

        print("\n[demo] Success. Hidden-state collection summary:")
        print(json.dumps(summary, indent=2))

        if args.summary_json:
            out_path = Path(args.summary_json).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[demo] Summary written to: {out_path}")
    except Exception as exc:
        print(f"[demo] ERROR: {exc}", file=sys.stderr)
        _print_error_hints(exc)
        if args.print_traceback:
            traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
