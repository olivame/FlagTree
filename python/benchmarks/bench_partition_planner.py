# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Benchmark and calibration utility for FlagTune multi-device partition planning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import types
from types import SimpleNamespace
from typing import Any, Dict, List

REPO_PYTHON = Path(__file__).resolve().parents[1]
if str(REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPO_PYTHON))

try:
    from triton._C.libtriton import getenv as _g  # noqa: F401
except Exception:
    triton_pkg = types.ModuleType("triton")
    triton_pkg.__path__ = [str(REPO_PYTHON / "triton")]
    triton_pkg.__version__ = "3.3.0"
    triton_pkg.next_power_of_2 = lambda n: 1 if n <= 1 else 1 << (int(n) - 1).bit_length()
    triton_pkg.cdiv = lambda x, y: (x + y - 1) // y
    c_pkg = types.ModuleType("triton._C")
    c_pkg.__path__ = []
    lib_stub = types.ModuleType("triton._C.libtriton")
    lib_stub.__path__ = []
    lib_stub.get_cache_invalidating_env_vars = lambda: {}
    lib_stub.native_specialize_impl = lambda *a, **kw: None
    lib_stub.ir = SimpleNamespace()
    lib_stub.passes = SimpleNamespace()
    lib_stub.llvm = SimpleNamespace()
    lib_stub.interpreter = SimpleNamespace()
    ll_mod = types.ModuleType("triton._C.libtriton.linear_layout")
    ll_mod.LinearLayout = object
    lib_stub.linear_layout = ll_mod
    lib_stub.getenv = lambda key, default=None: os.environ.get(key, default)
    lib_stub.getenv_bool = lambda key, default=None: (
        default
        if key not in os.environ
        else os.environ[key].strip().lower() in ("1", "true", "yes", "y", "on")
    )
    c_pkg.libtriton = lib_stub
    triton_pkg._C = c_pkg
    sys.modules["triton"] = triton_pkg
    sys.modules["triton._C"] = c_pkg
    sys.modules["triton._C.libtriton"] = lib_stub
    sys.modules["triton._C.libtriton.linear_layout"] = ll_mod

from triton.flagtune.runtime.partition_planner import (  # noqa: E402
    BucketedOperatorCostModel,
    CollectiveCostModel,
    HardwareClusterSpec,
    InferenceConstraints,
    ModelArchitectureSpec,
    OperatorBucketMeasurement,
    PartitionPlanner,
    TrainingConstraints,
)


PRESET_MODELS: Dict[str, ModelArchitectureSpec] = {
    "qwen3_embedding_0_6b": ModelArchitectureSpec(
        name="Qwen3-Embedding-0.6B",
        num_layers=28,
        hidden_size=1024,
        ffn_hidden_size=3072,
        num_attention_heads=16,
        num_kv_heads=8,
        vocab_size=151936,
    ),
    "bge_m3_encoder": ModelArchitectureSpec(
        name="BAAI-bge-m3",
        num_layers=24,
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        num_kv_heads=16,
        vocab_size=250002,
    ),
    "qwen3_reranker_8b": ModelArchitectureSpec(
        name="Qwen3-Reranker-8B",
        num_layers=36,
        hidden_size=4096,
        ffn_hidden_size=12288,
        num_attention_heads=32,
        num_kv_heads=8,
        vocab_size=151936,
    ),
    "onerec_moe_1_7b": ModelArchitectureSpec(
        name="OneRec-MoE-1.7B",
        num_layers=24,
        hidden_size=2048,
        ffn_hidden_size=1536,
        num_attention_heads=16,
        num_kv_heads=8,
        num_experts=8,
        top_k_experts=2,
        vocab_size=65536,
    ),
}


def _maybe_live_gemm_calibration(hidden_size: int) -> List[OperatorBucketMeasurement]:
    """Optionally calibrate representative bucket GEMMs if PyTorch accelerator is active."""
    try:
        import torch

        try:
            import torch_npu  # noqa: F401
        except Exception:
            pass
        has_npu = hasattr(torch, "npu") and torch.npu.is_available()
        has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
        has_cuda = torch.cuda.is_available()
        if not (has_npu or has_xpu or has_cuda):
            return []
        if has_npu:
            dev, sync = "npu", torch.npu.synchronize
        elif has_xpu:
            dev, sync = "xpu", torch.xpu.synchronize
        else:
            dev, sync = "cuda", torch.cuda.synchronize
        samples: List[OperatorBucketMeasurement] = []
        for tokens in (128, 512, 2048):
            a = torch.randn((tokens, hidden_size), dtype=torch.bfloat16, device=dev)
            b = torch.randn((hidden_size, hidden_size * 4), dtype=torch.bfloat16, device=dev)
            for _ in range(3):
                _ = a @ b
            sync()
            t0 = time.perf_counter()
            iters = 10
            for _ in range(iters):
                _ = a @ b
            sync()
            gemm_ms = ((time.perf_counter() - t0) * 1e3) / iters
            samples.append(
                OperatorBucketMeasurement(
                    op_name="transformer_layer_fwd",
                    bucket_tokens=tokens,
                    hidden_size=hidden_size,
                    tp_degree=1,
                    latency_ms=gemm_ms * 2.8,
                )
            )
        return samples
    except Exception:
        return []


def run_benchmark_suite(num_devices: int, memory_gb: int, enable_tle: bool) -> Dict[str, Any]:
    cluster = HardwareClusterSpec(
        num_devices=num_devices,
        device_memory_bytes=int(memory_gb) * (1024 ** 3),
        supports_tle=enable_tle,
    )
    live_samples = _maybe_live_gemm_calibration(1024) + _maybe_live_gemm_calibration(4096)
    planner = PartitionPlanner(
        cluster=cluster,
        compute_model=BucketedOperatorCostModel(measurements=live_samples),
        collective_model=CollectiveCostModel(),
    )

    results: Dict[str, Any] = {
        "cluster": {
            "num_devices": num_devices,
            "device_memory_gb": memory_gb,
            "supports_tle": enable_tle,
            "live_calibrated_buckets": len(live_samples),
        },
        "inference_reports": {},
        "training_reports": {},
    }

    # 1. Inference workloads across small retrieval encoders, 8B reranker/LLM, and MoE
    inf_small = planner.plan_inference(
        PRESET_MODELS["qwen3_embedding_0_6b"],
        InferenceConstraints(batch_size=32, prefill_seq_len=512, ttft_sla_ms=50.0),
    )
    inf_8b = planner.plan_inference(
        PRESET_MODELS["qwen3_reranker_8b"],
        InferenceConstraints(
            batch_size=8,
            prefill_seq_len=2048,
            decode_kv_len=4096,
            output_tokens=64,
            ttft_sla_ms=260.0,
            tpot_sla_ms=10.0,
        ),
    )
    inf_moe = planner.plan_inference(
        PRESET_MODELS["onerec_moe_1_7b"],
        InferenceConstraints(batch_size=16, prefill_seq_len=1024, output_tokens=32, moe_dispatch_skew=1.25),
    )
    for key, rep in (
        ("qwen3_embedding_0_6b", inf_small),
        ("qwen3_reranker_8b", inf_8b),
        ("onerec_moe_1_7b", inf_moe),
    ):
        results["inference_reports"][key] = rep.to_dict()

    # 2. Training workloads across dense encoder, 8B LLM/reranker, and MoE
    for key, gbs, seq_len in (
        ("bge_m3_encoder", 64, 512),
        ("qwen3_reranker_8b", 16, 4096),
        ("onerec_moe_1_7b", 32, 2048),
    ):
        rep = planner.plan_training(
            PRESET_MODELS[key],
            TrainingConstraints(global_batch_size=gbs, seq_len=seq_len, activation_checkpointing=(key == "qwen3_reranker_8b")),
        )
        results["training_reports"][key] = rep.to_dict()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FlagTune train/infer partition planner.")
    parser.add_argument("--devices", type=int, default=8, help="Number of devices in cluster.")
    parser.add_argument("--memory-gb", type=int, default=64, help="Per-device HBM capacity in GiB.")
    parser.add_argument("--no-tle", action="store_true", help="Disable TLE comm-compute overlap candidates.")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
    args = parser.parse_args()

    payload = run_benchmark_suite(args.devices, args.memory_gb, enable_tle=not args.no_tle)
    for mode_key in ("inference_reports", "training_reports"):
        print(f"=== {mode_key} ===")
        for name, rep in payload[mode_key].items():
            rec = rep["recommended"]
            rec_str = rec["plan_id"] if rec else "NONE"
            tps = rec["throughput_tokens_per_sec"] if rec else 0.0
            print(f"  {name}: {rec_str} ({tps:,.1f} tokens/s)")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
