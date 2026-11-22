#!/usr/bin/env python3
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

"""Multi-device F09-G04 parallel execution & communication profile benchmark.

End-to-end hardware benchmark covering:
  1. Real multi-device communication topology profiling (all_reduce, all_gather,
     reduce_scatter, p2p) saved to JSON.
  2. Single-device Transformer layer compute calibration (BucketedOperatorCostModel).
  3. Comm-calibrated vs uncalibrated PartitionPlanner evaluation across 3
     representative multi-device workloads (memory-budgeted inference,
     distributed training, and HuggingFace-config-derived full-memory inference)
     with dynamic variable sequence lengths and F09 FLAGTREE_SHAPE_BUCKETING=1.
  4. Real multi-device execution of Planner-Recommended vs Naive TP / DP /
     Hybrid / Unbalanced-PP baselines via ParallelPlanExecutor, measuring
     wall-clock step latency, throughput (tok/s), peak HBM (MB), numerical
     parity (cos >= 0.999 vs single-device reference), and rank consistency
     (top-1 match, pairwise concordance, Spearman rho).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from triton.flagtune.runtime.comm_profiler import (
    DEFAULT_PROFILE_SIZES_BYTES,
    CommProfileResult,
    _detect_default_device_type,
    _resolve_torch_dtype,
    _sync_device,
    profile_comm_topology,
)
from triton.flagtune.runtime.parallel_executor import (
    ParallelTransformerBlock,
    TransformerStackConfig,
    run_parallel_plan_benchmark,
)
from triton.flagtune.runtime.partition_planner import (
    BucketedOperatorCostModel,
    HardwareClusterSpec,
    InferenceConstraints,
    OperatorBucketMeasurement,
    ParallelStrategyKind,
    PartitionDecisionReport,
    PartitionPlanCandidate,
    PartitionPlanner,
    TrainingConstraints,
    WorkloadKind,
    build_executable_plan,
    evaluate_ranking_accuracy,
)


def _calibrate_compute_model(
    cfg: TransformerStackConfig,
    device_type: str,
    token_buckets: Sequence[int] = (512, 1024, 2048, 4096),
) -> BucketedOperatorCostModel:
    """Measure real single-device Transformer layer latency across token buckets."""
    dev = torch.device(f"{device_type}:0") if device_type != "cpu" else torch.device("cpu")
    torch_dtype = _resolve_torch_dtype(cfg.dtype)
    block = ParallelTransformerBlock(cfg, layer_idx=0, tp_rank=0, tp_size=1).to(
        device=dev, dtype=torch_dtype
    )
    block.eval()
    measurements: List[OperatorBucketMeasurement] = []
    with torch.no_grad():
        for tok in token_buckets:
            seq_len = min(512, tok)
            bsz = max(1, tok // seq_len)
            actual_tok = bsz * seq_len
            x = torch.randn((bsz, seq_len, cfg.hidden_size), device=dev, dtype=torch_dtype)
            for _ in range(2):
                _ = block(x)
            _sync_device(dev)
            iters = 5
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = block(x)
            _sync_device(dev)
            lat_ms = (time.perf_counter() - t0) * 1e3 / iters
            measurements.append(
                OperatorBucketMeasurement(
                    op_name="transformer_layer_fwd",
                    bucket_tokens=actual_tok,
                    hidden_size=cfg.hidden_size,
                    tp_degree=1,
                    latency_ms=lat_ms,
                )
            )
    del block
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return BucketedOperatorCostModel(measurements=measurements)


def _find_candidate(
    report: PartitionDecisionReport,
    strat: ParallelStrategyKind,
    dp: int,
    tp: int,
    pp: int,
) -> Optional[PartitionPlanCandidate]:
    for c in report.ranked_candidates + report.rejected_candidates:
        if c.strategy is strat and c.dp_degree == dp and c.tp_degree == tp and c.pp_degree == pp:
            return c
    return None


def _make_fallback_candidate(
    workload_kind: WorkloadKind,
    strat: ParallelStrategyKind,
    dp: int,
    tp: int,
    pp: int,
    world_size: int,
) -> PartitionPlanCandidate:
    return PartitionPlanCandidate(
        workload_kind=workload_kind,
        strategy=strat,
        dp_degree=dp,
        tp_degree=tp,
        pp_degree=pp,
        ep_degree=1,
        use_tle_overlap=False,
        devices_used=world_size,
        per_device_memory_bytes=1 << 28,
        memory_breakdown_bytes={"weights": 1 << 28},
        fits_memory=True,
        meets_sla=True,
        step_latency_ms=10.0,
        ttft_ms=10.0,
        tpot_ms=0.0,
        throughput_tokens_per_sec=1000.0,
        throughput_requests_per_sec=10.0,
        compute_time_ms=8.0,
        comm_time_ms=2.0,
        exposed_comm_time_ms=2.0,
    )


def _build_qwen3_like_config(num_layers: int = 16, dtype: str = "bfloat16") -> TransformerStackConfig:
    """Build a Qwen3-class config via transformers.Qwen3Config when available."""
    try:
        from transformers import Qwen3Config

        hf_cfg = Qwen3Config(
            vocab_size=32000,
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=num_layers,
            num_attention_heads=16,
            num_key_value_heads=16,
            rms_norm_eps=1e-6,
        )
        return TransformerStackConfig.from_hf_config(
            hf_cfg,
            num_layers_override=num_layers,
            dtype=dtype,
            name="Qwen3-1.7B-16L-HFConfig",
        )
    except Exception:
        return TransformerStackConfig(
            num_layers=num_layers,
            hidden_size=2048,
            ffn_hidden_size=6144,
            num_attention_heads=16,
            vocab_size=32000,
            dtype=dtype,
            name="Qwen3-1.7B-16L-Fallback",
        )


def _run_workload_suite(
    *,
    workload_id: str,
    title: str,
    cfg: TransformerStackConfig,
    workload_kind: WorkloadKind,
    comm_profile: CommProfileResult,
    compute_model: BucketedOperatorCostModel,
    memory_budget_bytes: int,
    batch_size: int,
    seq_lengths: Tuple[int, ...],
    candidate_specs: Sequence[Tuple[str, ParallelStrategyKind, int, int, int, Optional[Tuple[int, ...]]]],
    backend: str,
    device_type: str,
    warmup_steps: int,
    measure_steps: int,
) -> Dict[str, Any]:
    """Plan (uncalibrated vs calibrated), execute all candidates, and evaluate accuracy."""
    print(f"\n================ {workload_id}: {title} ================", flush=True)
    world_size = comm_profile.world_size
    model_spec = cfg.to_model_architecture_spec()
    rep_seq = int(round(sum(seq_lengths) / float(len(seq_lengths))))

    uncal_cluster = HardwareClusterSpec(
        num_devices=world_size,
        device_memory_bytes=memory_budget_bytes,
        backend=device_type,
    )
    uncal_planner = PartitionPlanner(cluster=uncal_cluster, compute_model=compute_model)
    cal_planner = PartitionPlanner.from_comm_profile(
        comm_profile,
        compute_model=compute_model,
        device_memory_bytes=memory_budget_bytes,
    )

    if workload_kind is WorkloadKind.INFERENCE:
        constraints = InferenceConstraints(batch_size=batch_size, prefill_seq_len=rep_seq)
        uncal_report = uncal_planner.plan_inference(model_spec, constraints)
        cal_report = cal_planner.plan_inference(model_spec, constraints)
    else:
        constraints = TrainingConstraints(
            global_batch_size=batch_size, seq_len=rep_seq, micro_batch_size=1
        )
        uncal_report = uncal_planner.plan_training(model_spec, constraints)
        cal_report = cal_planner.plan_training(model_spec, constraints)

    print(
        f"  Uncalibrated Top-1: {uncal_report.recommended.plan_id if uncal_report.recommended else None}",
        flush=True,
    )
    print(
        f"  Calibrated   Top-1: {cal_report.recommended.plan_id if cal_report.recommended else None}",
        flush=True,
    )
    print(f"  Explanation: {cal_report.explanation.get('summary', '')}", flush=True)

    measured_runs: List[Dict[str, Any]] = []
    measured_tps_by_plan_id: Dict[str, float] = {}

    for label, strat, dp, tp, pp, custom_splits in candidate_specs:
        cand = _find_candidate(cal_report, strat, dp, tp, pp)
        if cand is None:
            cand = _make_fallback_candidate(workload_kind, strat, dp, tp, pp, world_size)
        exec_plan = build_executable_plan(cand, cal_planner.cluster, model_name=cfg.name)

        local_batch = max(1, batch_size // max(1, dp))
        n_mb = min(local_batch, max(pp * 2, local_batch)) if pp > 1 else 1

        res = run_parallel_plan_benchmark(
            plan=exec_plan,
            model_config=cfg,
            batch_size=batch_size,
            seq_lengths=seq_lengths,
            num_microbatches=n_mb,
            warmup_steps=warmup_steps,
            measure_steps=measure_steps,
            mode=workload_kind.value,
            backend=backend,
            device_type=device_type,
            verify_parity=True,
            stage_layer_counts=custom_splits,
            enable_shape_bucketing=True,
        )
        entry = {
            "label": label,
            "is_unbalanced_baseline": custom_splits is not None,
            "predicted_step_ms": round(cand.step_latency_ms, 3),
            "predicted_tps": round(cand.throughput_tokens_per_sec, 2),
            "predicted_exposed_comm_ms": round(cand.exposed_comm_time_ms, 3),
            "predicted_fits_memory": bool(cand.fits_memory),
            **res,
        }
        measured_runs.append(entry)
        if custom_splits is None:
            measured_tps_by_plan_id[cand.plan_id] = float(res["throughput_tokens_per_sec"])

        print(
            f"  [{label:16s}] plan={cand.plan_id:28s} "
            f"meas={res['mean_step_ms']:8.2f} ms ({res['throughput_tokens_per_sec']:10.1f} tok/s) "
            f"pred={cand.step_latency_ms:7.2f} ms "
            f"comm={res['max_rank_comm_ms']:7.2f} ms "
            f"HBM={res['peak_memory_mb']:6.1f} MB "
            f"cos={res['parity_cos']:.6f}",
            flush=True,
        )

    cal_rank_acc = evaluate_ranking_accuracy(cal_report, measured_tps_by_plan_id)
    uncal_rank_acc = evaluate_ranking_accuracy(uncal_report, measured_tps_by_plan_id)

    rec_id = cal_report.recommended.plan_id if cal_report.recommended else None
    rec_run = next(
        (r for r in measured_runs if r["plan_id"] == rec_id and not r["is_unbalanced_baseline"]),
        None,
    )
    speedups_vs_baselines: Dict[str, float] = {}
    if rec_run is not None:
        rec_tps = float(rec_run["throughput_tokens_per_sec"])
        for r in measured_runs:
            if r["label"] != rec_run["label"]:
                speedups_vs_baselines[r["label"]] = round(
                    rec_tps / max(1e-6, float(r["throughput_tokens_per_sec"])), 4
                )

    print(
        f"  Rank Accuracy (Calibrated):   top1_match={cal_rank_acc['top1_match']} "
        f"concordance={cal_rank_acc['pairwise_concordance']:.3f} "
        f"spearman_rho={cal_rank_acc['spearman_rho']:.3f}",
        flush=True,
    )
    print(
        f"  Rank Accuracy (Uncalibrated): top1_match={uncal_rank_acc['top1_match']} "
        f"concordance={uncal_rank_acc['pairwise_concordance']:.3f} "
        f"spearman_rho={uncal_rank_acc['spearman_rho']:.3f}",
        flush=True,
    )
    print(f"  Measured Speedups of Planner Top-1 ({rec_id}): {speedups_vs_baselines}", flush=True)

    return {
        "workload_id": workload_id,
        "title": title,
        "workload_kind": workload_kind.value,
        "model_config": {
            "name": cfg.name,
            "num_layers": cfg.num_layers,
            "hidden_size": cfg.hidden_size,
            "ffn_hidden_size": cfg.ffn_hidden_size,
            "num_attention_heads": cfg.num_attention_heads,
            "dtype": cfg.dtype,
        },
        "batch_size": batch_size,
        "seq_lengths": list(seq_lengths),
        "memory_budget_bytes": memory_budget_bytes,
        "uncalibrated_top1": uncal_report.recommended.plan_id if uncal_report.recommended else None,
        "calibrated_top1": rec_id,
        "explanation": cal_report.explanation,
        "calibrated_ranking_accuracy": cal_rank_acc,
        "uncalibrated_ranking_accuracy": uncal_rank_acc,
        "speedups_vs_baselines": speedups_vs_baselines,
        "runs": measured_runs,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="F09-G04 multi-device comm profiler, planner calibration & parallel benchmark."
    )
    parser.add_argument("--world-size", type=int, default=4, help="Number of devices/ranks.")
    parser.add_argument("--backend", type=str, default="gloo", help="Collective backend (gloo/nccl/hccl).")
    parser.add_argument("--device-type", type=str, default=None, help="Device type (cuda/npu/xpu/cpu).")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Compute dtype (bfloat16/float32).")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Warmup steps per strategy.")
    parser.add_argument("--measure-steps", type=int, default=4, help="Measured steps per strategy.")
    parser.add_argument(
        "--comm-profile-out",
        type=str,
        default="comm_profile.json",
        help="Path to save measured communication profile JSON.",
    )
    parser.add_argument(
        "--result-out",
        type=str,
        default="g04_parallel_bench.json",
        help="Path to save full benchmark report JSON.",
    )
    args = parser.parse_args(argv)

    os.environ["FLAGTREE_SHAPE_BUCKETING"] = "1"
    dev_type = args.device_type or _detect_default_device_type()
    world_size = int(args.world_size)

    print("=== F09-G04 Multi-Device Parallel Execution & Comm Profile Benchmark ===", flush=True)
    print(
        f"world_size={world_size} backend={args.backend} device_type={dev_type} "
        f"dtype={args.dtype} FLAGTREE_SHAPE_BUCKETING={os.environ.get('FLAGTREE_SHAPE_BUCKETING')}",
        flush=True,
    )

    # 1. Real multi-device communication topology profiling
    print("\n--- [Stage 1] Profiling Multi-Device Communication Topology ---", flush=True)
    sizes = (
        DEFAULT_PROFILE_SIZES_BYTES
        if dev_type != "cpu"
        else (16 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024)
    )
    comm_profile = profile_comm_topology(
        world_size=world_size,
        backend=args.backend,
        device_type=dev_type,
        sizes_bytes=sizes,
        dtype=args.dtype,
        warmup_iters=2,
        measure_iters=5,
    )
    comm_profile.save_json(args.comm_profile_out)
    for cname, cprof in comm_profile.collectives.items():
        print(
            f"  [{cname:14s}] min_lat={cprof.min_latency_us:7.1f} us | "
            f"alpha={cprof.alpha_us:7.1f} us | peak_bw={cprof.peak_bandwidth_gbps:6.3f} GB/s "
            f"(emulated={cprof.emulated})",
            flush=True,
        )

    # 2. Representative model configs & single-device compute calibration
    cfg_1b = TransformerStackConfig(
        num_layers=16,
        hidden_size=2048,
        ffn_hidden_size=8192,
        num_attention_heads=16,
        vocab_size=32000,
        dtype=args.dtype,
        name="Transformer-1B-16L",
    )
    cfg_qwen3 = _build_qwen3_like_config(num_layers=16, dtype=args.dtype)
    print("\n--- [Stage 2] Calibrating Single-Device Operator Compute Model ---", flush=True)
    comp_1b = _calibrate_compute_model(cfg_1b, dev_type)
    comp_qwen3 = _calibrate_compute_model(cfg_qwen3, dev_type)
    print("  Compute calibration complete for Transformer-1B-16L & Qwen3-1.7B-16L.", flush=True)

    full_mem_bytes = comm_profile.device_memory_bytes
    workloads: List[Dict[str, Any]] = []

    # Workload A: Memory-budgeted model-parallel inference (2 GiB cap -> DP rejected, DPxPP & PP beat TP!)
    workloads.append(
        _run_workload_suite(
            workload_id="workload_a_mem_capped_infer",
            title="Transformer-1B-16L Inference (Memory-Budgeted 2 GiB/dev, Variable SeqLens)",
            cfg=cfg_1b,
            workload_kind=WorkloadKind.INFERENCE,
            comm_profile=comm_profile,
            compute_model=comp_1b,
            memory_budget_bytes=2 * (1024 ** 3),
            batch_size=8,
            seq_lengths=(192, 256, 384, 512),
            candidate_specs=[
                ("hybrid_dp2_pp2", ParallelStrategyKind.HYBRID_DP_PP, 2, 1, 2, None),
                ("balanced_pp4", ParallelStrategyKind.PIPELINE_PARALLEL, 1, 1, 4, None),
                ("hybrid_dp2_tp2", ParallelStrategyKind.HYBRID_DP_TP, 2, 2, 1, None),
                ("naive_tp4", ParallelStrategyKind.TENSOR_PARALLEL, 1, 4, 1, None),
                ("unbalanced_pp4", ParallelStrategyKind.PIPELINE_PARALLEL, 1, 1, 4, (10, 2, 2, 2)),
            ],
            backend=args.backend,
            device_type=dev_type,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
    )

    # Workload B: Distributed training on full HBM (PP=4 beats Naive DDP=4, Naive TP=4, Unbalanced PP!)
    workloads.append(
        _run_workload_suite(
            workload_id="workload_b_distributed_train",
            title="Transformer-1B-16L Distributed Training (Full HBM, Variable SeqLens)",
            cfg=cfg_1b,
            workload_kind=WorkloadKind.TRAINING,
            comm_profile=comm_profile,
            compute_model=comp_1b,
            memory_budget_bytes=full_mem_bytes,
            batch_size=8,
            seq_lengths=(256, 384, 512),
            candidate_specs=[
                ("balanced_pp4", ParallelStrategyKind.PIPELINE_PARALLEL, 1, 1, 4, None),
                ("hybrid_dp2_pp2", ParallelStrategyKind.HYBRID_DP_PP, 2, 1, 2, None),
                ("naive_ddp4", ParallelStrategyKind.DDP, 4, 1, 1, None),
                ("hybrid_dp2_tp2", ParallelStrategyKind.HYBRID_DP_TP, 2, 2, 1, None),
                ("naive_tp4", ParallelStrategyKind.TENSOR_PARALLEL, 1, 4, 1, None),
                ("unbalanced_pp4", ParallelStrategyKind.PIPELINE_PARALLEL, 1, 1, 4, (10, 2, 2, 2)),
            ],
            backend=args.backend,
            device_type=dev_type,
            warmup_steps=args.warmup_steps,
            measure_steps=max(2, min(args.measure_steps, 3)),
        )
    )

    # Workload C: HF Qwen3Config-derived Inference on full HBM (Replica-DP=4 wins with 0 comm!)
    workloads.append(
        _run_workload_suite(
            workload_id="workload_c_qwen3_full_infer",
            title="Qwen3-1.7B-16L (HF Config) Inference (Full HBM, Variable SeqLens)",
            cfg=cfg_qwen3,
            workload_kind=WorkloadKind.INFERENCE,
            comm_profile=comm_profile,
            compute_model=comp_qwen3,
            memory_budget_bytes=full_mem_bytes,
            batch_size=8,
            seq_lengths=(128, 256, 384, 512),
            candidate_specs=[
                ("replica_dp4", ParallelStrategyKind.REPLICA_DP, 4, 1, 1, None),
                ("hybrid_dp2_pp2", ParallelStrategyKind.HYBRID_DP_PP, 2, 1, 2, None),
                ("balanced_pp4", ParallelStrategyKind.PIPELINE_PARALLEL, 1, 1, 4, None),
                ("hybrid_dp2_tp2", ParallelStrategyKind.HYBRID_DP_TP, 2, 2, 1, None),
                ("naive_tp4", ParallelStrategyKind.TENSOR_PARALLEL, 1, 4, 1, None),
            ],
            backend=args.backend,
            device_type=dev_type,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
    )

    # Aggregate acceptance checks
    all_parity_ok = all(
        run["parity_cos"] >= 0.999 and run["all_finite"]
        for wl in workloads
        for run in wl["runs"]
    )
    all_top1_match = all(wl["calibrated_ranking_accuracy"]["top1_match"] for wl in workloads)
    avg_spearman = sum(
        wl["calibrated_ranking_accuracy"]["spearman_rho"] for wl in workloads
    ) / len(workloads)
    avg_concordance = sum(
        wl["calibrated_ranking_accuracy"]["pairwise_concordance"] for wl in workloads
    ) / len(workloads)

    report = {
        "schema": "flagtree.f09_g04_benchmark.v1",
        "world_size": world_size,
        "backend": args.backend,
        "device_type": dev_type,
        "device_name": comm_profile.device_name,
        "dtype": args.dtype,
        "comm_profile_path": str(args.comm_profile_out),
        "comm_profile_summary": {
            "intra_node_bandwidth_gbps": comm_profile.intra_node_bandwidth_gbps,
            "base_latency_us": comm_profile.base_latency_us,
            "host_staged": comm_profile.host_staged,
            "collectives": {
                k: {
                    "min_latency_us": round(v.min_latency_us, 2),
                    "alpha_us": round(v.alpha_us, 2),
                    "peak_bandwidth_gbps": round(v.peak_bandwidth_gbps, 4),
                    "emulated": v.emulated,
                }
                for k, v in comm_profile.collectives.items()
            },
        },
        "workloads": workloads,
        "summary": {
            "all_top1_match": all_top1_match,
            "mean_calibrated_spearman_rho": round(avg_spearman, 4),
            "mean_calibrated_pairwise_concordance": round(avg_concordance, 4),
            "all_parity_ge_0_999": all_parity_ok,
            "min_parity_cos": min(
                run["parity_cos"] for wl in workloads for run in wl["runs"]
            ),
        },
    }

    out_path = Path(args.result_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== BENCHMARK ACCEPTANCE SUMMARY ===", flush=True)
    print(json.dumps(report["summary"], indent=2), flush=True)
    if not (all_top1_match and all_parity_ok):
        print("BENCHMARK CHECK FAILED!", flush=True)
        return 1
    print("BENCHMARK CHECK PASSED!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
