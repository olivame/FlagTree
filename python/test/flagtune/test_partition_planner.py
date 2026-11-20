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

import json
import os
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest


def _ensure_triton_importable() -> None:
    repo_python = Path(__file__).resolve().parents[2]
    if str(repo_python) not in sys.path:
        sys.path.insert(0, str(repo_python))
    try:
        from triton._C.libtriton import getenv as _g  # noqa: F401
    except Exception:
        triton_pkg = types.ModuleType("triton")
        triton_pkg.__path__ = [str(repo_python / "triton")]
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


_ensure_triton_importable()

from triton.flagtune import PartitionPlanner as TopLevelPartitionPlanner  # noqa: E402
from triton.flagtune.runtime.partition_planner import (  # noqa: E402
    BucketedOperatorCostModel,
    CollectiveCostModel,
    CollectiveKind,
    HardwareClusterSpec,
    InferenceConstraints,
    ModelArchitectureSpec,
    OperatorBucketMeasurement,
    ParallelStrategyKind,
    PartitionPlanner,
    PiecewiseAlphaBetaCurve,
    TrainingConstraints,
    WorkloadKind,
    evaluate_ranking_accuracy,
)


class TestCalibratedPartitionPlanner(unittest.TestCase):

    def test_public_reexport_and_piecewise_collective_curve_fitting(self):
        self.assertIs(TopLevelPartitionPlanner, PartitionPlanner)

        samples = [
            (4 * 1024, 9.2),
            (32 * 1024, 10.1),
            (512 * 1024, 18.5),
            (2 * 1024 * 1024, 34.0),
            (16 * 1024 * 1024, 118.0),
            (64 * 1024 * 1024, 410.0),
        ]
        curve = PiecewiseAlphaBetaCurve.fit_from_samples(samples)
        self.assertEqual(len(curve.segments), 3)

        comm = CollectiveCostModel({CollectiveKind.ALL_REDUCE: curve})
        roundtrip = CollectiveCostModel.from_dict(comm.to_dict())
        self.assertEqual(comm.estimate_ms(CollectiveKind.ALL_REDUCE, 1024 * 1024, 1), 0.0)
        t_tp2 = roundtrip.estimate_ms(CollectiveKind.ALL_REDUCE, 4 * 1024 * 1024, 2)
        t_tp8 = roundtrip.estimate_ms(CollectiveKind.ALL_REDUCE, 4 * 1024 * 1024, 8)
        self.assertGreater(t_tp8, t_tp2)

    def test_inference_small_retrieval_model_locks_replica_dp_avoiding_tp_negative_scaling(self):
        # Qwen3-Embedding-0.6B / bge-m3 class encoder on 8 cards:
        # Cross-card TP AllReduce latency degrades per-device efficiency, whereas Replica-DP
        # runs 8 independent replicas with zero cross-card communication!
        cluster = HardwareClusterSpec(
            num_devices=8,
            device_memory_bytes=64 * (1024 ** 3),
            supports_tle=True,
        )
        planner = PartitionPlanner(cluster=cluster)
        small_embed = ModelArchitectureSpec(
            name="Qwen3-Embedding-0.6B",
            num_layers=28,
            hidden_size=1024,
            ffn_hidden_size=3072,
            num_attention_heads=16,
            num_kv_heads=8,
            vocab_size=151936,
        )
        report = planner.plan_inference(
            small_embed,
            InferenceConstraints(batch_size=16, prefill_seq_len=256),
        )
        self.assertEqual(report.workload_kind, WorkloadKind.INFERENCE)
        self.assertIsNotNone(report.recommended)
        self.assertEqual(report.recommended.strategy, ParallelStrategyKind.REPLICA_DP)
        self.assertEqual(report.recommended.dp_degree, 8)
        self.assertEqual(report.recommended.tp_degree, 1)
        self.assertEqual(report.recommended.comm_time_ms, 0.0)

        tp8_plan = next(
            c for c in report.ranked_candidates
            if c.strategy is ParallelStrategyKind.TENSOR_PARALLEL and c.tp_degree == 8 and not c.use_tle_overlap
        )
        self.assertGreater(
            report.recommended.throughput_tokens_per_sec,
            tp8_plan.throughput_tokens_per_sec * 1.5,
        )

    def test_inference_8b_tight_sla_and_kv_cache_selects_tp_with_tle_overlap(self):
        # 8B Reranker / VL model with long KV context and tight TTFT/TPOT SLA:
        # Single-card Replica-DP violates TTFT/TPOT SLA (or memory ceiling), so TP + TLE wins!
        cluster = HardwareClusterSpec(
            num_devices=8,
            device_memory_bytes=32 * (1024 ** 3),
            supports_tle=True,
        )
        planner = PartitionPlanner(cluster=cluster)
        model_8b = ModelArchitectureSpec(
            name="Qwen3-Reranker-8B",
            num_layers=36,
            hidden_size=4096,
            ffn_hidden_size=12288,
            num_attention_heads=32,
            num_kv_heads=8,
            vocab_size=151936,
        )
        # First check single-device baseline TTFT/TPOT to set a realistic strict online SLA
        probe = planner.plan_inference(
            model_8b,
            InferenceConstraints(
                batch_size=8,
                prefill_seq_len=2048,
                decode_kv_len=8192,
                output_tokens=64,
            ),
        )
        all_cands = probe.ranked_candidates + probe.rejected_candidates
        tp1_cand = next(c for c in all_cands if c.tp_degree == 1)

        strict_report = planner.plan_inference(
            model_8b,
            InferenceConstraints(
                batch_size=8,
                prefill_seq_len=2048,
                decode_kv_len=8192,
                output_tokens=64,
                ttft_sla_ms=tp1_cand.ttft_ms * 0.55,
                tpot_sla_ms=tp1_cand.tpot_ms * 0.60,
            ),
        )
        rec = strict_report.recommended
        self.assertIsNotNone(rec)
        self.assertGreater(rec.tp_degree, 1)
        self.assertTrue(rec.use_tle_overlap)
        self.assertTrue(rec.meets_sla)
        self.assertTrue(rec.fits_memory)

        # Verify TLE candidate strictly outperforms non-TLE counterpart of identical (dp, tp)
        non_tle = next(
            c for c in (strict_report.ranked_candidates + strict_report.rejected_candidates)
            if c.dp_degree == rec.dp_degree and c.tp_degree == rec.tp_degree and not c.use_tle_overlap
        )
        self.assertLess(rec.ttft_ms, non_tle.ttft_ms)
        self.assertLess(rec.exposed_comm_time_ms, non_tle.exposed_comm_time_ms)

    def test_training_ddp_vs_memory_bound_sharding_and_tle_overlap(self):
        # 1. Small/medium model on 8x64GB cards -> DDP wins with gradient bucketing + backward overlap
        cluster_64g = HardwareClusterSpec(
            num_devices=8,
            device_memory_bytes=64 * (1024 ** 3),
            supports_tle=True,
        )
        compute = BucketedOperatorCostModel(
            measurements=[
                OperatorBucketMeasurement("transformer_layer_fwd", 512, 1024, 1, 0.28),
                OperatorBucketMeasurement("transformer_layer_fwd", 2048, 1024, 1, 0.92),
                OperatorBucketMeasurement("transformer_layer_fwd", 4096, 1024, 1, 1.80),
            ]
        )
        planner = PartitionPlanner(cluster=cluster_64g, compute_model=compute)
        bge = ModelArchitectureSpec(
            name="BAAI-bge-m3",
            num_layers=24,
            hidden_size=1024,
            ffn_hidden_size=4096,
            num_attention_heads=16,
            vocab_size=128000,
        )
        rep_small = planner.plan_training(
            bge,
            TrainingConstraints(global_batch_size=32, seq_len=512),
        )
        self.assertIsNotNone(rep_small.recommended)
        self.assertEqual(rep_small.recommended.strategy, ParallelStrategyKind.DDP)
        self.assertLess(rep_small.recommended.exposed_comm_time_ms, rep_small.recommended.comm_time_ms)

        # 2. Large 13B model on 8x32GB cards -> DDP OOMs (weights+grads+AdamW = 16B/param > 200GB!),
        # so planner automatically prunes DDP and picks sharded FSDP or TP+TLE / PP!
        cluster_32g = HardwareClusterSpec(
            num_devices=8,
            device_memory_bytes=32 * (1024 ** 3),
            supports_tle=True,
        )
        planner_32g = PartitionPlanner(cluster=cluster_32g)
        model_7b = ModelArchitectureSpec(
            name="SearchAds-Ranker-7B",
            num_layers=32,
            hidden_size=4096,
            ffn_hidden_size=11008,
            num_attention_heads=32,
            num_kv_heads=8,
            vocab_size=64000,
        )
        rep_large = planner_32g.plan_training(
            model_7b,
            TrainingConstraints(
                global_batch_size=16,
                seq_len=2048,
                micro_batch_size=1,
                activation_checkpointing=True,
            ),
        )
        ddp_rejected = [c for c in rep_large.rejected_candidates if c.strategy is ParallelStrategyKind.DDP]
        self.assertEqual(len(ddp_rejected), 1)
        self.assertFalse(ddp_rejected[0].fits_memory)

        self.assertIsNotNone(rep_large.recommended)
        self.assertTrue(rep_large.recommended.fits_memory)
        self.assertIn(
            rep_large.recommended.strategy,
            {
                ParallelStrategyKind.ZERO_FSDP,
                ParallelStrategyKind.HYBRID_DP_TP,
                ParallelStrategyKind.TENSOR_PARALLEL,
                ParallelStrategyKind.PIPELINE_PARALLEL,
            },
        )

        # Verify JSON report export and ranking calibration concordance evaluator
        measured = {
            c.plan_id: c.throughput_tokens_per_sec * (1.02 if idx == 0 else 0.97)
            for idx, c in enumerate(rep_large.ranked_candidates)
        }
        acc = evaluate_ranking_accuracy(rep_large, measured)
        self.assertTrue(acc["top1_match"])
        self.assertGreaterEqual(acc["pairwise_concordance"], 0.95)

        with tempfile.TemporaryDirectory() as tmpd:
            out_json = Path(tmpd) / "plan_report.json"
            rep_large.save_json(out_json)
            loaded = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(loaded["recommended"]["plan_id"], rep_large.recommended.plan_id)


if __name__ == "__main__":
    unittest.main()
