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

"""Unit and multi-process tests for F09-G04 comm profiler, calibrated planner, and executor."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest


def _ensure_triton_importable() -> None:
    try:
        from triton._C.libtriton import getenv as _g  # noqa: F401
        from triton.flagtune.runtime import comm_profiler as _cp  # noqa: F401

        return
    except Exception:
        pass
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

import triton.flagtune as flagtune_pkg  # noqa: E402
from triton.flagtune.runtime.comm_profiler import (  # noqa: E402
    CollectiveProfile,
    CommProfileResult,
    CommSample,
    needs_host_staging,
    profile_comm_topology,
)
from triton.flagtune.runtime.partition_planner import (  # noqa: E402
    CollectiveCostModel,
    CollectiveKind,
    HardwareClusterSpec,
    HardwareTopologySpec,
    InferenceConstraints,
    ModelArchitectureSpec,
    ParallelStrategyKind,
    PartitionPlanCandidate,
    PartitionPlanner,
    TrainingConstraints,
    WorkloadKind,
    build_executable_plan,
    evaluate_ranking_accuracy,
)

try:
    import torch
    from triton.flagtune.runtime.parallel_executor import (  # noqa: E402
        ParallelPlanExecutor,
        TransformerStackConfig,
        compute_stage_layer_splits,
        run_parallel_plan_benchmark,
    )

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _make_synthetic_host_staged_comm_profile(world_size: int = 4) -> CommProfileResult:
    """Create a realistic host-staged gloo 4-device communication profile."""
    ar_samples = (
        CommSample(64 * 1024, 550.0, 540.0, 590.0, 0.178, 0.119),
        CommSample(1024 * 1024, 1580.0, 1560.0, 1640.0, 0.995, 0.664),
        CommSample(8 * 1024 * 1024, 7350.0, 7300.0, 7600.0, 1.712, 1.141),
        CommSample(32 * 1024 * 1024, 30200.0, 30100.0, 31000.0, 1.667, 1.111),
    )
    ag_samples = (
        CommSample(64 * 1024, 420.0, 415.0, 450.0, 0.117, 0.156),
        CommSample(1024 * 1024, 1120.0, 1100.0, 1180.0, 0.702, 0.936),
        CommSample(8 * 1024 * 1024, 5200.0, 5150.0, 5400.0, 1.210, 1.613),
        CommSample(32 * 1024 * 1024, 20400.0, 20300.0, 21000.0, 1.233, 1.645),
    )
    rs_samples = (
        CommSample(64 * 1024, 580.0, 570.0, 610.0, 0.085, 0.113),
        CommSample(1024 * 1024, 1650.0, 1630.0, 1710.0, 0.477, 0.635),
        CommSample(8 * 1024 * 1024, 7600.0, 7550.0, 7900.0, 0.828, 1.104),
        CommSample(32 * 1024 * 1024, 31000.0, 30800.0, 32000.0, 0.812, 1.082),
    )
    p2p_samples = (
        CommSample(64 * 1024, 190.0, 185.0, 210.0, 0.345, 0.345),
        CommSample(1024 * 1024, 560.0, 550.0, 590.0, 1.872, 1.872),
        CommSample(8 * 1024 * 1024, 2800.0, 2750.0, 2950.0, 2.996, 2.996),
        CommSample(32 * 1024 * 1024, 11100.0, 11000.0, 11400.0, 3.023, 3.023),
    )
    return CommProfileResult(
        world_size=world_size,
        backend="gloo",
        device_type="cuda",
        device_name="Generic-Accelerator-96GB",
        device_memory_bytes=96 * (1024 ** 3),
        host_staged=True,
        dtype="bfloat16",
        collectives={
            "all_reduce": CollectiveProfile("all_reduce", world_size, ar_samples, 500.0, 5.9e-4, 1.712, 550.0),
            "all_gather": CollectiveProfile("all_gather", world_size, ag_samples, 380.0, 8.0e-4, 1.233, 420.0),
            "reduce_scatter": CollectiveProfile(
                "reduce_scatter", world_size, rs_samples, 520.0, 1.2e-3, 0.828, 580.0, emulated=True
            ),
            "p2p": CollectiveProfile("p2p", world_size, p2p_samples, 160.0, 3.3e-4, 3.023, 190.0),
        },
        intra_node_bandwidth_gbps=1.712,
        base_latency_us=550.0,
    )


class TestCommProfilerAndCalibratedPlanner(unittest.TestCase):

    def test_public_reexports_and_host_staging_detection(self):
        self.assertIs(HardwareTopologySpec, HardwareClusterSpec)
        self.assertIs(flagtune_pkg.HardwareTopologySpec, HardwareClusterSpec)
        self.assertIs(flagtune_pkg.CommProfileResult, CommProfileResult)
        self.assertIs(flagtune_pkg.profile_comm_topology, profile_comm_topology)

        self.assertTrue(needs_host_staging("gloo", "cuda"))
        self.assertTrue(needs_host_staging("gloo", "npu"))
        self.assertFalse(needs_host_staging("gloo", "cpu"))
        self.assertFalse(needs_host_staging("nccl", "cuda"))
        self.assertFalse(needs_host_staging("hccl", "npu"))
        self.assertFalse(needs_host_staging("xccl", "xpu"))

    def test_comm_profile_json_roundtrip_and_collective_calibration(self):
        prof = _make_synthetic_host_staged_comm_profile(world_size=4)
        with tempfile.TemporaryDirectory() as tmpd:
            path = Path(tmpd) / "comm_profile.json"
            prof.save_json(path)
            loaded = CommProfileResult.load_json(path)
            self.assertEqual(loaded.world_size, 4)
            self.assertEqual(loaded.backend, "gloo")
            self.assertTrue(loaded.host_staged)
            self.assertEqual(set(loaded.collectives.keys()), {"all_reduce", "all_gather", "reduce_scatter", "p2p"})

            topo = HardwareTopologySpec.from_comm_profile(path)
            self.assertEqual(topo.num_devices, 4)
            self.assertEqual(topo.comm_backend, "gloo")
            self.assertTrue(topo.host_staged_collectives)
            self.assertAlmostEqual(topo.intra_node_bandwidth_gbps, 1.712, places=3)

            cmodel = CollectiveCostModel.from_comm_profile(path)
            # Check calibrated 8MB AllReduce (~7.35ms) and 8MB P2P (~2.80ms) accuracy
            est_ar_8m = cmodel.estimate_ms(CollectiveKind.ALL_REDUCE, 8 * 1024 * 1024, 4)
            est_p2p_8m = cmodel.estimate_ms(CollectiveKind.P2P, 8 * 1024 * 1024, 2)
            self.assertGreater(est_ar_8m, 5.5)
            self.assertLess(est_ar_8m, 9.5)
            self.assertGreater(est_p2p_8m, 2.0)
            self.assertLess(est_p2p_8m, 3.8)
            self.assertGreater(est_ar_8m, est_p2p_8m * 1.8)

    def test_calibrated_planner_replaces_nvlink_misranking_and_emits_explanation(self):
        prof = _make_synthetic_host_staged_comm_profile(world_size=4)
        model = ModelArchitectureSpec(
            name="Transformer-1B-16L",
            num_layers=16,
            hidden_size=2048,
            ffn_hidden_size=8192,
            num_attention_heads=16,
            num_kv_heads=16,
            vocab_size=32000,
            dtype_bytes=2,
        )

        # 1. Memory-constrained inference (2 GiB/dev): uncalibrated picks DPxTP/TP, calibrated picks PP/DPxPP
        inf_c = InferenceConstraints(batch_size=8, prefill_seq_len=512)
        uncal_inf = PartitionPlanner(
            HardwareClusterSpec(num_devices=4, device_memory_bytes=2 * (1024 ** 3), backend="cuda")
        ).plan_inference(model, inf_c)
        cal_inf = PartitionPlanner.from_comm_profile(
            prof, device_memory_bytes=2 * (1024 ** 3)
        ).plan_inference(model, inf_c)

        self.assertIsNotNone(uncal_inf.recommended)
        self.assertIsNotNone(cal_inf.recommended)
        self.assertEqual(cal_inf.recommended.tp_degree, 1)
        self.assertGreaterEqual(cal_inf.recommended.pp_degree, 2)

        # 2. Training (96 GiB/dev): uncalibrated picks DDP=4, calibrated picks PP=4
        tr_c = TrainingConstraints(global_batch_size=8, seq_len=512, micro_batch_size=1)
        uncal_tr = PartitionPlanner(
            HardwareClusterSpec(num_devices=4, device_memory_bytes=96 * (1024 ** 3), backend="cuda")
        ).plan_training(model, tr_c)
        cal_tr = PartitionPlanner.from_comm_profile(
            prof, device_memory_bytes=96 * (1024 ** 3)
        ).plan_training(model, tr_c)

        self.assertEqual(uncal_tr.recommended.strategy, ParallelStrategyKind.DDP)
        self.assertEqual(cal_tr.recommended.strategy, ParallelStrategyKind.PIPELINE_PARALLEL)
        self.assertEqual(cal_tr.recommended.pp_degree, 4)

        # 3. Verify structured explanation & baseline comparisons
        expl = cal_tr.explanation
        self.assertEqual(expl["selected_plan_id"], cal_tr.recommended.plan_id)
        self.assertTrue(expl["calibration_summary"]["calibrated"])
        self.assertEqual(expl["calibration_summary"]["comm_backend"], "gloo")
        self.assertIn("pp(dp=1,tp=1,pp=4,ep=1)", expl["summary"])
        baseline_strats = {b["baseline_strategy"] for b in expl["baseline_comparisons"]}
        self.assertIn("ddp", baseline_strats)
        self.assertIn("tp", baseline_strats)

        # 4. Verify Spearman rank correlation in evaluate_ranking_accuracy
        perfect = {
            c.plan_id: c.throughput_tokens_per_sec for c in cal_tr.ranked_candidates
        }
        acc_perf = evaluate_ranking_accuracy(cal_tr, perfect)
        self.assertTrue(acc_perf["top1_match"])
        self.assertAlmostEqual(acc_perf["spearman_rho"], 1.0, places=5)

        reversed_m = {
            c.plan_id: float(idx + 1)
            for idx, c in enumerate(cal_tr.ranked_candidates)
        }
        acc_rev = evaluate_ranking_accuracy(cal_tr, reversed_m)
        self.assertFalse(acc_rev["top1_match"])
        self.assertAlmostEqual(acc_rev["spearman_rho"], -1.0, places=5)

    def test_hybrid_dp_pp_device_mesh_and_stage_splits(self):
        cluster = HardwareClusterSpec(num_devices=4, device_memory_bytes=64 * (1024 ** 3), backend="generic")
        cand = PartitionPlanCandidate(
            workload_kind=WorkloadKind.INFERENCE,
            strategy=ParallelStrategyKind.HYBRID_DP_PP,
            dp_degree=2,
            tp_degree=1,
            pp_degree=2,
            ep_degree=1,
            use_tle_overlap=False,
            devices_used=4,
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
        exec_plan = build_executable_plan(cand, cluster, model_name="hybrid_model")
        self.assertEqual(exec_plan.mesh.mesh_shape, (2, 2))
        self.assertEqual(exec_plan.mesh.mesh_dim_names, ("dp", "pp"))
        self.assertEqual(exec_plan.mesh.dp_groups, ((0, 2), (1, 3)))
        self.assertEqual(exec_plan.mesh.pp_groups, ((0, 1), (2, 3)))

        if _HAS_TORCH:
            self.assertEqual(compute_stage_layer_splits(8, 4), ((0, 1), (2, 3), (4, 5), (6, 7)))
            self.assertEqual(
                compute_stage_layer_splits(8, 4, (5, 1, 1, 1)),
                ((0, 1, 2, 3, 4), (5,), (6,), (7,)),
            )
            with self.assertRaises(ValueError):
                compute_stage_layer_splits(8, 4, (5, 2, 2, 0))

            hf_like = SimpleNamespace(
                num_hidden_layers=12,
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
                vocab_size=30522,
                layer_norm_eps=1e-5,
                model_type="bert",
            )
            stack_cfg = TransformerStackConfig.from_hf_config(hf_like, num_layers_override=4)
            self.assertEqual(stack_cfg.num_layers, 4)
            self.assertEqual(stack_cfg.hidden_size, 768)
            self.assertEqual(stack_cfg.ffn_hidden_size, 3072)


@unittest.skipUnless(_HAS_TORCH, "PyTorch is required for live multi-process tests")
class TestLiveMultiprocessCommAndParallelExecutor(unittest.TestCase):

    def _make_plan(self, world_size: int, strat: ParallelStrategyKind, dp: int, tp: int, pp: int, kind: WorkloadKind):
        cluster = HardwareClusterSpec(num_devices=world_size, device_memory_bytes=16 * (1024 ** 3), backend="cpu")
        cand = PartitionPlanCandidate(
            workload_kind=kind,
            strategy=strat,
            dp_degree=dp,
            tp_degree=tp,
            pp_degree=pp,
            ep_degree=1,
            use_tle_overlap=False,
            devices_used=world_size,
            per_device_memory_bytes=1 << 24,
            memory_breakdown_bytes={"weights": 1 << 24},
            fits_memory=True,
            meets_sla=True,
            step_latency_ms=5.0,
            ttft_ms=5.0,
            tpot_ms=0.0,
            throughput_tokens_per_sec=100.0,
            throughput_requests_per_sec=10.0,
            compute_time_ms=4.0,
            comm_time_ms=1.0,
            exposed_comm_time_ms=1.0,
        )
        return build_executable_plan(cand, cluster, model_name="unit_stack")

    def test_live_multiprocess_comm_profiler_cpu_gloo(self):
        prof = profile_comm_topology(
            world_size=2,
            backend="gloo",
            device_type="cpu",
            sizes_bytes=(4096, 65536),
            dtype="float32",
            warmup_iters=1,
            measure_iters=2,
        )
        self.assertEqual(prof.world_size, 2)
        self.assertEqual(prof.backend, "gloo")
        self.assertFalse(prof.host_staged)
        for cname in ("all_reduce", "all_gather", "reduce_scatter", "p2p"):
            self.assertIn(cname, prof.collectives)
            self.assertGreater(prof.collectives[cname].min_latency_us, 0.0)
            self.assertGreater(prof.collectives[cname].peak_bandwidth_gbps, 0.0)
        planner = PartitionPlanner.from_comm_profile(prof)
        self.assertEqual(planner.cluster.num_devices, 2)

    def test_live_parallel_executor_dp_tp_pp_hybrid_numerical_parity_and_training(self):
        cfg = TransformerStackConfig(
            num_layers=4,
            hidden_size=64,
            ffn_hidden_size=128,
            num_attention_heads=4,
            dtype="float32",
            seed=1234,
        )
        # 1. Inference parity across DP=2, TP=2, PP=2, Unbalanced PP=[3,1], 2D Hybrid DP=2xPP=2 & DP=2xTP=2
        inf_cases = [
            (2, ParallelStrategyKind.REPLICA_DP, 2, 1, 1, 1, None),
            (2, ParallelStrategyKind.TENSOR_PARALLEL, 1, 2, 1, 1, None),
            (2, ParallelStrategyKind.PIPELINE_PARALLEL, 1, 1, 2, 2, None),
            (2, ParallelStrategyKind.PIPELINE_PARALLEL, 1, 1, 2, 2, (3, 1)),
            (4, ParallelStrategyKind.HYBRID_DP_PP, 2, 1, 2, 2, None),
            (4, ParallelStrategyKind.HYBRID_DP_TP, 2, 2, 1, 1, None),
        ]
        for ws, strat, dp, tp, pp, n_mb, splits in inf_cases:
            plan = self._make_plan(ws, strat, dp, tp, pp, WorkloadKind.INFERENCE)
            res = run_parallel_plan_benchmark(
                plan=plan,
                model_config=cfg,
                batch_size=4,
                seq_lengths=(16, 24),
                num_microbatches=n_mb,
                warmup_steps=1,
                measure_steps=2,
                mode="inference",
                backend="gloo",
                device_type="cpu",
                verify_parity=True,
                stage_layer_counts=splits,
            )
            self.assertTrue(res["all_finite"], f"{strat} produced non-finite values")
            self.assertGreaterEqual(
                res["parity_cos"],
                0.99999,
                f"{strat} parity_cos={res['parity_cos']} < 0.99999",
            )
            self.assertLess(res["parity_max_abs_diff"], 1e-4)
            self.assertGreater(res["throughput_tokens_per_sec"], 0.0)

        # 2. Training end-to-end across DDP=2, TP=2, PP=2, Hybrid DP=2xPP=2
        train_cases = [
            (2, ParallelStrategyKind.DDP, 2, 1, 1, 1),
            (2, ParallelStrategyKind.TENSOR_PARALLEL, 1, 2, 1, 1),
            (2, ParallelStrategyKind.PIPELINE_PARALLEL, 1, 1, 2, 2),
            (4, ParallelStrategyKind.HYBRID_DP_PP, 2, 1, 2, 2),
        ]
        for ws, strat, dp, tp, pp, n_mb in train_cases:
            plan = self._make_plan(ws, strat, dp, tp, pp, WorkloadKind.TRAINING)
            res = run_parallel_plan_benchmark(
                plan=plan,
                model_config=cfg,
                batch_size=4,
                seq_lengths=(16,),
                num_microbatches=n_mb,
                warmup_steps=1,
                measure_steps=2,
                mode="training",
                backend="gloo",
                device_type="cpu",
                verify_parity=True,
            )
            self.assertTrue(res["all_finite"])
            self.assertGreaterEqual(res["parity_cos"], 0.99999)
            self.assertGreater(res["last_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
