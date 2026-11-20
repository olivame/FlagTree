# Copyright 2025-     FlagOS Contributors
"""Unit tests covering F09 cross-backend portability, AST recursive heuristics,
mixed-predicate evaluation, N-D PGO policy, prewarm closure, disk binary cache
governance, mainline dispatch/OOM wiring, GraphBucketCompiler, and generic
non-Transformer PartitionPlanner + ExecutableParallelPlan export.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

import triton
from triton import knobs
from triton.runtime import (
    Autotuner,
    BackendBucketingCapabilities,
    BucketingMode,
    Config,
    DecodeGraphBucketRunner,
    DistributedDispatchGuard,
    GLOBAL_FALLBACK_REGISTRY,
    GraphBucketCompiler,
    IntervalPolicy,
    OnlineBucketPolicy,
    ProfileMode,
    RankDispatchState,
    ShapeProfiler,
    TokenBudgetBatcher,
    bucket_axes,
    compile_with_bucketing,
    get_backend_capabilities,
    register_backend_capabilities,
    warmup_buckets,
)
from triton.runtime.adjust_kernel_param import AxisRole, infer_bucketable_axes
from triton.runtime.backend_caps import resolve_collective_device, strip_int_alignment_spec
from triton.runtime.cache import FileCacheManager, enforce_binary_cache_budget
from triton.runtime.shape_bucketing import (
    IntervalSegment,
    classify_heuristic_fn,
    evaluate_heuristics_for_bucketing,
)
from triton.flagtune import (
    GraphWorkloadSpec,
    HardwareClusterSpec,
    ModelArchitectureSpec,
    OperatorStageSpec,
    ParallelStrategyKind,
    PartitionPlanner,
    TrainingConstraints,
    WorkloadKind,
    build_executable_plan,
)


def _nested_div_helper(a):
    return int(a["M"]) % 64 == 0


def _wrapper_div_heuristic(a):
    return _nested_div_helper(a)


def _compound_mixed_heuristic(a):
    return int(a["M"]) >= 64 and int(a["M"]) % 16 == 0


class _DummyJIT:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.arg_names = ["X", "M"]

    def run(self, *args, **kwargs):
        return ("ok", args, kwargs)


class TestF09PortabilityAndGenerality(unittest.TestCase):
    def setUp(self):
        GLOBAL_FALLBACK_REGISTRY.clear()

    def test_backend_capability_registry_and_custom_plugin(self):
        custom = BackendBucketingCapabilities(
            backend="custom_npu_test",
            divisibility=32,
            min_dot_size=(32, 32, 16),
            min_load_size=8,
            bakes_static_grid=True,
            default_num_stages=2,
            alignment_spec_tags=("D", "D32"),
        )
        register_backend_capabilities("custom_npu_test", custom)
        got = get_backend_capabilities("custom_npu_test")
        self.assertEqual(got.divisibility, 32)
        self.assertEqual(got.min_dot_size, (32, 32, 16))
        self.assertTrue(got.bakes_static_grid)

        norm, was_aligned, is_unaligned = strip_int_alignment_spec(("i32", "D32"), got)
        self.assertTrue(was_aligned)
        self.assertFalse(is_unaligned)
        self.assertEqual(norm, ("i32", ""))

        pol = OnlineBucketPolicy(backend="custom_npu_test")
        self.assertEqual(pol.divisibility, 32)
        self.assertEqual(resolve_collective_device("gloo"), "cpu")

    def test_ast_recursive_heuristics_and_compound_mixed_eval(self):
        # Nested helper call must be classified as divisibility, NOT invariant!
        self.assertEqual(classify_heuristic_fn(_wrapper_div_heuristic), "divisibility")
        self.assertEqual(classify_heuristic_fn(_compound_mixed_heuristic), "mixed")

        pol = OnlineBucketPolicy(mode=BucketingMode.INFERENCE, dynamic_axes=["M"])
        # Even for aligned M=384, dynamic axis M must evaluate %16==0 as False (via 1)
        # while M>=64 evaluates True (via bucket_hi=384) -> compound `and` is False!
        res = evaluate_heuristics_for_bucketing(
            {
                "WRAP_DIV": _wrapper_div_heuristic,
                "MIXED_AND": _compound_mixed_heuristic,
            },
            {"M": 384},
            pol,
        )
        self.assertFalse(res["WRAP_DIV"])
        self.assertFalse(res["MIXED_AND"])

        # Static axis K=128 keeps exact True
        res_static = evaluate_heuristics_for_bucketing(
            {"STATIC_MIXED": lambda a: a["K"] >= 64 and a["K"] % 16 == 0},
            {"K": 128},
            pol,
        )
        self.assertTrue(res_static["STATIC_MIXED"])

    def test_explicit_bucket_axes_and_nd_pgo_interval_policy(self):
        @bucket_axes(weird_dim="invariant_const", dyn_axis="dynamic_tile_dim")
        def custom_op(ptr, weird_dim, dyn_axis):
            pass

        specs = infer_bucketable_axes(custom_op, keys=["weird_dim", "dyn_axis"])
        self.assertEqual(specs["weird_dim"].role, AxisRole.INVARIANT_CONST)
        self.assertEqual(specs["dyn_axis"].role, AxisRole.DYNAMIC_TILE_DIM)

        # 2D N-D PGO IntervalPolicy on (B, S)
        seg_b = [IntervalSegment(1, 8, 0, 4), IntervalSegment(9, 64, 1, 32)]
        seg_s = [IntervalSegment(1, 256, 0, 128), IntervalSegment(257, 1024, 1, 512)]
        gid = "::dtypes="
        nd_pol = IntervalPolicy(
            dynamic_axis="B",
            groups={gid: seg_b},
            dynamic_axes=("B", "S"),
            axis_groups={"B": {gid: seg_b}, "S": {gid: seg_s}},
            backend="test_bk",
        )
        r1 = nd_pol.lookup_autotune_key({"B": 4, "S": 100}, ["B", "S"], extra_dtypes=())
        r2 = nd_pol.lookup_autotune_key({"B": 6, "S": 200}, ["B", "S"], extra_dtypes=())
        r3 = nd_pol.lookup_autotune_key({"B": 32, "S": 600}, ["B", "S"], extra_dtypes=())
        self.assertEqual(r1.bucket_key, r2.bucket_key)
        self.assertNotEqual(r1.bucket_key, r3.bucket_key)
        self.assertEqual(r3.representative_dims, {"B": 32, "S": 512})

    def test_prewarm_mark_dynamic_axes_eliminates_state_flip(self):
        pol = OnlineBucketPolicy(mode=BucketingMode.INFERENCE)
        seen = []
        warmup_buckets(
            lambda m: seen.append(m),
            {"M": (64, 512)},
            lambda pt: ((pt["M"],), {}),
            policy=pol,
            include_unaligned_tail=True,
        )
        # Immediately after warmup, M is dynamic & unaligned across buckets -> no post-warm flip!
        self.assertTrue(pol.is_axis_dynamic("M", 288))
        tok_aligned = pol.normalize_entry_scalar("M", 288, is_autotuned_key=True)
        tok_unaligned = pol.normalize_entry_scalar("M", 264, is_autotuned_key=True)
        self.assertEqual(tok_aligned, tok_unaligned)

    def test_disk_binary_cache_budget_eviction_in_cache_py(self):
        with tempfile.TemporaryDirectory() as tmpd:
            with knobs.cache.scope():
                knobs.cache.dir = tmpd
                knobs.cache.max_cache_entries = 2
                knobs.cache.max_cache_bytes = None
                for idx in range(4):
                    mgr = FileCacheManager(f"key_{idx}")
                    p_bin = mgr.put(b"x" * 128, f"kernel_{idx}.hgbin", binary=True)
                    mgr.put_group(f"kernel_{idx}.json", {f"kernel_{idx}.hgbin": p_bin})
                remaining = [d for d in os.listdir(tmpd) if os.path.isdir(os.path.join(tmpd, d))]
                self.assertLessEqual(len(remaining), 2)
                self.assertIn("key_3", remaining)

    def test_autotuner_mainline_dispatch_guard_and_oom_recovery(self):
        def _k(X, M):
            pass

        calls = []

        class _OOMOnceJIT(_DummyJIT):
            def run(self, *args, **kwargs):
                calls.append(kwargs.get("BLOCK"))
                if kwargs.get("BLOCK") == 256:
                    raise RuntimeError("CUDA out of memory")
                return ("ok", args, kwargs)

        barrier_hits = []
        guard = DistributedDispatchGuard(
            world_size=2,
            rank=0,
            all_reduce_or_fn=lambda m: m,
            barrier_fn=lambda: barrier_hits.append(1),
        )
        tuner = Autotuner(
            _OOMOnceJIT(_k),
            ["X", "M"],
            [Config({"BLOCK": 256}), Config({"BLOCK": 64})],
            ["M"],
            reset_to_zero=None,
            restore_value=None,
            do_bench=lambda fn_call, quantiles=None: (fn_call(), [0.1, 0.1, 0.1])[1],
            cache_results=False,
        )
        tuner.dispatch_guard = guard
        with knobs.autotuning.scope():
            knobs.autotuning.adjust_block_size = False
            knobs.autotuning.shape_bucketing = True
            knobs.autotuning.distributed_dispatch_guard = True
            knobs.autotuning.oom_recovery = True
            tuner.run(SimpleNamespace(dtype="torch.bfloat16", data_ptr=lambda: 4096), 256)

        self.assertEqual(tuner.best_config.kwargs["BLOCK"], 64)
        self.assertEqual(guard.negotiation_count, 1)
        # Pre-compile barrier + post-compile barrier = 2
        self.assertEqual(len(barrier_hits), 2)
        self.assertGreaterEqual(len(GLOBAL_FALLBACK_REGISTRY.records()), 1)

    def test_graph_bucket_compiler_generic_profiler_and_multimodal_batcher(self):
        import torch
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        wrapped = compile_with_bucketing(
            model,
            dynamic_dims={0: {0: "B", 1: "S"}},
            mode=BucketingMode.INFERENCE,
            use_torch_compile=False,
            enable_profiler=True,
            profiler_mode=ProfileMode.GENERIC_GRAPH,
        )
        for s_len in (24, 28, 30, 50, 60):
            x = torch.randn(2, s_len, 16)
            y = wrapped(x)
            self.assertEqual(tuple(y.shape), (2, s_len, 16))
        summary = wrapped.summary()
        self.assertEqual(summary["steps_executed"], 5)
        self.assertLess(summary["unique_model_cache_keys"], 5)
        comb = summary["profiler"]["model_shape_combinations"]
        self.assertEqual(comb["unique_raw_signatures"], 5)
        self.assertLess(comb["unique_bucketed_signatures"], 5)

        # Generalized TokenBudgetBatcher on multimodal/RecSys dicts
        items = [
            {"id": 0, "tokens": 12, "image_patches": 16},
            {"id": 1, "tokens": 180, "image_patches": 64},
            {"id": 2, "tokens": 20, "image_patches": 16},
        ]
        batcher = TokenBudgetBatcher(
            max_tokens_per_batch=128,
            length_fn=lambda it: it["tokens"] + it["image_patches"],
        )
        batches = batcher.build_generic_batches(items)
        self.assertEqual(len(batches), 2)

        # Pytree dict DecodeGraphBucketRunner
        runner = DecodeGraphBucketRunner(
            lambda inp, kvs: {"logits": inp["tok"] * 2},
            batch_buckets=(2, 4),
        )
        out = runner.run_decode_step({"tok": torch.ones(3, 8)}, [10, 20, 30])
        self.assertEqual(tuple(out["logits"].shape), (3, 8))

    def test_partition_planner_generic_graph_3d_and_executable_plan(self):
        import torch.nn as nn

        # 1. RecSys DLRM-like model (huge embedding tables + MLP)
        dlrm_graph = GraphWorkloadSpec(
            name="dlrm_ctr",
            stages=(
                OperatorStageSpec("emb_sparse", "embedding_table", 1e8, 0, 1 << 24, embedding_table_bytes=120 << 30, comm_bytes_per_step=16 << 20),
                OperatorStageSpec("bottom_mlp", "mlp", 2e10, 64 << 20, 32 << 20),
                OperatorStageSpec("top_mlp", "mlp", 4e10, 128 << 20, 32 << 20),
            ),
            batch_size=1024,
            workload_kind=WorkloadKind.INFERENCE,
        )
        cluster = HardwareClusterSpec(num_devices=4, device_memory_bytes=64 << 30, backend="ppu")
        planner = PartitionPlanner(cluster)
        rep = planner.plan_graph(dlrm_graph)
        self.assertIsNotNone(rep.recommended)
        self.assertEqual(rep.recommended.strategy, ParallelStrategyKind.TABLE_PARALLEL_DP)

        # 2. Automatic from_torch_module introspection on CNN
        cnn = nn.Sequential(nn.Conv2d(3, 16, 3), nn.Flatten(), nn.Linear(16, 8))
        cnn_spec = GraphWorkloadSpec.from_torch_module(cnn, batch_size=32)
        self.assertGreaterEqual(len(cnn_spec.stages), 2)

        # 3. 8-device 3D Hybrid Parallelism + ExecutableParallelPlan export
        cluster8 = HardwareClusterSpec(num_devices=8, device_memory_bytes=96 << 30, backend="ascend", devices_per_node=8)
        planner8 = PartitionPlanner(cluster8)
        m70b = ModelArchitectureSpec("Llama-70B", 80, 8192, 28672, 64, 8, 128256)
        train_rep = planner8.plan_training(m70b, TrainingConstraints(global_batch_size=32, seq_len=4096, activation_checkpointing=True))
        all_strats = {c.strategy for c in train_rep.ranked_candidates + train_rep.rejected_candidates}
        self.assertIn(ParallelStrategyKind.HYBRID_3D, all_strats)

        exec_plan = build_executable_plan(rep.recommended, cluster, model_name="dlrm_ctr")
        self.assertEqual(exec_plan.num_devices, 4)
        self.assertTrue(exec_plan.runtime_knobs["FLAGTREE_DISTRIBUTED_DISPATCH_GUARD"])
        with tempfile.TemporaryDirectory() as tmpd:
            p = Path(tmpd) / "exec_plan.json"
            exec_plan.save_json(p)
            loaded = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema"], "flagtree.executable_parallel_plan.v1")


if __name__ == "__main__":
    unittest.main()
