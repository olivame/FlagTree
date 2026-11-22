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
import tempfile
import unittest

from triton.backends.compiler import BaseBackend
from triton.runtime.backend_caps import (
    BackendBucketingCapabilities,
    get_backend_capabilities,
    register_backend_capabilities,
    strip_int_alignment_spec,
)
from triton.runtime.shape_bucketing import (
    BucketCacheBudget,
    BucketingMode,
    IntervalPolicy,
    IntervalSegment,
    OnlineBucketPolicy,
    classify_heuristic_fn,
    evaluate_heuristics_for_bucketing,
)


class _DummyTensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape = tuple(shape)
        self.dtype = dtype

    def data_ptr(self):
        return 1024


def _outer_is_div16(x: int) -> bool:
    return (x % 16) == 0


def _outer_compound_guard(m: int, bm: int) -> bool:
    return (m % bm == 0) and (m >= 64)


class TestBackendCapabilities(unittest.TestCase):
    def test_plugin_registration_and_cache_invalidation(self):
        custom_caps = BackendBucketingCapabilities(
            backend="mock_asic",
            divisibility=32,
            min_dot_size=(32, 32, 16),
            bakes_static_grid=True,
            alignment_spec_tags=("D", "D32"),
            collective_device="cpu",
        )
        register_backend_capabilities("mock_asic", custom_caps)
        resolved = get_backend_capabilities("mock_asic")
        self.assertEqual(resolved.divisibility, 32)
        self.assertEqual(resolved.min_dot_size, (32, 32, 16))
        self.assertTrue(resolved.bakes_static_grid)

        # Updating registration must invalidate active caps cache cleanly
        updated = BackendBucketingCapabilities(backend="mock_asic", divisibility=64)
        register_backend_capabilities("mock_asic", updated)
        self.assertEqual(get_backend_capabilities("mock_asic").divisibility, 64)

    def test_env_overrides_and_base_backend_hook(self):
        self.assertTrue(hasattr(BaseBackend, "get_bucketing_capabilities"))
        prev_div = os.environ.get("FLAGTREE_BUCKET_DIVISIBILITY")
        prev_grid = os.environ.get("FLAGTREE_BAKES_STATIC_GRID")
        try:
            os.environ["FLAGTREE_BUCKET_DIVISIBILITY"] = "64"
            os.environ["FLAGTREE_BAKES_STATIC_GRID"] = "1"
            caps = get_backend_capabilities()
            self.assertEqual(caps.divisibility, 64)
            self.assertTrue(caps.bakes_static_grid)
        finally:
            if prev_div is None:
                os.environ.pop("FLAGTREE_BUCKET_DIVISIBILITY", None)
            else:
                os.environ["FLAGTREE_BUCKET_DIVISIBILITY"] = prev_div
            if prev_grid is None:
                os.environ.pop("FLAGTREE_BAKES_STATIC_GRID", None)
            else:
                os.environ["FLAGTREE_BAKES_STATIC_GRID"] = prev_grid

    def test_strip_int_alignment_spec_single_and_multi_tag(self):
        single = BackendBucketingCapabilities(backend="t1", alignment_spec_tags=("D",))
        self.assertEqual(strip_int_alignment_spec(("i32", "D"), single), (("i32", ""), True, False))
        self.assertEqual(strip_int_alignment_spec(("i32", ""), single), (("i32", ""), False, True))
        self.assertEqual(strip_int_alignment_spec(("*bf16", 16), single), (("*bf16", 16), False, False))

        # Longest tag must be stripped first so "D32" does not leave "32"
        multi = BackendBucketingCapabilities(backend="t2", alignment_spec_tags=("D", "D32"))
        self.assertEqual(strip_int_alignment_spec(("i32", "D32"), multi), (("i32", ""), True, False))
        self.assertEqual(strip_int_alignment_spec(("i64", "D"), multi), (("i64", ""), True, False))


class TestOnlineShapeBucketing(unittest.TestCase):
    def test_bucket_compression_and_singleton_isolation(self):
        pol = OnlineBucketPolicy(mode=BucketingMode.INFERENCE)
        # Distinct lengths in the same bucket collapse to identical autotune key
        k264 = pol.lookup_autotune_key({"X": _DummyTensor((2, 264)), "M": 264}, ["M"]).bucket_key
        k288 = pol.lookup_autotune_key({"X": _DummyTensor((2, 288)), "M": 288}, ["M"]).bucket_key
        k364 = pol.lookup_autotune_key({"X": _DummyTensor((2, 364)), "M": 364}, ["M"]).bucket_key
        self.assertEqual(k264, k288)
        self.assertEqual(k288, k364)

        # Singleton == 1 must stay isolated from > 1
        k1 = pol.lookup_autotune_key({"X": _DummyTensor((1, 1)), "M": 1}, ["M"]).bucket_key
        k2 = pol.lookup_autotune_key({"X": _DummyTensor((1, 2)), "M": 2}, ["M"]).bucket_key
        self.assertNotEqual(k1, k2)

    def test_premark_dynamic_buckets_and_entry_scalar_normalization(self):
        pol = OnlineBucketPolicy(mode=BucketingMode.TRAINING)
        pol.mark_dynamic_axes(["M"])
        t_aligned = pol.normalize_entry_scalar("M", 256, is_autotuned_key=False)
        t_unaligned = pol.normalize_entry_scalar("M", 264, is_autotuned_key=False)
        self.assertEqual(t_aligned, t_unaligned)

        # Invariant const / flags remain exact
        self.assertEqual(
            pol.normalize_entry_scalar("head_dim", 64, is_autotuned_key=False),
            (int, 64),
        )
        self.assertEqual(
            pol.normalize_entry_scalar("is_causal", 1, is_autotuned_key=False),
            (int, 1),
        )

    def test_ast_recursive_heuristics_and_mixed_predicates(self):
        self.assertEqual(classify_heuristic_fn(lambda args: _outer_is_div16(args["M"])), "divisibility")
        self.assertEqual(
            classify_heuristic_fn(lambda args: _outer_compound_guard(args["M"], args["BLOCK_M"])),
            "mixed",
        )
        self.assertEqual(classify_heuristic_fn(lambda args: args["M"] >= 128), "scale_bound")
        self.assertEqual(classify_heuristic_fn(lambda args: 64), "invariant")

        pol = OnlineBucketPolicy(mode=BucketingMode.INFERENCE)
        pol.mark_dynamic_axes(["M"])
        heur_map = {
            "EVEN_M": lambda args: _outer_is_div16(args["M"]),
            "GUARD_M": lambda args: _outer_compound_guard(args["M"], args["BLOCK_M"]),
        }
        # Even for aligned M=256, premarked dynamic M must conservatively evaluate modulo -> False
        ev_aligned = evaluate_heuristics_for_bucketing(heur_map, {"M": 256, "BLOCK_M": 64}, pol)
        ev_unaligned = evaluate_heuristics_for_bucketing(heur_map, {"M": 264, "BLOCK_M": 64}, pol)
        self.assertFalse(ev_aligned["EVEN_M"])
        self.assertFalse(ev_aligned["GUARD_M"])
        self.assertEqual(ev_aligned, ev_unaligned)

    def test_multiaxis_pgo_interval_policy_roundtrip(self):
        m_segs = (
            IntervalSegment(lo=1, hi=128, config_index=0, representative=128),
            IntervalSegment(lo=129, hi=512, config_index=1, representative=512),
        )
        n_segs = (
            IntervalSegment(lo=1, hi=1024, config_index=0, representative=1024),
            IntervalSegment(lo=1025, hi=4096, config_index=1, representative=4096),
        )
        pgo = IntervalPolicy(
            dynamic_axis="M",
            groups={"default": m_segs},
            dynamic_axes=("M", "N"),
            axis_groups={
                "M": {"default": m_segs},
                "N": {"default": n_segs},
            },
            op_name="test_gemm",
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "policy.json")
            pgo.save_json(path)
            loaded = IntervalPolicy.load_json(path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("axis_groups", data)

        b1 = loaded.lookup_autotune_key({"M": 80, "N": 800}, ["M", "N"]).bucket_key
        b2 = loaded.lookup_autotune_key({"M": 92, "N": 1000}, ["M", "N"]).bucket_key
        b3 = loaded.lookup_autotune_key({"M": 256, "N": 2048}, ["M", "N"]).bucket_key
        self.assertEqual(b1, b2)
        self.assertNotEqual(b1, b3)

    def test_bucket_cache_budget_lru_eviction(self):
        budget = BucketCacheBudget(max_entries=2)
        cache = {}
        for k in ("b1", "b2", "b3"):
            cache[k] = k
            budget.touch_and_evict(cache, k, max_entries=2)
        self.assertEqual(len(cache), 2)
        self.assertNotIn("b1", cache)
        self.assertIn("b2", cache)
        self.assertIn("b3", cache)


if __name__ == "__main__":
    unittest.main()



def test_interval_policy_delegates_online_helpers(tmp_path):
    from triton.runtime.shape_bucketing import IntervalPolicy, IntervalSegment

    pol = IntervalPolicy(
        dynamic_axis="M",
        groups={
            "N=512::dtypes=bf16": [
                IntervalSegment(lo=1, hi=256, config_index=0, representative=128, regret_ms=0.01)
            ]
        },
        backend="ppu",
    )
    pol.observe_args({"M": 64, "N": 512})
    pol.observe_args({"M": 192, "N": 512})
    pol.mark_dynamic_axes(["M", "S"], unaligned=True)
    assert pol.is_axis_dynamic("M", 64)
    assert pol.is_axis_dynamic("S", 128)
    b_val, _ = pol.bucket_dim_value("M", 130)
    assert b_val >= 130
    assert len(pol._ladder_for("M")) > 0
    out_json = tmp_path / "pgo_policy.json"
    pol.save_json(out_json)
    loaded = IntervalPolicy.load_json(out_json)
    loaded.observe_args({"M": 128, "N": 512})
    assert loaded.is_axis_dynamic("M", 128)


def test_cross_bucket_dynamic_axis_promotion():
    from triton.runtime.shape_bucketing import OnlineBucketPolicy

    pol = OnlineBucketPolicy(divisibility=16, backend="ppu")
    pol.observe_args({"M": 92, "N": 1024})
    assert not pol.is_axis_dynamic("M", 92)
    # Second observation in a DIFFERENT bucket (168 -> bucket 192 vs 92 -> bucket 96)
    pol.observe_args({"M": 168, "N": 1024})
    assert pol.is_axis_dynamic("M", 92)
    assert pol.is_axis_dynamic("M", 336)
    assert not pol.is_axis_dynamic("N", 1024)
    # Aligned value in a new bucket of dynamic unaligned axis M must strip alignment
    tok_m = pol.normalize_entry_scalar("M", 336, is_autotuned_key=True)
    tok_n = pol.normalize_entry_scalar("N", 1024, is_autotuned_key=True)
    assert tok_m[3] is False  # is_aligned stripped for dynamic unaligned M
    assert tok_n[3] is True   # is_aligned preserved for static N=1024


def test_constexpr_function_deterministic_cache_key(tmp_path):
    import subprocess, sys, textwrap
    script = tmp_path / "constexpr_k.py"
    script.write_text(textwrap.dedent('''
        import triton, triton.language as tl

        @tl.constexpr
        def _helper_dtype(x):
            return tl.float32

        @triton.jit
        def _k(p, N, B: tl.constexpr):
            dt: tl.constexpr = _helper_dtype(p.type.element_ty)
            offs = tl.arange(0, B)
            tl.store(p + offs, tl.zeros((B,), dtype=dt), mask=offs < N)

        print(_k.cache_key)
    '''), encoding="utf-8")
    h1 = subprocess.check_output([sys.executable, str(script)], text=True).strip()
    h2 = subprocess.check_output([sys.executable, str(script)], text=True).strip()
    assert h1 == h2 and len(h1) == 64


def test_heuristics_never_treats_block_tile_config_as_dynamic():
    from triton.runtime.shape_bucketing import OnlineBucketPolicy, evaluate_heuristics_for_bucketing

    pol = OnlineBucketPolicy(divisibility=16, backend="ppu")
    # Simulate autotuner observing multiple configs (BLOCK_M=64 and BLOCK_M=128) across dynamic M
    pol.observe_args({"M": 72, "N": 72, "K": 128, "BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32})
    pol.observe_args({"M": 116, "N": 116, "K": 128, "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64})
    assert pol.is_axis_dynamic("M", 116)
    assert not pol.is_axis_dynamic("BLOCK_M", 64)
    assert not pol.is_axis_dynamic("BLOCK_N", 128)
    assert not pol.is_axis_dynamic("BLOCK_K", 64)

    def _full_m_tiles(args):
        return int(args["M"]) % int(args["BLOCK_M"]) == 0

    def _full_k_tiles(args):
        return int(args["K"]) % int(args["BLOCK_K"]) == 0

    ev = evaluate_heuristics_for_bucketing(
        {"FULL_M_TILES": _full_m_tiles, "FULL_K_TILES": _full_k_tiles},
        {"M": 116, "N": 116, "K": 128, "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
        pol,
    )
    # Must NEVER evaluate 1 % 1 == 0 -> True!
    assert ev["FULL_M_TILES"] is False
    assert ev["FULL_K_TILES"] is True
def test_interval_policy_1d_align_sig_and_singleton_isolation():
    from triton.runtime.shape_bucketing import IntervalPolicy, IntervalSegment

    gid = "N=512|K=512::dtypes=torch.bfloat16"
    pol = IntervalPolicy(
        dynamic_axis="M",
        groups={
            gid: [
                IntervalSegment(lo=1, hi=128, config_index=0, representative=64),
                IntervalSegment(lo=129, hi=512, config_index=1, representative=256),
            ]
        },
        divisibility=16,
        backend="ppu",
    )
    k_aligned = pol.lookup_autotune_key(
        {"M": 256, "N": 512, "K": 512}, ["M", "N", "K"], extra_dtypes=("torch.bfloat16",)
    ).bucket_key
    k_unaligned = pol.lookup_autotune_key(
        {"M": 264, "N": 512, "K": 512}, ["M", "N", "K"], extra_dtypes=("torch.bfloat16",)
    ).bucket_key
    assert k_aligned == k_unaligned

    k_one = pol.lookup_autotune_key(
        {"M": 1, "N": 512, "K": 512}, ["M", "N", "K"], extra_dtypes=("torch.bfloat16",)
    ).bucket_key
    k_two = pol.lookup_autotune_key(
        {"M": 2, "N": 512, "K": 512}, ["M", "N", "K"], extra_dtypes=("torch.bfloat16",)
    ).bucket_key
    assert k_one != k_two


def test_group_frontier_contiguous_partition_and_gap_routing():
    from triton.runtime.shape_bucketing import IntervalPolicy, group_frontier

    shapes = [32, 64, 128, 256, 512, 1024]
    timings = [
        [1.0, 5.0, 9.0],
        [1.1, 5.0, 9.0],
        [6.0, 1.0, 5.0],
        [6.0, 1.2, 5.0],
        [9.0, 6.0, 1.0],
        [9.0, 6.0, 1.1],
    ]
    segs = group_frontier(shapes, timings, max_buckets=3)
    assert len(segs) == 3
    for idx in range(1, len(segs)):
        assert segs[idx].lo == segs[idx - 1].hi + 1
    assert segs[0].lo == 32 and segs[-1].hi == 1024

    gid = "N=512::dtypes=torch.bfloat16"
    pol = IntervalPolicy(dynamic_axis="M", groups={gid: segs}, backend="ppu")
    r96 = pol.lookup_autotune_key({"M": 96, "N": 512}, ["M", "N"], extra_dtypes=("torch.bfloat16",))
    r384 = pol.lookup_autotune_key({"M": 384, "N": 512}, ["M", "N"], extra_dtypes=("torch.bfloat16",))
    assert r96.bucket_key[5] == 1
    assert r384.bucket_key[5] == 2


def test_interval_policy_normalize_entry_scalar_group_aware_and_nd():
    from triton.runtime.shape_bucketing import IntervalPolicy, IntervalSegment

    gid_a = "N=512|K=512::dtypes=torch.bfloat16"
    gid_b = "N=4096|K=4096::dtypes=torch.bfloat16"
    pol = IntervalPolicy(
        dynamic_axis="M",
        groups={
            gid_a: [IntervalSegment(1, 128, 0, 64), IntervalSegment(129, 1024, 1, 512)],
            gid_b: [
                IntervalSegment(1, 64, 0, 32),
                IntervalSegment(65, 256, 1, 128),
                IntervalSegment(257, 1024, 2, 512),
            ],
        },
        divisibility=16,
        backend="ppu",
    )
    t_bf16 = _DummyTensor((100, 4096), "torch.bfloat16")
    tok_b = pol.normalize_entry_scalar(
        "M", 100, is_autotuned_key=True, all_args={"A": t_bf16, "M": 100, "N": 4096, "K": 4096}
    )
    tok_a = pol.normalize_entry_scalar(
        "M", 100, is_autotuned_key=True, all_args={"A": t_bf16, "M": 100, "N": 512, "K": 512}
    )
    tok_unseen = pol.normalize_entry_scalar(
        "M", 100, is_autotuned_key=True, all_args={"A": t_bf16, "M": 100, "N": 8192, "K": 8192}
    )
    assert tok_b[:3] == ("ft_pgo_int", 65, 256)
    assert tok_a[:3] == ("ft_pgo_int", 1, 128)
    assert tok_unseen[0] == "ft_tune_int"

    nd_gid = "::dtypes="
    nd_pol = IntervalPolicy(
        dynamic_axis="B",
        groups={nd_gid: [IntervalSegment(1, 8, 0, 4)]},
        dynamic_axes=("B", "S"),
        axis_groups={
            "B": {nd_gid: [IntervalSegment(1, 8, 0, 4)]},
            "S": {nd_gid: [IntervalSegment(1, 200, 0, 128), IntervalSegment(201, 1024, 1, 512)]},
        },
        backend="ppu",
    )
    tok_s = nd_pol.normalize_entry_scalar("S", 150, is_autotuned_key=True, all_args={"B": 4, "S": 150})
    assert tok_s[:3] == ("ft_pgo_int", 1, 200)


def test_interval_policy_per_op_fallback_isolation(tmp_path):
    from triton.runtime.adjust_kernel_param import bucket_axes
    from triton.runtime.shape_bucketing import (
        IntervalPolicy,
        IntervalSegment,
        clear_policy_caches,
        resolve_tuner_policy,
    )

    @bucket_axes(M="dynamic_tile_dim", N="dynamic_tile_dim", head_dim="invariant_const")
    def op_bmm(A, M, N, head_dim):
        return A

    @bucket_axes(M="dynamic_tile_dim", N="invariant_const", K="invariant_const")
    def op_mm(A, M, N, K):
        return A

    class _MockTuner:
        def __init__(self, fn, keys):
            self.fn = fn
            self.base_fn = fn
            self.jit_function = fn
            self.keys = keys
            self.configs = []

    clear_policy_caches()
    pgo_file = tmp_path / "shared.json"
    IntervalPolicy(
        dynamic_axis="M",
        groups={"N=512::dtypes=torch.bfloat16": [IntervalSegment(1, 256, 0, 128)]},
        backend="ppu",
    ).save_json(pgo_file)

    pol_bmm = resolve_tuner_policy(_MockTuner(op_bmm, ["M", "N", "head_dim"]), policy_path=str(pgo_file))
    pol_mm = resolve_tuner_policy(_MockTuner(op_mm, ["M", "N", "K"]), policy_path=str(pgo_file))
    assert pol_bmm.fallback_policy is not pol_mm.fallback_policy
    assert len(pol_mm.fallback_policy.axis_specs) > 0

    pol_bmm.observe_args({"M": 72, "N": 72, "head_dim": 64})
    pol_bmm.observe_args({"M": 116, "N": 116, "head_dim": 64})
    assert not pol_mm.is_axis_dynamic("N", 768)
    assert pol_mm.normalize_entry_scalar("N", 768, is_autotuned_key=True) == (int, 768)
def test_backend_caps_cudagraph_bench_extra_keys_and_fp8_min_dot_size():
    from triton.runtime.adjust_kernel_param import get_backend_min_dot_size
    from triton.runtime.backend_caps import (
        BackendBucketingCapabilities,
        get_backend_capabilities,
        get_min_dot_size,
        register_backend_capabilities,
    )
    from triton.runtime.shape_bucketing import _is_config_constexpr_key

    ppu = get_backend_capabilities("ppu")
    thead = get_backend_capabilities("thead")
    nv = get_backend_capabilities("nvidia")
    mt = get_backend_capabilities("mthreads")
    ascend = get_backend_capabilities("ascend")

    assert ppu.supports_cudagraph_bench is False
    assert thead.supports_cudagraph_bench is False
    assert nv.supports_cudagraph_bench is True
    assert ascend.torch_extension_module == "torch_npu"

    assert _is_config_constexpr_key("aiu_load_mask", backend="ppu") is True
    assert _is_config_constexpr_key("aiu_load_mask", backend="nvidia") is False

    register_backend_capabilities(
        "mock_fp8_chip",
        BackendBucketingCapabilities(
            backend="mock_fp8_chip",
            min_dot_size=(16, 16, 16),
            fp8_min_dot_size=(16, 16, 32),
            extra_config_constexpr_keys=("custom_hw_mask",),
            supports_cudagraph_bench=False,
        ),
    )
    assert _is_config_constexpr_key("custom_hw_mask", backend="mock_fp8_chip") is True
    assert get_min_dot_size("nvidia", dtype="float8_e4m3fn")[2] >= 32
    assert get_min_dot_size("ppu", dtype="float8_e5m2") == (16, 16, 32)
    assert get_min_dot_size("mthreads", dtype="fp8e4nv") == (64, 16, 32)
    assert get_min_dot_size("nvidia", dtype="bfloat16") == (16, 16, 16)
    assert get_backend_min_dot_size("ppu", dtype="float8_e4m3fn") == (16, 16, 32)
