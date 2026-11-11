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

import ast
import json
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path


def _ensure_triton_importable_for_unit_test() -> None:
    """Allow pure-Python runtime unit tests to run even without compiled libtriton."""
    repo_python = Path(__file__).resolve().parents[3]
    if str(repo_python) not in sys.path:
        sys.path.insert(0, str(repo_python))
    try:
        from triton._C.libtriton import getenv as _real_getenv  # noqa: F401
    except Exception:
        triton_pkg = types.ModuleType("triton")
        triton_pkg.__path__ = [str(repo_python / "triton")]
        triton_pkg.__version__ = "3.3.0"
        triton_pkg.next_power_of_2 = lambda n: 1 if n <= 1 else 1 << (int(n) - 1).bit_length()
        triton_pkg.cdiv = lambda x, y: (x + y - 1) // y
        sys.modules["triton"] = triton_pkg
        c_pkg = types.ModuleType("triton._C")
        c_pkg.__path__ = []
        lib_stub = types.ModuleType("triton._C.libtriton")
        lib_stub.__path__ = []
        lib_stub.get_cache_invalidating_env_vars = lambda: {}
        lib_stub.native_specialize_impl = lambda *a, **kw: None
        lib_stub.ir = types.SimpleNamespace()
        lib_stub.passes = types.SimpleNamespace()
        lib_stub.llvm = types.SimpleNamespace()
        lib_stub.interpreter = types.SimpleNamespace()
        ll_mod = types.ModuleType("triton._C.libtriton.linear_layout")
        ll_mod.LinearLayout = object
        lib_stub.linear_layout = ll_mod
        sys.modules["triton._C.libtriton.linear_layout"] = ll_mod
        lib_stub.getenv = lambda key, default=None: os.environ.get(key, default)
        lib_stub.getenv_bool = lambda key, default=None: (
            default
            if key not in os.environ
            else os.environ[key].strip().lower() in ("1", "true", "yes", "y", "on")
        )
        c_pkg.libtriton = lib_stub
        triton_pkg._C = c_pkg
        sys.modules["triton._C"] = c_pkg
        sys.modules["triton._C.libtriton"] = lib_stub


_ensure_triton_importable_for_unit_test()

from triton import knobs  # noqa: E402
from triton.runtime.adjust_kernel_param import (  # noqa: E402
    AxisRole,
    clear_analysis_cache,
    get_backend_min_dot_size,
    infer_bucketable_axes,
)
from triton.runtime.autotuner import Autotuner, Config  # noqa: E402
from triton.runtime.jit import JITFunction  # noqa: E402
from triton.runtime.shape_bucketing import (  # noqa: E402
    BucketingMode,
    DECODE_FINE_LADDER,
    DispatchStatus,
    GLOBAL_BUCKET_TELEMETRY,
    IntervalPolicy,
    IntervalSegment,
    OnlineBucketPolicy,
    allocate_budget,
    clear_policy_caches,
    enforce_disk_cache_budget,
    group_frontier,
    warmup_buckets,
)


class _FakeTensor:

    def __init__(self, dtype: str = "torch.bfloat16", ptr: int = 4096):
        self.dtype = dtype
        self._ptr = ptr

    def data_ptr(self) -> int:
        return self._ptr


class _ParsedKernel:

    def __init__(self, source: str, extra_globals: dict | None = None, name: str = "parsed_kernel"):
        self.source = textwrap.dedent(source)
        self.__globals__ = dict(extra_globals or {})
        self.__name__ = name
        fndef = next(n for n in ast.walk(ast.parse(self.source)) if isinstance(n, ast.FunctionDef))
        self.__name__ = fndef.name
        self.arg_names = [a.arg for a in fndef.args.args]
        self.cache_key = f"cache_{self.__name__}"

    def parse(self):
        return ast.parse(self.source)


class _DummyJITRunner(JITFunction):
    """Minimal JITFunction stand-in for testing Autotuner.run and check_disk_cache."""

    def __init__(self, parsed_kernel: _ParsedKernel):
        self._parsed = parsed_kernel
        self.__name__ = parsed_kernel.__name__
        self.__globals__ = parsed_kernel.__globals__
        self.arg_names = parsed_kernel.arg_names

        def _base_callable():
            return None

        _base_callable.__name__ = parsed_kernel.__name__
        self.fn = _base_callable
        self.launches = []

    @property
    def cache_key(self) -> str:
        return self._parsed.cache_key

    def parse(self):
        return self._parsed.parse()

    def run(self, *args, **kwargs):
        self.launches.append((args, dict(kwargs)))
        return ("launched", args, kwargs)


class TestShapeBucketingKnobs(unittest.TestCase):

    def test_knobs_defaults_and_env_roundtrip(self):
        with knobs.autotuning.scope():
            for env_k in (
                "FLAGTREE_SHAPE_BUCKETING",
                "FLAGTREE_SHAPE_BUCKETING_MODE",
                "FLAGTREE_SHAPE_POLICY_PATH",
                "FLAGTREE_MAX_BUCKET_ARTIFACTS",
                "FLAGTREE_COMPILE_TIMEOUT_SEC",
            ):
                os.environ.pop(env_k, None)
            knobs.autotuning.reset()
            self.assertFalse(knobs.autotuning.shape_bucketing)
            self.assertEqual(knobs.autotuning.shape_bucketing_mode, "auto")
            self.assertIsNone(knobs.autotuning.shape_policy_path)
            self.assertIsNone(knobs.autotuning.max_bucket_artifacts)
            self.assertIsNone(knobs.autotuning.compile_timeout_sec)

            os.environ["FLAGTREE_SHAPE_BUCKETING"] = "1"
            os.environ["FLAGTREE_SHAPE_BUCKETING_MODE"] = "inference"
            os.environ["FLAGTREE_MAX_BUCKET_ARTIFACTS"] = "256"
            os.environ["FLAGTREE_COMPILE_TIMEOUT_SEC"] = "45"
            self.assertTrue(knobs.autotuning.shape_bucketing)
            self.assertEqual(knobs.autotuning.shape_bucketing_mode, "inference")
            self.assertEqual(knobs.autotuning.max_bucket_artifacts, 256)
            self.assertEqual(knobs.autotuning.compile_timeout_sec, 45)

            # Direct attribute assignment & copy safety
            knobs.autotuning.max_bucket_artifacts = 64
            copied = knobs.autotuning.copy()
            self.assertEqual(copied.max_bucket_artifacts, 64)
            for env_k in (
                "FLAGTREE_SHAPE_BUCKETING",
                "FLAGTREE_SHAPE_BUCKETING_MODE",
                "FLAGTREE_MAX_BUCKET_ARTIFACTS",
                "FLAGTREE_COMPILE_TIMEOUT_SEC",
            ):
                os.environ.pop(env_k, None)


class TestASTAxisClassification(unittest.TestCase):

    def setUp(self):
        clear_analysis_cache()
        clear_policy_caches()

    def test_gemm_fwd_and_bwd_dw_stride_pairing(self):
        helper = _ParsedKernel(
            """
            def prev_multiple_of(a, b):
                return tl.cdiv(a, b) * b - b
            """
        )
        mm_src = """
        def mm_kernel_general(
            A, B, C, M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, IS_FP64: tl.constexpr = False,
        ):
            pid = tl.program_id(0)
            grid_m = tl.cdiv(M, BLOCK_M)
            grid_n = tl.cdiv(N, BLOCK_N)
            rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
            prev_m = prev_multiple_of(K, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for start_k in range(0, prev_m, BLOCK_K):
                rk = start_k + tl.arange(0, BLOCK_K)
                a = tl.load(A + (rm[:, None] * stride_am + rk[None, :] * stride_ak))
                b = tl.load(B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn))
                acc += tl.dot(a, b)
            tl.store(C + rm[:, None] * stride_cm + rn[None, :] * stride_cn, acc)
        """
        kernel = _ParsedKernel(mm_src, extra_globals={"prev_multiple_of": helper})
        cfgs = [
            Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}),
            Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}),
        ]
        specs = infer_bucketable_axes(
            kernel,
            configs=cfgs,
            keys=["M", "N", "K", "stride_am", "stride_bk"],
        )
        for dim in ("M", "N", "K"):
            self.assertEqual(specs[dim].role, AxisRole.DYNAMIC_TILE_DIM)
            self.assertTrue(specs[dim].feeds_dot)
            self.assertGreaterEqual(specs[dim].min_block_size, 16)
        self.assertEqual(specs["stride_am"].role, AxisRole.DYNAMIC_STRIDE)
        self.assertIn("K", specs["stride_am"].paired_dims)
        self.assertEqual(specs["stride_bk"].role, AxisRole.DYNAMIC_STRIDE)
        self.assertIn("N", specs["stride_bk"].paired_dims)
        self.assertEqual(specs["IS_FP64"].role, AxisRole.INVARIANT_CONST)

    def test_multi_dot_attention_with_inner_helper_preserves_kv_ctx_and_head_dim_invariant(self):
        inner = _ParsedKernel(
            """
            def _attn_fwd_inner(
                acc, q, K_ptr, V_ptr, stride_k, stride_v,
                BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                offs_n, KV_CTX,
            ):
                lo, hi = 0, KV_CTX
                for start_n in range(lo, hi, BLOCK_N):
                    kv_mask = (start_n + offs_n) < KV_CTX
                    k = tl.load(K_ptr, mask=kv_mask[None, :], other=0.0)
                    qk = tl.dot(q, k)
                    p = tl.math.exp2(qk)
                    v = tl.load(V_ptr, mask=kv_mask[:, None], other=0.0)
                    acc = tl.dot(p, v, acc)
                    K_ptr += BLOCK_N * stride_k
                    V_ptr += BLOCK_N * stride_v
                return acc
            """
        )
        outer = _ParsedKernel(
            """
            def _attn_fwd(
                Q, K, V, Out,
                stride_q_seqlen, stride_k_seqlen, stride_v_seqlen,
                Q_CTX, KV_CTX,
                HEAD_DIM: tl.constexpr,
                BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr,
                HAS_ATTN_MASK: tl.constexpr,
            ):
                tl.static_assert(BLOCK_N <= HEAD_DIM)
                offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_d = tl.arange(0, HEAD_DIM)
                offs_n = tl.arange(0, BLOCK_N)
                q = tl.load(Q + offs_m[:, None] * stride_q_seqlen + offs_d[None, :], mask=(offs_m < Q_CTX)[:, None])
                acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
                acc = _attn_fwd_inner(
                    acc, q, K, V, stride_k_seqlen, stride_v_seqlen,
                    BLOCK_M, HEAD_DIM, BLOCK_N, offs_n, KV_CTX,
                )
                tl.store(Out + offs_m[:, None] * stride_q_seqlen + offs_d[None, :], acc)
            """,
            extra_globals={"_attn_fwd_inner": inner},
        )
        cfgs = [
            Config({"BLOCK_M": 64, "BLOCK_N": 32}),
            Config({"BLOCK_M": 128, "BLOCK_N": 64}),
        ]
        specs = infer_bucketable_axes(outer, configs=cfgs, keys=["KV_CTX", "HEAD_DIM"])
        self.assertEqual(specs["KV_CTX"].role, AxisRole.DYNAMIC_TILE_DIM)
        self.assertIn("BLOCK_N", specs["KV_CTX"].block_param_names)
        self.assertTrue(specs["KV_CTX"].feeds_dot)
        self.assertEqual(specs["Q_CTX"].role, AxisRole.DYNAMIC_TILE_DIM)
        self.assertEqual(specs["HEAD_DIM"].role, AxisRole.INVARIANT_CONST)
        self.assertEqual(specs["HAS_ATTN_MASK"].role, AxisRole.INVARIANT_CONST)
        self.assertEqual(specs["stride_k_seqlen"].role, AxisRole.DYNAMIC_STRIDE)

    def test_reduction_and_conv_spatial_strides_and_backends(self):
        rms_src = """
        def rms_norm_loop_kernel(out_ptr, in_ptr, w_ptr, N, eps, TILE_N: tl.constexpr):
            pid = tl.program_id(0)
            num_steps = tl.cdiv(N, TILE_N)
            acc = tl.zeros((TILE_N,), dtype=tl.float32)
            for step in range(0, num_steps):
                offs = step * TILE_N + tl.arange(0, TILE_N)
                x = tl.load(in_ptr + pid * N + offs, mask=offs < N, other=0.0)
                acc += x * x
            tl.store(out_ptr + pid, tl.sum(acc))
        """
        specs = infer_bucketable_axes(
            _ParsedKernel(rms_src),
            configs=[Config({"TILE_N": 256}), Config({"TILE_N": 1024})],
            keys=["N"],
        )
        self.assertEqual(specs["N"].role, AxisRole.DYNAMIC_TILE_DIM)
        self.assertFalse(specs["N"].feeds_dot)
        self.assertEqual(specs["N"].min_block_size, 1)

        conv_src = """
        def conv2d_kernel(x_ptr, w_ptr, y_ptr, out_nhw, stride_height, stride_width, groups, BLOCK_NIHO: tl.constexpr):
            offs = tl.program_id(0) * BLOCK_NIHO + tl.arange(0, BLOCK_NIHO)
            val = tl.load(x_ptr + offs * stride_height, mask=offs < out_nhw)
            tl.store(y_ptr + offs, val, mask=offs < out_nhw)
        """
        conv_specs = infer_bucketable_axes(
            _ParsedKernel(conv_src),
            configs=[Config({"BLOCK_NIHO": 64})],
            keys=["out_nhw", "stride_height", "stride_width", "groups"],
        )
        self.assertEqual(conv_specs["out_nhw"].role, AxisRole.DYNAMIC_TILE_DIM)
        self.assertEqual(conv_specs["stride_height"].role, AxisRole.INVARIANT_CONST)
        self.assertEqual(conv_specs["stride_width"].role, AxisRole.INVARIANT_CONST)
        self.assertEqual(conv_specs["groups"].role, AxisRole.INVARIANT_CONST)

        self.assertEqual(get_backend_min_dot_size("sunrise"), (8, 8, 16))
        self.assertEqual(get_backend_min_dot_size("default"), (16, 16, 16))


class TestOnlineBucketPolicyTrainAndInfer(unittest.TestCase):

    def test_decode_fine_ladder_and_1000_step_kv_ctx_growth(self):
        policy = OnlineBucketPolicy(mode=BucketingMode.INFERENCE)
        self.assertEqual(DECODE_FINE_LADDER, (1, 2, 4, 8, 16, 24, 32, 48, 64))
        # M=1, M=2, M=8, M=32 get distinct Decode buckets (unlike coarse align32!)
        b1, _ = policy.bucket_dim_value("M", 1)
        b2, _ = policy.bucket_dim_value("M", 2)
        b8, _ = policy.bucket_dim_value("M", 8)
        b32, _ = policy.bucket_dim_value("M", 32)
        self.assertEqual((b1, b2, b8, b32), (1, 2, 8, 32))

        # Simulate 1,000 autoregressive Decode steps: KV_CTX = 513..1512, HEAD_DIM = 128
        q = _FakeTensor("torch.bfloat16")
        k = _FakeTensor("torch.bfloat16")
        unique_bucket_keys = set()
        for kv_ctx in range(513, 1513):
            res = policy.lookup_autotune_key(
                {"Q": q, "K": k, "KV_CTX": kv_ctx, "HEAD_DIM": 128, "IS_CAUSAL": True},
                ["KV_CTX", "HEAD_DIM", "IS_CAUSAL"],
            )
            # Must be 100% JSON serializable for Autotuner.check_disk_cache
            serialized = json.dumps({"key": res.bucket_key})
            self.assertIsInstance(serialized, str)
            unique_bucket_keys.add(res.bucket_key)

        # 3 ceiling steps (768, 1024, 1536) x 2 alignment states (divisible by 16 vs not) = 6 keys across 1,000 tokens!
        self.assertLessEqual(len(unique_bucket_keys), 6)

        # HEAD_DIM=64 vs HEAD_DIM=128 must NEVER collide; IS_CAUSAL=True vs False must NEVER collide;
        # FP16 vs BF16 must NEVER collide.
        r_hd64 = policy.lookup_autotune_key({"Q": q, "KV_CTX": 600, "HEAD_DIM": 64, "IS_CAUSAL": True}, ["KV_CTX", "HEAD_DIM", "IS_CAUSAL"])
        r_hd128 = policy.lookup_autotune_key({"Q": q, "KV_CTX": 600, "HEAD_DIM": 128, "IS_CAUSAL": True}, ["KV_CTX", "HEAD_DIM", "IS_CAUSAL"])
        r_nocausal = policy.lookup_autotune_key({"Q": q, "KV_CTX": 600, "HEAD_DIM": 128, "IS_CAUSAL": False}, ["KV_CTX", "HEAD_DIM", "IS_CAUSAL"])
        r_fp16 = policy.lookup_autotune_key({"Q": _FakeTensor("torch.float16"), "KV_CTX": 600, "HEAD_DIM": 128, "IS_CAUSAL": True}, ["KV_CTX", "HEAD_DIM", "IS_CAUSAL"])
        self.assertNotEqual(r_hd64.bucket_key, r_hd128.bucket_key)
        self.assertNotEqual(r_hd128.bucket_key, r_nocausal.bucket_key)
        self.assertNotEqual(r_hd128.bucket_key, r_fp16.bucket_key)

    def test_training_backward_dw_dynamic_k_and_contiguous_stride_collapse(self):
        mm_src = """
        def mm_bwd_dw(A, B, C, M, N, K, stride_am, stride_bk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
            rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
            rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                rk = k0 + tl.arange(0, BLOCK_K)
                a = tl.load(A + rm[:, None] * stride_am + rk[None, :], mask=(rk < K)[None, :])
                b = tl.load(B + rk[:, None] * stride_bk + rn[None, :], mask=(rk < K)[:, None])
                acc += tl.dot(a, b)
            tl.store(C + rm[:, None] * N + rn[None, :], acc)
        """
        specs = infer_bucketable_axes(
            _ParsedKernel(mm_src),
            configs=[Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64})],
            keys=["M", "N", "K", "stride_am", "stride_bk"],
        )
        policy = OnlineBucketPolicy(specs, mode=BucketingMode.TRAINING)
        a_t, b_t, c_t = _FakeTensor(), _FakeTensor(), _FakeTensor()

        # Simulate dW = dY^T @ X where M=4096, N=4096 are static model dims and
        # contraction K = B*S varies dynamically across training steps (1040, 1056, ..., 1520)
        # while A has contiguous row stride_am == K!
        keys_seen = set()
        for k_tokens in range(1040, 1536, 16):
            res = policy.lookup_autotune_key(
                {
                    "A": a_t,
                    "B": b_t,
                    "C": c_t,
                    "M": 4096,
                    "N": 4096,
                    "K": k_tokens,
                    "stride_am": k_tokens,
                    "stride_bk": 4096,
                },
                ["M", "N", "K", "stride_am", "stride_bk"],
            )
            keys_seen.add(res.bucket_key)
        # Every 16-aligned K in (1024, 1536] maps to the SINGLE bucket 1536!
        self.assertEqual(len(keys_seen), 1)


class TestAutotunerFiveStateAndQuotaGovernance(unittest.TestCase):

    def setUp(self):
        GLOBAL_BUCKET_TELEMETRY.reset()
        clear_analysis_cache()
        clear_policy_caches()

    def test_five_states_unpadded_launch_warmup_and_disk_quota(self):
        src = """
        def vec_norm_kernel(X, Y, N, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr):
            tl.static_assert(BLOCK_N <= HEAD_DIM)
            offs = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
            v = tl.load(X + offs, mask=offs < N, other=0.0)
            tl.store(Y + offs, v, mask=offs < N)
        """
        parsed = _ParsedKernel(src)
        runner = _DummyJITRunner(parsed)
        bench_calls = []

        def fake_bench(fn_call, quantiles=None):
            bench_calls.append(1)
            fn_call()
            return [0.1, 0.09, 0.11]

        def prune_fn(configs, named_args, **kwargs):
            # Simulate dynamic guard: when N == 240 (16-aligned in bucket 256), disallow BLOCK_N == 64
            n_val = named_args.get("N", kwargs.get("N", 0))
            if n_val == 240:
                return [c for c in configs if c.kwargs["BLOCK_N"] <= 32]
            return [c for c in configs if c.kwargs["BLOCK_N"] >= 64]

        policy = OnlineBucketPolicy(
            infer_bucketable_axes(
                parsed,
                configs=[Config({"BLOCK_N": 32}), Config({"BLOCK_N": 64})],
                keys=["N", "HEAD_DIM"],
            ),
            mode=BucketingMode.INFERENCE,
            domain_bounds={"N": (8, 4096)},
        )
        tuner = Autotuner(
            runner,
            parsed.arg_names,
            [Config({"BLOCK_N": 32}), Config({"BLOCK_N": 64})],
            ["N", "HEAD_DIM"],
            reset_to_zero=None,
            restore_value=None,
            prune_configs_by={"early_config_prune": prune_fn},
            do_bench=fake_bench,
            cache_results=False,
            shape_policy=policy,
        )
        x_t, y_t = _FakeTensor(), _FakeTensor()

        with knobs.autotuning.scope():
            knobs.autotuning.adjust_block_size = False
            knobs.autotuning.max_bucket_artifacts = 4

            # 1. Cold tune at N=208 -> bucket 256 -> COLD_TUNED
            tuner.run(x_t, y_t, 208, HEAD_DIM=128)
            self.assertEqual(tuner.last_dispatch_result.status, DispatchStatus.COLD_TUNED)
            # Verify kernel launched with EXACT unpadded N=208 (never padded to 256!)
            self.assertEqual(runner.launches[-1][0][2], 208)

            # 2. Nearby aligned shape N=224 -> same bucket 256 -> HIT without bench!
            prev_benches = len(bench_calls)
            tuner.run(x_t, y_t, 224, HEAD_DIM=128)
            self.assertEqual(tuner.last_dispatch_result.status, DispatchStatus.HIT)
            self.assertEqual(len(bench_calls), prev_benches)
            self.assertEqual(runner.launches[-1][0][2], 224)

            # 3. Guard mismatch at N=240 (also 16-aligned in bucket 256, but prune_fn rejects BLOCK_N=64!)
            tuner.run(x_t, y_t, 240, HEAD_DIM=128)
            self.assertEqual(tuner.last_dispatch_result.status, DispatchStatus.GUARD_MISMATCH)
            self.assertEqual(tuner.best_config.kwargs["BLOCK_N"], 32)

            # 4. UNDERFLOW (N=2 < lo=8) & OVERFLOW (N=8192 > hi=4096)
            tuner.run(x_t, y_t, 2, HEAD_DIM=128)
            self.assertEqual(tuner.last_dispatch_result.status, DispatchStatus.UNDERFLOW)
            tuner.run(x_t, y_t, 8192, HEAD_DIM=128)
            self.assertEqual(tuner.last_dispatch_result.status, DispatchStatus.OVERFLOW)

            # LRU cap check
            self.assertLessEqual(len(tuner.cache), 4)

        summary = GLOBAL_BUCKET_TELEMETRY.summary()
        self.assertGreaterEqual(summary["counts"]["HIT"], 1)
        self.assertGreaterEqual(summary["counts"]["COLD_TUNED"], 1)
        self.assertGreaterEqual(summary["counts"]["GUARD_MISMATCH"], 1)
        self.assertGreaterEqual(summary["counts"]["UNDERFLOW"], 1)
        self.assertGreaterEqual(summary["counts"]["OVERFLOW"], 1)

    def test_warmup_buckets_zero_cold_misses_and_pgo_interval_policy(self):
        src = """
        def decode_proj(X, Y, M, BLOCK_M: tl.constexpr):
            offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
            v = tl.load(X + offs, mask=offs < M, other=0.0)
            tl.store(Y + offs, v, mask=offs < M)
        """
        parsed = _ParsedKernel(src)
        runner = _DummyJITRunner(parsed)
        tuner = Autotuner(
            runner,
            parsed.arg_names,
            [Config({"BLOCK_M": 16}), Config({"BLOCK_M": 64})],
            ["M"],
            reset_to_zero=None,
            restore_value=None,
            do_bench=lambda fn_call, quantiles=None: [0.05, 0.04, 0.06],
            cache_results=False,
            shape_policy=OnlineBucketPolicy(mode=BucketingMode.INFERENCE),
        )
        x_t, y_t = _FakeTensor(), _FakeTensor()

        with knobs.autotuning.scope():
            knobs.autotuning.adjust_block_size = False
            warmed = warmup_buckets(
                lambda *a, **kw: tuner.run(*a, **kw),
                {"M": (1, 128)},
                lambda pt: ((x_t, y_t, pt["M"]), {}),
                policy=tuner.shape_policy,
            )
            self.assertGreater(len(warmed), 0)

            GLOBAL_BUCKET_TELEMETRY.reset()
            for m_val in range(1, 129):
                tuner.run(x_t, y_t, m_val)
                self.assertEqual(tuner.last_dispatch_result.status, DispatchStatus.HIT)
            self.assertEqual(GLOBAL_BUCKET_TELEMETRY.summary()["hit_rate"], 1.0)

        # Verify generalized PGO DP (`group_frontier` + `allocate_budget` + `IntervalPolicy` JSON + disk quota)
        shapes = [16, 32, 64, 128, 256, 512]
        timings = [
            [0.10, 0.30],
            [0.12, 0.28],
            [0.15, 0.25],
            [0.40, 0.18],
            [0.65, 0.20],
            [1.10, 0.22],
        ]
        segs = group_frontier(shapes, timings, max_buckets=2)
        self.assertEqual(len(segs), 2)
        self.assertEqual((segs[0].lo, segs[0].hi, segs[0].config_index), (16, 64, 0))
        self.assertEqual((segs[1].lo, segs[1].hi, segs[1].config_index), (128, 512, 1))

        alloc = allocate_budget({"g1": [10.0, 2.0, 0.5], "g2": [3.0, 2.5, 2.4]}, total_budget=3)
        self.assertEqual(alloc, {"g1": 2, "g2": 1})

        with tempfile.TemporaryDirectory() as tmpd:
            gid = IntervalPolicy.format_group_id({"M": 32, "N": 1024}, ["M", "N"], "M", ("torch.bfloat16",))
            ipol = IntervalPolicy("M", {gid: segs})
            json_path = Path(tmpd) / "policy.json"
            ipol.save_json(json_path)
            loaded = IntervalPolicy.load_json(json_path)

            hit_res = loaded.lookup_autotune_key({"M": 48, "N": 1024}, ["M", "N"], extra_dtypes=("torch.bfloat16",))
            self.assertEqual(hit_res.status, DispatchStatus.HIT)
            unseen_res = loaded.lookup_autotune_key({"M": 48, "N": 2048}, ["M", "N"], extra_dtypes=("torch.bfloat16",))
            self.assertEqual(unseen_res.status, DispatchStatus.GUARD_MISMATCH)

            # Disk quota enforcement test
            for idx in range(5):
                p = Path(tmpd) / f"sub_{idx}" / "kern.autotune.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("{}", encoding="utf-8")
            removed = enforce_disk_cache_budget(tmpd, max_artifacts=2)
            self.assertEqual(removed, 3)


    def test_ascend_inner_constexpr_kv_ctx_helper_propagation(self):
        # Mirrors FlagGems _ascend/ops/attention.py where _attn_fwd_inner annotates
        # KV_CTX: tl.constexpr while outer _attn_fwd receives dynamic runtime KV_CTX.
        inner = _ParsedKernel(
            """
            def _ascend_attn_fwd_inner(
                acc, K_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                STAGE: tl.constexpr, offs_n: tl.constexpr, KV_CTX: tl.constexpr
            ):
                if STAGE == 1:
                    lo, hi = 0, BLOCK_M
                else:
                    lo, hi = 0, KV_CTX
                for start_n in range(lo, hi, BLOCK_N):
                    mask = (start_n + offs_n) < KV_CTX
                    k = tl.load(K_ptr + start_n, mask=mask)
                    acc = tl.dot(acc, k)
                return acc
            """
        )
        outer = _ParsedKernel(
            """
            def _ascend_attn_fwd(
                Q, K, Z, Q_CTX, KV_CTX,
                HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr
            ):
                tl.static_assert(BLOCK_N <= HEAD_DIM)
                offs_m = tl.arange(0, BLOCK_M)
                q_mask = offs_m < Q_CTX
                offs_n = tl.arange(0, BLOCK_N)
                q = tl.load(Q + offs_m, mask=q_mask)
                _ascend_attn_fwd_inner(q, K, BLOCK_M, BLOCK_N, STAGE, offs_n, KV_CTX)
            """,
            extra_globals={"_ascend_attn_fwd_inner": inner},
        )
        specs = infer_bucketable_axes(
            outer,
            configs=[Config({"BLOCK_M": 64, "BLOCK_N": 64})],
            keys=["Z", "Q_CTX", "KV_CTX", "HEAD_DIM"],
            backend="ascend",
        )
        self.assertEqual(specs["KV_CTX"].role, AxisRole.DYNAMIC_TILE_DIM)
        self.assertIn("BLOCK_N", specs["KV_CTX"].block_param_names)
        self.assertTrue(specs["KV_CTX"].feeds_dot)
        self.assertEqual(specs["HEAD_DIM"].role, AxisRole.INVARIANT_CONST)


if __name__ == "__main__":
    unittest.main()
