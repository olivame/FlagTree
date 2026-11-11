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
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


def _ensure_triton_importable() -> None:
    repo_python = Path(__file__).resolve().parents[3]
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

from triton.runtime.bucket_guard import (  # noqa: E402
    DecodeGraphBucketRunner,
    SymbolicBucketGuard,
    TokenBudgetBatcher,
)
from triton.runtime.shape_bucketing import BucketingMode, DispatchStatus  # noqa: E402
from triton.runtime.shape_profiler import ProfileMode, ShapeProfiler  # noqa: E402


class _FakeGraph:

    def __init__(self, fn, out_holder):
        self._fn = fn
        self._holder = out_holder
        self.replay_calls = 0
        self.output = self._fn()

    def replay(self):
        self.replay_calls += 1
        out = self._fn()
        self._holder[:] = out
        self.output = self._holder


class _FakeGraphAdapter:

    def __init__(self):
        self.captured = []

    def capture(self, fn):
        holder = []
        g = _FakeGraph(fn, holder)
        holder[:] = g.output
        g.output = holder
        self.captured.append(g)
        return g


class TestThreeModeShapeProfiler(unittest.TestCase):

    def test_llm_serving_prefill_decode_and_moe_profile(self):
        prof = ShapeProfiler(mode=ProfileMode.INFER_LLM_SERVING)
        # Prefill prompts: M = 512, 1400
        prof.record_op("mm", [(512, 4096), (4096, 4096)], dtypes=["bf16", "bf16"])
        prof.record_op("mm", [(1400, 4096), (4096, 4096)], dtypes=["bf16", "bf16"])
        # Decode steps: batch=4, KV_CTX growing 1025..1030
        for kv in range(1025, 1031):
            prof.record_op("attention", [(4, 1, 32, 128), (4, kv, 8, 128)], kv_len=kv)
        # MoE routing skew
        prof.record_op("fused_moe", [(64, 4096)], expert_counts=[2, 4, 30, 28])

        rep = prof.summary()
        self.assertEqual(rep["llm_serving"]["prefill_tokens"]["count"], 2)
        self.assertEqual(rep["llm_serving"]["decode_batch_size"]["count"], 6)
        self.assertEqual(rep["llm_serving"]["decode_kv_len"]["max"], 1030.0)
        self.assertGreater(rep["llm_serving"]["moe_expert_skew"]["max"], 1.5)

        pol = prof.recommend_bucket_policy()
        self.assertEqual(pol.mode, BucketingMode.INFERENCE)
        self.assertIn("KV_CTX", pol.domain_bounds)

        with tempfile.TemporaryDirectory() as tmpd:
            out_p = Path(tmpd) / "llm_profile.json"
            prof.save_json(out_p)
            loaded = json.loads(out_p.read_text(encoding="utf-8"))
            self.assertEqual(loaded["mode"], "infer_llm_serving")

    def test_retrieval_2d_batch_and_training_fwd_bwd_multi_rank_skew(self):
        ret_prof = ShapeProfiler(mode=ProfileMode.INFER_RETRIEVAL_BATCH)
        for b, s in [(8, 96), (16, 120), (32, 240), (32, 500)]:
            ret_prof.record_op("encoder_mm", [(b, s, 1024)])
        ret_summary = ret_prof.summary()
        self.assertEqual(ret_summary["retrieval_batch"]["unique_2d_shapes"], 4)
        ret_pol = ret_prof.recommend_bucket_policy()
        # Below min bound 8 -> UNDERFLOW
        _, st_under = ret_pol.bucket_dim_value("B", 2)
        self.assertEqual(st_under, DispatchStatus.UNDERFLOW)

        trn_prof = ShapeProfiler(mode=ProfileMode.TRAIN_FWD_BWD, rank=0)
        trn_prof.step()
        trn_prof.record_op("linear_fwd", [(8, 300, 4096)])
        trn_prof.set_backward_context(True)
        trn_prof.record_op("mm_dw", [(4096, 2400), (2400, 4096)])
        trn_prof.set_backward_context(False)
        skew = trn_prof.record_rank_tokens([1200, 1400, 1300, 2600, 1100, 1250, 1350, 1400])
        self.assertGreater(skew.skew_ratio, 1.7)
        trn_summary = trn_prof.summary()
        self.assertEqual(trn_summary["training"]["forward_tokens"]["max"], 2400.0)
        self.assertEqual(trn_summary["training"]["backward_contraction_k"]["max"], 2400.0)


class TestBucketGuardsTrainAndInfer(unittest.TestCase):

    def test_symbolic_bucket_guard_zero_padding_pairing_and_memory_cap(self):
        # 1 MB budget, 1024 bytes/token -> at B=2, max safe S = 512!
        guard = SymbolicBucketGuard(
            mode=BucketingMode.TRAINING,
            activation_budget_bytes=1024 * 1024,
            bytes_per_token_activation=1024,
        )
        # S=520 normally buckets up to 768, which at B=2 would take 1.5 MB > 1.0 MB!
        # Since 520 > 512 (even exact shape exceeds 1 MB), guard recommends 2 microbatch splits!
        over_plan = guard.plan_step({"S": 520}, batch_size=2, seq_axis="S", step_id=10)
        self.assertTrue(over_plan.memory_capped)
        self.assertGreaterEqual(over_plan.recommended_microbatch_splits, 2)
        bwd_plan = guard.get_paired_backward_plan(10)
        self.assertEqual(bwd_plan, over_plan)

        # S=400 normally buckets up to 512; with tighter budget 900 KB (max_safe_s = 450 >= 400),
        # ceiling tightens from 512 down to 450 without needing microbatch splits!
        guard_tight = SymbolicBucketGuard(
            mode=BucketingMode.TRAINING,
            activation_budget_bytes=900 * 1024,
            bytes_per_token_activation=1024,
        )
        tight_plan = guard_tight.plan_step({"S": 400}, batch_size=2, seq_axis="S", step_id=11)
        self.assertTrue(tight_plan.memory_capped)
        self.assertEqual(tight_plan.recommended_microbatch_splits, 1)
        self.assertEqual(tight_plan.axis_intervals["S"][1], 450)

    def test_decode_graph_bucket_runner_dummy_mask_and_rl_release(self):
        observed_kv_masks = []

        def decode_step(tokens, kv_lens):
            observed_kv_masks.append(list(kv_lens))
            # Output equals token + valid_kv_len (dummy slots with kv_len == 0 produce 0!)
            return [tok + kv if kv > 0 else -999 for tok, kv in zip(tokens, kv_lens)]

        adapter = _FakeGraphAdapter()
        runner = DecodeGraphBucketRunner(
            decode_step,
            batch_buckets=(1, 2, 4, 8, 16),
            graph_adapter=adapter,
        )

        # Batch of 3 snaps to bucket 4: slot 3 MUST receive kv_len == 0!
        out_3 = runner.run_decode_step([10, 20, 30], [100, 200, 300])
        self.assertEqual(out_3, [110, 220, 330])
        self.assertEqual(observed_kv_masks[-1], [100, 200, 300, 0])
        self.assertEqual(runner.capture_count, 1)

        # Next Decode token for batch of 4 reuses the ALREADY captured bucket-4 graph!
        out_4 = runner.run_decode_step([11, 21, 31, 41], [101, 201, 301, 50])
        self.assertEqual(out_4, [112, 222, 332, 91])
        self.assertEqual(runner.capture_count, 1)
        self.assertEqual(runner.replay_count, 1)

        # Batch of 20 (> max bucket 16) safely falls back to unpadded eager execution
        out_20 = runner.run_decode_step(list(range(20)), [10] * 20)
        self.assertEqual(len(out_20), 20)
        self.assertEqual(runner.eager_overflow_count, 1)

        # Simulate RL Rollout -> Policy Update transition releasing graph buffers
        released = runner.release_graphs()
        self.assertEqual(released, 1)

    def test_token_budget_batcher_preserves_masks_and_restores_exact_order(self):
        # Mix short and long queries as seen across 500+ retrieval/reranker workloads
        queries = [
            [1] * 300,  # idx 0 (long)
            [2] * 12,   # idx 1 (short)
            [3] * 280,  # idx 2 (long)
            [4] * 15,   # idx 3 (short)
            [5] * 10,   # idx 4 (short)
        ]
        batcher = TokenBudgetBatcher(max_tokens_per_batch=640, max_batch_size=4)
        batches = batcher.build_batches(queries)
        # Short queries [4, 1, 3] cluster into a small bucket (padded_seq_len=16),
        # instead of being padded to 384 alongside idx 0 and idx 2!
        self.assertGreaterEqual(len(batches), 2)
        self.assertEqual(batches[0].padded_seq_len, 16)
        for b in batches:
            for seq_row, mask_row in zip(b.padded_sequences, b.attention_mask):
                self.assertEqual(len(seq_row), b.padded_seq_len)
                self.assertEqual(len(mask_row), b.padded_seq_len)
                # Every padded tail position MUST have mask == 0
                valid_cnt = sum(mask_row)
                self.assertEqual(mask_row[:valid_cnt], (1,) * valid_cnt)
                self.assertEqual(mask_row[valid_cnt:], (0,) * (b.padded_seq_len - valid_cnt))

        # Simulate model inference returning sum of masked valid tokens per row
        batch_outputs = [
            [sum(tok * m for tok, m in zip(row, mrow)) for row, mrow in zip(b.padded_sequences, b.attention_mask)]
            for b in batches
        ]
        restored = batcher.restore_order(batch_outputs, batches)
        expected = [sum(q) for q in queries]
        self.assertEqual(restored, expected)


if __name__ == "__main__":
    unittest.main()
