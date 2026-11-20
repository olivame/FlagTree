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
import threading
import time
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

from triton import knobs  # noqa: E402
from triton.runtime import (  # noqa: E402
    CompileTimeoutError,
    DistributedDispatchGuard,
    GLOBAL_COMPILE_BLACKLIST,
    GLOBAL_FALLBACK_REGISTRY,
    OOMRecoveryController,
    OutOfResources,
    PerfRiskLevel,
    RankDispatchState,
    run_with_compile_timeout,
)
from triton.runtime.autotuner import Autotuner, Config  # noqa: E402
from triton.runtime.jit import JITFunction  # noqa: E402


def _hung_subprocess_task():
    time.sleep(5.0)
    return 42


class _MiniJIT(JITFunction):

    def __init__(self):
        self.__name__ = "slow_or_fast_kernel"
        self.__globals__ = {}
        self.arg_names = ["X", "N", "BLOCK"]
        self._src = (
            "def slow_or_fast_kernel(X, N, BLOCK: tl.constexpr):\n"
            "    offs = tl.arange(0, BLOCK)\n"
            "    mask = offs < N\n"
            "    tl.load(X + offs, mask=mask)\n"
        )
        self.fn = lambda: None
        self.fn.__name__ = self.__name__

    @property
    def cache_key(self) -> str:
        return "ck_slow_or_fast"

    def run(self, *args, **kwargs):
        return ("ok", args, kwargs)


class TestDistributedDispatchAndResilience(unittest.TestCase):

    def setUp(self):
        GLOBAL_COMPILE_BLACKLIST.clear()
        GLOBAL_FALLBACK_REGISTRY.clear()

    def test_8_rank_ddp_and_tp_moe_skew_zero_deadlock_negotiation(self):
        # Simulate an 8-card collective across threads where Rank 3 hits a cold compile
        # (e.g., long tail sequence in DDP training or MoE expert spike in TP inference)
        # and Rank 6 requests a controlled fallback!
        world_size = 8
        barrier = threading.Barrier(world_size)
        shared_masks = [0] * world_size
        decisions = [None] * world_size

        def rank_worker(r: int, local_state: RankDispatchState) -> None:
            def all_reduce_or(val: int) -> int:
                shared_masks[r] = val
                barrier.wait()
                merged = 0
                for m in shared_masks:
                    merged |= m
                barrier.wait()
                return merged

            guard = DistributedDispatchGuard(
                world_size=world_size,
                rank=r,
                all_reduce_or_fn=all_reduce_or,
                barrier_fn=lambda: barrier.wait(),
            )
            decisions[r] = guard.negotiate(local_state)

        states = [RankDispatchState.ALL_HOT_HIT] * world_size
        states[3] = RankDispatchState.ANY_COLD_COMPILE
        states[6] = RankDispatchState.ANY_FALLBACK

        threads = [
            threading.Thread(target=rank_worker, args=(r, states[r]))
            for r in range(world_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            self.assertFalse(t.is_alive())

        # Every single rank MUST reach the exact same agreed decision without deadlock!
        for r in range(world_size):
            self.assertIsNotNone(decisions[r])
            self.assertFalse(decisions[r].all_hot_hit)
            self.assertTrue(decisions[r].extend_collective_window)
            self.assertTrue(decisions[r].agreed_fallback)
            self.assertTrue(decisions[r].barrier_completed)

    def test_subprocess_and_autotuner_compile_timeout_circuit_breaker(self):
        # 1. Real subprocess kill on hung compilation
        t0 = time.monotonic()
        with self.assertRaises(CompileTimeoutError):
            run_with_compile_timeout(
                _hung_subprocess_task,
                0.15,
                kernel_name="hung_backend_kernel",
                config_desc="BLOCK_M: 256",
                use_subprocess=True,
            )
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 4.0)
        self.assertTrue(GLOBAL_COMPILE_BLACKLIST.is_blacklisted("hung_backend_kernel", "BLOCK_M: 256"))

        # Persist and reload blacklist JSON
        with tempfile.TemporaryDirectory() as tmpd:
            bl_path = Path(tmpd) / "blacklist.json"
            GLOBAL_COMPILE_BLACKLIST.save_json(bl_path)
            GLOBAL_COMPILE_BLACKLIST.clear()
            self.assertEqual(GLOBAL_COMPILE_BLACKLIST.load_json(bl_path), 1)
            self.assertTrue(GLOBAL_COMPILE_BLACKLIST.is_blacklisted("hung_backend_kernel", "BLOCK_M: 256"))

        # 2. Autotuner circuit breaker skips hung Config(BLOCK=256) and selects fast Config(BLOCK=64)
        def bench_fn(kernel_call, quantiles=None):
            kernel_call()
            return [0.05, 0.04, 0.06]

        slow_cfg = Config({"BLOCK": 256})
        fast_cfg = Config({"BLOCK": 64})

        class _SlowOnceJIT(_MiniJIT):

            def run(self, *args, **kwargs):
                if kwargs.get("BLOCK") == 256:
                    time.sleep(1.3)
                return ("ok", args, kwargs)

        tuner2 = Autotuner(
            _SlowOnceJIT(),
            ["X", "N", "BLOCK"],
            [slow_cfg, fast_cfg],
            ["N"],
            reset_to_zero=None,
            restore_value=None,
            do_bench=bench_fn,
            cache_results=False,
        )
        with knobs.autotuning.scope():
            knobs.autotuning.adjust_block_size = False
            knobs.autotuning.shape_bucketing = True
            knobs.autotuning.compile_timeout_sec = 1
            tuner2.run(SimpleNamespace(dtype="torch.bfloat16", data_ptr=lambda: 4096), 512)
            self.assertEqual(tuner2.best_config.kwargs["BLOCK"], 64)
            self.assertTrue(GLOBAL_COMPILE_BLACKLIST.is_blacklisted(tuner2.base_fn.__name__, str(slow_cfg)))

    def test_differentiated_infer_and_train_oom_recovery_and_fallback_audit(self):
        ctrl = OOMRecoveryController(min_split_unit=16, max_split_depth=3)
        released_caches = [0]

        # 1. Inference Prefill OOM -> automatic chunked prefill split (MEDIUM risk)
        res_infer = ctrl.execute(
            "flash_prefill",
            "prefill",
            work_units=4096,
            primary_fn=lambda: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
            split_fn=lambda chunks: f"prefill_ok_in_{chunks}_chunks",
            release_cache_fn=lambda: released_caches.__setitem__(0, released_caches[0] + 1),
            bucket_key=("ft_bkt_v1", 4096),
            rank=0,
        )
        self.assertEqual(res_infer, "prefill_ok_in_2_chunks")
        self.assertEqual(released_caches[0], 1)

        # 2. Training Backward OOM where splits also OOM -> controlled ATen fallback (HIGH risk)
        res_train = ctrl.execute(
            "mm_dw",
            "backward",
            work_units=1024,
            primary_fn=lambda: (_ for _ in ()).throw(OutOfResources(256000, 163840, "shared_memory")),
            split_fn=lambda splits: (_ for _ in ()).throw(RuntimeError("XPU out of memory")),
            fallback_fn=lambda: "aten_mm_out",
            bucket_key=("ft_bkt_v1", 1024),
            rank=2,
        )
        self.assertEqual(res_train, "aten_mm_out")

        summary = GLOBAL_FALLBACK_REGISTRY.summary()
        self.assertEqual(summary["by_risk_level"]["medium"], 1)
        self.assertEqual(summary["by_risk_level"]["high"], 1)

        with tempfile.TemporaryDirectory() as tmpd:
            audit_p = Path(tmpd) / "fallbacks.json"
            GLOBAL_FALLBACK_REGISTRY.save_json(audit_p)
            saved = json.loads(audit_p.read_text(encoding="utf-8"))
            self.assertEqual(saved["total_fallbacks"], 2)


if __name__ == "__main__":
    unittest.main()
