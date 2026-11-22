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
    try:
        from triton._C.libtriton import getenv as _g  # noqa: F401
        from triton.runtime import _distributed as _d  # noqa: F401
        return
    except Exception:
        pass
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


def test_default_dispatch_guard_creates_isolated_gloo_pg(monkeypatch):
    import sys
    import types
    import triton.runtime._distributed as dmod

    sentinel_pg = object()
    world_pg_1 = object()
    world_pg_2 = object()
    calls = []

    fake_group = types.SimpleNamespace(WORLD=world_pg_1)
    fake_dist = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 4,
        get_rank=lambda: 1,
        is_gloo_available=lambda: True,
        group=fake_group,
        new_group=lambda backend=None: (calls.append(backend), sentinel_pg)[1],
    )
    fake_torch = types.SimpleNamespace(distributed=fake_dist)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    dmod.reset_default_dispatch_guard()

    # 1. Lazy get_default_dispatch_guard() MUST be strictly non-collective (no dist.new_group)
    lazy_guard = dmod.get_default_dispatch_guard()
    assert lazy_guard is not None
    assert lazy_guard.world_size == 4
    assert lazy_guard.rank == 1
    assert lazy_guard.process_group is None
    assert calls == []

    # 2. Eager init_default_dispatch_guard() collectively creates an isolated gloo PG
    eager_guard = dmod.init_default_dispatch_guard()
    assert eager_guard is not None
    assert eager_guard.world_size == 4
    assert eager_guard.rank == 1
    assert eager_guard.process_group is sentinel_pg
    assert calls == ["gloo"]

    # 3. Subsequent get_default_dispatch_guard() reuses the eager isolated guard
    assert dmod.get_default_dispatch_guard() is eager_guard
    assert calls == ["gloo"]

    # 4. World PG re-initialization invalidates stale guard without calling new_group
    fake_group.WORLD = world_pg_2
    refreshed = dmod.get_default_dispatch_guard()
    assert refreshed is not eager_guard
    assert refreshed.process_group is None
    assert calls == ["gloo"]

    # 5. Caller-supplied PG and opt-in create_pg=True work cleanly
    custom_pg = object()
    custom_guard = dmod.init_default_dispatch_guard(process_group=custom_pg)
    assert custom_guard is not None and custom_guard.process_group is custom_pg
    assert calls == ["gloo"]

    dmod.reset_default_dispatch_guard()
    opt_in = dmod.get_default_dispatch_guard(create_pg=True)
    assert opt_in is not None and opt_in.process_group is sentinel_pg
    assert calls == ["gloo", "gloo"]
    dmod.reset_default_dispatch_guard()


def test_dispatch_guard_scope_reentrancy_and_autotuner_integration():
    import triton.runtime._distributed as dmod

    assert not dmod.is_in_dispatch_guard_scope()
    barriers = []
    guard = DistributedDispatchGuard(
        world_size=2,
        rank=0,
        all_reduce_or_fn=lambda m: m,
        barrier_fn=lambda: barriers.append(1),
    )

    # 1. Nested coordinate() inside coordinate() negotiates and syncs strictly once
    with guard.coordinate(RankDispatchState.ANY_COLD_COMPILE) as outer_dec:
        assert dmod.is_in_dispatch_guard_scope()
        assert outer_dec is not None and outer_dec.extend_collective_window
        with guard.coordinate(RankDispatchState.ALL_HOT_HIT) as inner_dec:
            assert inner_dec is None
            assert dmod.is_in_dispatch_guard_scope()
    assert not dmod.is_in_dispatch_guard_scope()
    assert guard.negotiation_count == 1
    assert len(barriers) == 2

    # 2. Exception safety: post_compile_sync still runs in finally and scope resets
    barriers.clear()
    try:
        with guard.coordinate(RankDispatchState.ANY_COLD_COMPILE):
            raise RuntimeError("simulated compile failure")
    except RuntimeError:
        pass
    assert not dmod.is_in_dispatch_guard_scope()
    assert guard.negotiation_count == 2
    assert len(barriers) == 2

    # 3. Autotuner skips duplicate negotiation inside outer dispatch_guard_scope()
    barriers.clear()
    tuner = Autotuner(
        _MiniJIT(),
        ["X", "N", "BLOCK"],
        [Config({"BLOCK": 64}), Config({"BLOCK": 128})],
        ["N"],
        reset_to_zero=None,
        restore_value=None,
        do_bench=lambda fn_call, quantiles=None: [0.1, 0.1, 0.1],
        cache_results=False,
    )
    tuner.dispatch_guard = guard
    dummy_x = SimpleNamespace(dtype="torch.bfloat16", data_ptr=lambda: 4096)

    with knobs.autotuning.scope():
        knobs.autotuning.adjust_block_size = False
        knobs.autotuning.shape_bucketing = True
        # Guard OFF -> zero collectives
        knobs.autotuning.distributed_dispatch_guard = False
        tuner.run(dummy_x, 128)
        assert guard.negotiation_count == 2
        assert len(barriers) == 0

        # Guard ON inside outer scope -> nested Autotuner skips duplicate negotiate
        knobs.autotuning.distributed_dispatch_guard = True
        with dmod.dispatch_guard_scope():
            tuner.run(dummy_x, 256)
        assert guard.negotiation_count == 2
        assert len(barriers) == 0

        # Guard ON standalone (cold miss) -> Autotuner negotiates + 2 barriers
        tuner.run(dummy_x, 512)
        assert guard.negotiation_count == 3
        assert len(barriers) == 2

        # Guard ON standalone (warm hit) -> negotiates ALL_HOT_HIT with 0 new barriers
        tuner.run(dummy_x, 512)
        assert guard.negotiation_count == 4
        assert len(barriers) == 2


def test_two_rank_gloo_asymmetric_get_default_dispatch_guard_no_deadlock():
    import subprocess

    try:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_gloo_available()):
            return
    except Exception:
        return

    worker_code = (
        "import datetime, os, sys, time, torch.distributed as dist\n"
        "from triton.runtime import _distributed as dmod\n"
        "rank = int(sys.argv[1])\n"
        "init_file = sys.argv[2]\n"
        "done_file = sys.argv[3]\n"
        "dist.init_process_group("
        "backend='gloo', init_method=f'file://{init_file}', "
        "rank=rank, world_size=2, timeout=datetime.timedelta(seconds=6))\n"
        "try:\n"
        "    dist.barrier()\n"
        "    if rank == 0:\n"
        "        t0 = time.monotonic()\n"
        "        while not os.path.exists(done_file):\n"
        "            assert time.monotonic() - t0 < 5.0, 'rank 1 hung'\n"
        "            time.sleep(0.02)\n"
        "    else:\n"
        "        t0 = time.monotonic()\n"
        "        g = dmod.get_default_dispatch_guard()\n"
        "        assert g is not None and g.process_group is None\n"
        "        assert time.monotonic() - t0 < 2.0\n"
        "        open(done_file, 'w').write('ok')\n"
        "    dist.barrier()\n"
        "    eg = dmod.init_default_dispatch_guard(create_gloo_group=True)\n"
        "    assert eg is not None and eg.process_group is not None\n"
        "    st = (dmod.RankDispatchState.ALL_HOT_HIT if rank == 0 "
        "else dmod.RankDispatchState.ANY_COLD_COMPILE)\n"
        "    with eg.coordinate(st) as dec:\n"
        "        assert dec is not None and dec.extend_collective_window\n"
        "    dmod.reset_default_dispatch_guard()\n"
        "finally:\n"
        "    dist.destroy_process_group()\n"
    )
    with tempfile.TemporaryDirectory(prefix="ft_gloo_guard_test_") as tmpd:
        init_f = os.path.join(tmpd, "rdv")
        done_f = os.path.join(tmpd, "r1_done")
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", worker_code, str(r), init_f, done_f],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for r in (0, 1)
        ]
        for r, p in enumerate(procs):
            out, err = p.communicate(timeout=15)
            assert p.returncode == 0, f"rank {r} failed ({p.returncode}):\n{out}\n{err}"
def test_distributed_fault_errors_attributes_and_pickling():
    import pickle
    from triton.runtime import (
        DistributedCommTimeoutError,
        DistributedRankFaultError,
    )

    err_to = DistributedCommTimeoutError(
        rank=2,
        world_size=4,
        op="all_reduce_bitmask",
        phase="dispatch_negotiate",
        timeout_s=3.0,
        cause="Operation timed out!",
    )
    assert err_to.rank == 2
    assert err_to.world_size == 4
    assert err_to.op == "all_reduce_bitmask"
    assert err_to.phase == "dispatch_negotiate"
    assert err_to.timeout_s == 3.0
    assert "timed out" in str(err_to).lower()
    rt_to = pickle.loads(pickle.dumps(err_to))
    assert (rt_to.rank, rt_to.world_size, rt_to.op, rt_to.phase, rt_to.timeout_s, rt_to.cause) == (
        2,
        4,
        "all_reduce_bitmask",
        "dispatch_negotiate",
        3.0,
        "Operation timed out!",
    )

    err_rf = DistributedRankFaultError(
        rank=1,
        world_size=4,
        op="check_step_health",
        phase="training",
        timeout_s=3.0,
        cause="Connection closed by peer",
    )
    assert err_rf.rank == 1
    assert err_rf.world_size == 4
    assert err_rf.timeout_s == 3.0
    assert "rank fault" in str(err_rf).lower()
    rt_rf = pickle.loads(pickle.dumps(err_rf))
    assert (rt_rf.rank, rt_rf.world_size, rt_rf.op, rt_rf.phase, rt_rf.timeout_s, rt_rf.cause) == (
        1,
        4,
        "check_step_health",
        "training",
        3.0,
        "Connection closed by peer",
    )


def test_dispatch_guard_comm_timeout_and_degraded_solo_circuit_breaker():
    from triton.runtime import (
        DistributedCommTimeoutError,
        DistributedRankFaultError,
    )
    from triton.runtime._distributed import FallbackRegistry

    # 1. raise_on_fault=True trips DistributedCommTimeoutError within bounded time
    reg1 = FallbackRegistry()

    def _stall_reduce(_mask: int) -> int:
        time.sleep(2.0)
        return 0

    guard1 = DistributedDispatchGuard(
        world_size=4,
        rank=1,
        all_reduce_or_fn=_stall_reduce,
        comm_timeout_s=0.25,
        raise_on_fault=True,
        fallback_registry=reg1,
    )
    t0 = time.monotonic()
    raised = None
    try:
        guard1.negotiate(RankDispatchState.ALL_HOT_HIT)
    except DistributedCommTimeoutError as exc:
        raised = exc
    elapsed = time.monotonic() - t0
    assert raised is not None
    assert elapsed < 1.2
    assert raised.rank == 1 and raised.world_size == 4 and raised.timeout_s == 0.25
    assert guard1.degraded_solo
    assert guard1.fault_count == 1
    assert reg1.summary()["total_fallbacks"] == 1
    assert reg1.summary()["records"][0]["recovery_action"] == "abort_comm_timeout"

    # 2. raise_on_fault=False degrades cleanly to degraded_solo without hanging
    reg2 = FallbackRegistry()
    barrier_calls = []
    guard2 = DistributedDispatchGuard(
        world_size=4,
        rank=0,
        all_reduce_or_fn=_stall_reduce,
        barrier_fn=lambda: barrier_calls.append(1),
        comm_timeout_s=0.2,
        raise_on_fault=False,
        fallback_registry=reg2,
    )
    dec = guard2.negotiate(RankDispatchState.ANY_COLD_COMPILE)
    assert dec.degraded_solo
    assert dec.agreed_fallback
    assert dec.fault_kind == "comm_timeout"
    assert not dec.barrier_completed
    assert guard2.post_compile_sync(dec) is False
    assert barrier_calls == []
    assert reg2.summary()["records"][0]["recovery_action"] == "degrade_solo_comm_timeout"
    guard2.reset_fault_state()
    assert not guard2.degraded_solo

    # 3. Peer rank fault maps to DistributedRankFaultError + structured audit
    reg3 = FallbackRegistry()

    def _broken_peer(_mask: int) -> int:
        raise ConnectionResetError("Connection reset by peer")

    guard3 = DistributedDispatchGuard(
        world_size=4,
        rank=2,
        all_reduce_or_fn=_broken_peer,
        comm_timeout_s=1.0,
        raise_on_fault=True,
        fallback_registry=reg3,
    )
    rf_exc = None
    try:
        guard3.negotiate(RankDispatchState.ALL_HOT_HIT)
    except DistributedRankFaultError as exc:
        rf_exc = exc
    assert rf_exc is not None
    assert rf_exc.rank == 2 and rf_exc.world_size == 4
    assert reg3.summary()["records"][0]["recovery_action"] == "abort_rank_fault"


def test_coordinated_step_guard_8_rank_asymmetric_oom_and_error_recovery():
    from triton.runtime import (
        CoordinatedStepGuard,
        DistributedCommTimeoutError,
        DistributedFaultCoordinator,
    )
    from triton.runtime._distributed import FallbackRegistry

    assert DistributedFaultCoordinator is CoordinatedStepGuard

    world_size = 8
    barrier = threading.Barrier(world_size)
    shared = [[0] * world_size for _ in range(world_size)]
    reg = FallbackRegistry()

    def make_reduce_fn(r: int):
        def _reduce(vec):
            shared[r] = list(vec)
            barrier.wait()
            out = [max(shared[k][i] for k in range(world_size)) for i in range(world_size)]
            barrier.wait()
            return out

        return _reduce

    # 1. Rank 5 hits OOM -> all 8 ranks synchronously split microbatches & commit
    committed = [0] * world_size
    splits_used = [0] * world_size
    results = [None] * world_size

    def oom_worker(r: int) -> None:
        sg = CoordinatedStepGuard(
            world_size=world_size,
            rank=r,
            comm_timeout_s=2.0,
            fallback_registry=reg,
            all_reduce_status_fn=make_reduce_fn(r),
        )

        def step_fn():
            if r == 5:
                raise RuntimeError("CUDA out of memory on long-tail S=2047")
            return "full"

        def split_fn(chunks: int):
            splits_used[r] = chunks
            return f"split_{chunks}"

        def post_ok_fn(val):
            committed[r] += 1
            return val

        res, health = sg.run_coordinated_step(
            "varlen_train",
            "training",
            work_units=8,
            step_fn=step_fn,
            split_fn=split_fn,
            post_ok_fn=post_ok_fn,
            shape=(8, 2047, 128),
            bucket_key=("ft_bkt_v1", 2047),
        )
        assert health.all_ok
        results[r] = res

    threads = [threading.Thread(target=oom_worker, args=(r,)) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()

    assert all(c == 1 for c in committed)
    assert all(s == 2 for s in splits_used)
    assert all(v == "split_2" for v in results)
    assert reg.summary()["total_fallbacks"] >= world_size * 2

    # 2. Rank 2 hits non-OOM error -> all 8 ranks safely skip commit without deadlock
    skipped_commits = [0] * world_size
    skip_decisions = [None] * world_size

    def err_worker(r: int) -> None:
        sg = CoordinatedStepGuard(
            world_size=world_size,
            rank=r,
            comm_timeout_s=2.0,
            fallback_registry=reg,
            all_reduce_status_fn=make_reduce_fn(r),
        )

        def step_fn():
            if r == 2:
                raise ValueError("transient NaN corruption")
            return "ok"

        _, health = sg.run_coordinated_step(
            "varlen_train_err",
            "training",
            work_units=8,
            step_fn=step_fn,
            post_ok_fn=lambda v: skipped_commits.__setitem__(r, skipped_commits[r] + 1),
        )
        skip_decisions[r] = health

    threads2 = [threading.Thread(target=err_worker, args=(r,)) for r in range(world_size)]
    for t in threads2:
        t.start()
    for t in threads2:
        t.join(timeout=5.0)
        assert not t.is_alive()

    assert all(c == 0 for c in skipped_commits)
    assert all(d.any_error and d.error_ranks == (2,) and d.recommended_action == "skip_step" for d in skip_decisions)

    # 3. Stalled status reduction trips DistributedCommTimeoutError cleanly
    reg_to = FallbackRegistry()
    sg_to = CoordinatedStepGuard(
        world_size=4,
        rank=0,
        comm_timeout_s=0.2,
        raise_on_fault=True,
        fallback_registry=reg_to,
        all_reduce_status_fn=lambda vec: (time.sleep(2.0), vec)[1],
    )
    caught = None
    try:
        sg_to.check_step_health(local_ok=True)
    except DistributedCommTimeoutError as exc:
        caught = exc
    assert caught is not None
    assert sg_to.faulted
    assert reg_to.summary()["total_fallbacks"] == 1


def test_multiprocess_gloo_sigkill_stall_and_asymmetric_oom_recovery():
    import signal
    import subprocess

    try:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_gloo_available()):
            return
    except Exception:
        return

    worker_code = (
        "import datetime, os, signal, sys, time, torch.distributed as dist\n"
        "from triton.runtime import (\n"
        "    CoordinatedStepGuard,\n"
        "    DistributedCommTimeoutError,\n"
        "    DistributedRankFaultError,\n"
        "    GLOBAL_FALLBACK_REGISTRY,\n"
        "    RankDispatchState,\n"
        "    init_default_dispatch_guard,\n"
        "    reset_default_dispatch_guard,\n"
        ")\n"
        "mode = sys.argv[1]\n"
        "rank = int(sys.argv[2])\n"
        "init_file = sys.argv[3]\n"
        "GLOBAL_FALLBACK_REGISTRY.clear()\n"
        "dist.init_process_group("
        "backend='gloo', init_method=f'file://{init_file}', "
        "rank=rank, world_size=2, timeout=datetime.timedelta(seconds=20))\n"
        "g = init_default_dispatch_guard("
        "create_gloo_group=True, comm_timeout_s=1.2, raise_on_fault=True)\n"
        "sg = CoordinatedStepGuard(dispatch_guard=g, comm_timeout_s=1.2, raise_on_fault=True)\n"
        "dist.barrier()\n"
        "if mode == 'sigkill':\n"
        "    if rank == 1:\n"
        "        time.sleep(0.15)\n"
        "        os.kill(os.getpid(), signal.SIGKILL)\n"
        "    t0 = time.monotonic()\n"
        "    try:\n"
        "        sg.check_step_health(local_ok=True, op_name='mp_sigkill')\n"
        "        sys.exit(2)\n"
        "    except (DistributedRankFaultError, DistributedCommTimeoutError):\n"
        "        assert time.monotonic() - t0 < 4.0\n"
        "        assert GLOBAL_FALLBACK_REGISTRY.summary()['total_fallbacks'] >= 1\n"
        "        sg.cleanup_after_fault(destroy_world=True)\n"
        "        reset_default_dispatch_guard()\n"
        "        sys.exit(0)\n"
        "elif mode == 'stall':\n"
        "    if rank == 1:\n"
        "        time.sleep(6.0)\n"
        "        sys.exit(0)\n"
        "    t0 = time.monotonic()\n"
        "    try:\n"
        "        g.negotiate(RankDispatchState.ALL_HOT_HIT)\n"
        "        sys.exit(2)\n"
        "    except DistributedCommTimeoutError:\n"
        "        assert time.monotonic() - t0 < 3.5\n"
        "        assert GLOBAL_FALLBACK_REGISTRY.summary()['total_fallbacks'] >= 1\n"
        "        g.cleanup_after_fault(destroy_world=True)\n"
        "        reset_default_dispatch_guard()\n"
        "        sys.exit(0)\n"
        "elif mode == 'oom':\n"
        "    def step_fn():\n"
        "        if rank == 1:\n"
        "            raise RuntimeError('CUDA out of memory')\n"
        "        return 10\n"
        "    val, health = sg.run_coordinated_step(\n"
        "        'mp_oom', 'training', 4, step_fn,\n"
        "        split_fn=lambda s: s * 100,\n"
        "    )\n"
        "    assert health.all_ok and val == 200\n"
        "    assert GLOBAL_FALLBACK_REGISTRY.summary()['total_fallbacks'] >= 2\n"
        "    dist.barrier()\n"
        "    reset_default_dispatch_guard()\n"
        "    dist.destroy_process_group()\n"
        "    sys.exit(0)\n"
    )

    for mode in ("sigkill", "stall", "oom"):
        with tempfile.TemporaryDirectory(prefix=f"ft_gloo_g05_{mode}_") as tmpd:
            init_f = os.path.join(tmpd, "rdv")
            procs = [
                subprocess.Popen(
                    [sys.executable, "-c", worker_code, mode, str(r), init_f],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for r in (0, 1)
            ]
            try:
                out0, err0 = procs[0].communicate(timeout=12)
                assert procs[0].returncode == 0, f"mode={mode} r0 rc={procs[0].returncode}:\n{out0}\n{err0}"
                if mode == "oom":
                    out1, err1 = procs[1].communicate(timeout=12)
                    assert procs[1].returncode == 0, f"mode={mode} r1 rc={procs[1].returncode}:\n{out1}\n{err1}"
                elif mode == "sigkill":
                    procs[1].wait(timeout=5)
                    assert procs[1].returncode == -signal.SIGKILL
                else:
                    if procs[1].poll() is None:
                        procs[1].kill()
                    procs[1].wait(timeout=5)
            finally:
                for p in procs:
                    if p.poll() is None:
                        p.kill()
                        p.wait(timeout=3)
