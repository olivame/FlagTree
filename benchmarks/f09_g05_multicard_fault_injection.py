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
"""F09-G05 Real Multi-Card Fault-Injection & Long-Tail Varlen Soak Benchmark.

Validates FlagTree multi-card fault tolerance end-to-end across 4 devices:
  1. Scenario 1: Real 4-rank SIGKILL peer crash -> bounded detection, audit, clean exit.
  2. Scenario 2: Real 4-rank comm stall (12s) vs comm_timeout_s=3.0 -> timeout circuit breaker.
  3. Scenario 3: Real 4-rank asymmetric long-tail OOM (S=2047) -> CoordinatedStepGuard +
     OOMRecoveryController microbatch halving with exact weight parity (diff == 0.0).
  4. Scenario 4: Real 4-rank continuous long-tail varlen soak
     (S = 1, 7, 17, 63, 128, 257, 509, 1023, 2047) with F09 shape bucketing ON and
     periodic transient fault injection (OOM recovery, cold-compile skew, step skip).

Portable & hardcoded-path-free; uses gloo for control/collective health plane and real
accelerator device tensors for model/Triton kernel computation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple


LONG_TAIL_SHAPES: Tuple[int, ...] = (1, 7, 17, 63, 128, 257, 509, 1023, 2047)


def _build_model(hidden_dim: int = 128, device: str = "cuda:0"):
    import torch
    import torch.nn as nn

    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim, bias=True),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim, bias=True),
    ).to(device=device, dtype=torch.float32)


def _make_batch(
    rank: int,
    step_id: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    device: str,
):
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(50000 + step_id * 97 + rank)
    x = torch.randn((batch_size, seq_len, hidden_dim), generator=gen, dtype=torch.float32)
    y = torch.randn((batch_size, seq_len, hidden_dim), generator=gen, dtype=torch.float32)
    return x.to(device), y.to(device)


def _forward_backward(
    model,
    x,
    y,
    *,
    scale: float = 1.0,
    use_gems: bool = True,
) -> float:
    import torch.nn.functional as F

    if use_gems:
        try:
            import flag_gems

            with flag_gems.use_gems(include=["layer_norm", "gelu"]):
                out = model(x)
                loss = F.mse_loss(out, y) * scale
                loss.backward()
                return float(loss.detach().item())
        except Exception:
            pass
    out = model(x)
    loss = F.mse_loss(out, y) * scale
    loss.backward()
    return float(loss.detach().item())


def _allreduce_grads_gloo(model, world_size: int) -> None:
    import torch.distributed as dist

    for param in model.parameters():
        if param.grad is None:
            continue
        grad_cpu = param.grad.detach().cpu().contiguous()
        dist.all_reduce(grad_cpu, op=dist.ReduceOp.SUM)
        grad_cpu.div_(float(world_size))
        param.grad.copy_(grad_cpu.to(param.device))


def _flush_device() -> None:
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _worker_sigkill(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
    comm_timeout_s: float,
) -> int:
    import torch
    import torch.distributed as dist
    from triton.runtime import (
        CoordinatedStepGuard,
        DistributedCommTimeoutError,
        DistributedRankFaultError,
        GLOBAL_FALLBACK_REGISTRY,
        init_default_dispatch_guard,
        reset_default_dispatch_guard,
    )

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    GLOBAL_FALLBACK_REGISTRY.clear()

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    guard = init_default_dispatch_guard(
        create_gloo_group=True,
        timeout=datetime.timedelta(seconds=comm_timeout_s),
        comm_timeout_s=comm_timeout_s,
        raise_on_fault=True,
    )
    step_guard = CoordinatedStepGuard(
        dispatch_guard=guard,
        comm_timeout_s=comm_timeout_s,
        raise_on_fault=True,
    )

    model = _build_model(128, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    # Warm step S=63
    opt.zero_grad(set_to_none=True)
    x1, y1 = _make_batch(rank, 1, 2, 63, 128, device)
    _forward_backward(model, x1, y1)
    _allreduce_grads_gloo(model, world_size)
    opt.step()
    torch.cuda.synchronize(rank)
    dist.barrier()
    if rank == 0:
        (Path(out_dir) / "step1_done").write_text("ok", encoding="utf-8")

    # Fault step S=509: Rank (world_size - 1) SIGKILLs mid-step
    if rank == world_size - 1:
        x2, y2 = _make_batch(rank, 2, 2, 509, 128, device)
        _forward_backward(model, x2, y2)
        torch.cuda.synchronize(rank)
        time.sleep(0.25)
        os.kill(os.getpid(), signal.SIGKILL)

    t0 = time.monotonic()
    opt.zero_grad(set_to_none=True)
    x2, y2 = _make_batch(rank, 2, 2, 509, 128, device)
    _forward_backward(model, x2, y2)
    torch.cuda.synchronize(rank)

    try:
        step_guard.check_step_health(
            local_ok=True,
            local_oom=False,
            op_name="drill_sigkill_s509",
            phase="training",
            shape=(2, 509, 128),
        )
        return 2
    except (DistributedRankFaultError, DistributedCommTimeoutError) as exc:
        detect_s = time.monotonic() - t0
        _flush_device()
        _write_json(
            Path(out_dir) / f"rank_{rank}.json",
            {
                "rank": rank,
                "status": "fault_detected_clean_exit",
                "error_type": type(exc).__name__,
                "detect_s": round(detect_s, 4),
                "fallback_summary": GLOBAL_FALLBACK_REGISTRY.summary(),
            },
        )
        step_guard.cleanup_after_fault(destroy_world=True)
        guard.cleanup_after_fault(destroy_world=True)
        reset_default_dispatch_guard()
        return 0


def _worker_stall(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
    comm_timeout_s: float,
    stall_s: float,
) -> int:
    import torch
    import torch.distributed as dist
    from triton.runtime import (
        DistributedCommTimeoutError,
        GLOBAL_FALLBACK_REGISTRY,
        RankDispatchState,
        init_default_dispatch_guard,
        reset_default_dispatch_guard,
    )

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    GLOBAL_FALLBACK_REGISTRY.clear()

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    guard = init_default_dispatch_guard(
        create_gloo_group=True,
        timeout=datetime.timedelta(seconds=comm_timeout_s),
        comm_timeout_s=comm_timeout_s,
        raise_on_fault=True,
    )

    model = _build_model(128, device=device)
    x1, y1 = _make_batch(rank, 1, 2, 63, 128, device)
    _forward_backward(model, x1, y1)
    torch.cuda.synchronize(rank)
    dist.barrier()
    if rank == 0:
        (Path(out_dir) / "step1_done").write_text("ok", encoding="utf-8")

    stalled_rank = 2 % world_size
    if rank == stalled_rank:
        _write_json(
            Path(out_dir) / f"rank_{rank}.json",
            {"rank": rank, "status": "intentionally_stalled", "stall_s": stall_s},
        )
        time.sleep(stall_s)
        return 0

    t0 = time.monotonic()
    try:
        guard.negotiate(RankDispatchState.ALL_HOT_HIT)
        return 2
    except DistributedCommTimeoutError as exc:
        trip_s = time.monotonic() - t0
        _flush_device()
        _write_json(
            Path(out_dir) / f"rank_{rank}.json",
            {
                "rank": rank,
                "status": "timeout_circuit_breaker_tripped",
                "error_type": type(exc).__name__,
                "timeout_s": exc.timeout_s,
                "trip_s": round(trip_s, 4),
                "fallback_summary": GLOBAL_FALLBACK_REGISTRY.summary(),
            },
        )
        reset_default_dispatch_guard()
        return 0


def _worker_oom(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
    comm_timeout_s: float,
) -> int:
    import torch
    import torch.distributed as dist
    from triton.runtime import (
        CoordinatedStepGuard,
        GLOBAL_FALLBACK_REGISTRY,
        init_default_dispatch_guard,
        reset_default_dispatch_guard,
    )

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    GLOBAL_FALLBACK_REGISTRY.clear()

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    guard = init_default_dispatch_guard(
        create_gloo_group=True,
        timeout=datetime.timedelta(seconds=comm_timeout_s),
        comm_timeout_s=comm_timeout_s,
        raise_on_fault=True,
    )
    step_guard = CoordinatedStepGuard(
        dispatch_guard=guard,
        comm_timeout_s=comm_timeout_s,
        raise_on_fault=True,
    )

    model = _build_model(128, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    # Step 1 warm S=63
    opt.zero_grad(set_to_none=True)
    x1, y1 = _make_batch(rank, 1, 4, 63, 128, device)
    _forward_backward(model, x1, y1)
    _allreduce_grads_gloo(model, world_size)
    opt.step()
    torch.cuda.synchronize(rank)
    dist.barrier()
    if rank == 0:
        (Path(out_dir) / "step1_done").write_text("ok", encoding="utf-8")

    # Step 2 long-tail S=2047 with Rank 1 OOM on full batch -> coordinated microbatch halving
    t0 = time.monotonic()
    x2, y2 = _make_batch(rank, 2, 4, 2047, 128, device)
    injected = {"done": False}

    def full_step() -> float:
        opt.zero_grad(set_to_none=True)
        if rank == 1 and not injected["done"]:
            injected["done"] = True
            raise torch.OutOfMemoryError("Injected long-tail PPU OOM at S=2047")
        loss_v = _forward_backward(model, x2, y2)
        torch.cuda.synchronize(rank)
        return loss_v

    def split_step(num_splits: int) -> float:
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for cx, cy in zip(x2.chunk(num_splits, dim=0), y2.chunk(num_splits, dim=0)):
            total += _forward_backward(model, cx, cy, scale=1.0 / float(num_splits))
        torch.cuda.synchronize(rank)
        return total

    def commit_grads(loss_v: float) -> float:
        _allreduce_grads_gloo(model, world_size)
        opt.step()
        torch.cuda.synchronize(rank)
        return loss_v

    loss_val, final_health = step_guard.run_coordinated_step(
        "drill_oom_s2047",
        "training",
        work_units=4,
        step_fn=full_step,
        split_fn=split_step,
        on_oom_reset_fn=lambda: opt.zero_grad(set_to_none=True),
        post_ok_fn=commit_grads,
        release_cache_fn=_flush_device,
        shape=(4, 2047, 128),
        bucket_key=("ft_bkt_v1", 2047),
    )
    assert final_health.all_ok, f"expected recovered health all_ok=True, got {final_health}"

    flat = torch.cat([p.detach().float().cpu().reshape(-1) for p in model.parameters()])
    all_finite = bool(torch.isfinite(flat).all().item())
    torch.save(flat, Path(out_dir) / f"params_rank_{rank}.pt")
    _write_json(
        Path(out_dir) / f"rank_{rank}.json",
        {
            "rank": rank,
            "status": "coordinated_oom_recovered",
            "loss": round(float(loss_val), 6),
            "all_finite": all_finite,
            "oom_recoveries": step_guard.oom_recoveries,
            "elapsed_s": round(time.monotonic() - t0, 4),
            "fallback_summary": GLOBAL_FALLBACK_REGISTRY.summary(),
        },
    )
    dist.barrier()
    reset_default_dispatch_guard()
    dist.destroy_process_group()
    return 0


def _worker_soak(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
    comm_timeout_s: float,
    soak_rounds: int,
) -> int:
    import torch
    import torch.distributed as dist
    from triton.runtime import (
        CoordinatedStepGuard,
        GLOBAL_FALLBACK_REGISTRY,
        RankDispatchState,
        init_default_dispatch_guard,
        reset_default_dispatch_guard,
    )

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    GLOBAL_FALLBACK_REGISTRY.clear()

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    guard = init_default_dispatch_guard(
        create_gloo_group=True,
        timeout=datetime.timedelta(seconds=comm_timeout_s),
        comm_timeout_s=comm_timeout_s,
        raise_on_fault=True,
    )
    step_guard = CoordinatedStepGuard(
        dispatch_guard=guard,
        comm_timeout_s=comm_timeout_s,
        raise_on_fault=True,
    )

    model = _build_model(128, device=device)
    opt = torch.optim.SGD(model.parameters(), lr=5e-3)

    dist.barrier()
    if rank == 0:
        (Path(out_dir) / "step1_done").write_text("ok", encoding="utf-8")

    t0 = time.monotonic()
    step_idx = 0
    steps_committed = 0
    steps_skipped = 0
    oom_events = 0
    skew_syncs = 0
    shape_trace: List[Dict[str, Any]] = []

    for round_id in range(soak_rounds):
        for seq_len in LONG_TAIL_SHAPES:
            step_idx += 1
            x, y = _make_batch(rank, step_idx, 4, seq_len, 128, device)

            # Inject asymmetric cold-compile dispatch skew on S=257
            local_disp = (
                RankDispatchState.ANY_COLD_COMPILE
                if (seq_len == 257 and rank == (round_id % world_size))
                else RankDispatchState.ALL_HOT_HIT
            )
            with guard.coordinate(local_disp) as dec:
                if dec is not None and dec.extend_collective_window:
                    skew_syncs += 1
                    if local_disp is RankDispatchState.ANY_COLD_COMPILE:
                        time.sleep(0.05)

            inject_oom = (
                (seq_len == 509 and rank == (2 % world_size))
                or (seq_len == 2047 and rank == (3 % world_size))
            )
            inject_transient_err = round_id == 0 and seq_len == 1023 and rank == 0
            oom_fired = {"flag": False}

            def step_fn() -> float:
                opt.zero_grad(set_to_none=True)
                if inject_oom and not oom_fired["flag"]:
                    oom_fired["flag"] = True
                    raise torch.OutOfMemoryError(
                        f"Soak transient OOM round={round_id} S={seq_len} rank={rank}"
                    )
                if inject_transient_err:
                    raise RuntimeError(
                        f"Soak transient non-OOM fault round={round_id} S={seq_len}"
                    )
                lv = _forward_backward(model, x, y)
                torch.cuda.synchronize(rank)
                return lv

            def split_fn(num_splits: int) -> float:
                opt.zero_grad(set_to_none=True)
                total = 0.0
                for cx, cy in zip(x.chunk(num_splits, dim=0), y.chunk(num_splits, dim=0)):
                    total += _forward_backward(model, cx, cy, scale=1.0 / float(num_splits))
                torch.cuda.synchronize(rank)
                return total

            def post_ok_fn(lv: float) -> float:
                _allreduce_grads_gloo(model, world_size)
                opt.step()
                torch.cuda.synchronize(rank)
                return lv

            prev_oom_rec = step_guard.oom_recoveries
            loss_val, health = step_guard.run_coordinated_step(
                f"soak_s{seq_len}",
                "training",
                work_units=4,
                step_fn=step_fn,
                split_fn=split_fn,
                on_oom_reset_fn=lambda: opt.zero_grad(set_to_none=True),
                post_ok_fn=post_ok_fn,
                release_cache_fn=_flush_device,
                shape=(4, seq_len, 128),
                bucket_key=("ft_bkt_v1", seq_len),
            )
            recovered_oom = step_guard.oom_recoveries > prev_oom_rec
            if recovered_oom:
                oom_events += 1
            if health.all_ok:
                steps_committed += 1
            else:
                steps_skipped += 1
                opt.zero_grad(set_to_none=True)

            shape_trace.append(
                {
                    "step": step_idx,
                    "round": round_id,
                    "seq_len": seq_len,
                    "committed": bool(health.all_ok),
                    "recovered_oom": recovered_oom,
                    "loss": None if loss_val is None else round(float(loss_val), 6),
                }
            )

    flat = torch.cat([p.detach().float().cpu().reshape(-1) for p in model.parameters()])
    all_finite = bool(torch.isfinite(flat).all().item())
    torch.save(flat, Path(out_dir) / f"params_rank_{rank}.pt")
    elapsed_s = time.monotonic() - t0
    _write_json(
        Path(out_dir) / f"rank_{rank}.json",
        {
            "rank": rank,
            "status": "soak_completed",
            "total_steps": step_idx,
            "steps_committed": steps_committed,
            "steps_skipped": steps_skipped,
            "oom_recoveries": oom_events,
            "skew_syncs": skew_syncs,
            "all_finite": all_finite,
            "elapsed_s": round(elapsed_s, 4),
            "shapes_covered": list(LONG_TAIL_SHAPES),
            "fallback_summary": GLOBAL_FALLBACK_REGISTRY.summary(),
            "trace_tail": shape_trace[-4:],
        },
    )
    dist.barrier()
    reset_default_dispatch_guard()
    dist.destroy_process_group()
    return 0


def _reap_all(procs: Sequence[subprocess.Popen]) -> int:
    lingering = 0
    for p in procs:
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass
            try:
                p.wait(timeout=3.0)
            except Exception:
                lingering += 1
        else:
            try:
                p.wait(timeout=1.0)
            except Exception:
                pass
    return lingering


def _run_drill_scenario(
    name: str,
    world_size: int,
    comm_timeout_s: float,
    stall_s: float,
    soak_rounds: int,
    watchdog_s: float,
    expect_killed_rank: Optional[int] = None,
    stalled_rank: Optional[int] = None,
) -> Dict[str, Any]:
    print(f"\n==================== {name} ====================", flush=True)
    with tempfile.TemporaryDirectory(prefix=f"f09_g05_drill_{name}_") as td:
        rdv = os.path.join(td, "rdv")
        t0 = time.monotonic()
        procs = [
            subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    name,
                    str(r),
                    str(world_size),
                    rdv,
                    td,
                    str(comm_timeout_s),
                    str(stall_s),
                    str(soak_rounds),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for r in range(world_size)
        ]

        marker = Path(td) / "step1_done"
        startup_deadline = t0 + 90.0
        while time.monotonic() < startup_deadline:
            if marker.exists():
                break
            if any(p.poll() is not None for p in procs):
                break
            time.sleep(0.05)

        step1_ok = marker.exists()
        t_phase2 = time.monotonic()
        watch_ranks = [
            r for r in range(world_size) if r != expect_killed_rank and r != stalled_rank
        ]
        deadline = t_phase2 + watchdog_s
        while time.monotonic() < deadline:
            if all(procs[r].poll() is not None for r in watch_ranks):
                break
            time.sleep(0.05)

        wall_s = round(time.monotonic() - t_phase2, 4)
        hung = [r for r in watch_ranks if procs[r].poll() is None]

        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass

        rank_data: Dict[str, Any] = {}
        rcs: Dict[int, int] = {}
        for r, p in enumerate(procs):
            out, err = p.communicate(timeout=5.0)
            rcs[r] = p.returncode
            jf = Path(td) / f"rank_{r}.json"
            rdata = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else None
            rank_data[str(r)] = rdata
            rd = rdata or {}
            timing = rd.get("detect_s") or rd.get("trip_s") or rd.get("elapsed_s")
            print(
                f"  [rank {r}] rc={p.returncode} status={rd.get('status')} timing={timing}",
                flush=True,
            )
            if p.returncode != 0 and r != expect_killed_rank and r != stalled_rank:
                print(f"  [rank {r} stderr]: {err.strip()[-800:]}", flush=True)

        lingering = _reap_all(procs)

        extra: Dict[str, Any] = {}
        passed = False
        if name == "sigkill_rank_fault":
            killed_ok = rcs[world_size - 1] == -signal.SIGKILL
            survivors = [r for r in range(world_size - 1)]
            surv_ok = (
                not hung
                and wall_s <= 6.0
                and all(
                    rcs[r] == 0
                    and (rank_data[str(r)] or {}).get("status") == "fault_detected_clean_exit"
                    and (rank_data[str(r)] or {})
                    .get("fallback_summary", {})
                    .get("total_fallbacks", 0)
                    >= 1
                    for r in survivors
                )
            )
            extra["killed_rank_sigkill"] = killed_ok
            extra["max_detect_s"] = max(
                float((rank_data[str(r)] or {}).get("detect_s", 99.0)) for r in survivors
            )
            passed = step1_ok and killed_ok and surv_ok and lingering == 0
        elif name == "comm_timeout_stall":
            survivors = [r for r in range(world_size) if r != stalled_rank]
            surv_ok = (
                not hung
                and all(
                    rcs[r] == 0
                    and (rank_data[str(r)] or {}).get("status")
                    == "timeout_circuit_breaker_tripped"
                    and float((rank_data[str(r)] or {}).get("trip_s", 99.0))
                    <= comm_timeout_s + 1.5
                    and (rank_data[str(r)] or {})
                    .get("fallback_summary", {})
                    .get("total_fallbacks", 0)
                    >= 1
                    for r in survivors
                )
            )
            extra["max_trip_s"] = max(
                float((rank_data[str(r)] or {}).get("trip_s", 99.0)) for r in survivors
            )
            passed = step1_ok and surv_ok and lingering == 0
        elif name in ("asymmetric_longtail_oom_recovery", "continuous_longtail_varlen_soak"):
            import torch

            expected_status = (
                "coordinated_oom_recovered"
                if name == "asymmetric_longtail_oom_recovery"
                else "soak_completed"
            )
            all_ok = (
                not hung
                and all(
                    rcs[r] == 0
                    and (rank_data[str(r)] or {}).get("status") == expected_status
                    and (rank_data[str(r)] or {}).get("all_finite") is True
                    and (rank_data[str(r)] or {})
                    .get("fallback_summary", {})
                    .get("total_fallbacks", 0)
                    >= 1
                    for r in range(world_size)
                )
            )
            max_diff = None
            if all_ok:
                params = [
                    torch.load(Path(td) / f"params_rank_{r}.pt", weights_only=True)
                    for r in range(world_size)
                ]
                max_diff = max(
                    float((params[r] - params[0]).abs().max().item())
                    for r in range(1, world_size)
                )
                extra["max_rank_param_diff"] = max_diff
                extra["all_finite"] = True
            passed = step1_ok and all_ok and max_diff == 0.0 and lingering == 0

        res = {
            "scenario": name,
            "step1_completed": step1_ok,
            "passed": passed,
            "wall_s": wall_s,
            "hung_ranks": hung,
            "lingering_pids": lingering,
            **extra,
            "ranks": rank_data,
        }
        print(
            f"  => {name}: {'PASS' if passed else 'FAIL'} "
            f"(wall={wall_s:.2f}s, hung={hung}, lingering={lingering}, extra={extra})",
            flush=True,
        )
        return res


def main() -> int:
    parser = argparse.ArgumentParser(
        description="F09-G05 Multi-Card Fault-Injection & Continuous Varlen Soak Drill"
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--comm-timeout-s", type=float, default=3.0)
    parser.add_argument("--stall-s", type=float, default=12.0)
    parser.add_argument("--soak-rounds", type=int, default=2)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument(
        "--worker",
        nargs=8,
        metavar=(
            "SCENARIO",
            "RANK",
            "WORLD_SIZE",
            "INIT_FILE",
            "OUT_DIR",
            "TIMEOUT_S",
            "STALL_S",
            "SOAK_ROUNDS",
        ),
    )
    args = parser.parse_args()

    if args.worker is not None:
        sc, rk_s, ws_s, rdv, out_d, to_s, st_s, sr_s = args.worker
        rk, ws, to_f, st_f, sr_i = int(rk_s), int(ws_s), float(to_s), float(st_s), int(sr_s)
        if sc == "sigkill_rank_fault":
            return _worker_sigkill(rk, ws, rdv, out_d, to_f)
        if sc == "comm_timeout_stall":
            return _worker_stall(rk, ws, rdv, out_d, to_f, st_f)
        if sc == "asymmetric_longtail_oom_recovery":
            return _worker_oom(rk, ws, rdv, out_d, to_f)
        if sc == "continuous_longtail_varlen_soak":
            return _worker_soak(rk, ws, rdv, out_d, to_f, sr_i)
        raise ValueError(f"Unknown scenario: {sc}")

    import torch
    import triton

    assert torch.cuda.is_available(), "CUDA/PPU accelerator required"
    dev_cnt = torch.cuda.device_count()
    assert dev_cnt >= args.world_size, f"Need >= {args.world_size} devices, got {dev_cnt}"

    print(
        f"F09-G05 4-Card Fault-Injection Drill | devices={dev_cnt} "
        f"({torch.cuda.get_device_name(0)}) | torch={torch.__version__} | "
        f"bucketing={os.getenv('FLAGTREE_SHAPE_BUCKETING', '0')} | "
        f"triton={triton.__file__}",
        flush=True,
    )

    # Pre-warm FlagGems Triton kernels once on device 0 across long-tail buckets
    warm_model = _build_model(128, device="cuda:0")
    for seq_len in (1, 63, 257, 509, 2047):
        wx, wy = _make_batch(0, 0, 2, seq_len, 128, "cuda:0")
        _forward_backward(warm_model, wx, wy)
    torch.cuda.synchronize(0)
    del warm_model
    torch.cuda.empty_cache()

    scenarios = [
        _run_drill_scenario(
            "sigkill_rank_fault",
            args.world_size,
            args.comm_timeout_s,
            args.stall_s,
            args.soak_rounds,
            watchdog_s=7.0,
            expect_killed_rank=args.world_size - 1,
        ),
        _run_drill_scenario(
            "comm_timeout_stall",
            args.world_size,
            args.comm_timeout_s,
            args.stall_s,
            args.soak_rounds,
            watchdog_s=9.0,
            stalled_rank=2 % args.world_size,
        ),
        _run_drill_scenario(
            "asymmetric_longtail_oom_recovery",
            args.world_size,
            args.comm_timeout_s,
            args.stall_s,
            args.soak_rounds,
            watchdog_s=15.0,
        ),
        _run_drill_scenario(
            "continuous_longtail_varlen_soak",
            args.world_size,
            args.comm_timeout_s,
            args.stall_s,
            args.soak_rounds,
            watchdog_s=90.0,
        ),
    ]

    all_passed = all(s["passed"] for s in scenarios)
    report = {
        "benchmark": "f09_g05_multicard_fault_injection",
        "device_name": torch.cuda.get_device_name(0),
        "world_size": args.world_size,
        "torch_version": torch.__version__,
        "flagtree_shape_bucketing": os.getenv("FLAGTREE_SHAPE_BUCKETING", "0"),
        "comm_timeout_s": args.comm_timeout_s,
        "stall_s": args.stall_s,
        "soak_rounds": args.soak_rounds,
        "long_tail_shapes": list(LONG_TAIL_SHAPES),
        "all_passed": all_passed,
        "scenarios": scenarios,
    }

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nSaved JSON report to: {out_p}", flush=True)

    print("\n=== F09-G05 FAULT DRILL SUMMARY ===", flush=True)
    for s in scenarios:
        print(
            f"  {s['scenario']}: {'PASS' if s['passed'] else 'FAIL'} "
            f"(wall={s['wall_s']:.2f}s, hung={s['hung_ranks']}, lingering={s['lingering_pids']})",
            flush=True,
        )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
