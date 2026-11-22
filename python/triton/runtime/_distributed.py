# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

import functools


class DistributedRtContext:
    _init_count = 0

    def __init__(self, _comm_ptr=None, _mem_ptr=None):
        if _comm_ptr is not None and _mem_ptr is not None:
            type(self)._init_count += 1
        self._comm_ptr = _comm_ptr
        self._mem_ptr = _mem_ptr
        self._registered_buffer = None

    def register_buffer(self, buffer) -> None:
        self._registered_buffer = buffer

    @property
    def registered_buffer(self):
        return getattr(self, "_registered_buffer", None)

    def get_packed_data(self):
        return int(self._mem_ptr), int(self._comm_ptr)

    @property
    def comm_ptr(self) -> int | None:
        """Communication runtime pointer."""
        return self._comm_ptr

    def _get_needed_params(self):
        return {"device_comm_ptr": self._comm_ptr, "device_mem_ptr": self._mem_ptr}

    @property
    def mem_ptr(self) -> int | None:
        """Distributed memory runtime pointer."""
        return self._mem_ptr

    @property
    def is_lite_mode(self) -> bool:
        import os
        user_action = os.getenv("FLAGTREE_LITE_DIST", "").strip().upper() in {
            "1",
            "ON",
            "TRUE",
        }
        inner_action = self._init_count == 1
        return inner_action and user_action

    def __getitem__(self, index=0):
        return list(self._get_needed_params().values())[index]

    def add_args_to_jitfunction(self, **kwargs):
        params = kwargs['params']  # list
        _kwargs = kwargs['kwargs']
        if (isinstance(params, list) and len(params) > 0):
            needed_params = self._get_needed_params()
            needed_params_size = len(needed_params)
            template_ele = params[0]
            KernelParam = type(template_ele)  # KernelParam type
            Parameter = type(template_ele._param)  # inspect.Parameter type

            dist_params = []
            for i, (name, val) in enumerate(needed_params.items()):
                _kwargs[name] = val
                param = Parameter(name, kind=template_ele._param.kind)
                dist_params.append(KernelParam(i, param, False, False))

            new_params = []
            for param in params:
                new_loc = param.num + needed_params_size
                #num: int, param: inspect.Parameter, do_not_specialize: bool, do_not_specialize_on_alignment: bool
                new_params.append(
                    KernelParam(new_loc, param._param, param.do_not_specialize, param.do_not_specialize_on_alignment))
            new_params = dist_params + new_params
            params[:] = new_params


@functools.lru_cache(maxsize=256)
def _parse_node_buffer_bindings(encoded_bindings):
    if not isinstance(encoded_bindings, str):
        raise RuntimeError("the compiled kernel information is invalid or out of date. "
                           "Please clear the Triton/FlagTree kernel cache and recompile. "
                           "If the problem persists, report it as a compiler/runtime issue.")

    roles = {"s": "source", "d": "destination"}
    bindings = []
    for encoded in encoded_bindings.split(","):
        try:
            role_code, ordinal_text, handle_text = encoded.split(":", 2)
            ordinal = int(ordinal_text)
            handle = int(handle_text)
        except (TypeError, ValueError):
            raise RuntimeError("the compiled kernel information is invalid or out of date. "
                               "Please clear the Triton/FlagTree kernel cache and recompile. "
                               "If the problem persists, report it as a compiler/runtime issue.") from None
        role = roles.get(role_code)
        if role is None:
            raise RuntimeError("the compiled kernel information is invalid or out of date. "
                               "Please clear the Triton/FlagTree kernel cache and recompile. "
                               "If the problem persists, report it as a compiler/runtime issue.")
        if ordinal < 0:
            raise RuntimeError("the compiled kernel information is invalid or out of date. "
                               "Please clear the Triton/FlagTree kernel cache and recompile. "
                               "If the problem persists, report it as a compiler/runtime issue.")
        bindings.append((encoded, role, ordinal, handle))
    return tuple(bindings)


def _collect_distributed_contexts(values):
    return tuple({id(value): value for value in values if isinstance(value, DistributedRtContext)}.values())


def _collect_runtime_items(bound_args, specialization):
    return tuple(
        (name, value) for (name, value), spec in zip(bound_args.items(), specialization) if spec[0] != "constexpr")


def _node_buffer_metadata_error(kernel_name):
    return RuntimeError(f"Could not launch kernel {kernel_name!r}: its compiled information is "
                        "invalid or out of date. Please clear the Triton/FlagTree kernel cache "
                        "and recompile. If the problem persists, report it as a compiler/runtime "
                        "issue.")


def validate_node_buffer_bindings(
    contexts,
    runtime_args,
    encoded_bindings,
    kernel_name: str,
    arg_names=None,
) -> None:
    if not encoded_bindings:
        return

    contexts_by_handle = {int(ctx[0]): ctx for ctx in contexts}
    try:
        parsed_bindings = _parse_node_buffer_bindings(encoded_bindings)
    except RuntimeError:
        raise _node_buffer_metadata_error(kernel_name) from None
    for encoded, role, ordinal, handle in parsed_bindings:
        if ordinal >= len(runtime_args):
            raise _node_buffer_metadata_error(kernel_name)

        ctx = contexts_by_handle.get(handle)
        if ctx is None:
            raise RuntimeError(f"kernel {kernel_name!r} did not receive the node context "
                               f"with mem handle 0x{handle:x}. Pass the same ctx returned by "
                               "ctx = tle.create_dist_tensor(buffer) to this launch")
        registered = ctx.registered_buffer
        if registered is None:
            raise RuntimeError(f"kernel {kernel_name!r} node context has no registered buffer. "
                               "Create it with ctx = tle.create_dist_tensor(buffer) instead of "
                               "constructing DistributedRtContext directly")

        value = runtime_args[ordinal]
        data_ptr = getattr(value, "data_ptr", None)
        if not isinstance(value, int) and not callable(data_ptr):
            raise TypeError(f"kernel {kernel_name!r} node local {role} buffer must be a tensor "
                            "with data_ptr() or an integer device pointer. Pass the same buffer "
                            "used in ctx = tle.create_dist_tensor(buffer)")
        actual = value if isinstance(value, int) else data_ptr()
        expected = registered.data_ptr()
        if actual == expected:
            continue

        name = (arg_names[ordinal]
                if arg_names is not None and ordinal < len(arg_names) else f"runtime ordinal {ordinal}")
        raise ValueError(f"kernel {kernel_name!r} node local {role} buffer argument "
                         f"{name!r} does not match its context buffer: expected "
                         f"0x{expected:x}, got 0x{actual:x}. Use ctx = "
                         "tle.create_dist_tensor(buffer) and launch with that same buffer/ctx "
                         "pair; a ctx created for a different buffer cannot be used")


def validate_node_launch_bindings(kernel, bound_args, specialization):
    """Validate node buffer bindings for one kernel launch."""
    encoded_bindings = getattr(kernel.metadata, "tle_node_buffer_bindings", "")
    if not encoded_bindings:
        return

    contexts = _collect_distributed_contexts(bound_args.values())
    runtime_items = _collect_runtime_items(bound_args, specialization)
    validate_node_buffer_bindings(
        contexts,
        tuple(value for _, value in runtime_items),
        encoded_bindings,
        kernel.name,
        tuple(name for name, _ in runtime_items),
    )


# =====================================================================
# Multi-Rank Dispatch Coordination, OOM Recovery & Fallback Risk Guard
# =====================================================================

import contextlib
from dataclasses import asdict, dataclass
import datetime
from enum import Enum, IntFlag
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .errors import (
    DistributedCommTimeoutError,
    DistributedRankFaultError,
    OutOfResources,
)

_dist_logger = logging.getLogger(__name__)


class RankDispatchState(IntFlag):
    """Bitmask exchanged across ranks prior to collective-sensitive regions."""

    ALL_HOT_HIT = 0
    ANY_COLD_COMPILE = 1
    ANY_OOM_RETRY = 2
    ANY_TIMEOUT = 4
    ANY_FALLBACK = 8
    ANY_RANK_FAULT = 16


class PerfRiskLevel(str, Enum):
    """Structured performance-degradation severity for controlled fallbacks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class FallbackRecord:
    """Structured audit record emitted whenever a kernel or batch degrades."""

    op_name: str
    phase: str
    shape: Tuple[Any, ...]
    bucket_key: Tuple[Any, ...]
    reason: str
    recovery_action: str
    perf_risk_level: PerfRiskLevel
    rank: int = 0
    timestamp: float = 0.0


class FallbackRegistry:
    """Thread-safe collector and JSON exporter for controlled fallback events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[FallbackRecord] = []

    def record(
        self,
        op_name: str,
        phase: str,
        shape: Sequence[Any],
        bucket_key: Sequence[Any],
        *,
        reason: str,
        recovery_action: str,
        perf_risk_level: PerfRiskLevel = PerfRiskLevel.MEDIUM,
        rank: int = 0,
    ) -> FallbackRecord:
        entry = FallbackRecord(
            op_name=str(op_name),
            phase=str(phase),
            shape=tuple(shape),
            bucket_key=tuple(str(x) for x in bucket_key),
            reason=str(reason),
            recovery_action=str(recovery_action),
            perf_risk_level=PerfRiskLevel(perf_risk_level),
            rank=int(rank),
            timestamp=time.time(),
        )
        with self._lock:
            self._records.append(entry)
        _dist_logger.warning(
            "[FlagTree Fallback][risk=%s][rank=%d] op=%s phase=%s shape=%s reason=%s -> %s",
            entry.perf_risk_level.value,
            entry.rank,
            entry.op_name,
            entry.phase,
            entry.shape,
            entry.reason,
            entry.recovery_action,
        )
        return entry

    def records(self) -> List[FallbackRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            by_risk = {lvl.value: 0 for lvl in PerfRiskLevel}
            for rec in self._records:
                by_risk[rec.perf_risk_level.value] += 1
            return {
                "total_fallbacks": len(self._records),
                "by_risk_level": by_risk,
                "records": [
                    {
                        **asdict(r),
                        "perf_risk_level": r.perf_risk_level.value,
                    }
                    for r in self._records
                ],
            }

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.summary(), indent=2, sort_keys=True), encoding="utf-8")


GLOBAL_FALLBACK_REGISTRY = FallbackRegistry()


def _parse_env_comm_timeout() -> Optional[float]:
    raw = os.getenv("FLAGTREE_COMM_TIMEOUT_SEC", "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
        return val if val > 0.0 else None
    except ValueError:
        return None


def _is_timeout_exception(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, DistributedCommTimeoutError)):
        return True
    cls_name = type(exc).__name__.lower()
    if "timeout" in cls_name:
        return True
    msg = str(exc).lower()
    return "timed out" in msg or "timeout" in msg or "time out" in msg


def _is_rank_fault_exception(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            EOFError,
            DistributedRankFaultError,
        ),
    ):
        return True
    cls_name = type(exc).__name__.lower()
    if any(tok in cls_name for tok in ("distbackend", "distnetwork", "diststore", "disterror")):
        return True
    msg = str(exc).lower()
    markers = (
        "connection closed by peer",
        "connection reset by peer",
        "broken pipe",
        "remote rank",
        "communicator was aborted",
        "device fault",
        "read error",
        "write error",
        "socket",
        "transport",
    )
    return any(tok in msg for tok in markers)


def _classify_collective_exception(
    exc: BaseException,
    *,
    rank: int,
    world_size: int,
    op: str,
    phase: str,
    timeout_s: Optional[float],
) -> Tuple[str, Exception]:
    if isinstance(exc, DistributedCommTimeoutError):
        return "comm_timeout", exc
    if isinstance(exc, DistributedRankFaultError):
        return "rank_fault", exc
    if _is_timeout_exception(exc):
        eff_t = float(timeout_s) if timeout_s is not None else 0.0
        return (
            "comm_timeout",
            DistributedCommTimeoutError(
                rank=rank,
                world_size=world_size,
                op=op,
                phase=phase,
                timeout_s=eff_t,
                cause=f"{type(exc).__name__}: {exc}",
            ),
        )
    return (
        "rank_fault",
        DistributedRankFaultError(
            rank=rank,
            world_size=world_size,
            op=op,
            phase=phase,
            timeout_s=timeout_s,
            cause=f"{type(exc).__name__}: {exc}",
        ),
    )


def _release_device_memory_best_effort() -> None:
    """Flush device allocator caches portably across accelerator backends."""
    try:
        import torch
    except Exception:
        return
    for attr in ("cuda", "xpu", "npu", "mlu", "musa"):
        backend = getattr(torch, attr, None)
        if backend is None:
            continue
        try:
            is_avail = getattr(backend, "is_available", None)
            if callable(is_avail) and not is_avail():
                continue
            empty_cache = getattr(backend, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
        except Exception:
            continue


def _run_callable_with_timeout(
    fn: Callable[[], Any],
    timeout_s: Optional[float],
    *,
    rank: int,
    world_size: int,
    op: str,
    phase: str,
) -> Any:
    if timeout_s is None or timeout_s <= 0.0:
        return fn()

    result_box: List[Any] = []
    error_box: List[BaseException] = []
    done = threading.Event()

    def _runner() -> None:
        try:
            result_box.append(fn())
        except BaseException as exc:  # noqa: BLE001
            error_box.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    if not done.wait(timeout=float(timeout_s)):
        raise DistributedCommTimeoutError(
            rank=rank,
            world_size=world_size,
            op=op,
            phase=phase,
            timeout_s=float(timeout_s),
            cause=f"callable '{op}' exceeded {float(timeout_s):.2f}s deadline",
        )
    if error_box:
        raise error_box[0]
    return result_box[0] if result_box else None


def _wait_work_bounded(
    work: Any,
    timeout_s: Optional[float],
    *,
    rank: int,
    world_size: int,
    op: str,
    phase: str,
) -> None:
    if work is None:
        return
    if timeout_s is None or timeout_s <= 0.0:
        work.wait()
        return

    td = datetime.timedelta(seconds=float(timeout_s))
    try:
        ok = work.wait(timeout=td)
    except TypeError:
        is_completed = getattr(work, "is_completed", None)
        if callable(is_completed):
            deadline = time.monotonic() + float(timeout_s)
            while not is_completed():
                if time.monotonic() >= deadline:
                    raise DistributedCommTimeoutError(
                        rank=rank,
                        world_size=world_size,
                        op=op,
                        phase=phase,
                        timeout_s=float(timeout_s),
                        cause="work polling deadline exceeded",
                    )
                time.sleep(0.01)
            work.wait()
            return
        _run_callable_with_timeout(
            work.wait,
            timeout_s,
            rank=rank,
            world_size=world_size,
            op=op,
            phase=phase,
        )
        return

    if ok is False:
        raise DistributedCommTimeoutError(
            rank=rank,
            world_size=world_size,
            op=op,
            phase=phase,
            timeout_s=float(timeout_s),
            cause="work.wait returned False",
        )


def _compute_work_wait_timeout(timeout_s: float, pg_timeout_applied: bool) -> float:
    """Give the PG socket timeout a small grace margin so C++ worker threads finish cleanly.

    When ProcessGroup.set_timeout(timeout_s) succeeds, giving work.wait() a
    slightly larger ceiling guarantees the backend C++ thread (e.g. Gloo TCP
    transport) marks the Work completed-with-timeout BEFORE work.wait() returns,
    eliminating C++ ~ProcessGroupGloo std::terminate races on teardown.
    """
    t = float(timeout_s)
    if not pg_timeout_applied:
        return t
    grace = min(0.5, max(0.2, t * 0.15))
    return t + grace


def _apply_pg_timeout_best_effort(
    dist_mod: Any,
    process_group: Any,
    timeout_s: Optional[float],
) -> Tuple[Any, bool]:
    """Apply ``timeout_s`` on the target ProcessGroup so C++ threads unwind cleanly."""
    if timeout_s is None or timeout_s <= 0.0:
        return None, False
    target_pg = process_group
    using_world = False
    if target_pg is None:
        target_pg = getattr(getattr(dist_mod, "group", None), "WORLD", None)
        using_world = target_pg is not None
    set_timeout = getattr(target_pg, "set_timeout", None)
    if callable(set_timeout):
        try:
            set_timeout(datetime.timedelta(seconds=float(timeout_s)))
            return target_pg, using_world
        except Exception:
            return None, False
    return None, False


def _restore_world_pg_timeout_best_effort(
    dist_mod: Any,
    target_pg: Any,
    using_world: bool,
) -> None:
    if not using_world or target_pg is None:
        return
    default_to = getattr(dist_mod, "default_pg_timeout", None)
    set_timeout = getattr(target_pg, "set_timeout", None)
    if default_to is not None and callable(set_timeout):
        try:
            set_timeout(default_to)
        except Exception:
            pass


def _resolve_pg_device(dist_mod: Any, process_group: Any) -> str:
    pg_backend = ""
    try:
        pg_backend = str(dist_mod.get_backend(process_group)).lower()
    except Exception:
        pg_backend = ""
    from .backend_caps import resolve_collective_device

    return resolve_collective_device(pg_backend)


@dataclass(frozen=True)
class CoordinatedDispatchDecision:
    """Agreed cross-rank execution plan preventing collective deadlocks."""

    global_mask: RankDispatchState
    all_hot_hit: bool
    extend_collective_window: bool
    agreed_fallback: bool
    barrier_completed: bool
    degraded_solo: bool = False
    fault_kind: Optional[str] = None
    fault_reason: Optional[str] = None


_DISPATCH_GUARD_SCOPE_LOCAL = threading.local()


def is_in_dispatch_guard_scope() -> bool:
    """Return True when the current thread is already inside a dispatch guard scope."""
    return getattr(_DISPATCH_GUARD_SCOPE_LOCAL, "depth", 0) > 0


@contextlib.contextmanager
def dispatch_guard_scope() -> Iterator[None]:
    """Re-entrant thread-local scope preventing duplicate nested guard collectives."""
    depth = getattr(_DISPATCH_GUARD_SCOPE_LOCAL, "depth", 0)
    _DISPATCH_GUARD_SCOPE_LOCAL.depth = depth + 1
    try:
        yield
    finally:
        _DISPATCH_GUARD_SCOPE_LOCAL.depth = depth


class DistributedDispatchGuard:
    """Synchronize bucket/compile/fallback states across DDP/FSDP/TP/EP ranks.

    Prevents collective watchdog deadlocks when a single rank encounters a cold
    bucket compilation, MoE expert spike, or OOM fallback while peer ranks hit
    warm cache, and enforces bounded per-call timeouts with structured fault
    auditing so peer crashes or stalls never hang indefinitely.
    """

    def __init__(
        self,
        *,
        world_size: int = 1,
        rank: int = 0,
        process_group: Any = None,
        all_reduce_or_fn: Optional[Callable[[int], int]] = None,
        barrier_fn: Optional[Callable[[], None]] = None,
        comm_timeout_s: Optional[float] = None,
        raise_on_fault: bool = True,
        fallback_registry: FallbackRegistry = GLOBAL_FALLBACK_REGISTRY,
        release_memory_on_fault: bool = True,
    ) -> None:
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.process_group = process_group
        self._all_reduce_or_fn = all_reduce_or_fn
        self._barrier_fn = barrier_fn
        resolved_timeout = _parse_env_comm_timeout() if comm_timeout_s is None else float(comm_timeout_s)
        self.comm_timeout_s: Optional[float] = (
            resolved_timeout if (resolved_timeout is not None and resolved_timeout > 0.0) else None
        )
        self.raise_on_fault = bool(raise_on_fault)
        self.fallback_registry = fallback_registry
        self.release_memory_on_fault = bool(release_memory_on_fault)
        self.negotiation_count = 0
        self.slow_path_syncs = 0
        self.fault_count = 0
        self._degraded_solo = False
        self._last_fault_kind: Optional[str] = None
        self._last_fault_reason: Optional[str] = None

    @property
    def degraded_solo(self) -> bool:
        """Return True after the fault circuit breaker has tripped."""
        return self._degraded_solo

    def reset_fault_state(self) -> None:
        """Re-arm the circuit breaker after process group recovery."""
        self._degraded_solo = False
        self._last_fault_kind = None
        self._last_fault_reason = None

    def mark_faulted(self, fault_kind: str, reason: str) -> None:
        """Latch fault state so subsequent barrier/sync calls cannot hang."""
        self._degraded_solo = True
        self._last_fault_kind = str(fault_kind)
        self._last_fault_reason = str(reason)

    def cleanup_after_fault(self, *, destroy_world: bool = False) -> None:
        """Release device memory and tear down faulted ProcessGroups cleanly.

        Explicitly destroying faulted side-channel (and optionally WORLD)
        ProcessGroups before interpreter exit avoids C++ static-destruction
        races on in-flight Gloo worker threads.
        """
        self._degraded_solo = True
        _release_device_memory_best_effort()
        try:
            import torch.distributed as dist

            dist_active = bool(dist.is_available() and dist.is_initialized())
        except Exception:
            dist_active = False
            dist = None  # type: ignore[assignment]

        pg = self.process_group
        self.process_group = None
        if pg is not None and dist_active and dist is not None:
            try:
                dist.destroy_process_group(pg)
            except Exception:
                pass

        if destroy_world and dist_active and dist is not None:
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    def _handle_collective_fault(
        self,
        exc: BaseException,
        *,
        op: str,
        phase: str,
        local_state: RankDispatchState,
    ) -> CoordinatedDispatchDecision:
        fault_kind, wrapped = _classify_collective_exception(
            exc,
            rank=self.rank,
            world_size=self.world_size,
            op=op,
            phase=phase,
            timeout_s=self.comm_timeout_s,
        )
        self.fault_count += 1
        self.mark_faulted(fault_kind, str(wrapped))
        if self.release_memory_on_fault:
            _release_device_memory_best_effort()
        action = (
            f"abort_{fault_kind}"
            if self.raise_on_fault
            else f"degrade_solo_{fault_kind}"
        )
        self.fallback_registry.record(
            "distributed_dispatch_guard",
            phase,
            (self.world_size,),
            (f"rank={self.rank}", f"op={op}"),
            reason=str(wrapped),
            recovery_action=action,
            perf_risk_level=PerfRiskLevel.HIGH,
            rank=self.rank,
        )
        if self.raise_on_fault:
            raise wrapped from exc
        fault_bit = (
            RankDispatchState.ANY_TIMEOUT
            if fault_kind == "comm_timeout"
            else RankDispatchState.ANY_RANK_FAULT
        )
        return CoordinatedDispatchDecision(
            global_mask=local_state | fault_bit | RankDispatchState.ANY_FALLBACK,
            all_hot_hit=False,
            extend_collective_window=False,
            agreed_fallback=True,
            barrier_completed=False,
            degraded_solo=True,
            fault_kind=fault_kind,
            fault_reason=str(wrapped),
        )

    def _reduce_bitmask(self, local_mask: int) -> int:
        if self.world_size <= 1 and self._all_reduce_or_fn is None and self.process_group is None:
            try:
                import torch.distributed as dist

                if not (dist.is_available() and dist.is_initialized()):
                    return int(local_mask)
            except Exception:
                return int(local_mask)

        if self._all_reduce_or_fn is not None:
            return int(
                _run_callable_with_timeout(
                    lambda: self._all_reduce_or_fn(int(local_mask)),
                    self.comm_timeout_s,
                    rank=self.rank,
                    world_size=self.world_size,
                    op="all_reduce_bitmask",
                    phase="dispatch_negotiate",
                )
            )

        try:
            import torch
            import torch.distributed as dist

            dist_active = bool(dist.is_available() and dist.is_initialized())
        except Exception:
            dist_active = False

        if not dist_active:
            return int(local_mask)

        dev = _resolve_pg_device(dist, self.process_group)
        bits_tensor = torch.tensor(
            [(1 if (local_mask & (1 << b)) else 0) for b in range(5)],
            dtype=torch.int32,
            device=dev,
        )
        target_pg, using_world = _apply_pg_timeout_best_effort(
            dist, self.process_group, self.comm_timeout_s
        )
        if self.comm_timeout_s is not None:
            try:
                work = dist.all_reduce(
                    bits_tensor,
                    op=dist.ReduceOp.MAX,
                    group=self.process_group,
                    async_op=True,
                )
            except TypeError:
                _run_callable_with_timeout(
                    lambda: dist.all_reduce(
                        bits_tensor, op=dist.ReduceOp.MAX, group=self.process_group
                    ),
                    self.comm_timeout_s,
                    rank=self.rank,
                    world_size=self.world_size,
                    op="all_reduce_bitmask",
                    phase="dispatch_negotiate",
                )
            else:
                _wait_work_bounded(
                    work,
                    _compute_work_wait_timeout(self.comm_timeout_s, target_pg is not None),
                    rank=self.rank,
                    world_size=self.world_size,
                    op="all_reduce_bitmask",
                    phase="dispatch_negotiate",
                )
            _restore_world_pg_timeout_best_effort(dist, target_pg, using_world)
        else:
            dist.all_reduce(bits_tensor, op=dist.ReduceOp.MAX, group=self.process_group)

        out_mask = 0
        for b, v in enumerate(bits_tensor.tolist()):
            if v > 0:
                out_mask |= 1 << b
        return out_mask

    def _run_barrier(self, *, phase: str = "dispatch_barrier") -> bool:
        if self._degraded_solo:
            return False
        if self._barrier_fn is not None:
            _run_callable_with_timeout(
                self._barrier_fn,
                self.comm_timeout_s,
                rank=self.rank,
                world_size=self.world_size,
                op="barrier",
                phase=phase,
            )
            return True

        try:
            import torch.distributed as dist

            dist_active = bool(dist.is_available() and dist.is_initialized())
        except Exception:
            dist_active = False

        if not dist_active:
            return False

        target_pg, using_world = _apply_pg_timeout_best_effort(
            dist, self.process_group, self.comm_timeout_s
        )
        if self.comm_timeout_s is not None:
            try:
                work = dist.barrier(group=self.process_group, async_op=True)
            except TypeError:
                _run_callable_with_timeout(
                    lambda: dist.barrier(group=self.process_group),
                    self.comm_timeout_s,
                    rank=self.rank,
                    world_size=self.world_size,
                    op="barrier",
                    phase=phase,
                )
            else:
                _wait_work_bounded(
                    work,
                    _compute_work_wait_timeout(self.comm_timeout_s, target_pg is not None),
                    rank=self.rank,
                    world_size=self.world_size,
                    op="barrier",
                    phase=phase,
                )
            _restore_world_pg_timeout_best_effort(dist, target_pg, using_world)
            return True

        dist.barrier(group=self.process_group)
        return True

    def negotiate(self, local_state: RankDispatchState) -> CoordinatedDispatchDecision:
        """Synchronize ``local_state`` across ranks and return an agreed decision."""
        self.negotiation_count += 1
        if self._degraded_solo:
            if self.raise_on_fault:
                kind = self._last_fault_kind or "rank_fault"
                reason = self._last_fault_reason or "dispatch guard circuit breaker tripped"
                if kind == "comm_timeout":
                    raise DistributedCommTimeoutError(
                        rank=self.rank,
                        world_size=self.world_size,
                        op="negotiate",
                        phase="circuit_breaker_open",
                        timeout_s=float(self.comm_timeout_s or 0.0),
                        cause=reason,
                    )
                raise DistributedRankFaultError(
                    rank=self.rank,
                    world_size=self.world_size,
                    op="negotiate",
                    phase="circuit_breaker_open",
                    timeout_s=self.comm_timeout_s,
                    cause=reason,
                )
            return CoordinatedDispatchDecision(
                global_mask=local_state | RankDispatchState.ANY_FALLBACK,
                all_hot_hit=False,
                extend_collective_window=False,
                agreed_fallback=True,
                barrier_completed=False,
                degraded_solo=True,
                fault_kind=self._last_fault_kind,
                fault_reason=self._last_fault_reason,
            )

        try:
            merged = RankDispatchState(self._reduce_bitmask(int(local_state)))
        except Exception as exc:
            return self._handle_collective_fault(
                exc,
                op="all_reduce_bitmask",
                phase="dispatch_negotiate",
                local_state=local_state,
            )

        all_hot = merged is RankDispatchState.ALL_HOT_HIT
        need_window = bool(
            merged
            & (
                RankDispatchState.ANY_COLD_COMPILE
                | RankDispatchState.ANY_OOM_RETRY
                | RankDispatchState.ANY_TIMEOUT
            )
        )
        agreed_fb = bool(
            merged & (RankDispatchState.ANY_FALLBACK | RankDispatchState.ANY_RANK_FAULT)
        )
        did_barrier = False
        if need_window or agreed_fb:
            self.slow_path_syncs += 1
            try:
                did_barrier = self._run_barrier(phase="dispatch_negotiate_barrier")
            except Exception as exc:
                return self._handle_collective_fault(
                    exc,
                    op="barrier",
                    phase="dispatch_negotiate_barrier",
                    local_state=local_state,
                )

        return CoordinatedDispatchDecision(
            global_mask=merged,
            all_hot_hit=all_hot,
            extend_collective_window=need_window,
            agreed_fallback=agreed_fb,
            barrier_completed=did_barrier,
        )

    def post_compile_sync(self, decision: Optional[CoordinatedDispatchDecision]) -> bool:
        """Synchronize all ranks AFTER a cold-compile or fallback window completes."""
        if decision is None or decision.all_hot_hit or decision.degraded_solo or self._degraded_solo:
            return False
        if decision.extend_collective_window or decision.agreed_fallback:
            try:
                return self._run_barrier(phase="post_compile_sync")
            except Exception as exc:
                self._handle_collective_fault(
                    exc,
                    op="barrier",
                    phase="post_compile_sync",
                    local_state=decision.global_mask,
                )
                return False
        return False

    @contextlib.contextmanager
    def coordinate(
        self, local_state: RankDispatchState
    ) -> Iterator[Optional[CoordinatedDispatchDecision]]:
        """Negotiate once at an outer dispatch boundary and sync on exit.

        Nested invocations inside an active :func:`dispatch_guard_scope` yield
        ``None`` without issuing duplicate collectives. ``post_compile_sync``
        always executes in ``finally`` so peer ranks never hang on compile errors.
        """
        if is_in_dispatch_guard_scope():
            yield None
            return
        decision = self.negotiate(local_state)
        try:
            with dispatch_guard_scope():
                yield decision
        finally:
            self.post_compile_sync(decision)


_DEFAULT_GUARD_LOCK = threading.Lock()
_DEFAULT_DISPATCH_GUARD: Optional[DistributedDispatchGuard] = None
_DEFAULT_DISPATCH_WORLD_PG: Any = None


def _get_active_dist_context():
    """Return ``(dist, world_size, rank, world_pg)`` when active (ws > 1), else None."""
    try:
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return None
        ws = dist.get_world_size()
        if ws <= 1:
            return None
        rk = dist.get_rank()
        world_pg = getattr(getattr(dist, "group", None), "WORLD", None)
        return dist, ws, rk, world_pg
    except Exception:
        return None


def reset_default_dispatch_guard() -> None:
    """Reset the cached default DistributedDispatchGuard singleton."""
    global _DEFAULT_DISPATCH_GUARD, _DEFAULT_DISPATCH_WORLD_PG
    with _DEFAULT_GUARD_LOCK:
        _DEFAULT_DISPATCH_GUARD = None
        _DEFAULT_DISPATCH_WORLD_PG = None


def init_default_dispatch_guard(
    process_group: Any = None,
    create_gloo_group: bool = True,
    *,
    backend: Optional[str] = None,
    timeout: Any = None,
    comm_timeout_s: Optional[float] = None,
    raise_on_fault: bool = True,
) -> Optional[DistributedDispatchGuard]:
    """Eagerly initialize the shared DistributedDispatchGuard across ranks.

    When ``process_group`` is None and ``create_gloo_group`` is True, this
    collectively creates a dedicated side-channel ``gloo`` ProcessGroup via
    ``dist.new_group`` and MUST be called by all ranks (e.g. immediately after
    ``torch.distributed.init_process_group``).
    """
    global _DEFAULT_DISPATCH_GUARD, _DEFAULT_DISPATCH_WORLD_PG
    ctx = _get_active_dist_context()
    if ctx is None:
        return None
    dist, ws, rk, world_pg = ctx

    eff_timeout = timeout
    if eff_timeout is None and comm_timeout_s is not None and float(comm_timeout_s) > 0.0:
        eff_timeout = datetime.timedelta(seconds=float(comm_timeout_s))

    pg = process_group
    if pg is None and create_gloo_group:
        ng_kwargs: Dict[str, Any] = {}
        if eff_timeout is not None:
            ng_kwargs["timeout"] = eff_timeout
        try:
            if backend is not None:
                pg = dist.new_group(backend=backend, **ng_kwargs)
            elif getattr(dist, "is_gloo_available", lambda: False)():
                pg = dist.new_group(backend="gloo", **ng_kwargs)
            else:
                pg = dist.new_group(**ng_kwargs)
        except TypeError:
            try:
                if backend is not None:
                    pg = dist.new_group(backend=backend)
                elif getattr(dist, "is_gloo_available", lambda: False)():
                    pg = dist.new_group(backend="gloo")
                else:
                    pg = dist.new_group()
            except Exception:
                pg = None
        except Exception:
            pg = None

    with _DEFAULT_GUARD_LOCK:
        _DEFAULT_DISPATCH_GUARD = DistributedDispatchGuard(
            world_size=ws,
            rank=rk,
            process_group=pg,
            comm_timeout_s=comm_timeout_s,
            raise_on_fault=raise_on_fault,
        )
        _DEFAULT_DISPATCH_WORLD_PG = world_pg
        return _DEFAULT_DISPATCH_GUARD


def get_default_dispatch_guard(*, create_pg: bool = False) -> Optional[DistributedDispatchGuard]:
    """Return a shared DistributedDispatchGuard when torch.distributed is active (ws > 1).

    Strictly non-collective by default (``create_pg=False``): returns the guard
    eagerly registered by :func:`init_default_dispatch_guard` if present, or
    lazily binds to the already-initialized default ProcessGroup
    (``process_group=None``) without calling ``dist.new_group()``.
    """
    global _DEFAULT_DISPATCH_GUARD, _DEFAULT_DISPATCH_WORLD_PG
    ctx = _get_active_dist_context()
    if ctx is None:
        return None
    _, ws, rk, world_pg = ctx

    with _DEFAULT_GUARD_LOCK:
        guard = _DEFAULT_DISPATCH_GUARD
        stale = (
            guard is None
            or guard.world_size != ws
            or guard.rank != rk
            or (
                world_pg is not None
                and _DEFAULT_DISPATCH_WORLD_PG is not None
                and _DEFAULT_DISPATCH_WORLD_PG is not world_pg
            )
        )
        if not stale:
            if not (create_pg and guard.process_group is None):
                return guard
        elif not create_pg:
            _DEFAULT_DISPATCH_GUARD = DistributedDispatchGuard(
                world_size=ws,
                rank=rk,
                process_group=None,
            )
            _DEFAULT_DISPATCH_WORLD_PG = world_pg
            return _DEFAULT_DISPATCH_GUARD

    return init_default_dispatch_guard(create_gloo_group=True)


def _is_oom_exception(exc: BaseException) -> bool:
    if isinstance(exc, OutOfResources):
        return True
    cls_name = type(exc).__name__.lower()
    if "outofmemory" in cls_name:
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "out of resource" in msg


class OOMRecoveryController:
    """Differentiated inference/training OOM self-healing with structured risk tracking."""

    def __init__(
        self,
        *,
        min_split_unit: int = 1,
        max_split_depth: int = 3,
        registry: FallbackRegistry = GLOBAL_FALLBACK_REGISTRY,
    ) -> None:
        self.min_split_unit = max(1, int(min_split_unit))
        self.max_split_depth = max(1, int(max_split_depth))
        self.registry = registry

    def execute(
        self,
        op_name: str,
        phase: str,
        work_units: int,
        primary_fn: Callable[[], Any],
        *,
        split_fn: Optional[Callable[[int], Any]] = None,
        fallback_fn: Optional[Callable[[], Any]] = None,
        release_cache_fn: Optional[Callable[[], None]] = None,
        bucket_key: Sequence[Any] = (),
        rank: int = 0,
    ) -> Any:
        """Run ``primary_fn``, recovering on OOM via chunk/microbatch splitting or fallback."""
        try:
            return primary_fn()
        except BaseException as exc:
            if not _is_oom_exception(exc):
                raise
            oom_reason = f"{type(exc).__name__}: {exc}"

        if release_cache_fn is not None:
            release_cache_fn()

        is_infer = str(phase).lower() in ("prefill", "decode", "inference", "batch_infer")
        if split_fn is not None and work_units >= 2 * self.min_split_unit:
            for depth in range(1, self.max_split_depth + 1):
                num_chunks = 1 << depth
                if work_units // num_chunks < self.min_split_unit:
                    break
                action = (
                    f"inference_chunk_split(chunks={num_chunks})"
                    if is_infer
                    else f"training_microbatch_accum(splits={num_chunks})"
                )
                try:
                    result = split_fn(num_chunks)
                    self.registry.record(
                        op_name,
                        phase,
                        (work_units,),
                        bucket_key,
                        reason=oom_reason,
                        recovery_action=action,
                        perf_risk_level=PerfRiskLevel.MEDIUM,
                        rank=rank,
                    )
                    return result
                except BaseException as retry_exc:
                    if not _is_oom_exception(retry_exc):
                        raise
                    if release_cache_fn is not None:
                        release_cache_fn()
                    continue

        if fallback_fn is not None:
            res = fallback_fn()
            self.registry.record(
                op_name,
                phase,
                (work_units,),
                bucket_key,
                reason=oom_reason,
                recovery_action="aten_reference_fallback",
                perf_risk_level=PerfRiskLevel.HIGH,
                rank=rank,
            )
            return res

        raise OutOfResources(work_units, self.min_split_unit, f"{op_name}:{phase}")


@dataclass(frozen=True)
class StepHealthDecision:
    """Agreed cross-rank health state prior to gradient collectives or step commit."""

    all_ok: bool
    any_oom: bool
    any_error: bool
    oom_ranks: Tuple[int, ...] = ()
    error_ranks: Tuple[int, ...] = ()
    degraded_solo: bool = False
    fault_kind: Optional[str] = None
    fault_reason: Optional[str] = None
    recommended_action: str = "proceed"


class CoordinatedStepGuard:
    """Multi-card fault & asymmetric-OOM step coordinator with bounded timeouts.

    Performs a single bounded pre-collective health consensus across all ranks
    so that:
      - If ANY rank hits OOM on a long-tail shape, ALL ranks learn ``any_oom=True``
        before entering gradient ``all_reduce`` and synchronously run
        :class:`OOMRecoveryController` (empty cache + microbatch/chunk step-down)
        with zero deadlock.
      - If ANY peer rank crashes (``SIGKILL`` / device fault) or stalls past
        ``comm_timeout_s``, surviving ranks detect the fault within bounded time,
        record a structured :class:`FallbackRegistry` audit event, release device
        memory, and abort/degrade deterministically without hanging.
    """

    _STATUS_OK = 0
    _STATUS_OOM = 1
    _STATUS_ERROR = 2

    def __init__(
        self,
        *,
        dispatch_guard: Optional[DistributedDispatchGuard] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        process_group: Any = None,
        comm_timeout_s: Optional[float] = 5.0,
        raise_on_fault: bool = True,
        oom_controller: Optional[OOMRecoveryController] = None,
        fallback_registry: FallbackRegistry = GLOBAL_FALLBACK_REGISTRY,
        all_reduce_status_fn: Optional[Callable[[List[int]], List[int]]] = None,
        release_memory_on_fault: bool = True,
    ) -> None:
        self.dispatch_guard = dispatch_guard
        if dispatch_guard is not None:
            self.world_size = max(1, int(world_size if world_size is not None else dispatch_guard.world_size))
            self.rank = int(rank if rank is not None else dispatch_guard.rank)
            self.process_group = (
                process_group if process_group is not None else dispatch_guard.process_group
            )
            default_timeout = dispatch_guard.comm_timeout_s
        else:
            ctx = _get_active_dist_context()
            if ctx is not None:
                _, det_ws, det_rk, _ = ctx
                self.world_size = max(1, int(world_size if world_size is not None else det_ws))
                self.rank = int(rank if rank is not None else det_rk)
            else:
                self.world_size = max(1, int(world_size if world_size is not None else 1))
                self.rank = int(rank if rank is not None else 0)
            self.process_group = process_group
            default_timeout = _parse_env_comm_timeout()

        resolved_to = default_timeout if comm_timeout_s is None else float(comm_timeout_s)
        self.comm_timeout_s: Optional[float] = (
            resolved_to if (resolved_to is not None and resolved_to > 0.0) else None
        )
        self.raise_on_fault = bool(raise_on_fault)
        self.fallback_registry = fallback_registry
        self.oom_controller = oom_controller or OOMRecoveryController(registry=fallback_registry)
        self._all_reduce_status_fn = all_reduce_status_fn
        self.release_memory_on_fault = bool(release_memory_on_fault)
        self.checks_completed = 0
        self.oom_recoveries = 0
        self.fault_count = 0
        self._faulted = False
        self._last_fault_kind: Optional[str] = None
        self._last_fault_reason: Optional[str] = None

    @property
    def faulted(self) -> bool:
        return self._faulted

    def reset_fault_state(self) -> None:
        self._faulted = False
        self._last_fault_kind = None
        self._last_fault_reason = None
        if self.dispatch_guard is not None:
            self.dispatch_guard.reset_fault_state()

    def cleanup_after_fault(self, *, destroy_world: bool = False) -> None:
        """Flush device memory and tear down faulted ProcessGroups cleanly."""
        self._faulted = True
        if self.dispatch_guard is not None:
            self.dispatch_guard.cleanup_after_fault(destroy_world=destroy_world)
            self.process_group = None
            return
        _release_device_memory_best_effort()
        try:
            import torch.distributed as dist

            dist_active = bool(dist.is_available() and dist.is_initialized())
        except Exception:
            dist_active = False
            dist = None  # type: ignore[assignment]
        pg = self.process_group
        self.process_group = None
        if pg is not None and dist_active and dist is not None:
            try:
                dist.destroy_process_group(pg)
            except Exception:
                pass
        if destroy_world and dist_active and dist is not None:
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    def _reduce_status_vector(
        self,
        local_vec: List[int],
        *,
        phase: str,
    ) -> List[int]:
        if self._all_reduce_status_fn is not None:
            out = _run_callable_with_timeout(
                lambda: self._all_reduce_status_fn(list(local_vec)),
                self.comm_timeout_s,
                rank=self.rank,
                world_size=self.world_size,
                op="all_reduce_health",
                phase=phase,
            )
            return [int(v) for v in out]

        if self.world_size <= 1 and self.process_group is None:
            return list(local_vec)

        try:
            import torch
            import torch.distributed as dist

            dist_active = bool(dist.is_available() and dist.is_initialized())
        except Exception:
            dist_active = False

        if not dist_active:
            return list(local_vec)

        dev = _resolve_pg_device(dist, self.process_group)
        status_tensor = torch.tensor(local_vec, dtype=torch.int32, device=dev)
        target_pg, using_world = _apply_pg_timeout_best_effort(
            dist, self.process_group, self.comm_timeout_s
        )
        if self.comm_timeout_s is not None:
            try:
                work = dist.all_reduce(
                    status_tensor,
                    op=dist.ReduceOp.MAX,
                    group=self.process_group,
                    async_op=True,
                )
            except TypeError:
                _run_callable_with_timeout(
                    lambda: dist.all_reduce(
                        status_tensor, op=dist.ReduceOp.MAX, group=self.process_group
                    ),
                    self.comm_timeout_s,
                    rank=self.rank,
                    world_size=self.world_size,
                    op="all_reduce_health",
                    phase=phase,
                )
            else:
                _wait_work_bounded(
                    work,
                    _compute_work_wait_timeout(self.comm_timeout_s, target_pg is not None),
                    rank=self.rank,
                    world_size=self.world_size,
                    op="all_reduce_health",
                    phase=phase,
                )
            _restore_world_pg_timeout_best_effort(dist, target_pg, using_world)
        else:
            dist.all_reduce(status_tensor, op=dist.ReduceOp.MAX, group=self.process_group)
        return [int(v) for v in status_tensor.tolist()]

    def _handle_health_fault(
        self,
        exc: BaseException,
        *,
        op_name: str,
        phase: str,
        shape: Sequence[Any],
        bucket_key: Sequence[Any],
    ) -> StepHealthDecision:
        fault_kind, wrapped = _classify_collective_exception(
            exc,
            rank=self.rank,
            world_size=self.world_size,
            op="check_step_health",
            phase=phase,
            timeout_s=self.comm_timeout_s,
        )
        self.fault_count += 1
        self._faulted = True
        self._last_fault_kind = fault_kind
        self._last_fault_reason = str(wrapped)
        if self.dispatch_guard is not None:
            self.dispatch_guard.mark_faulted(fault_kind, str(wrapped))
        if self.release_memory_on_fault:
            _release_device_memory_best_effort()
        action = f"abort_{fault_kind}" if self.raise_on_fault else f"degrade_solo_{fault_kind}"
        self.fallback_registry.record(
            op_name,
            phase,
            shape if shape else (self.world_size,),
            bucket_key if bucket_key else (f"rank={self.rank}",),
            reason=str(wrapped),
            recovery_action=action,
            perf_risk_level=PerfRiskLevel.HIGH,
            rank=self.rank,
        )
        if self.raise_on_fault:
            raise wrapped from exc
        return StepHealthDecision(
            all_ok=False,
            any_oom=False,
            any_error=True,
            degraded_solo=True,
            fault_kind=fault_kind,
            fault_reason=str(wrapped),
            recommended_action="degrade_solo",
        )

    def check_step_health(
        self,
        local_ok: bool = True,
        local_oom: bool = False,
        local_error: Optional[BaseException] = None,
        *,
        op_name: str = "coordinated_step",
        phase: str = "training",
        shape: Sequence[Any] = (),
        bucket_key: Sequence[Any] = (),
    ) -> StepHealthDecision:
        """Exchange per-rank OK/OOM/ERROR status with bounded timeout before collectives."""
        if self._faulted:
            reason = self._last_fault_reason or "coordinated step guard already faulted"
            kind = self._last_fault_kind or "rank_fault"
            if self.raise_on_fault:
                if kind == "comm_timeout":
                    raise DistributedCommTimeoutError(
                        rank=self.rank,
                        world_size=self.world_size,
                        op="check_step_health",
                        phase=phase,
                        timeout_s=float(self.comm_timeout_s or 0.0),
                        cause=reason,
                    )
                raise DistributedRankFaultError(
                    rank=self.rank,
                    world_size=self.world_size,
                    op="check_step_health",
                    phase=phase,
                    timeout_s=self.comm_timeout_s,
                    cause=reason,
                )
            return StepHealthDecision(
                all_ok=False,
                any_oom=False,
                any_error=True,
                degraded_solo=True,
                fault_kind=kind,
                fault_reason=reason,
                recommended_action="degrade_solo",
            )

        is_oom = bool(local_oom) or (local_error is not None and _is_oom_exception(local_error))
        is_err = (not bool(local_ok) or local_error is not None) and not is_oom

        local_code = self._STATUS_OK
        if is_oom:
            local_code = self._STATUS_OOM
        elif is_err:
            local_code = self._STATUS_ERROR

        vec = [self._STATUS_OK] * self.world_size
        if 0 <= self.rank < self.world_size:
            vec[self.rank] = local_code

        try:
            merged = self._reduce_status_vector(vec, phase=phase)
        except Exception as exc:
            return self._handle_health_fault(
                exc,
                op_name=op_name,
                phase=phase,
                shape=shape,
                bucket_key=bucket_key,
            )

        self.checks_completed += 1
        oom_ranks = tuple(r for r, code in enumerate(merged) if code == self._STATUS_OOM)
        err_ranks = tuple(r for r, code in enumerate(merged) if code == self._STATUS_ERROR)
        any_oom = len(oom_ranks) > 0
        any_error = len(err_ranks) > 0
        all_ok = not any_oom and not any_error

        if all_ok:
            recommended = "proceed"
        elif any_oom and not any_error:
            recommended = "coordinated_oom_recovery"
            self.fallback_registry.record(
                op_name,
                phase,
                shape if shape else (self.world_size,),
                bucket_key if bucket_key else (f"oom_ranks={oom_ranks}",),
                reason=f"asymmetric_rank_oom(oom_ranks={oom_ranks})",
                recovery_action="coordinated_oom_consensus",
                perf_risk_level=PerfRiskLevel.MEDIUM,
                rank=self.rank,
            )
        else:
            recommended = "skip_step"
            self.fallback_registry.record(
                op_name,
                phase,
                shape if shape else (self.world_size,),
                bucket_key if bucket_key else (f"error_ranks={err_ranks}",),
                reason=f"peer_step_error(error_ranks={err_ranks}, oom_ranks={oom_ranks})",
                recovery_action="coordinated_step_skip",
                perf_risk_level=PerfRiskLevel.HIGH,
                rank=self.rank,
            )

        return StepHealthDecision(
            all_ok=all_ok,
            any_oom=any_oom,
            any_error=any_error,
            oom_ranks=oom_ranks,
            error_ranks=err_ranks,
            recommended_action=recommended,
        )

    def recover_coordinated_oom(
        self,
        op_name: str,
        phase: str,
        work_units: int,
        *,
        split_fn: Optional[Callable[[int], Any]] = None,
        fallback_fn: Optional[Callable[[], Any]] = None,
        release_cache_fn: Optional[Callable[[], None]] = None,
        bucket_key: Sequence[Any] = (),
        oom_reason: str = "coordinated_peer_oom",
    ) -> Any:
        """Synchronously execute OOMRecoveryController step-down across all ranks."""
        self.oom_recoveries += 1
        release_fn = release_cache_fn or _release_device_memory_best_effort

        def _trigger_coordinated_recovery() -> Any:
            raise OutOfResources(work_units, self.oom_controller.min_split_unit, oom_reason)

        return self.oom_controller.execute(
            op_name,
            phase,
            work_units,
            primary_fn=_trigger_coordinated_recovery,
            split_fn=split_fn,
            fallback_fn=fallback_fn,
            release_cache_fn=release_fn,
            bucket_key=bucket_key,
            rank=self.rank,
        )

    def run_coordinated_step(
        self,
        op_name: str,
        phase: str,
        work_units: int,
        step_fn: Callable[[], Any],
        *,
        split_fn: Optional[Callable[[int], Any]] = None,
        fallback_fn: Optional[Callable[[], Any]] = None,
        on_oom_reset_fn: Optional[Callable[[], None]] = None,
        post_ok_fn: Optional[Callable[[Any], Any]] = None,
        release_cache_fn: Optional[Callable[[], None]] = None,
        shape: Sequence[Any] = (),
        bucket_key: Sequence[Any] = (),
    ) -> Tuple[Any, StepHealthDecision]:
        """Run one multi-card step with pre-collective health & OOM consensus."""
        local_ok = True
        local_oom = False
        local_err: Optional[BaseException] = None
        step_result: Any = None
        try:
            step_result = step_fn()
        except BaseException as exc:  # noqa: BLE001
            local_ok = False
            local_oom = _is_oom_exception(exc)
            local_err = exc

        health = self.check_step_health(
            local_ok=local_ok,
            local_oom=local_oom,
            local_error=local_err,
            op_name=op_name,
            phase=phase,
            shape=shape,
            bucket_key=bucket_key,
        )

        if health.all_ok:
            if post_ok_fn is not None:
                step_result = post_ok_fn(step_result)
            return step_result, health

        if health.any_oom and not health.any_error and (split_fn is not None or fallback_fn is not None):
            if on_oom_reset_fn is not None:
                on_oom_reset_fn()
            rec_ok = True
            rec_oom = False
            rec_err: Optional[BaseException] = None
            try:
                step_result = self.recover_coordinated_oom(
                    op_name=op_name,
                    phase=phase,
                    work_units=work_units,
                    split_fn=split_fn,
                    fallback_fn=fallback_fn,
                    release_cache_fn=release_cache_fn,
                    bucket_key=bucket_key,
                    oom_reason=f"coordinated_oom(oom_ranks={health.oom_ranks})",
                )
            except BaseException as exc:  # noqa: BLE001
                rec_ok = False
                rec_oom = _is_oom_exception(exc)
                rec_err = exc

            post_health = self.check_step_health(
                local_ok=rec_ok,
                local_oom=rec_oom,
                local_error=rec_err,
                op_name=f"{op_name}_oom_retry",
                phase=phase,
                shape=shape,
                bucket_key=bucket_key,
            )
            if post_health.all_ok and post_ok_fn is not None:
                step_result = post_ok_fn(step_result)
            return step_result, post_health

        return None, health


DistributedFaultCoordinator = CoordinatedStepGuard
