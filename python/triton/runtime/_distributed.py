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

from dataclasses import asdict, dataclass
from enum import Enum, IntFlag
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .errors import OutOfResources

_dist_logger = logging.getLogger(__name__)


class RankDispatchState(IntFlag):
    """Bitmask exchanged across ranks prior to collective-sensitive regions."""

    ALL_HOT_HIT = 0
    ANY_COLD_COMPILE = 1
    ANY_OOM_RETRY = 2
    ANY_TIMEOUT = 4
    ANY_FALLBACK = 8


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


@dataclass(frozen=True)
class CoordinatedDispatchDecision:
    """Agreed cross-rank execution plan preventing collective deadlocks."""

    global_mask: RankDispatchState
    all_hot_hit: bool
    extend_collective_window: bool
    agreed_fallback: bool
    barrier_completed: bool


class DistributedDispatchGuard:
    """Synchronize bucket/compile/fallback states across DDP/FSDP/TP/EP ranks.

    Prevents FlagCX / NCCL collective watchdog deadlocks when a single rank
    encounters a cold bucket compilation, MoE expert spike, or OOM fallback
    while peer ranks hit warm cache.
    """

    def __init__(
        self,
        *,
        world_size: int = 1,
        rank: int = 0,
        process_group: Any = None,
        all_reduce_or_fn: Optional[Callable[[int], int]] = None,
        barrier_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.process_group = process_group
        self._all_reduce_or_fn = all_reduce_or_fn
        self._barrier_fn = barrier_fn
        self.negotiation_count = 0
        self.slow_path_syncs = 0

    def _reduce_bitmask(self, local_mask: int) -> int:
        if self.world_size <= 1 and self._all_reduce_or_fn is None and self.process_group is None:
            try:
                import torch.distributed as dist
                if not (dist.is_available() and dist.is_initialized()):
                    return int(local_mask)
            except Exception:
                return int(local_mask)

        if self._all_reduce_or_fn is not None:
            return int(self._all_reduce_or_fn(int(local_mask)))

        try:
            import torch
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                try:
                    import torch_npu  # noqa: F401
                except Exception:
                    pass
                pg_backend = ""
                try:
                    pg_backend = str(dist.get_backend(self.process_group)).lower()
                except Exception:
                    pg_backend = ""
                if pg_backend == "gloo":
                    dev = "cpu"
                elif pg_backend == "hccl" or (hasattr(torch, "npu") and torch.npu.is_available()):
                    dev = "npu"
                elif hasattr(torch, "xpu") and torch.xpu.is_available():
                    dev = "xpu"
                elif hasattr(torch, "mlu") and torch.mlu.is_available():
                    dev = "mlu"
                elif hasattr(torch, "musa") and torch.musa.is_available():
                    dev = "musa"
                elif torch.cuda.is_available():
                    dev = "cuda"
                else:
                    dev = "cpu"
                # Exchange bitmask using MAX over individual bit powers or BOR if supported
                bits_tensor = torch.tensor(
                    [(1 if (local_mask & (1 << b)) else 0) for b in range(4)],
                    dtype=torch.int32,
                    device=dev,
                )
                dist.all_reduce(bits_tensor, op=dist.ReduceOp.MAX, group=self.process_group)
                out_mask = 0
                for b, v in enumerate(bits_tensor.tolist()):
                    if v > 0:
                        out_mask |= (1 << b)
                return out_mask
        except Exception:
            pass
        return int(local_mask)

    def _run_barrier(self) -> bool:
        if self._barrier_fn is not None:
            self._barrier_fn()
            return True
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier(group=self.process_group)
                return True
        except Exception:
            pass
        return False

    def negotiate(self, local_state: RankDispatchState) -> CoordinatedDispatchDecision:
        """Synchronize ``local_state`` across ranks and return an agreed decision."""
        self.negotiation_count += 1
        merged = RankDispatchState(self._reduce_bitmask(int(local_state)))
        all_hot = merged is RankDispatchState.ALL_HOT_HIT
        need_window = bool(
            merged & (RankDispatchState.ANY_COLD_COMPILE | RankDispatchState.ANY_OOM_RETRY | RankDispatchState.ANY_TIMEOUT)
        )
        agreed_fb = bool(merged & RankDispatchState.ANY_FALLBACK)
        did_barrier = False
        if need_window or agreed_fb:
            self.slow_path_syncs += 1
            did_barrier = self._run_barrier()
        return CoordinatedDispatchDecision(
            global_mask=merged,
            all_hot_hit=all_hot,
            extend_collective_window=need_window,
            agreed_fallback=agreed_fb,
            barrier_completed=did_barrier,
        )


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
