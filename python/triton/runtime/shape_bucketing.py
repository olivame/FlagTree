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

"""Phase-aware AST-guided online shape bucketing and optional PGO interval policies.

Supports both inference workloads (LLM prefill/decode with dynamic KV cache
growth, MoE token routing, and multi-batch retrieval/reranker/CTR models)
and training workloads (forward/backward dynamic sequence lengths, transposed
weight-gradient contractions, and multi-device collectives) without relying on
offline chip-specific sweep tables.
"""

from __future__ import annotations

import bisect
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class DispatchStatus(str, Enum):
    """Explicit 5-state bucket dispatch outcome (plus DISABLED)."""

    HIT = "HIT"
    COLD_TUNED = "COLD_TUNED"
    UNDERFLOW = "UNDERFLOW"
    OVERFLOW = "OVERFLOW"
    GUARD_MISMATCH = "GUARD_MISMATCH"
    DISABLED = "DISABLED"


class BucketingMode(str, Enum):
    """Execution phase controlling low-end Decode vs. throughput ladders."""

    AUTO = "auto"
    INFERENCE = "inference"
    TRAINING = "training"

    @classmethod
    def parse(cls, raw: Optional[str]) -> "BucketingMode":
        if not raw:
            return cls.AUTO
        norm = raw.strip().lower()
        for member in cls:
            if member.value == norm:
                return member
        return cls.AUTO


# Fine-grained Decode / low-batch ladder (1..64) distinguishing memory-bound
# single-token / small-batch decode from TensorCore/Cube tile transitions.
DECODE_FINE_LADDER: Tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 48, 64)

# Half-power-of-two (1.0x and 1.5x * 2^k) throughput ladder covering Prefill,
# long-context KV cache (up to 128K+ statically, extensible arbitrarily),
# batch retrieval, and forward/backward training token counts.
THROUGHPUT_GEOMETRIC_LADDER: Tuple[int, ...] = (
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
)


def _active_backend_name() -> str:
    import os

    try:
        from triton._flagtree_backend import FLAGTREE_BACKEND
        if FLAGTREE_BACKEND:
            return FLAGTREE_BACKEND
    except Exception:
        pass
    env_bk = os.environ.get("FLAGTREE_BACKEND", "").strip().lower()
    if env_bk:
        return env_bk
    try:
        from .driver import driver
        target = driver.active.get_current_target()
        bk = getattr(target, "backend", None)
        if bk:
            return "ascend" if bk in ("ascend", "cann", "npu") else str(bk)
    except Exception:
        pass
    return "default"


def _freeze_hashable(value: Any) -> Any:
    """Convert nested containers into hashable tuples for telemetry sets."""
    if isinstance(value, tuple):
        return tuple(_freeze_hashable(v) for v in value)
    if isinstance(value, list):
        return tuple(_freeze_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze_hashable(v)) for k, v in value.items()))
    try:
        hash(value)
        return value
    except TypeError:
        return str(value)


def _canonical_dtype_str(value: Any) -> Optional[str]:
    """Extract canonical tensor dtype string while ignoring non-tensor objects."""
    if hasattr(value, "data_ptr") and hasattr(value, "dtype"):
        return str(value.dtype)
    # Handle Triton TensorDescriptor wrapping a base tensor
    base = getattr(value, "base", None)
    if base is not None and hasattr(base, "data_ptr") and hasattr(base, "dtype"):
        return str(base.dtype)
    return None


def extract_tensor_dtypes(values: Iterable[Any]) -> Tuple[str, ...]:
    """Return canonical tensor dtype strings in argument order."""
    out: List[str] = []
    for val in values:
        dt = _canonical_dtype_str(val)
        if dt is not None:
            out.append(dt)
    return tuple(out)


def int_width_tag(val: int) -> str:
    """Match Triton / LibEntry integer specialization width buckets."""
    if -(2**31) <= val <= 2**31 - 1:
        return "i32"
    if 2**63 <= val <= 2**64 - 1:
        return "u64"
    return "i64"


def geometric_bucket_ceil(
    value: int,
    ladder: Sequence[int],
    min_floor: int = 1,
) -> int:
    """Map positive integer ``value`` to the smallest ladder step >= ``value``.

    For values exceeding ``ladder[-1]``, continues deterministically along the
    half-power-of-two progression ``{2^k, 3 * 2^(k-1)}`` so ultra-long contexts
    or large flattened element counts never degenerate back to exact keys.
    """
    clamped = max(int(value), max(1, int(min_floor)))
    if ladder:
        idx = bisect.bisect_left(ladder, clamped)
        if idx < len(ladder):
            return int(ladder[idx])
    if clamped <= 1:
        return 1
    bit_len = clamped.bit_length()
    pow2_lo = 1 << (bit_len - 1)
    if pow2_lo == clamped:
        return clamped
    half_step = pow2_lo + (pow2_lo >> 1)
    if clamped <= half_step:
        return half_step
    return 1 << bit_len


def build_axis_ladder(
    *,
    mode: BucketingMode = BucketingMode.AUTO,
    candidate_blocks: Sequence[int] = (),
    min_block_size: int = 1,
    feeds_dot: bool = False,
) -> Tuple[int, ...]:
    """Construct a sorted monotonically increasing bucket ladder for one axis."""
    points = set(THROUGHPUT_GEOMETRIC_LADDER)
    if mode is BucketingMode.TRAINING and feeds_dot and min_block_size >= 8:
        # In pure training mode on matrix-multiply axes, snap tiny sizes up to
        # the hardware dot lower bound while keeping fine steps above it.
        floor = max(1, int(min_block_size))
        for p in DECODE_FINE_LADDER:
            if p >= floor:
                points.add(p)
        points.add(floor)
    else:
        # AUTO and INFERENCE preserve fine-grained Decode buckets (1, 2, 4, 8,
        # 16, 24, 32, 48, 64) so autoregressive decode and small-batch queries
        # select memory-optimal tiles instead of coarse large-batch tiles.
        points.update(DECODE_FINE_LADDER)
        if min_block_size > 1:
            points.add(int(min_block_size))
    for blk in candidate_blocks:
        if type(blk) is int and blk > 0:
            points.add(blk)
            half_blk = blk + (blk >> 1)
            if half_blk > blk:
                points.add(half_blk)
    return tuple(sorted(points))


@dataclass(frozen=True)
class BucketLookupResult:
    """Result of normalizing runtime arguments into a JSON-safe bucket key."""

    bucket_key: Tuple[Any, ...]
    status: DispatchStatus
    representative_dims: Dict[str, int] = field(default_factory=dict)
    raw_dims: Dict[str, Any] = field(default_factory=dict)
    clamped_axes: Tuple[str, ...] = ()


class BucketTelemetry:
    """Thread-safe observability counters for bucket hits, cold tunes, and fallbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts: Dict[DispatchStatus, int] = {s: 0 for s in DispatchStatus}
        self._raw_shapes_by_op: Dict[str, set] = {}
        self._buckets_by_op: Dict[str, set] = {}
        self.evictions: int = 0

    def record(
        self,
        op_name: str,
        status: DispatchStatus,
        raw_key: Any,
        bucket_key: Any,
    ) -> None:
        frozen_raw = _freeze_hashable(raw_key)
        frozen_bkt = _freeze_hashable(bucket_key)
        with self._lock:
            self.counts[status] = self.counts.get(status, 0) + 1
            self._raw_shapes_by_op.setdefault(op_name, set()).add(frozen_raw)
            self._buckets_by_op.setdefault(op_name, set()).add(frozen_bkt)

    def record_eviction(self, count: int = 1) -> None:
        with self._lock:
            self.evictions += count

    def reset(self) -> None:
        with self._lock:
            for s in DispatchStatus:
                self.counts[s] = 0
            self._raw_shapes_by_op.clear()
            self._buckets_by_op.clear()
            self.evictions = 0

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            total = sum(v for k, v in self.counts.items() if k is not DispatchStatus.DISABLED)
            hits = self.counts.get(DispatchStatus.HIT, 0)
            unique_raw = sum(len(v) for v in self._raw_shapes_by_op.values())
            unique_buckets = sum(len(v) for v in self._buckets_by_op.values())
            return {
                "counts": {k.value: v for k, v in self.counts.items()},
                "total_dispatches": total,
                "hit_rate": (hits / total) if total > 0 else 0.0,
                "unique_raw_shapes": unique_raw,
                "unique_buckets": unique_buckets,
                "compression_ratio": (unique_raw / unique_buckets) if unique_buckets > 0 else 1.0,
                "evictions": self.evictions,
            }


GLOBAL_BUCKET_TELEMETRY = BucketTelemetry()


class BucketCacheBudget:
    """LRU capacity controller for in-memory bucket caches."""

    def __init__(self, max_entries: Optional[int] = None) -> None:
        self.max_entries = max_entries if (max_entries is not None and max_entries > 0) else None
        self._lru: "OrderedDict[Any, None]" = OrderedDict()
        self._lock = threading.Lock()

    def touch_and_evict(
        self,
        cache_dict: Dict[Any, Any],
        key: Any,
        max_entries: Optional[int] = None,
    ) -> int:
        limit = max_entries if max_entries is not None else self.max_entries
        if limit is None or limit <= 0:
            return 0
        evicted = 0
        with self._lock:
            self._lru.pop(key, None)
            self._lru[key] = None
            while len(cache_dict) > limit and self._lru:
                oldest_key, _ = self._lru.popitem(last=False)
                if oldest_key == key:
                    self._lru[key] = None
                    break
                if oldest_key in cache_dict:
                    del cache_dict[oldest_key]
                    evicted += 1
            if len(cache_dict) > limit:
                for k in list(cache_dict.keys()):
                    if len(cache_dict) <= limit:
                        break
                    if k != key:
                        del cache_dict[k]
                        self._lru.pop(k, None)
                        evicted += 1
        if evicted:
            GLOBAL_BUCKET_TELEMETRY.record_eviction(evicted)
        return evicted


def enforce_disk_cache_budget(
    cache_root: Optional[str | Path],
    max_artifacts: Optional[int],
    *,
    pattern: str = "*.autotune.json",
) -> int:
    """Evict oldest on-disk autotune JSON artifacts when count exceeds ``max_artifacts``."""
    if cache_root is None or max_artifacts is None or max_artifacts <= 0:
        return 0
    root = Path(cache_root)
    if not root.exists():
        return 0
    files = list(root.rglob(pattern))
    if len(files) <= max_artifacts:
        return 0
    files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    to_remove = len(files) - max_artifacts
    removed = 0
    for victim in files[:to_remove]:
        try:
            victim.unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    if removed:
        GLOBAL_BUCKET_TELEMETRY.record_eviction(removed)
    return removed


class OnlineBucketPolicy:
    """AST-guided universal online shape bucketing policy across chips and ops.

    Produces strictly JSON-serializable nested-tuple bucket keys suitable for
    ``Autotuner.cache``, ``Autotuner.check_disk_cache``, ``Flagtuner``, and
    ``FlagGems`` ``LibTuner`` / ``LibEntry``.
    """

    def __init__(
        self,
        axis_specs: Optional[Mapping[str, Any]] = None,
        *,
        mode: BucketingMode = BucketingMode.AUTO,
        divisibility: int = 16,
        domain_bounds: Optional[Mapping[str, Tuple[int, int]]] = None,
        backend: Optional[str] = None,
        op_name: str = "unknown",
    ) -> None:
        self.axis_specs: Dict[str, Any] = dict(axis_specs or {})
        self.mode = mode
        self.divisibility = max(1, int(divisibility))
        self.domain_bounds: Dict[str, Tuple[int, int]] = {
            k: (int(lo), int(hi)) for k, (lo, hi) in (domain_bounds or {}).items()
        }
        self.backend = backend if backend is not None else _active_backend_name()
        self.op_name = op_name
        self._ladders: Dict[str, Tuple[int, ...]] = {}
        for name, spec in self.axis_specs.items():
            role_val = getattr(getattr(spec, "role", None), "value", getattr(spec, "role", ""))
            if role_val in ("dynamic_tile_dim", "dynamic_stride"):
                self._ladders[name] = build_axis_ladder(
                    mode=self.mode,
                    candidate_blocks=getattr(spec, "candidate_blocks", ()),
                    min_block_size=getattr(spec, "min_block_size", 1),
                    feeds_dot=getattr(spec, "feeds_dot", False),
                )

    def _ladder_for(self, axis_name: str) -> Tuple[int, ...]:
        ladder = self._ladders.get(axis_name)
        if ladder is None:
            ladder = build_axis_ladder(mode=self.mode)
            self._ladders[axis_name] = ladder
        return ladder

    def bucket_dim_value(self, axis_name: str, value: int) -> Tuple[int, Optional[DispatchStatus]]:
        """Bucket one integer dimension, returning ``(bucket_hi, boundary_status)``."""
        iv = int(value)
        status: Optional[DispatchStatus] = None
        bounds = self.domain_bounds.get(axis_name)
        if bounds is not None:
            lo, hi = bounds
            if iv < lo:
                status = DispatchStatus.UNDERFLOW
                iv = lo
            elif iv > hi:
                status = DispatchStatus.OVERFLOW
                iv = hi
        spec = self.axis_specs.get(axis_name)
        min_floor = 1
        if (
            self.mode is BucketingMode.TRAINING
            and spec is not None
            and getattr(spec, "feeds_dot", False)
        ):
            min_floor = max(1, int(getattr(spec, "min_block_size", 1)))
        bucket_hi = geometric_bucket_ceil(iv, self._ladder_for(axis_name), min_floor=min_floor)
        return bucket_hi, status

    def _normalize_stride_value(
        self,
        axis_name: str,
        value: int,
        all_args: Mapping[str, Any],
    ) -> Tuple[Any, ...]:
        iv = int(value)
        # Preserve broadcast (0), unit contiguous (1), and small spatial strides (<=4)
        if iv <= 4:
            return ("s_exact", iv)
        align_ok = (iv % self.divisibility) == 0
        spec = self.axis_specs.get(axis_name)
        paired_dims: Tuple[str, ...] = ()
        if spec is not None:
            pd = getattr(spec, "paired_dim", None)
            if isinstance(pd, str) and pd:
                paired_dims = (pd,)
            pds = getattr(spec, "paired_dims", ())
            if pds:
                paired_dims = tuple(dict.fromkeys(paired_dims + tuple(pds)))
        for pdim in paired_dims:
            pval = all_args.get(pdim)
            if type(pval) is int and pval == iv:
                # Contiguous row-major stride matching tensor dimension `pdim`
                return ("s_contig", pdim, align_ok)
        bucket_hi = geometric_bucket_ceil(iv, self._ladder_for(axis_name), min_floor=1)
        return ("s_bucket", bucket_hi, align_ok)

    def normalize_scalar_for_autotune(
        self,
        axis_name: str,
        value: Any,
        all_args: Mapping[str, Any],
    ) -> Tuple[Any, Optional[int], Optional[DispatchStatus]]:
        """Normalize one key value into a JSON-safe autotune key component."""
        # Never bucket booleans (`isinstance(True, int)` is True in Python!)
        if type(value) is not int:
            if isinstance(value, (str, float, bool)) or value is None:
                return value, None, None
            return str(value), None, None

        spec = self.axis_specs.get(axis_name)
        role_val = getattr(getattr(spec, "role", None), "value", getattr(spec, "role", None))
        if role_val == "invariant_const":
            return value, None, None
        if role_val == "dynamic_stride":
            return self._normalize_stride_value(axis_name, value, all_args), None, None
        if role_val == "dynamic_tile_dim" or role_val is None:
            lower = axis_name.lower()
            if role_val is None and (
                lower.startswith(("is_", "has_", "have_", "do_", "use_", "apply_", "swap_"))
                or lower in (
                    "head_dim",
                    "head_dim_actual",
                    "block_dmodel",
                    "dqk",
                    "groups",
                    "topk",
                    "top_k",
                    "mode",
                    "dtype_key",
                )
            ):
                return value, None, None
            if role_val is None and "stride" in lower and not lower.startswith(
                ("stride_h", "stride_w", "stride_d", "stride_l")
            ):
                return self._normalize_stride_value(axis_name, value, all_args), None, None
            if value <= 0:
                return ("d_nonpos", value), value, DispatchStatus.UNDERFLOW
            bucket_hi, b_status = self.bucket_dim_value(axis_name, value)
            align_sig = (value == 1, (value % self.divisibility) == 0)
            return ("d_bucket", bucket_hi, align_sig[0], align_sig[1]), bucket_hi, b_status
        return value, None, None

    def lookup_autotune_key(
        self,
        args: Mapping[str, Any],
        keys: Sequence[str],
        *,
        extra_dtypes: Optional[Sequence[str]] = None,
    ) -> BucketLookupResult:
        """Build a normalized, dtype- and backend-isolated autotune bucket key."""
        components: List[Any] = ["ft_bkt_v1", self.backend, self.mode.value]
        rep_dims: Dict[str, int] = {}
        raw_dims: Dict[str, Any] = {}
        clamped: List[str] = []
        boundary_status: Optional[DispatchStatus] = None

        for k in keys:
            if k not in args:
                continue
            raw_val = args[k]
            raw_dims[k] = raw_val
            norm_comp, rep_hi, st = self.normalize_scalar_for_autotune(k, raw_val, args)
            components.append(norm_comp)
            if rep_hi is not None:
                rep_dims[k] = rep_hi
            if st is not None:
                clamped.append(k)
                if boundary_status is None or st is DispatchStatus.OVERFLOW:
                    boundary_status = st

        dtypes = tuple(extra_dtypes) if extra_dtypes is not None else extract_tensor_dtypes(args.values())
        components.append(("dtypes",) + dtypes)
        status = boundary_status if boundary_status is not None else DispatchStatus.HIT
        return BucketLookupResult(
            bucket_key=tuple(components),
            status=status,
            representative_dims=rep_dims,
            raw_dims=raw_dims,
            clamped_axes=tuple(clamped),
        )

    def normalize_entry_scalar(
        self,
        param_name: str,
        value: int,
        *,
        is_autotuned_key: bool,
        all_args: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Any, ...]:
        """Normalize a specialized integer parameter for ``LibEntry.key``.

        Preserves Triton's exact backend specialization contract
        ``(value == 1, value % divisibility == 0, int_width_tag)`` for pure
        JIT kernels, and additionally binds the autotune bucket ceiling when the
        parameter participates in autotuner config selection.
        """
        width = int_width_tag(value)
        is_one = value == 1
        is_aligned = (value % self.divisibility) == 0

        spec = self.axis_specs.get(param_name)
        role_val = getattr(getattr(spec, "role", None), "value", getattr(spec, "role", None))
        if role_val == "invariant_const":
            return (int, value)

        lower = param_name.lower()
        if role_val is None and (
            lower.startswith(("is_", "has_", "have_", "do_", "use_", "apply_", "swap_"))
            or lower in (
                "head_dim",
                "head_dim_actual",
                "block_dmodel",
                "dqk",
                "groups",
                "topk",
                "top_k",
                "mode",
                "dtype_key",
            )
        ):
            return (int, value)

        if not is_autotuned_key:
            # Pure JIT / Heuristics / Pointwise parameter: Triton's compiler
            # only specializes on == 1, divisibility alignment, and i32/i64 width!
            if value <= 0:
                return (int, value)
            return ("ft_jit_int", is_one, is_aligned, width)

        if role_val == "dynamic_stride" or (
            role_val is None
            and "stride" in lower
            and not lower.startswith(("stride_h", "stride_w", "stride_d", "stride_l"))
        ):
            stride_tok = self._normalize_stride_value(param_name, value, all_args or {})
            return ("ft_tune_stride", stride_tok, is_one, is_aligned, width)

        if value <= 0:
            return (int, value)
        bucket_hi, _ = self.bucket_dim_value(param_name, value)
        return ("ft_tune_int", bucket_hi, is_one, is_aligned, width)


# ---------------------------------------------------------------------------
# Secondary PGO Interval DP Policy (generalized: N-D, any dtype, pre_hook safe)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntervalSegment:
    """Closed integer interval ``[lo, hi]`` mapped to a chosen config index."""

    lo: int
    hi: int
    config_index: int
    representative: int
    regret_ms: float = 0.0


def group_frontier(
    shapes: Sequence[int],
    timings_matrix: Sequence[Sequence[float]],
    max_buckets: int,
    weights: Optional[Sequence[float]] = None,
) -> List[IntervalSegment]:
    """Optimal 1D contiguous interval partitioning minimizing weighted latency regret.

    Generalized for arbitrary config sets (no ``BLOCK_M`` or ``FP32`` assumption).
    ``timings_matrix[i][c]`` is the latency of ``shapes[i]`` under config ``c``.
    """
    n = len(shapes)
    if n == 0:
        return []
    num_configs = len(timings_matrix[0])
    if num_configs == 0:
        raise ValueError("timings_matrix must contain at least one candidate config")
    k_max = max(1, min(int(max_buckets), n))
    w = [float(x) for x in weights] if weights is not None else [1.0] * n

    oracle = [min(timings_matrix[i]) for i in range(n)]

    cost = [[0.0] * n for _ in range(n)]
    best_cfg = [[0] * n for _ in range(n)]
    for i in range(n):
        acc_c = [0.0] * num_configs
        oracle_sum = 0.0
        for j in range(i, n):
            wj = w[j]
            oracle_sum += wj * oracle[j]
            row = timings_matrix[j]
            best_val = math.inf
            best_idx = 0
            for c in range(num_configs):
                val = row[c]
                acc_c[c] = (
                    math.inf
                    if (not math.isfinite(val) or not math.isfinite(acc_c[c]))
                    else (acc_c[c] + wj * val)
                )
                if acc_c[c] < best_val:
                    best_val = acc_c[c]
                    best_idx = c
            cost[i][j] = (best_val - oracle_sum) if math.isfinite(best_val) else 1e18
            best_cfg[i][j] = best_idx

    dp = [[math.inf] * n for _ in range(k_max + 1)]
    split = [[-1] * n for _ in range(k_max + 1)]
    for j in range(n):
        dp[1][j] = cost[0][j]
        split[1][j] = 0

    for b in range(2, k_max + 1):
        for j in range(b - 1, n):
            best_total = math.inf
            best_s = b - 1
            for s in range(b - 1, j + 1):
                cand = dp[b - 1][s - 1] + cost[s][j]
                if cand < best_total:
                    best_total = cand
                    best_s = s
            dp[b][j] = best_total
            split[b][j] = best_s

    target_b = 1
    best_final = dp[1][n - 1]
    for b in range(2, k_max + 1):
        if dp[b][n - 1] + 1e-9 < best_final:
            best_final = dp[b][n - 1]
            target_b = b

    segments: List[IntervalSegment] = []
    cur_end = n - 1
    for b in range(target_b, 0, -1):
        cur_start = split[b][cur_end]
        cfg_idx = best_cfg[cur_start][cur_end]
        mid_idx = (cur_start + cur_end) // 2
        segments.append(
            IntervalSegment(
                lo=int(shapes[cur_start]),
                hi=int(shapes[cur_end]),
                config_index=cfg_idx,
                representative=int(shapes[mid_idx]),
                regret_ms=float(cost[cur_start][cur_end]),
            )
        )
        cur_end = cur_start - 1
    segments.reverse()
    return segments


def allocate_budget(
    group_regrets: Mapping[str, Sequence[float]],
    total_budget: int,
) -> Dict[str, int]:
    """Allocate a total bucket budget across multiple static operator groups via greedy marginal gain."""
    if not group_regrets:
        return {}
    alloc = {g: 1 for g in group_regrets}
    remaining = max(0, int(total_budget) - len(alloc))
    while remaining > 0:
        best_group = None
        best_gain = 0.0
        for g, curve in group_regrets.items():
            cur_k = alloc[g]
            if cur_k < len(curve):
                gain = float(curve[cur_k - 1] - curve[cur_k])
                if gain > best_gain:
                    best_gain = gain
                    best_group = g
        if best_group is None or best_gain <= 1e-12:
            break
        alloc[best_group] += 1
        remaining -= 1
    return alloc


class IntervalPolicy:
    """Profile-guided interval bucket policy with online fallback for unseen groups."""

    def __init__(
        self,
        dynamic_axis: str,
        groups: Mapping[str, Sequence[IntervalSegment]],
        *,
        fallback_policy: Optional[OnlineBucketPolicy] = None,
        divisibility: int = 16,
        backend: Optional[str] = None,
        op_name: str = "unknown",
    ) -> None:
        self.dynamic_axis = dynamic_axis
        self.groups: Dict[str, Tuple[IntervalSegment, ...]] = {
            str(gk): tuple(sorted(segs, key=lambda s: s.lo))
            for gk, segs in groups.items()
        }
        self.divisibility = max(1, int(divisibility))
        self.backend = backend if backend is not None else _active_backend_name()
        self.op_name = op_name
        self.fallback_policy = fallback_policy or OnlineBucketPolicy(
            divisibility=self.divisibility,
            backend=self.backend,
            op_name=self.op_name,
        )

    @staticmethod
    def format_group_id(
        args: Mapping[str, Any],
        keys: Sequence[str],
        dynamic_axis: str,
        dtypes: Sequence[str],
    ) -> str:
        static_parts = [f"{k}={args[k]}" for k in keys if k != dynamic_axis and k in args]
        dt_part = ",".join(dtypes)
        return f"{'|'.join(static_parts)}::dtypes={dt_part}"

    def lookup_autotune_key(
        self,
        args: Mapping[str, Any],
        keys: Sequence[str],
        *,
        extra_dtypes: Optional[Sequence[str]] = None,
    ) -> BucketLookupResult:
        dtypes = tuple(extra_dtypes) if extra_dtypes is not None else extract_tensor_dtypes(args.values())
        if self.dynamic_axis not in args or type(args[self.dynamic_axis]) is not int:
            return self.fallback_policy.lookup_autotune_key(args, keys, extra_dtypes=dtypes)

        group_id = self.format_group_id(args, keys, self.dynamic_axis, dtypes)
        segs = self.groups.get(group_id)
        if not segs:
            res = self.fallback_policy.lookup_autotune_key(args, keys, extra_dtypes=dtypes)
            return BucketLookupResult(
                bucket_key=res.bucket_key,
                status=DispatchStatus.GUARD_MISMATCH,
                representative_dims=res.representative_dims,
                raw_dims=res.raw_dims,
                clamped_axes=res.clamped_axes,
            )

        val = int(args[self.dynamic_axis])
        status = DispatchStatus.HIT
        clamped: Tuple[str, ...] = ()
        if val < segs[0].lo:
            chosen = segs[0]
            status = DispatchStatus.UNDERFLOW
            clamped = (self.dynamic_axis,)
        elif val > segs[-1].hi:
            chosen = segs[-1]
            status = DispatchStatus.OVERFLOW
            clamped = (self.dynamic_axis,)
        else:
            chosen = segs[-1]
            for seg in segs:
                if seg.lo <= val <= seg.hi:
                    chosen = seg
                    break

        align_sig = (val == 1, (val % self.divisibility) == 0)
        bkey = (
            "ft_pgo_v1",
            self.backend,
            group_id,
            chosen.lo,
            chosen.hi,
            chosen.config_index,
            align_sig[0],
            align_sig[1],
        )
        return BucketLookupResult(
            bucket_key=bkey,
            status=status,
            representative_dims={self.dynamic_axis: chosen.representative},
            raw_dims={k: args[k] for k in keys if k in args},
            clamped_axes=clamped,
        )

    def normalize_entry_scalar(
        self,
        param_name: str,
        value: int,
        *,
        is_autotuned_key: bool,
        all_args: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Any, ...]:
        if not is_autotuned_key or param_name != self.dynamic_axis:
            return self.fallback_policy.normalize_entry_scalar(
                param_name,
                value,
                is_autotuned_key=is_autotuned_key,
                all_args=all_args,
            )
        width = int_width_tag(value)
        is_one = value == 1
        is_aligned = (value % self.divisibility) == 0
        for segs in self.groups.values():
            for seg in segs:
                if seg.lo <= value <= seg.hi:
                    return ("ft_pgo_int", seg.lo, seg.hi, is_one, is_aligned, width)
        return self.fallback_policy.normalize_entry_scalar(
            param_name,
            value,
            is_autotuned_key=is_autotuned_key,
            all_args=all_args,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "flagtree.shape_policy.v1",
            "dynamic_axis": self.dynamic_axis,
            "divisibility": self.divisibility,
            "backend": self.backend,
            "groups": {
                gid: [
                    {
                        "lo": s.lo,
                        "hi": s.hi,
                        "config_index": s.config_index,
                        "representative": s.representative,
                        "regret_ms": s.regret_ms,
                    }
                    for s in segs
                ]
                for gid, segs in self.groups.items()
            },
        }

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "IntervalPolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        groups = {
            gid: [IntervalSegment(**seg) for seg in seg_list]
            for gid, seg_list in data.get("groups", {}).items()
        }
        return cls(
            dynamic_axis=data["dynamic_axis"],
            groups=groups,
            divisibility=int(data.get("divisibility", 16)),
            backend=data.get("backend"),
        )


def warmup_buckets(
    runner_fn: Callable[..., Any],
    axis_ranges: Mapping[str, Tuple[int, int]],
    sample_builder: Callable[[Dict[str, int]], Tuple[Tuple[Any, ...], Dict[str, Any]]],
    *,
    policy: Optional[OnlineBucketPolicy] = None,
    mode: BucketingMode = BucketingMode.AUTO,
    include_unaligned_tail: bool = True,
    divisibility: int = 16,
) -> List[Dict[str, int]]:
    """Pre-warm closure-complete bucket representatives before online traffic.

    Generates the exact set of bucket ceiling points (and optional unaligned
    tail representatives ``bound - 1`` so both aligned and unaligned kernel
    specializations are warm in cache) within ``axis_ranges`` and invokes
    ``runner_fn(*args, **kwargs)`` once per representative coordinate.
    """
    active_policy = policy or OnlineBucketPolicy(mode=mode, divisibility=divisibility)
    axis_points: Dict[str, List[int]] = {}
    for axis, (lo, hi) in axis_ranges.items():
        lo_i, hi_i = max(1, int(lo)), max(1, int(hi))
        if hi_i < lo_i:
            lo_i, hi_i = hi_i, lo_i
        ladder = active_policy._ladder_for(axis)
        reps: List[int] = []
        seen_buckets = set()
        candidates = [p for p in ladder if lo_i <= p <= hi_i]
        _, _ = active_policy.bucket_dim_value(axis, hi_i)
        ceil_lo, _ = active_policy.bucket_dim_value(axis, lo_i)
        for p in sorted(set([lo_i] + candidates + [min(hi_i, ceil_lo), hi_i])):
            b_hi, _ = active_policy.bucket_dim_value(axis, p)
            if b_hi not in seen_buckets:
                seen_buckets.add(b_hi)
                aligned_rep = b_hi if b_hi <= hi_i else hi_i
                reps.append(aligned_rep)
                if include_unaligned_tail and aligned_rep > 2 and (aligned_rep % divisibility == 0):
                    unaligned = max(lo_i, aligned_rep - 1)
                    if unaligned % divisibility != 0 and unaligned != aligned_rep:
                        reps.append(unaligned)
        axis_points[axis] = sorted(set(reps))

    axes = list(axis_points.keys())
    executed: List[Dict[str, int]] = []

    def _recurse(idx: int, current: Dict[str, int]) -> None:
        if idx == len(axes):
            point = dict(current)
            args, kwargs = sample_builder(point)
            runner_fn(*args, **kwargs)
            executed.append(point)
            return
        ax = axes[idx]
        for val in axis_points[ax]:
            current[ax] = val
            _recurse(idx + 1, current)
        current.pop(ax, None)

    _recurse(0, {})
    return executed


_POLICY_FILE_CACHE: Dict[str, IntervalPolicy] = {}


def clear_policy_caches() -> None:
    """Clear cached policy files (useful in unit tests)."""
    _POLICY_FILE_CACHE.clear()


def resolve_tuner_policy(
    tuner: Any,
    *,
    explicit_policy: Optional[Any] = None,
    policy_path: Optional[str] = None,
    mode_str: Optional[str] = None,
    divisibility: int = 16,
) -> Any:
    """Resolve explicit, JSON-configured, or cached AST-guided ``OnlineBucketPolicy``."""
    if explicit_policy is not None:
        return explicit_policy
    if policy_path:
        cached_file = _POLICY_FILE_CACHE.get(policy_path)
        if cached_file is None:
            cached_file = IntervalPolicy.load_json(policy_path)
            _POLICY_FILE_CACHE[policy_path] = cached_file
        return cached_file

    mode = BucketingMode.parse(mode_str)
    slot_key = (mode.value, int(divisibility), _active_backend_name())
    local_cache = getattr(tuner, "_ft_shape_policies", None)
    if isinstance(local_cache, dict) and slot_key in local_cache:
        return local_cache[slot_key]

    from .adjust_kernel_param import infer_bucketable_axes

    jit_fn = getattr(tuner, "jit_function", None) or getattr(tuner, "fn", tuner)
    configs = getattr(tuner, "configs", None)
    keys = getattr(tuner, "keys", None)
    pre_hook_fn = getattr(tuner, "shared_config_pre_hook", None)
    if pre_hook_fn is None and configs:
        pre_hook_fn = getattr(configs[0], "pre_hook", None)

    specs = infer_bucketable_axes(
        jit_fn,
        configs=configs,
        keys=keys,
        pre_hook_fn=pre_hook_fn,
    )
    base_fn = getattr(tuner, "base_fn", jit_fn)
    while not callable(base_fn) and hasattr(base_fn, "fn"):
        base_fn = base_fn.fn
    op_name = getattr(base_fn, "__name__", "unknown")
    policy = OnlineBucketPolicy(
        specs,
        mode=mode,
        divisibility=divisibility,
        op_name=op_name,
    )
    try:
        if not isinstance(local_cache, dict):
            local_cache = {}
            setattr(tuner, "_ft_shape_policies", local_cache)
        local_cache[slot_key] = policy
    except Exception:
        pass
    return policy


__all__ = [
    "BucketCacheBudget",
    "BucketLookupResult",
    "BucketTelemetry",
    "BucketingMode",
    "DECODE_FINE_LADDER",
    "DispatchStatus",
    "GLOBAL_BUCKET_TELEMETRY",
    "IntervalPolicy",
    "IntervalSegment",
    "OnlineBucketPolicy",
    "THROUGHPUT_GEOMETRIC_LADDER",
    "allocate_budget",
    "build_axis_ladder",
    "clear_policy_caches",
    "enforce_disk_cache_budget",
    "extract_tensor_dtypes",
    "geometric_bucket_ceil",
    "group_frontier",
    "int_width_tag",
    "resolve_tuner_policy",
    "warmup_buckets",
]
