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

"""Non-intrusive shape distribution profiler across training and inference workloads.

Supports three workload profiles:
1. ``TRAIN_FWD_BWD``: forward/backward operator separation, transposed weight-gradient
   contraction tracking, activation footprint estimation, and multi-rank token skew.
2. ``INFER_LLM_SERVING``: Prefill vs. Decode automatic classification, KV-cache length
   trajectories, and MoE per-expert token routing skew.
3. ``INFER_RETRIEVAL_BATCH``: 2D ``(Batch, SeqLen)`` distributions, quantiles, and
   bucket-policy synthesis for multi-batch embedding, reranker, multimodal, and CTR models.
"""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .shape_bucketing import (
    BucketingMode,
    OnlineBucketPolicy,
    build_axis_ladder,
    geometric_bucket_ceil,
)


class ProfileMode(str, Enum):
    """Workload profiling mode."""

    TRAIN_FWD_BWD = "train_fwd_bwd"
    INFER_LLM_SERVING = "infer_llm_serving"
    INFER_RETRIEVAL_BATCH = "infer_retrieval_batch"


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = max(0.0, min(1.0, float(q))) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def compute_distribution_stats(values: Sequence[int | float]) -> Dict[str, float]:
    """Compute summary quantiles and tail ratio for a numeric sample sequence."""
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "tail_gt_p95_ratio": 0.0,
        }
    arr = sorted(float(v) for v in values)
    p95 = _quantile(arr, 0.95)
    tail_cnt = sum(1 for v in arr if v > p95)
    return {
        "count": len(arr),
        "min": arr[0],
        "mean": sum(arr) / len(arr),
        "p50": _quantile(arr, 0.50),
        "p90": _quantile(arr, 0.90),
        "p95": p95,
        "p99": _quantile(arr, 0.99),
        "max": arr[-1],
        "tail_gt_p95_ratio": tail_cnt / len(arr),
    }


@dataclass(frozen=True)
class OperatorShapeSample:
    """Single observed operator invocation."""

    op_name: str
    phase: str
    shapes: Tuple[Tuple[int, ...], ...]
    strides: Tuple[Tuple[int, ...], ...] = ()
    dtypes: Tuple[str, ...] = ()
    step: int = 0
    rank: int = 0
    kv_len: Optional[int] = None
    expert_counts: Tuple[int, ...] = ()


@dataclass
class MultiRankSkewSample:
    """Cross-rank token load snapshot for one training or inference step."""

    step: int
    rank_tokens: Tuple[int, ...]

    @property
    def mean_tokens(self) -> float:
        return (sum(self.rank_tokens) / len(self.rank_tokens)) if self.rank_tokens else 0.0

    @property
    def max_tokens(self) -> int:
        return max(self.rank_tokens) if self.rank_tokens else 0

    @property
    def min_tokens(self) -> int:
        return min(self.rank_tokens) if self.rank_tokens else 0

    @property
    def skew_ratio(self) -> float:
        mean = self.mean_tokens
        return (self.max_tokens / mean) if mean > 0 else 1.0


class ShapeProfiler:
    """Unified shape distribution collector with optional ``TorchDispatchMode`` hook."""

    def __init__(
        self,
        mode: ProfileMode = ProfileMode.TRAIN_FWD_BWD,
        *,
        rank: int = 0,
        max_samples: int = 100_000,
    ) -> None:
        self.mode = ProfileMode(mode)
        self.rank = int(rank)
        self.max_samples = max(1, int(max_samples))
        self._lock = threading.Lock()
        self._step: int = 0
        self._in_backward: bool = False
        self.samples: List[OperatorShapeSample] = []
        self._op_phase_counts: Counter = Counter()
        self._batch_seq_hist: Counter = Counter()
        self._prefill_tokens: List[int] = []
        self._decode_batches: List[int] = []
        self._decode_kv_lens: List[int] = []
        self._moe_skew_ratios: List[float] = []
        self._fwd_tokens: List[int] = []
        self._bwd_contraction_k: List[int] = []
        self._rank_skew_samples: List[MultiRankSkewSample] = []
        self._dispatch_ctx: Any = None

    def step(self) -> int:
        """Advance logical step counter."""
        with self._lock:
            self._step += 1
            return self._step

    def set_backward_context(self, in_backward: bool) -> None:
        with self._lock:
            self._in_backward = bool(in_backward)

    def _infer_phase(
        self,
        op_name: str,
        shapes: Sequence[Sequence[int]],
        explicit_phase: Optional[str],
        kv_len: Optional[int],
    ) -> str:
        if explicit_phase:
            return explicit_phase
        low = op_name.lower()
        if self.mode is ProfileMode.TRAIN_FWD_BWD:
            if self._in_backward or "backward" in low or low.endswith(("_bwd", "_grad")):
                return "backward"
            return "forward"
        if self.mode is ProfileMode.INFER_LLM_SERVING:
            if "moe" in low:
                return "moe_routing"
            first_m = shapes[0][0] if (shapes and len(shapes[0]) >= 1) else 1
            if kv_len is not None and first_m <= 64 and kv_len > first_m:
                return "decode"
            if len(shapes) >= 2 and len(shapes[0]) >= 3 and shapes[0][1] == 1 and shapes[1][1] > 1:
                # Attention Q: (B, 1, H, D), K: (B, KV_CTX, H_kv, D)
                return "decode"
            return "prefill" if first_m > 1 else "decode"
        return "batch_infer"

    def record_op(
        self,
        op_name: str,
        shapes: Sequence[Sequence[int]],
        *,
        strides: Sequence[Sequence[int]] = (),
        dtypes: Sequence[str] = (),
        phase: Optional[str] = None,
        kv_len: Optional[int] = None,
        expert_counts: Sequence[int] = (),
    ) -> None:
        """Record one operator's input shape/stride/dtype tuple."""
        norm_shapes = tuple(tuple(int(d) for d in s) for s in shapes)
        norm_strides = tuple(tuple(int(st) for st in s) for s in strides)
        norm_dtypes = tuple(str(dt) for dt in dtypes)
        norm_experts = tuple(int(c) for c in expert_counts)

        with self._lock:
            resolved_phase = self._infer_phase(op_name, norm_shapes, phase, kv_len)
            self._op_phase_counts[(op_name, resolved_phase)] += 1
            if len(self.samples) < self.max_samples:
                self.samples.append(
                    OperatorShapeSample(
                        op_name=op_name,
                        phase=resolved_phase,
                        shapes=norm_shapes,
                        strides=norm_strides,
                        dtypes=norm_dtypes,
                        step=self._step,
                        rank=self.rank,
                        kv_len=int(kv_len) if kv_len is not None else None,
                        expert_counts=norm_experts,
                    )
                )

            # Populate mode-specific analytic series
            if norm_experts:
                mean_e = sum(norm_experts) / len(norm_experts)
                if mean_e > 0:
                    self._moe_skew_ratios.append(max(norm_experts) / mean_e)

            if self.mode is ProfileMode.INFER_LLM_SERVING:
                m_dim = norm_shapes[0][0] if (norm_shapes and len(norm_shapes[0]) >= 1) else 1
                inferred_kv = kv_len
                if inferred_kv is None and len(norm_shapes) >= 2 and len(norm_shapes[1]) >= 2:
                    inferred_kv = norm_shapes[1][1]
                if resolved_phase == "prefill":
                    total_tok = (
                        norm_shapes[0][0] * norm_shapes[0][1]
                        if (norm_shapes and len(norm_shapes[0]) >= 3)
                        else m_dim
                    )
                    self._prefill_tokens.append(int(total_tok))
                elif resolved_phase == "decode":
                    self._decode_batches.append(int(m_dim))
                    if inferred_kv is not None:
                        self._decode_kv_lens.append(int(inferred_kv))

            elif self.mode is ProfileMode.INFER_RETRIEVAL_BATCH:
                if norm_shapes and len(norm_shapes[0]) >= 2:
                    b_sz, s_len = int(norm_shapes[0][0]), int(norm_shapes[0][1])
                    self._batch_seq_hist[(b_sz, s_len)] += 1
                elif norm_shapes and len(norm_shapes[0]) == 1:
                    self._batch_seq_hist[(1, int(norm_shapes[0][0]))] += 1

            elif self.mode is ProfileMode.TRAIN_FWD_BWD:
                if norm_shapes and len(norm_shapes[0]) >= 2:
                    if resolved_phase == "forward":
                        tok = (
                            norm_shapes[0][0] * norm_shapes[0][1]
                            if len(norm_shapes[0]) >= 3
                            else norm_shapes[0][0]
                        )
                        self._fwd_tokens.append(int(tok))
                    else:
                        # In backward weight grad dW = dY^T @ X, A is (N_out, K_tokens)
                        k_dim = norm_shapes[0][-1]
                        self._bwd_contraction_k.append(int(k_dim))

    def record_rank_tokens(self, rank_tokens: Sequence[int], step: Optional[int] = None) -> MultiRankSkewSample:
        """Record per-rank token loads to quantify multi-device straggler skew."""
        with self._lock:
            st = self._step if step is None else int(step)
            sample = MultiRankSkewSample(step=st, rank_tokens=tuple(int(t) for t in rank_tokens))
            self._rank_skew_samples.append(sample)
            return sample

    def __enter__(self) -> "ShapeProfiler":
        try:
            from torch.utils._python_dispatch import TorchDispatchMode
            import torch
        except Exception:
            self._dispatch_ctx = None
            return self

        profiler_ref = self

        class _DispatchCapture(TorchDispatchMode):

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                tensor_shapes = []
                tensor_strides = []
                tensor_dtypes = []

                def _scan(obj: Any) -> None:
                    if isinstance(obj, torch.Tensor):
                        tensor_shapes.append(tuple(int(d) for d in obj.shape))
                        tensor_strides.append(tuple(int(s) for s in obj.stride()))
                        tensor_dtypes.append(str(obj.dtype))
                    elif isinstance(obj, (list, tuple)):
                        for item in obj:
                            _scan(item)

                _scan(args)
                if tensor_shapes:
                    profiler_ref.record_op(
                        str(getattr(func, "__name__", func)),
                        tensor_shapes,
                        strides=tensor_strides,
                        dtypes=tensor_dtypes,
                    )
                return func(*args, **kwargs)

        self._dispatch_ctx = _DispatchCapture()
        self._dispatch_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._dispatch_ctx is not None:
            self._dispatch_ctx.__exit__(exc_type, exc_val, exc_tb)
            self._dispatch_ctx = None

    def recommend_bucket_policy(self, *, divisibility: int = 16) -> OnlineBucketPolicy:
        """Synthesize an ``OnlineBucketPolicy`` with domain bounds tailored to observed traffic."""
        with self._lock:
            if self.mode is ProfileMode.INFER_LLM_SERVING:
                b_mode = BucketingMode.INFERENCE
                bounds: Dict[str, Tuple[int, int]] = {}
                all_m = self._prefill_tokens + self._decode_batches
                if all_m:
                    bounds["M"] = (max(1, min(all_m)), geometric_bucket_ceil(max(all_m), build_axis_ladder(mode=b_mode)))
                if self._decode_kv_lens:
                    bounds["KV_CTX"] = (
                        max(1, min(self._decode_kv_lens)),
                        geometric_bucket_ceil(max(self._decode_kv_lens), build_axis_ladder(mode=b_mode)),
                    )
                return OnlineBucketPolicy(mode=b_mode, divisibility=divisibility, domain_bounds=bounds)

            if self.mode is ProfileMode.INFER_RETRIEVAL_BATCH:
                b_mode = BucketingMode.INFERENCE
                bounds = {}
                if self._batch_seq_hist:
                    bs_vals = [b for (b, _), c in self._batch_seq_hist.items() for _ in range(c)]
                    sq_vals = [s for (_, s), c in self._batch_seq_hist.items() for _ in range(c)]
                    ladder = build_axis_ladder(mode=b_mode)
                    bounds["B"] = (max(1, min(bs_vals)), geometric_bucket_ceil(max(bs_vals), ladder))
                    bounds["S"] = (max(1, min(sq_vals)), geometric_bucket_ceil(max(sq_vals), ladder))
                    tok_vals = [b * s for (b, s) in self._batch_seq_hist.keys()]
                    bounds["M"] = (max(1, min(tok_vals)), geometric_bucket_ceil(max(tok_vals), ladder))
                return OnlineBucketPolicy(mode=b_mode, divisibility=divisibility, domain_bounds=bounds)

            b_mode = BucketingMode.TRAINING
            bounds = {}
            ladder = build_axis_ladder(mode=b_mode)
            if self._fwd_tokens:
                bounds["M"] = (max(1, min(self._fwd_tokens)), geometric_bucket_ceil(max(self._fwd_tokens), ladder))
            if self._bwd_contraction_k:
                bounds["K"] = (
                    max(1, min(self._bwd_contraction_k)),
                    geometric_bucket_ceil(max(self._bwd_contraction_k), ladder),
                )
            return OnlineBucketPolicy(mode=b_mode, divisibility=divisibility, domain_bounds=bounds)

    def summary(self) -> Dict[str, Any]:
        """Export structured multi-mode distribution report."""
        with self._lock:
            op_counts = {f"{op}::{ph}": cnt for (op, ph), cnt in self._op_phase_counts.items()}
            batch_vals = [b for (b, _), c in self._batch_seq_hist.items() for _ in range(c)]
            seq_vals = [s for (_, s), c in self._batch_seq_hist.items() for _ in range(c)]
            skew_vals = [s.skew_ratio for s in self._rank_skew_samples]
            return {
                "schema": "flagtree.shape_profile.v1",
                "mode": self.mode.value,
                "rank": self.rank,
                "total_steps": self._step,
                "total_samples": len(self.samples),
                "operator_phase_counts": op_counts,
                "llm_serving": {
                    "prefill_tokens": compute_distribution_stats(self._prefill_tokens),
                    "decode_batch_size": compute_distribution_stats(self._decode_batches),
                    "decode_kv_len": compute_distribution_stats(self._decode_kv_lens),
                    "moe_expert_skew": compute_distribution_stats(self._moe_skew_ratios),
                },
                "retrieval_batch": {
                    "batch_size": compute_distribution_stats(batch_vals),
                    "seq_len": compute_distribution_stats(seq_vals),
                    "unique_2d_shapes": len(self._batch_seq_hist),
                    "top_2d_shapes": [
                        {"batch": b, "seq_len": s, "count": c}
                        for (b, s), c in self._batch_seq_hist.most_common(16)
                    ],
                },
                "training": {
                    "forward_tokens": compute_distribution_stats(self._fwd_tokens),
                    "backward_contraction_k": compute_distribution_stats(self._bwd_contraction_k),
                    "multi_rank_skew_ratio": compute_distribution_stats(skew_vals),
                },
            }

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.summary(), indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "MultiRankSkewSample",
    "OperatorShapeSample",
    "ProfileMode",
    "ShapeProfiler",
    "compute_distribution_stats",
]
