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

"""Graph- and batch-level shape bucket guards for training and inference.

Provides three complementary mechanisms:
1. ``SymbolicBucketGuard``: Zero-padding segmented ``torch._dynamo.mark_dynamic``
   range guards with coupled forward-backward bucket pairing and activation-memory
   budget pruning (training forward/backward, LLM prefill, variable-length encoders).
2. ``DecodeGraphBucketRunner``: Static batch-bucket Device Graph pool with
   ``seqused_k = 0`` dummy-slot masking and RL rollout-to-training graph lifecycle
   management for low-latency autoregressive Decode.
3. ``TokenBudgetBatcher``: Length-sorted, mask-preserving token-budget batcher
   for high-throughput multi-batch retrieval, reranker, multimodal, and CTR inference.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .shape_bucketing import (
    BucketingMode,
    OnlineBucketPolicy,
    build_axis_ladder,
    geometric_bucket_ceil,
)


DEFAULT_DECODE_BATCH_BUCKETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class BucketSegmentPlan:
    """Closed symbolic interval ``[lo, hi]`` and memory guard plan for one step."""

    axis_intervals: Dict[str, Tuple[int, int]]
    estimated_activation_bytes: int = 0
    memory_capped: bool = False
    recommended_microbatch_splits: int = 1


class SymbolicBucketGuard:
    """Zero-padding segmented dynamic-shape guard for ``torch.compile`` and training."""

    def __init__(
        self,
        policy: Optional[OnlineBucketPolicy] = None,
        *,
        mode: BucketingMode = BucketingMode.AUTO,
        activation_budget_bytes: Optional[int] = None,
        bytes_per_token_activation: int = 0,
    ) -> None:
        self.policy = policy or OnlineBucketPolicy(mode=mode)
        self.activation_budget_bytes = (
            int(activation_budget_bytes)
            if (activation_budget_bytes is not None and activation_budget_bytes > 0)
            else None
        )
        self.bytes_per_token_activation = max(0, int(bytes_per_token_activation))
        self._paired_step_plans: Dict[int, BucketSegmentPlan] = {}

    def segment_bounds(self, axis_name: str, value: int) -> Tuple[int, int]:
        """Return closed ``(lo, hi)`` bucket interval containing ``value``."""
        iv = max(1, int(value))
        ladder = self.policy._ladder_for(axis_name)
        hi, _ = self.policy.bucket_dim_value(axis_name, iv)
        idx = bisect.bisect_left(ladder, hi)
        if idx <= 0:
            lo = 1
        else:
            prev = ladder[idx - 1]
            lo = min(iv, prev + 1)
        hi = max(lo, hi)
        return (int(lo), int(hi))

    def plan_step(
        self,
        dims: Mapping[str, int],
        *,
        batch_size: int = 1,
        seq_axis: str = "S",
        step_id: Optional[int] = None,
    ) -> BucketSegmentPlan:
        """Compute segmented ``[lo, hi]`` bounds with optional activation memory cap."""
        intervals: Dict[str, Tuple[int, int]] = {
            ax: self.segment_bounds(ax, val) for ax, val in dims.items()
        }
        est_bytes = 0
        capped = False
        splits = 1

        if seq_axis in intervals and self.bytes_per_token_activation > 0:
            lo_s, hi_s = intervals[seq_axis]
            eff_b = max(1, int(batch_size))
            est_bytes = eff_b * hi_s * self.bytes_per_token_activation
            if self.activation_budget_bytes is not None and est_bytes > self.activation_budget_bytes:
                max_safe_s = self.activation_budget_bytes // max(1, eff_b * self.bytes_per_token_activation)
                actual_s = max(1, int(dims[seq_axis]))
                if max_safe_s >= actual_s:
                    # Tighten ceiling hi_s so Dynamo/Inductor upper-bound buffers stay under budget!
                    hi_s = max(lo_s, int(max_safe_s))
                    intervals[seq_axis] = (lo_s, hi_s)
                    est_bytes = eff_b * hi_s * self.bytes_per_token_activation
                    capped = True
                else:
                    # Single batch exceeds budget even at exact shape -> recommend microbatch split
                    splits = max(2, int(math.ceil(est_bytes / self.activation_budget_bytes)))
                    capped = True

        plan = BucketSegmentPlan(
            axis_intervals=intervals,
            estimated_activation_bytes=est_bytes,
            memory_capped=capped,
            recommended_microbatch_splits=splits,
        )
        if step_id is not None:
            self._paired_step_plans[int(step_id)] = plan
        return plan

    def get_paired_backward_plan(self, step_id: int) -> Optional[BucketSegmentPlan]:
        """Retrieve and consume the forward bucket plan paired with ``step_id``."""
        return self._paired_step_plans.pop(int(step_id), None)

    def mark_tensor_dynamic(
        self,
        tensor: Any,
        dim_to_axis: Mapping[int, str],
        *,
        plan: Optional[BucketSegmentPlan] = None,
    ) -> BucketSegmentPlan:
        """Apply ``torch._dynamo.mark_dynamic`` using segmented ``[lo, hi]`` bounds (zero padding)."""
        shape = getattr(tensor, "shape", ())
        inferred_dims = {
            axis_name: int(shape[dim_idx])
            for dim_idx, axis_name in dim_to_axis.items()
            if 0 <= dim_idx < len(shape)
        }
        active_plan = plan or self.plan_step(inferred_dims)
        try:
            import torch._dynamo as dynamo
        except Exception:
            dynamo = None

        if dynamo is not None and hasattr(dynamo, "mark_dynamic"):
            for dim_idx, axis_name in dim_to_axis.items():
                if axis_name in active_plan.axis_intervals:
                    lo, hi = active_plan.axis_intervals[axis_name]
                    if hi > lo:
                        try:
                            dynamo.mark_dynamic(tensor, int(dim_idx), min=int(lo), max=int(hi))
                        except Exception:
                            pass
        return active_plan


@dataclass
class _CapturedBucketSlot:
    bucket_batch: int
    static_inputs: Any
    static_kv_lens: List[int]
    static_output: Any = None
    graph_obj: Any = None
    replays: int = 0


class DecodeGraphBucketRunner:
    """Static batch-bucket runner with optional Device Graph capture/replay for Decode.

    Pads batch size ``B`` only up to ``B_bucket in (1, 2, 4, 8, 16, 32, 64, 128)``,
    masks inactive rows ``[B : B_bucket]`` with ``seqused_k = 0``, and slices the
    output back to ``[:B]``. Supports instant graph-pool release when switching
    from RL rollout inference to policy-gradient training in the same process.
    """

    def __init__(
        self,
        step_fn: Callable[[Any, Sequence[int]], Any],
        *,
        batch_buckets: Sequence[int] = DEFAULT_DECODE_BATCH_BUCKETS,
        buffer_allocator: Optional[Callable[[int], Any]] = None,
        buffer_copier: Optional[Callable[[Any, Any, int], None]] = None,
        output_slicer: Optional[Callable[[Any, int], Any]] = None,
        graph_adapter: Optional[Any] = None,
    ) -> None:
        self.step_fn = step_fn
        self.batch_buckets: Tuple[int, ...] = tuple(sorted(set(int(b) for b in batch_buckets if b > 0)))
        if not self.batch_buckets:
            raise ValueError("batch_buckets must contain at least one positive batch size")
        self._allocator = buffer_allocator
        self._copier = buffer_copier
        self._slicer = output_slicer
        self._graph_adapter = graph_adapter
        self._slots: Dict[int, _CapturedBucketSlot] = {}
        self.capture_count: int = 0
        self.replay_count: int = 0
        self.eager_overflow_count: int = 0

    def select_batch_bucket(self, batch_size: int) -> Optional[int]:
        b = max(1, int(batch_size))
        idx = bisect.bisect_left(self.batch_buckets, b)
        if idx < len(self.batch_buckets):
            return self.batch_buckets[idx]
        return None

    def _default_allocate(self, bucket_b: int, sample_inputs: Any) -> Any:
        if self._allocator is not None:
            return self._allocator(bucket_b)
        if hasattr(sample_inputs, "new_zeros") and hasattr(sample_inputs, "shape"):
            shape = (bucket_b,) + tuple(sample_inputs.shape[1:])
            return sample_inputs.new_zeros(shape)
        if isinstance(sample_inputs, list):
            pad_item = sample_inputs[0] if sample_inputs else 0
            return [pad_item for _ in range(bucket_b)]
        return [0] * bucket_b

    def _default_copy(self, dst: Any, src: Any, actual_b: int) -> None:
        if self._copier is not None:
            self._copier(dst, src, actual_b)
            return
        if hasattr(dst, "__setitem__"):
            dst[:actual_b] = src[:actual_b]

    def _default_slice(self, out: Any, actual_b: int) -> Any:
        if self._slicer is not None:
            return self._slicer(out, actual_b)
        return out[:actual_b]

    def run_decode_step(self, batch_inputs: Any, kv_seqlens: Sequence[int]) -> Any:
        """Execute one autoregressive Decode step using the tightest batch-bucket slot."""
        actual_b = len(kv_seqlens)
        if actual_b == 0:
            return batch_inputs
        bucket_b = self.select_batch_bucket(actual_b)
        if bucket_b is None:
            # Overflow above max configured Decode graph bucket -> safe unpadded eager fallback
            self.eager_overflow_count += 1
            return self.step_fn(batch_inputs, list(kv_seqlens))

        slot = self._slots.get(bucket_b)
        if slot is None:
            static_in = self._default_allocate(bucket_b, batch_inputs)
            static_kv = [0] * bucket_b
            self._default_copy(static_in, batch_inputs, actual_b)
            for i in range(actual_b):
                static_kv[i] = int(kv_seqlens[i])
            for i in range(actual_b, bucket_b):
                static_kv[i] = 0  # Dummy slot mask: zero valid KV tokens!

            graph_obj = None
            if self._graph_adapter is not None:
                graph_obj = self._graph_adapter.capture(
                    lambda: self.step_fn(static_in, static_kv)
                )
                static_out = graph_obj.output
            else:
                static_out = self.step_fn(static_in, static_kv)
            slot = _CapturedBucketSlot(
                bucket_batch=bucket_b,
                static_inputs=static_in,
                static_kv_lens=static_kv,
                static_output=static_out,
                graph_obj=graph_obj,
                replays=0,
            )
            self._slots[bucket_b] = slot
            self.capture_count += 1
            return self._default_slice(slot.static_output, actual_b)

        # In-place metadata & input update before replay
        self._default_copy(slot.static_inputs, batch_inputs, actual_b)
        for i in range(actual_b):
            slot.static_kv_lens[i] = int(kv_seqlens[i])
        for i in range(actual_b, bucket_b):
            slot.static_kv_lens[i] = 0

        if slot.graph_obj is not None:
            slot.graph_obj.replay()
        else:
            slot.static_output = self.step_fn(slot.static_inputs, slot.static_kv_lens)
        slot.replays += 1
        self.replay_count += 1
        return self._default_slice(slot.static_output, actual_b)

    def release_graphs(self) -> int:
        """Release all captured Decode graphs and static buffers (e.g. before RL training step)."""
        released = len(self._slots)
        self._slots.clear()
        return released


@dataclass(frozen=True)
class BucketedMicroBatch:
    """One length-clustered micro-batch with explicit valid-token mask."""

    original_indices: Tuple[int, ...]
    padded_seq_len: int
    padded_sequences: Tuple[Tuple[int, ...], ...]
    attention_mask: Tuple[Tuple[int, ...], ...]
    valid_tokens: int
    total_padded_tokens: int


class TokenBudgetBatcher:
    """Length-sorted, mask-preserving batcher for 500+ retrieval / reranker / CTR models."""

    def __init__(
        self,
        max_tokens_per_batch: int = 4096,
        *,
        max_batch_size: int = 64,
        pad_token_id: int = 0,
        snap_seq_to_bucket: bool = True,
        mode: BucketingMode = BucketingMode.INFERENCE,
    ) -> None:
        self.max_tokens_per_batch = max(1, int(max_tokens_per_batch))
        self.max_batch_size = max(1, int(max_batch_size))
        self.pad_token_id = int(pad_token_id)
        self.snap_seq_to_bucket = bool(snap_seq_to_bucket)
        self._ladder = build_axis_ladder(mode=mode)

    def _target_seq_len(self, max_len_in_batch: int) -> int:
        if not self.snap_seq_to_bucket:
            return max(1, int(max_len_in_batch))
        return geometric_bucket_ceil(max(1, int(max_len_in_batch)), self._ladder)

    def build_batches(self, sequences: Sequence[Sequence[int]]) -> List[BucketedMicroBatch]:
        """Sort sequences by length, pack under token budget, and emit explicit masks."""
        if not sequences:
            return []
        indexed = sorted(enumerate(sequences), key=lambda pair: len(pair[1]))
        batches: List[BucketedMicroBatch] = []
        cur_indices: List[int] = []
        cur_seqs: List[Sequence[int]] = []
        cur_max_len = 0

        def _flush() -> None:
            if not cur_seqs:
                return
            pad_len = self._target_seq_len(cur_max_len)
            padded_rows: List[Tuple[int, ...]] = []
            mask_rows: List[Tuple[int, ...]] = []
            valid_tok = 0
            for seq in cur_seqs:
                slen = len(seq)
                valid_tok += slen
                pad_needed = pad_len - slen
                padded_rows.append(tuple(int(x) for x in seq) + (self.pad_token_id,) * pad_needed)
                mask_rows.append((1,) * slen + (0,) * pad_needed)
            batches.append(
                BucketedMicroBatch(
                    original_indices=tuple(cur_indices),
                    padded_seq_len=pad_len,
                    padded_sequences=tuple(padded_rows),
                    attention_mask=tuple(mask_rows),
                    valid_tokens=valid_tok,
                    total_padded_tokens=len(cur_seqs) * pad_len,
                )
            )

        for orig_idx, seq in indexed:
            cand_max = max(cur_max_len, len(seq))
            cand_pad = self._target_seq_len(cand_max)
            cand_count = len(cur_seqs) + 1
            if cur_seqs and (
                cand_count > self.max_batch_size
                or cand_count * cand_pad > self.max_tokens_per_batch
            ):
                _flush()
                cur_indices = []
                cur_seqs = []
                cur_max_len = 0
                cand_max = len(seq)

            cur_indices.append(orig_idx)
            cur_seqs.append(seq)
            cur_max_len = cand_max

        _flush()
        return batches

    @staticmethod
    def restore_order(
        batch_outputs: Sequence[Sequence[Any]],
        batches: Sequence[BucketedMicroBatch],
    ) -> List[Any]:
        """Scatter micro-batch outputs back into the caller's original request order."""
        total = sum(len(b.original_indices) for b in batches)
        restored: List[Any] = [None] * total
        for outs, bmeta in zip(batch_outputs, batches):
            for row_out, orig_idx in zip(outs, bmeta.original_indices):
                restored[orig_idx] = row_out
        return restored


class TorchDeviceGraphAdapter:
    """Multi-vendor Device Graph adapter supporting CUDA, Huawei Ascend NPU, XPU, and MLU."""

    @staticmethod
    def capture(step_callable: Callable[[], Any]) -> Optional[Any]:
        try:
            import torch
        except ImportError:
            return None

        # Probe CUDA, Huawei Ascend NPU (torch_npu), XPU, MLU, MUSA graph APIs
        backends = (
            ("npu", "NPUGraph", "graph"),
            ("cuda", "CUDAGraph", "graph"),
            ("xpu", "XPUGraph", "graph"),
            ("mlu", "MLUGraph", "graph"),
            ("musa", "MUSAGraph", "graph"),
        )
        for mod_name, graph_cls_name, ctx_name in backends:
            dev_mod = getattr(torch, mod_name, None)
            if dev_mod is None or not callable(getattr(dev_mod, "is_available", None)):
                continue
            try:
                if not dev_mod.is_available():
                    continue
                graph_cls = getattr(dev_mod, graph_cls_name, None)
                graph_ctx = getattr(dev_mod, ctx_name, None)
                if graph_cls is None or graph_ctx is None:
                    continue
                # Warmup stream before capture
                _ = step_callable()
                if hasattr(dev_mod, "synchronize"):
                    dev_mod.synchronize()
                g = graph_cls()
                with graph_ctx(g):
                    captured_out = step_callable()
                if hasattr(dev_mod, "synchronize"):
                    dev_mod.synchronize()

                class _BoundGraph:
                    def __init__(self, raw_g: Any, out: Any) -> None:
                        self._g = raw_g
                        self.output = out

                    def replay(self) -> None:
                        self._g.replay()

                return _BoundGraph(g, captured_out)
            except Exception:
                return None
        return None


__all__ = [
    "BucketSegmentPlan",
    "BucketedMicroBatch",
    "DEFAULT_DECODE_BATCH_BUCKETS",
    "DecodeGraphBucketRunner",
    "SymbolicBucketGuard",
    "TokenBudgetBatcher",
    "TorchDeviceGraphAdapter",
]
