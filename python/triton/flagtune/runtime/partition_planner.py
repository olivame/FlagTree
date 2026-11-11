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

"""Calibrated multi-device partition and comm-compute overlap planner.

Combines bucketed operator execution profiles (FlagGems + FlagTree + TLE fused
remote kernels) with piecewise alpha-beta collective communication models
(FlagCX / torch.distributed) to select optimal parallel execution plans for
both online/batch inference and distributed forward-backward training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class WorkloadKind(str, Enum):
    """Target workload regime for multi-device partition planning."""

    INFERENCE = "inference"
    TRAINING = "training"


class ParallelStrategyKind(str, Enum):
    """Candidate parallelization topologies across devices."""

    SINGLE_DEVICE = "single_device"
    REPLICA_DP = "replica_dp"
    DDP = "ddp"
    ZERO_FSDP = "zero_fsdp"
    TENSOR_PARALLEL = "tp"
    EXPERT_PARALLEL = "ep"
    PIPELINE_PARALLEL = "pp"
    HYBRID_DP_TP = "dp_tp"


class CollectiveKind(str, Enum):
    """Collective communication primitives modeled via FlagCX curves."""

    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER = "all_gather"
    ALL_TO_ALL = "all_to_all"
    P2P = "p2p"


@dataclass(frozen=True)
class AlphaBetaSegment:
    """Linear latency segment ``time_us = alpha_us + bytes * beta_us_per_byte``."""

    max_bytes: int
    alpha_us: float
    bandwidth_gbps: float

    @property
    def beta_us_per_byte(self) -> float:
        bw_bytes_per_us = max(self.bandwidth_gbps, 1e-6) * 1e3
        return 1.0 / bw_bytes_per_us


@dataclass
class PiecewiseAlphaBetaCurve:
    """Piecewise alpha-beta collective latency curve across message sizes."""

    segments: Tuple[AlphaBetaSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("PiecewiseAlphaBetaCurve requires at least one segment")
        ordered = tuple(sorted(self.segments, key=lambda s: s.max_bytes))
        self.segments = ordered

    def select_segment(self, message_bytes: int) -> AlphaBetaSegment:
        nbytes = max(0, int(message_bytes))
        for seg in self.segments:
            if nbytes <= seg.max_bytes:
                return seg
        return self.segments[-1]

    def predict_raw_us(self, effective_bytes: float) -> float:
        if effective_bytes <= 0:
            return 0.0
        seg = self.select_segment(int(math.ceil(effective_bytes)))
        return float(seg.alpha_us + effective_bytes * seg.beta_us_per_byte)

    @classmethod
    def fit_from_samples(
        cls,
        samples: Sequence[Tuple[int, float]],
        *,
        boundaries: Sequence[int] = (64 * 1024, 4 * 1024 * 1024, 1 << 60),
    ) -> "PiecewiseAlphaBetaCurve":
        """Fit a piecewise alpha-beta curve from ``(message_bytes, latency_us)`` samples."""
        if not samples:
            raise ValueError("Cannot fit PiecewiseAlphaBetaCurve from empty samples")

        cleaned = sorted((max(1, int(b)), max(0.01, float(t))) for b, t in samples)
        built: List[AlphaBetaSegment] = []
        lower = 0
        fallback_alpha = cleaned[0][1]
        fallback_bw = max(1.0, (cleaned[-1][0] / max(cleaned[-1][1], 1e-6)) * 1e-3)

        for upper in boundaries:
            bucket = [(b, t) for b, t in cleaned if lower < b <= upper]
            if len(bucket) >= 2:
                mean_b = sum(b for b, _ in bucket) / len(bucket)
                mean_t = sum(t for _, t in bucket) / len(bucket)
                var_b = sum((b - mean_b) ** 2 for b, _ in bucket)
                if var_b > 0:
                    cov_bt = sum((b - mean_b) * (t - mean_t) for b, t in bucket)
                    beta = max(1e-7, cov_bt / var_b)
                    alpha = max(0.05, mean_t - beta * mean_b)
                    bw_gbps = max(0.1, 1.0 / (beta * 1e3))
                else:
                    alpha = max(0.05, mean_t * 0.2)
                    bw_gbps = fallback_bw
                fallback_alpha, fallback_bw = alpha, bw_gbps
            elif len(bucket) == 1:
                b0, t0 = bucket[0]
                alpha = min(fallback_alpha, t0 * 0.5)
                rem = max(t0 - alpha, 1e-3)
                bw_gbps = max(0.1, (b0 / rem) * 1e-3)
                fallback_alpha, fallback_bw = alpha, bw_gbps
            else:
                alpha, bw_gbps = fallback_alpha, fallback_bw

            built.append(AlphaBetaSegment(max_bytes=int(upper), alpha_us=float(alpha), bandwidth_gbps=float(bw_gbps)))
            lower = upper

        return cls(tuple(built))


class CollectiveCostModel:
    """Calibrated FlagCX / distributed collective latency estimator."""

    def __init__(
        self,
        curves: Optional[Mapping[CollectiveKind, PiecewiseAlphaBetaCurve]] = None,
        *,
        intra_node_bandwidth_gbps: float = 180.0,
        base_latency_us: float = 8.0,
    ) -> None:
        default_curve = PiecewiseAlphaBetaCurve(
            (
                AlphaBetaSegment(max_bytes=64 * 1024, alpha_us=base_latency_us, bandwidth_gbps=intra_node_bandwidth_gbps * 0.35),
                AlphaBetaSegment(max_bytes=4 * 1024 * 1024, alpha_us=base_latency_us * 1.5, bandwidth_gbps=intra_node_bandwidth_gbps * 0.78),
                AlphaBetaSegment(max_bytes=1 << 60, alpha_us=base_latency_us * 2.2, bandwidth_gbps=intra_node_bandwidth_gbps),
            )
        )
        self._curves: Dict[CollectiveKind, PiecewiseAlphaBetaCurve] = {
            kind: default_curve for kind in CollectiveKind
        }
        if curves:
            for k, v in curves.items():
                self._curves[CollectiveKind(k)] = v

    def set_curve(self, kind: CollectiveKind, curve: PiecewiseAlphaBetaCurve) -> None:
        self._curves[CollectiveKind(kind)] = curve

    def estimate_ms(
        self,
        kind: CollectiveKind,
        message_bytes: int,
        group_size: int,
        *,
        skew_factor: float = 1.0,
    ) -> float:
        """Predict collective completion time in milliseconds."""
        if group_size <= 1 or message_bytes <= 0:
            return 0.0

        ckind = CollectiveKind(kind)
        curve = self._curves[ckind]
        ring_ratio = (group_size - 1.0) / float(group_size)
        if ckind is CollectiveKind.ALL_REDUCE:
            effective_bytes = 2.0 * ring_ratio * float(message_bytes)
            hop_alpha_scale = 1.0 + 0.18 * math.log2(max(2, group_size))
        elif ckind in (CollectiveKind.REDUCE_SCATTER, CollectiveKind.ALL_GATHER):
            effective_bytes = ring_ratio * float(message_bytes)
            hop_alpha_scale = 1.0 + 0.10 * math.log2(max(2, group_size))
        elif ckind is CollectiveKind.ALL_TO_ALL:
            effective_bytes = ring_ratio * float(message_bytes) * max(1.0, float(skew_factor))
            hop_alpha_scale = 1.0 + 0.25 * math.log2(max(2, group_size))
        else:
            effective_bytes = float(message_bytes)
            hop_alpha_scale = 1.0

        seg = curve.select_segment(int(math.ceil(effective_bytes)))
        us = seg.alpha_us * hop_alpha_scale + effective_bytes * seg.beta_us_per_byte
        return us * 1e-3

    def to_dict(self) -> Dict[str, Any]:
        return {
            kind.value: [asdict(seg) for seg in curve.segments]
            for kind, curve in self._curves.items()
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CollectiveCostModel":
        curves: Dict[CollectiveKind, PiecewiseAlphaBetaCurve] = {}
        for key, seg_list in payload.items():
            kind = CollectiveKind(key)
            curves[kind] = PiecewiseAlphaBetaCurve(
                tuple(AlphaBetaSegment(**seg) for seg in seg_list)
            )
        return cls(curves=curves)


@dataclass(frozen=True)
class OperatorBucketMeasurement:
    """Measured kernel latency at a representative shape bucket."""

    op_name: str
    bucket_tokens: int
    hidden_size: int
    tp_degree: int
    latency_ms: float
    is_tle_fused: bool = False


class BucketedOperatorCostModel:
    """Compute cost estimator calibrated by FlagGems/FlagTree bucket measurements."""

    def __init__(
        self,
        *,
        peak_bf16_tflops: float = 280.0,
        hbm_bandwidth_gbps: float = 1600.0,
        kernel_launch_us: float = 6.5,
        tle_overlap_efficiency: float = 0.78,
        tle_sm_overhead: float = 0.04,
        measurements: Sequence[OperatorBucketMeasurement] = (),
    ) -> None:
        self.peak_bf16_tflops = max(1.0, float(peak_bf16_tflops))
        self.hbm_bandwidth_gbps = max(10.0, float(hbm_bandwidth_gbps))
        self.kernel_launch_us = max(0.0, float(kernel_launch_us))
        self.tle_overlap_efficiency = min(0.98, max(0.0, float(tle_overlap_efficiency)))
        self.tle_sm_overhead = max(0.0, float(tle_sm_overhead))
        self._table: Dict[Tuple[str, int, int, bool], List[Tuple[int, float]]] = {}
        for m in measurements:
            self.add_measurement(m)

    def add_measurement(self, sample: OperatorBucketMeasurement) -> None:
        key = (
            str(sample.op_name),
            int(sample.hidden_size),
            int(sample.tp_degree),
            bool(sample.is_tle_fused),
        )
        pts = self._table.setdefault(key, [])
        pts.append((max(1, int(sample.bucket_tokens)), max(1e-5, float(sample.latency_ms))))
        pts.sort(key=lambda item: item[0])

    def lookup_interpolated_ms(
        self,
        op_name: str,
        tokens: int,
        hidden_size: int,
        tp_degree: int,
        *,
        is_tle_fused: bool = False,
    ) -> Optional[float]:
        pts = self._table.get((op_name, int(hidden_size), int(tp_degree), bool(is_tle_fused)))
        if not pts:
            return None
        t = max(1, int(tokens))
        if t <= pts[0][0]:
            b0, lat0 = pts[0]
            return lat0 * max(0.35, float(t) / float(b0))
        if t >= pts[-1][0]:
            b1, lat1 = pts[-1]
            return lat1 * (float(t) / float(b1))
        for (b_lo, l_lo), (b_hi, l_hi) in zip(pts[:-1], pts[1:]):
            if b_lo <= t <= b_hi:
                if b_hi == b_lo:
                    return l_hi
                w = (t - b_lo) / float(b_hi - b_lo)
                return l_lo + w * (l_hi - l_lo)
        return pts[-1][1]

    def roofline_gemm_ms(
        self,
        m_tokens: int,
        n_dim: int,
        k_dim: int,
        dtype_bytes: int = 2,
    ) -> float:
        m = max(1, int(m_tokens))
        n = max(1, int(n_dim))
        k = max(1, int(k_dim))
        flops = 2.0 * m * n * k
        # Small-M or overly sharded GEMMs suffer tile wave quantization & lower tensor-core utilization
        m_util = min(1.0, (m / (m + 48.0)) + 0.15)
        shape_util = min(1.0, min(n, k) / (min(n, k) + 256.0) + 0.18)
        eff_tflops = self.peak_bf16_tflops * (2.0 / max(1, dtype_bytes)) * max(0.12, m_util * shape_util)
        comp_ms = (flops / (eff_tflops * 1e12)) * 1e3

        bytes_moved = float((m * k + k * n + m * n) * dtype_bytes)
        mem_ms = (bytes_moved / (self.hbm_bandwidth_gbps * 1e9)) * 1e3
        launch_ms = self.kernel_launch_us * 1e-3
        return max(comp_ms, mem_ms) + launch_ms

    def fuse_tle_overlap_ms(self, compute_ms: float, comm_ms: float) -> Tuple[float, float]:
        """Return ``(total_fused_ms, exposed_comm_ms)`` for TLE tile-pipelined GEMM+Comm."""
        if comm_ms <= 0.0:
            return compute_ms, 0.0
        adj_comp = compute_ms * (1.0 + self.tle_sm_overhead)
        dominant = max(adj_comp, comm_ms)
        residual = (1.0 - self.tle_overlap_efficiency) * min(adj_comp, comm_ms)
        total = dominant + residual
        exposed_comm = max(0.0, total - compute_ms)
        return total, exposed_comm


@dataclass(frozen=True)
class HardwareClusterSpec:
    """Cluster resource specification across accelerators."""

    num_devices: int
    device_memory_bytes: int
    memory_safety_margin: float = 0.90
    supports_tle: bool = False
    backend: str = "generic"

    @property
    def usable_memory_bytes(self) -> int:
        return int(self.device_memory_bytes * self.memory_safety_margin)


@dataclass(frozen=True)
class ModelArchitectureSpec:
    """Unified Transformer / Retrieval / MoE / Recommendation model descriptor."""

    name: str
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    num_experts: int = 1
    top_k_experts: int = 1
    dtype_bytes: int = 2

    @property
    def effective_kv_heads(self) -> int:
        return int(self.num_kv_heads or self.num_attention_heads)

    @property
    def head_dim(self) -> int:
        return max(1, self.hidden_size // max(1, self.num_attention_heads))

    @property
    def attn_params_per_layer(self) -> int:
        q_proj = self.hidden_size * self.hidden_size
        kv_proj = 2 * self.hidden_size * (self.effective_kv_heads * self.head_dim)
        o_proj = self.hidden_size * self.hidden_size
        return q_proj + kv_proj + o_proj

    @property
    def ffn_params_per_expert(self) -> int:
        # SwiGLU / gated MLP has 3 projections (gate, up, down)
        return 3 * self.hidden_size * self.ffn_hidden_size

    @property
    def total_params(self) -> int:
        embed = 2 * self.vocab_size * self.hidden_size
        per_layer = self.attn_params_per_layer + max(1, self.num_experts) * self.ffn_params_per_expert
        return embed + self.num_layers * per_layer

    @property
    def kv_bytes_per_token_per_layer(self) -> int:
        return 2 * self.effective_kv_heads * self.head_dim * self.dtype_bytes


@dataclass(frozen=True)
class InferenceConstraints:
    """Workload parameters and SLA constraints for online or batch inference."""

    batch_size: int
    prefill_seq_len: int
    decode_kv_len: int = 0
    output_tokens: int = 0
    max_concurrent_kv_tokens: int = 0
    ttft_sla_ms: Optional[float] = None
    tpot_sla_ms: Optional[float] = None
    moe_dispatch_skew: float = 1.0


@dataclass(frozen=True)
class TrainingConstraints:
    """Workload parameters and memory/overlap knobs for distributed training."""

    global_batch_size: int
    seq_len: int
    micro_batch_size: Optional[int] = None
    optimizer_state_multiplier: float = 6.0
    activation_checkpointing: bool = False
    ddp_bucket_cap_mb: float = 25.0
    ddp_backward_overlap: float = 0.78
    fsdp_prefetch_overlap: float = 0.68
    moe_dispatch_skew: float = 1.0


@dataclass(frozen=True)
class PartitionPlanCandidate:
    """Evaluated parallel execution candidate with latency, memory, and throughput metrics."""

    workload_kind: WorkloadKind
    strategy: ParallelStrategyKind
    dp_degree: int
    tp_degree: int
    pp_degree: int
    ep_degree: int
    use_tle_overlap: bool
    devices_used: int
    per_device_memory_bytes: int
    memory_breakdown_bytes: Dict[str, int]
    fits_memory: bool
    meets_sla: bool
    step_latency_ms: float
    ttft_ms: float
    tpot_ms: float
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float
    compute_time_ms: float
    comm_time_ms: float
    exposed_comm_time_ms: float
    pipeline_bubble_ratio: float = 0.0
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def feasible(self) -> bool:
        return self.fits_memory and self.meets_sla

    @property
    def plan_id(self) -> str:
        tle_sfx = "+tle" if self.use_tle_overlap else ""
        return (
            f"{self.strategy.value}(dp={self.dp_degree},tp={self.tp_degree},"
            f"pp={self.pp_degree},ep={self.ep_degree}{tle_sfx})"
        )


@dataclass
class PartitionDecisionReport:
    """Complete evaluation report across feasible and rejected partition candidates."""

    model_name: str
    workload_kind: WorkloadKind
    num_devices: int
    recommended: Optional[PartitionPlanCandidate]
    ranked_candidates: List[PartitionPlanCandidate]
    rejected_candidates: List[PartitionPlanCandidate]

    def to_dict(self) -> Dict[str, Any]:
        def _ser(c: Optional[PartitionPlanCandidate]) -> Optional[Dict[str, Any]]:
            if c is None:
                return None
            raw = asdict(c)
            raw["workload_kind"] = c.workload_kind.value
            raw["strategy"] = c.strategy.value
            raw["plan_id"] = c.plan_id
            raw["feasible"] = c.feasible
            return raw

        return {
            "model_name": self.model_name,
            "workload_kind": self.workload_kind.value,
            "num_devices": self.num_devices,
            "recommended": _ser(self.recommended),
            "ranked_candidates": [_ser(c) for c in self.ranked_candidates],
            "rejected_candidates": [_ser(c) for c in self.rejected_candidates],
        }

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _divisors(n: int) -> List[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


class PartitionPlanner:
    """Dual-mode (Inference + Training) calibrated partition & overlap planner."""

    def __init__(
        self,
        cluster: HardwareClusterSpec,
        compute_model: Optional[BucketedOperatorCostModel] = None,
        collective_model: Optional[CollectiveCostModel] = None,
    ) -> None:
        self.cluster = cluster
        self.compute_model = compute_model or BucketedOperatorCostModel()
        self.collective_model = collective_model or CollectiveCostModel()

    def _layer_roofline_comp_ms(
        self,
        model: ModelArchitectureSpec,
        tokens: int,
        seq_ctx: int,
        tp: int,
        ep: int,
        is_decode_step: bool = False,
    ) -> Tuple[float, float, float]:
        """Return ``(raw_layer_comp_ms, o_ms, ffn_down_ms)`` from roofline model."""
        h = model.hidden_size
        kv_h = max(1, model.effective_kv_heads // tp) * model.head_dim
        qkv_ms = self.compute_model.roofline_gemm_ms(tokens, (h // tp) + 2 * kv_h, h, model.dtype_bytes)
        o_ms = self.compute_model.roofline_gemm_ms(tokens, h, h // tp, model.dtype_bytes)

        attn_flops = 4.0 * tokens * max(1, seq_ctx) * (h // tp)
        kv_per_tok_sharded = max(1, model.kv_bytes_per_token_per_layer // tp)
        if is_decode_step:
            # Decode streams full KV history (batch_size == tokens) * seq_ctx from HBM
            attn_kv_bytes = float(tokens * max(1, seq_ctx) * kv_per_tok_sharded)
        else:
            # FlashAttention prefill tiles KV in SRAM; HBM reads/writes Q, K, V, O once
            attn_kv_bytes = float(tokens * (kv_per_tok_sharded + 2 * (h // tp) * model.dtype_bytes))
        attn_comp_ms = (attn_flops / (self.compute_model.peak_bf16_tflops * 0.62 * 1e12)) * 1e3
        attn_mem_ms = (attn_kv_bytes / (self.compute_model.hbm_bandwidth_gbps * 1e9)) * 1e3
        attn_ms = max(attn_comp_ms, attn_mem_ms) + self.compute_model.kernel_launch_us * 1e-3

        active_exp_tokens = int(math.ceil(tokens * model.top_k_experts / max(1, ep)))
        ffn_tp = max(1, tp if ep == 1 else 1)
        ffn_up_ms = self.compute_model.roofline_gemm_ms(
            active_exp_tokens, 2 * (model.ffn_hidden_size // ffn_tp), h, model.dtype_bytes
        )
        ffn_down_ms = self.compute_model.roofline_gemm_ms(
            active_exp_tokens, h, model.ffn_hidden_size // ffn_tp, model.dtype_bytes
        )
        return qkv_ms + attn_ms + o_ms + ffn_up_ms + ffn_down_ms, o_ms, ffn_down_ms

    def _layer_forward_ms(
        self,
        model: ModelArchitectureSpec,
        tokens: int,
        seq_ctx: int,
        tp: int,
        ep: int,
        use_tle: bool,
        moe_skew: float,
        is_decode_step: bool = False,
    ) -> Tuple[float, float, float]:
        """Return ``(total_layer_ms, compute_ms, exposed_comm_ms)`` for 1 Transformer/Encoder layer."""
        op_tag = "transformer_layer_decode" if is_decode_step else "transformer_layer_fwd"
        interp = self.compute_model.lookup_interpolated_ms(
            op_tag,
            tokens,
            model.hidden_size,
            tp,
            is_tle_fused=use_tle,
        )
        raw_comp_ms, o_ms, ffn_down_ms = self._layer_roofline_comp_ms(
            model, tokens, seq_ctx, tp, ep, is_decode_step=is_decode_step
        )
        if interp is not None:
            comp_ms = interp
            proj_scale = comp_ms / max(1e-6, raw_comp_ms)
            o_ms *= proj_scale
            ffn_down_ms *= proj_scale
        else:
            interp_tp1 = (
                self.compute_model.lookup_interpolated_ms(
                    op_tag,
                    tokens,
                    model.hidden_size,
                    1,
                    is_tle_fused=False,
                )
                if (tp > 1 or ep > 1 or use_tle)
                else None
            )
            if interp_tp1 is not None:
                raw_tp1_ms, _, _ = self._layer_roofline_comp_ms(
                    model, tokens, seq_ctx, 1, 1, is_decode_step=is_decode_step
                )
                mfu_anchor = interp_tp1 / max(1e-6, raw_tp1_ms)
                comp_ms = raw_comp_ms * mfu_anchor
                o_ms *= mfu_anchor
                ffn_down_ms *= mfu_anchor
            else:
                comp_ms = raw_comp_ms

        # Intra-layer collectives: 2 AllReduces per layer for TP (post-O-proj and post-FFN)
        h = model.hidden_size
        tp_msg_bytes = tokens * h * model.dtype_bytes
        raw_tp_comm = 0.0
        if tp > 1:
            ar_one = self.collective_model.estimate_ms(CollectiveKind.ALL_REDUCE, tp_msg_bytes, tp)
            raw_tp_comm = 2.0 * ar_one if ep == 1 else ar_one

        ep_comm = 0.0
        if ep > 1 and model.num_experts > 1:
            a2a_bytes = tokens * model.top_k_experts * h * model.dtype_bytes
            ep_comm = 2.0 * self.collective_model.estimate_ms(
                CollectiveKind.ALL_TO_ALL, a2a_bytes, ep, skew_factor=moe_skew
            )

        if use_tle and tp > 1:
            fused_proj_ms, exposed_tp = self.compute_model.fuse_tle_overlap_ms(o_ms + ffn_down_ms, raw_tp_comm)
            non_fused_comp = max(0.0, comp_ms - (o_ms + ffn_down_ms))
            total_ms = non_fused_comp + fused_proj_ms + ep_comm
            return total_ms, comp_ms, exposed_tp + ep_comm

        total_ms = comp_ms + raw_tp_comm + ep_comm
        return total_ms, comp_ms, raw_tp_comm + ep_comm

    def plan_inference(
        self,
        model: ModelArchitectureSpec,
        constraints: InferenceConstraints,
    ) -> PartitionDecisionReport:
        """Evaluate and rank multi-device inference partition plans."""
        ndev = max(1, self.cluster.num_devices)
        usable_mem = self.cluster.usable_memory_bytes
        b = max(1, constraints.batch_size)
        s_pre = max(1, constraints.prefill_seq_len)
        is_decode_workload = constraints.output_tokens > 0 or constraints.decode_kv_len > 0
        kv_ctx = max(s_pre, constraints.decode_kv_len + s_pre)
        total_kv_tokens = constraints.max_concurrent_kv_tokens or (b * kv_ctx)

        candidates: List[PartitionPlanCandidate] = []
        tle_choices = (False, True) if self.cluster.supports_tle else (False,)

        # Enumerate topology specs: (strategy, dp, tp, pp, ep)
        topologies: List[Tuple[ParallelStrategyKind, int, int, int, int]] = [
            (ParallelStrategyKind.SINGLE_DEVICE, 1, 1, 1, 1),
        ]
        if ndev > 1:
            topologies.append((ParallelStrategyKind.REPLICA_DP, ndev, 1, 1, 1))
            for d in _divisors(ndev):
                if d > 1 and model.num_attention_heads % d == 0:
                    topologies.append((ParallelStrategyKind.TENSOR_PARALLEL, 1, d, 1, 1))
                    if ndev // d > 1:
                        topologies.append((ParallelStrategyKind.HYBRID_DP_TP, ndev // d, d, 1, 1))
                if d > 1 and model.num_layers % d == 0:
                    topologies.append((ParallelStrategyKind.PIPELINE_PARALLEL, 1, 1, d, 1))
                if d > 1 and model.num_experts > 1 and model.num_experts % d == 0:
                    topologies.append((ParallelStrategyKind.EXPERT_PARALLEL, 1, 1, 1, d))

        single_tp1_ttft: Optional[float] = None

        for strat, dp, tp, pp, ep in topologies:
            for use_tle in tle_choices:
                if use_tle and tp <= 1:
                    continue

                devices_used = dp * tp * pp * ep
                sharded_attn_params = model.num_layers * model.attn_params_per_layer // (tp * pp)
                sharded_ffn_params = (
                    model.num_layers
                    * (max(1, model.num_experts) // ep)
                    * model.ffn_params_per_expert
                    // (max(1, tp if ep == 1 else 1) * pp)
                )
                embed_params = (2 * model.vocab_size * model.hidden_size) // (tp * pp)
                weight_bytes = (sharded_attn_params + sharded_ffn_params + embed_params) * model.dtype_bytes

                kv_heads_per_rank = max(1, math.ceil(model.effective_kv_heads / float(tp)))
                kv_bytes_per_tok_rank = (
                    (model.num_layers // pp) * 2 * kv_heads_per_rank * model.head_dim * model.dtype_bytes
                )
                kv_cache_bytes = total_kv_tokens * kv_bytes_per_tok_rank if is_decode_workload else 0
                act_tokens = b * s_pre
                act_bytes = int(act_tokens * model.hidden_size * model.dtype_bytes * (4.0 / tp))

                mem_total = weight_bytes + kv_cache_bytes + act_bytes
                fits_mem = mem_total <= usable_mem

                layers_per_stage = model.num_layers // pp
                pre_layer_ms, pre_comp_ms, pre_exp_comm_ms = self._layer_forward_ms(
                    model, act_tokens, s_pre, tp, ep, use_tle, constraints.moe_dispatch_skew
                )
                pp_p2p_ms = (
                    (pp - 1)
                    * self.collective_model.estimate_ms(
                        CollectiveKind.P2P, act_tokens * model.hidden_size * model.dtype_bytes, 2
                    )
                    if pp > 1
                    else 0.0
                )
                num_micro = max(1, min(b, pp * 2)) if pp > 1 else 1
                bubble_ratio = (pp - 1.0) / float(num_micro + pp - 1) if pp > 1 else 0.0

                ttft_ms = model.num_layers * pre_layer_ms + pp_p2p_ms
                if strat is ParallelStrategyKind.SINGLE_DEVICE and not use_tle:
                    single_tp1_ttft = ttft_ms

                if is_decode_workload:
                    dec_layer_ms, dec_comp_ms, dec_exp_comm_ms = self._layer_forward_ms(
                        model, b, kv_ctx, tp, ep, use_tle, constraints.moe_dispatch_skew, is_decode_step=True
                    )
                    dec_p2p_ms = (
                        (pp - 1)
                        * self.collective_model.estimate_ms(
                            CollectiveKind.P2P, b * model.hidden_size * model.dtype_bytes, 2
                        )
                        if pp > 1
                        else 0.0
                    )
                    tpot_ms = model.num_layers * dec_layer_ms + dec_p2p_ms
                    out_len = max(1, constraints.output_tokens)
                    req_batch_ms = ttft_ms + out_len * tpot_ms
                    tokens_per_batch = b * (s_pre + out_len)
                    comp_total = model.num_layers * (pre_comp_ms + out_len * dec_comp_ms)
                    exp_comm_total = model.num_layers * (pre_exp_comm_ms + out_len * dec_exp_comm_ms) + pp_p2p_ms + out_len * dec_p2p_ms
                else:
                    tpot_ms = 0.0
                    req_batch_ms = ttft_ms / (1.0 - bubble_ratio * 0.5)
                    tokens_per_batch = act_tokens
                    comp_total = model.num_layers * pre_comp_ms
                    exp_comm_total = model.num_layers * pre_exp_comm_ms + pp_p2p_ms

                meets_ttft = constraints.ttft_sla_ms is None or ttft_ms <= constraints.ttft_sla_ms
                meets_tpot = (
                    not is_decode_workload
                    or constraints.tpot_sla_ms is None
                    or tpot_ms <= constraints.tpot_sla_ms
                )
                meets_sla = meets_ttft and meets_tpot

                # Replica-DP scales whole-node throughput linearly by dp with ZERO collective overhead!
                stage_penalty = 1.0 - bubble_ratio
                qps = float(dp * b) / max(1e-6, req_batch_ms * 1e-3) * stage_penalty
                tps = float(dp * tokens_per_batch) / max(1e-6, req_batch_ms * 1e-3) * stage_penalty

                notes: List[str] = []
                if strat is ParallelStrategyKind.REPLICA_DP:
                    notes.append("zero_cross_device_collective_overhead")
                if tp > 1 and single_tp1_ttft is not None and ttft_ms >= single_tp1_ttft * 0.85:
                    notes.append("tp_collective_overhead_diminishes_small_model_scaling")
                if use_tle:
                    notes.append("tle_remote_gemm_comm_overlap_enabled")
                if not fits_mem:
                    notes.append("exceeds_device_memory_ceiling")
                if not meets_sla:
                    notes.append("violates_latency_sla")

                candidates.append(
                    PartitionPlanCandidate(
                        workload_kind=WorkloadKind.INFERENCE,
                        strategy=strat,
                        dp_degree=dp,
                        tp_degree=tp,
                        pp_degree=pp,
                        ep_degree=ep,
                        use_tle_overlap=use_tle,
                        devices_used=devices_used,
                        per_device_memory_bytes=mem_total,
                        memory_breakdown_bytes={
                            "weights": weight_bytes,
                            "kv_cache": kv_cache_bytes,
                            "activations": act_bytes,
                        },
                        fits_memory=fits_mem,
                        meets_sla=meets_sla,
                        step_latency_ms=req_batch_ms,
                        ttft_ms=ttft_ms,
                        tpot_ms=tpot_ms,
                        throughput_tokens_per_sec=tps,
                        throughput_requests_per_sec=qps,
                        compute_time_ms=comp_total * (layers_per_stage / float(model.num_layers)),
                        comm_time_ms=exp_comm_total,
                        exposed_comm_time_ms=exp_comm_total,
                        pipeline_bubble_ratio=bubble_ratio,
                        notes=tuple(notes),
                    )
                )

        feasible = [c for c in candidates if c.feasible]
        rejected = [c for c in candidates if not c.feasible]
        feasible.sort(key=lambda c: (c.throughput_tokens_per_sec, -c.step_latency_ms), reverse=True)
        rejected.sort(key=lambda c: (c.fits_memory, -c.per_device_memory_bytes), reverse=True)
        return PartitionDecisionReport(
            model_name=model.name,
            workload_kind=WorkloadKind.INFERENCE,
            num_devices=ndev,
            recommended=feasible[0] if feasible else None,
            ranked_candidates=feasible,
            rejected_candidates=rejected,
        )

    def plan_training(
        self,
        model: ModelArchitectureSpec,
        constraints: TrainingConstraints,
    ) -> PartitionDecisionReport:
        """Evaluate and rank multi-device forward+backward training partition plans."""
        ndev = max(1, self.cluster.num_devices)
        usable_mem = self.cluster.usable_memory_bytes
        gbs = max(1, constraints.global_batch_size)
        seq_len = max(1, constraints.seq_len)
        tle_choices = (False, True) if self.cluster.supports_tle else (False,)

        topologies: List[Tuple[ParallelStrategyKind, int, int, int, int]] = []
        if ndev == 1:
            topologies.append((ParallelStrategyKind.SINGLE_DEVICE, 1, 1, 1, 1))
        else:
            if gbs >= ndev:
                topologies.append((ParallelStrategyKind.DDP, ndev, 1, 1, 1))
                topologies.append((ParallelStrategyKind.ZERO_FSDP, ndev, 1, 1, 1))
            for d in _divisors(ndev):
                if d > 1 and model.num_attention_heads % d == 0:
                    dp_rem = ndev // d
                    if dp_rem == 1:
                        topologies.append((ParallelStrategyKind.TENSOR_PARALLEL, 1, d, 1, 1))
                    elif gbs >= dp_rem:
                        topologies.append((ParallelStrategyKind.HYBRID_DP_TP, dp_rem, d, 1, 1))
                if d > 1 and model.num_layers % d == 0 and gbs >= d:
                    topologies.append((ParallelStrategyKind.PIPELINE_PARALLEL, 1, 1, d, 1))
                if d > 1 and model.num_experts > 1 and model.num_experts % d == 0:
                    dp_rem = max(1, ndev // d)
                    topologies.append((ParallelStrategyKind.EXPERT_PARALLEL, dp_rem, 1, 1, d))

        candidates: List[PartitionPlanCandidate] = []
        for strat, dp, tp, pp, ep in topologies:
            for use_tle in tle_choices:
                if use_tle and tp <= 1:
                    continue

                devices_used = max(1, dp * tp * pp) if strat is not ParallelStrategyKind.EXPERT_PARALLEL else ndev
                dp_eff = dp
                per_dp_batch = max(1, math.ceil(gbs / float(dp_eff)))
                mbs = min(per_dp_batch, constraints.micro_batch_size or per_dp_batch)
                if pp > 1 and constraints.micro_batch_size is None:
                    mbs = max(1, per_dp_batch // max(4, pp * 2))
                num_micro = max(1, math.ceil(per_dp_batch / float(mbs)))
                micro_tokens = mbs * seq_len

                total_p = model.total_params
                if strat is ParallelStrategyKind.ZERO_FSDP:
                    param_shard = ndev
                elif strat is ParallelStrategyKind.EXPERT_PARALLEL:
                    non_exp = total_p - model.num_layers * model.num_experts * model.ffn_params_per_expert
                    exp_p = (model.num_layers * model.num_experts * model.ffn_params_per_expert) // ep
                    param_shard = 1
                    sharded_params = non_exp + exp_p
                else:
                    param_shard = tp * pp

                if strat is not ParallelStrategyKind.EXPERT_PARALLEL:
                    sharded_params = total_p // max(1, param_shard)

                weight_bytes = sharded_params * model.dtype_bytes
                grad_bytes = sharded_params * model.dtype_bytes
                opt_bytes = int(sharded_params * model.dtype_bytes * constraints.optimizer_state_multiplier)
                # Transient unsharded layer buffer for ZeRO/FSDP
                if strat is ParallelStrategyKind.ZERO_FSDP:
                    layer_p = total_p // max(1, model.num_layers)
                    weight_bytes += 2 * layer_p * model.dtype_bytes

                layers_on_rank = model.num_layers // pp
                act_factor = 2.0 if constraints.activation_checkpointing else 10.0
                # Sequence parallel with TP shards activation memory along tp
                sp_div = tp if tp > 1 else 1
                in_flight_micro = min(num_micro, pp) if pp > 1 else 1
                act_bytes = int(
                    in_flight_micro
                    * layers_on_rank
                    * micro_tokens
                    * model.hidden_size
                    * model.dtype_bytes
                    * act_factor
                    / float(sp_div)
                )

                mem_total = weight_bytes + grad_bytes + opt_bytes + act_bytes
                fits_mem = mem_total <= usable_mem

                # Forward + Backward compute (backward ~= 2.0x forward FLOPs, +0.35x if checkpoint recompute)
                fwd_layer_ms, fwd_comp_only_ms, _ = self._layer_forward_ms(
                    model, micro_tokens, seq_len, tp, ep, False, constraints.moe_dispatch_skew
                )
                bwd_mult = 2.35 if constraints.activation_checkpointing else 2.0
                per_micro_comp_ms = layers_on_rank * fwd_comp_only_ms * (1.0 + bwd_mult)
                total_comp_ms = num_micro * per_micro_comp_ms

                # Communication modeling per strategy
                raw_comm_ms = 0.0
                exposed_comm_ms = 0.0
                notes: List[str] = []

                if tp > 1:
                    # TP + Sequence Parallel: 2 AllGather + 2 ReduceScatter per layer in fwd, and again in bwd
                    sp_bytes = micro_tokens * model.hidden_size * model.dtype_bytes
                    ag_ms = self.collective_model.estimate_ms(CollectiveKind.ALL_GATHER, sp_bytes, tp)
                    rs_ms = self.collective_model.estimate_ms(CollectiveKind.REDUCE_SCATTER, sp_bytes, tp)
                    tp_comm_per_layer_step = 3.0 * (ag_ms + rs_ms)
                    raw_tp_comm = num_micro * layers_on_rank * tp_comm_per_layer_step
                    raw_comm_ms += raw_tp_comm
                    if use_tle:
                        _, exp_tp = self.compute_model.fuse_tle_overlap_ms(total_comp_ms * 0.65, raw_tp_comm)
                        exposed_comm_ms += exp_tp
                        notes.append("tle_gemm_reducescatter_allgather_overlap")
                    else:
                        exposed_comm_ms += raw_tp_comm

                if ep > 1 and model.num_experts > 1:
                    a2a_bytes = micro_tokens * model.top_k_experts * model.hidden_size * model.dtype_bytes
                    # 2 AllToAll in fwd + 2 AllToAll in bwd
                    ep_comm = num_micro * layers_on_rank * 4.0 * self.collective_model.estimate_ms(
                        CollectiveKind.ALL_TO_ALL, a2a_bytes, ep, skew_factor=constraints.moe_dispatch_skew
                    )
                    raw_comm_ms += ep_comm
                    exposed_comm_ms += ep_comm

                if strat in (ParallelStrategyKind.DDP, ParallelStrategyKind.HYBRID_DP_TP, ParallelStrategyKind.EXPERT_PARALLEL) and dp_eff > 1:
                    total_grad_bytes = sharded_params * model.dtype_bytes
                    bucket_bytes = max(1024 * 1024, int(constraints.ddp_bucket_cap_mb * 1024 * 1024))
                    num_buckets = max(1, math.ceil(total_grad_bytes / float(bucket_bytes)))
                    ddp_ar_ms = num_buckets * self.collective_model.estimate_ms(
                        CollectiveKind.ALL_REDUCE, min(bucket_bytes, total_grad_bytes), dp_eff
                    )
                    raw_comm_ms += ddp_ar_ms
                    bwd_window_ms = total_comp_ms * (bwd_mult / (1.0 + bwd_mult))
                    overlapped = min(bwd_window_ms * constraints.ddp_backward_overlap, ddp_ar_ms * 0.85)
                    exposed_comm_ms += max(0.0, ddp_ar_ms - overlapped)
                    notes.append(f"ddp_gradient_buckets={num_buckets}")

                if strat is ParallelStrategyKind.ZERO_FSDP:
                    layer_bytes = (total_p // max(1, model.num_layers)) * model.dtype_bytes
                    # FSDP: AllGather fwd + AllGather bwd + ReduceScatter bwd per layer
                    ag_ms = self.collective_model.estimate_ms(CollectiveKind.ALL_GATHER, layer_bytes, ndev)
                    rs_ms = self.collective_model.estimate_ms(CollectiveKind.REDUCE_SCATTER, layer_bytes, ndev)
                    fsdp_raw = num_micro * model.num_layers * (2.0 * ag_ms + rs_ms)
                    raw_comm_ms += fsdp_raw
                    overlapped = min(total_comp_ms * constraints.fsdp_prefetch_overlap, fsdp_raw * 0.80)
                    exposed_comm_ms += max(0.0, fsdp_raw - overlapped)
                    notes.append("fsdp_sharded_params_grads_opt")

                bubble_ratio = 0.0
                if pp > 1:
                    bubble_ratio = (pp - 1.0) / float(num_micro + pp - 1)
                    p2p_bytes = micro_tokens * model.hidden_size * model.dtype_bytes
                    pp_p2p = num_micro * 2.0 * (pp - 1) * self.collective_model.estimate_ms(
                        CollectiveKind.P2P, p2p_bytes, 2
                    )
                    raw_comm_ms += pp_p2p
                    exposed_comm_ms += pp_p2p
                    notes.append(f"1f1b_microbatches={num_micro}")

                bubble_ms = (total_comp_ms + exposed_comm_ms) * (bubble_ratio / max(1e-4, 1.0 - bubble_ratio))
                step_ms = total_comp_ms + exposed_comm_ms + bubble_ms
                global_tokens = gbs * seq_len
                tps = float(global_tokens) / max(1e-6, step_ms * 1e-3)
                sps = float(gbs) / max(1e-6, step_ms * 1e-3)

                if not fits_mem:
                    notes.append("exceeds_device_memory_ceiling")

                candidates.append(
                    PartitionPlanCandidate(
                        workload_kind=WorkloadKind.TRAINING,
                        strategy=strat,
                        dp_degree=dp_eff,
                        tp_degree=tp,
                        pp_degree=pp,
                        ep_degree=ep,
                        use_tle_overlap=use_tle,
                        devices_used=devices_used,
                        per_device_memory_bytes=mem_total,
                        memory_breakdown_bytes={
                            "weights": weight_bytes,
                            "gradients": grad_bytes,
                            "optimizer": opt_bytes,
                            "activations": act_bytes,
                        },
                        fits_memory=fits_mem,
                        meets_sla=True,
                        step_latency_ms=step_ms,
                        ttft_ms=0.0,
                        tpot_ms=0.0,
                        throughput_tokens_per_sec=tps,
                        throughput_requests_per_sec=sps,
                        compute_time_ms=total_comp_ms,
                        comm_time_ms=raw_comm_ms,
                        exposed_comm_time_ms=exposed_comm_ms,
                        pipeline_bubble_ratio=bubble_ratio,
                        notes=tuple(notes),
                    )
                )

        feasible = [c for c in candidates if c.feasible]
        rejected = [c for c in candidates if not c.feasible]
        feasible.sort(key=lambda c: c.throughput_tokens_per_sec, reverse=True)
        rejected.sort(key=lambda c: c.per_device_memory_bytes)
        return PartitionDecisionReport(
            model_name=model.name,
            workload_kind=WorkloadKind.TRAINING,
            num_devices=ndev,
            recommended=feasible[0] if feasible else None,
            ranked_candidates=feasible,
            rejected_candidates=rejected,
        )


def evaluate_ranking_accuracy(
    report: PartitionDecisionReport,
    measured_throughput_by_plan_id: Mapping[str, float],
) -> Dict[str, Any]:
    """Compare predicted candidate ranking against hardware-measured throughputs."""
    common = [c for c in report.ranked_candidates if c.plan_id in measured_throughput_by_plan_id]
    if not common:
        return {"matched_plans": 0, "top1_match": False, "pairwise_concordance": 0.0}

    pred_best = max(common, key=lambda c: c.throughput_tokens_per_sec).plan_id
    meas_best = max(common, key=lambda c: measured_throughput_by_plan_id[c.plan_id]).plan_id

    concordant = 0
    total_pairs = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            ci, cj = common[i], common[j]
            dp_pred = ci.throughput_tokens_per_sec - cj.throughput_tokens_per_sec
            dp_meas = measured_throughput_by_plan_id[ci.plan_id] - measured_throughput_by_plan_id[cj.plan_id]
            total_pairs += 1
            if dp_pred * dp_meas >= 0:
                concordant += 1

    rel_errors = [
        abs(c.throughput_tokens_per_sec - measured_throughput_by_plan_id[c.plan_id])
        / max(1e-6, measured_throughput_by_plan_id[c.plan_id])
        for c in common
    ]
    return {
        "matched_plans": len(common),
        "predicted_top1": pred_best,
        "measured_top1": meas_best,
        "top1_match": pred_best == meas_best,
        "pairwise_concordance": (concordant / float(total_pairs)) if total_pairs > 0 else 1.0,
        "mean_relative_error": sum(rel_errors) / len(rel_errors),
    }


__all__ = [
    "AlphaBetaSegment",
    "BucketedOperatorCostModel",
    "CollectiveCostModel",
    "CollectiveKind",
    "HardwareClusterSpec",
    "InferenceConstraints",
    "ModelArchitectureSpec",
    "OperatorBucketMeasurement",
    "ParallelStrategyKind",
    "PartitionDecisionReport",
    "PartitionPlanCandidate",
    "PartitionPlanner",
    "PiecewiseAlphaBetaCurve",
    "TrainingConstraints",
    "WorkloadKind",
    "evaluate_ranking_accuracy",
]
