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

"""Backend-agnostic multi-device parallel plan executor (DP / TP / PP / Hybrid).

Materializes an ``ExecutableParallelPlan`` (``DeviceMeshSpec``) onto real
multi-device process groups for:

- **Data Parallel (DP / DDP)**: replicated model with bucketed gradient
  ``all_reduce`` across DP ranks (zero collectives during inference).
- **Tensor Parallel (TP)**: Megatron-style ``ColumnParallelLinear`` +
  ``RowParallelLinear`` Transformer attention and gated MLP with real
  cross-rank ``all_reduce`` per layer (differentiable for training).
- **Pipeline Parallel (PP)**: stage-partitioned multi-layer execution with
  micro-batch P2P ``send``/``recv`` forward and backward schedules.
- **2D Hybrid Parallel (DP x PP, DP x TP)**: hierarchical sub-process-groups
  built directly from ``DeviceMeshSpec``.

Supports both direct accelerator collectives (``nccl`` / ``hccl`` / ``xccl``)
and automatic device<->host staging when running over ``gloo``, keeping all
compute on accelerators while remaining portable across heterogeneous backends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from triton.flagtune.runtime.comm_profiler import (
    _detect_default_device_type,
    _find_free_port,
    _prefer_loaded_triton_in_spawn,
    _resolve_torch_dtype,
    _set_device_for_rank,
    _sync_device,
    needs_host_staging,
)
from triton.flagtune.runtime.partition_planner import (
    DeviceMeshSpec,
    ExecutableParallelPlan,
    ModelArchitectureSpec,
)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cpu":
        return
    mod = getattr(torch, device.type, None)
    if mod is not None and callable(getattr(mod, "reset_peak_memory_stats", None)):
        try:
            mod.reset_peak_memory_stats(device)
        except Exception:
            pass


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cpu":
        return 0.0
    mod = getattr(torch, device.type, None)
    if mod is not None and callable(getattr(mod, "max_memory_allocated", None)):
        try:
            return float(mod.max_memory_allocated(device)) / (1024.0 * 1024.0)
        except Exception:
            return 0.0
    return 0.0


class _CollectiveAdapter:
    """Backend-agnostic collective wrapper with automatic host staging for gloo."""

    def __init__(self, backend: str, device: torch.device) -> None:
        self.backend = (backend or "gloo").strip().lower()
        self.device = device
        self.host_staged = needs_host_staging(self.backend, device.type)
        self.comm_time_ms: float = 0.0
        self.collective_calls: int = 0
        self._pending_works: List[Tuple[Any, Optional[torch.Tensor]]] = []

    def reset_stats(self) -> None:
        self.comm_time_ms = 0.0
        self.collective_calls = 0

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        group: Any = None,
        op: Any = None,
        average_by: int = 1,
    ) -> torch.Tensor:
        if op is None:
            op = dist.ReduceOp.SUM
        t0 = time.perf_counter()
        if self.host_staged:
            buf = tensor.detach().to("cpu", non_blocking=False).contiguous()
            dist.all_reduce(buf, op=op, group=group)
            if average_by > 1:
                buf.div_(float(average_by))
            out = buf.to(self.device, non_blocking=False)
            _sync_device(self.device)
        else:
            out = tensor.clone() if tensor.requires_grad else tensor
            dist.all_reduce(out, op=op, group=group)
            if average_by > 1:
                out.div_(float(average_by))
            _sync_device(self.device)
        self.comm_time_ms += (time.perf_counter() - t0) * 1e3
        self.collective_calls += 1
        return out

    def send(
        self,
        tensor: torch.Tensor,
        *,
        dst: int,
        tag: int = 0,
        async_op: bool = True,
    ) -> None:
        t0 = time.perf_counter()
        if self.host_staged:
            buf = tensor.detach().to("cpu", non_blocking=False).contiguous()
            if async_op:
                work = dist.isend(buf, dst=dst, tag=tag)
                self._pending_works.append((work, buf))
            else:
                dist.send(buf, dst=dst, tag=tag)
        else:
            buf = tensor.detach().contiguous()
            if async_op:
                work = dist.isend(buf, dst=dst, tag=tag)
                self._pending_works.append((work, buf))
            else:
                dist.send(buf, dst=dst, tag=tag)
        self.comm_time_ms += (time.perf_counter() - t0) * 1e3
        self.collective_calls += 1

    def recv(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        src: int,
        tag: int = 0,
    ) -> torch.Tensor:
        t0 = time.perf_counter()
        if self.host_staged:
            buf = torch.empty(tuple(shape), dtype=dtype, device="cpu")
            dist.recv(buf, src=src, tag=tag)
            out = buf.to(self.device, non_blocking=False)
            _sync_device(self.device)
        else:
            out = torch.empty(tuple(shape), dtype=dtype, device=self.device)
            dist.recv(out, src=src, tag=tag)
            _sync_device(self.device)
        self.comm_time_ms += (time.perf_counter() - t0) * 1e3
        self.collective_calls += 1
        return out

    def wait_pending(self) -> None:
        if not self._pending_works:
            return
        t0 = time.perf_counter()
        for work, _buf in self._pending_works:
            work.wait()
        self._pending_works.clear()
        self.comm_time_ms += (time.perf_counter() - t0) * 1e3


class _CopyToTPRegion(torch.autograd.Function):
    """Megatron-LM ``f`` operator: identity in forward, all-reduce in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_group: Any, adapter: _CollectiveAdapter, tp_size: int):
        ctx.tp_group = tp_group
        ctx.adapter = adapter
        ctx.tp_size = tp_size
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.tp_size <= 1:
            return grad_output, None, None, None
        reduced = ctx.adapter.all_reduce(grad_output, group=ctx.tp_group)
        return reduced, None, None, None


class _ReduceFromTPRegion(torch.autograd.Function):
    """Megatron-LM ``g`` operator: all-reduce in forward, identity in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_group: Any, adapter: _CollectiveAdapter, tp_size: int):
        if tp_size <= 1:
            return x
        return adapter.all_reduce(x, group=tp_group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None, None


class RMSNorm(nn.Module):
    """Portable RMSNorm matching modern Transformer blocks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * inv_rms).to(orig_dtype) * self.weight


class ColumnParallelLinear(nn.Module):
    """Output-dimension sharded linear layer for Tensor Parallelism."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_rank: int = 0,
        tp_size: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if out_features % tp_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.out_per_rank = self.out_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_per_rank))
        else:
            self.register_parameter("bias", None)

    def load_sharded_weight(
        self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor] = None
    ) -> None:
        start = self.tp_rank * self.out_per_rank
        end = start + self.out_per_rank
        with torch.no_grad():
            self.weight.copy_(full_weight[start:end, :])
            if self.bias is not None and full_bias is not None:
                self.bias.copy_(full_bias[start:end])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Input-dimension sharded linear layer with cross-rank all-reduce."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group: Any = None,
        adapter: Optional[_CollectiveAdapter] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if in_features % tp_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.tp_group = tp_group
        self.adapter = adapter
        self.in_per_rank = self.in_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_per_rank))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

    def load_sharded_weight(
        self, full_weight: torch.Tensor, full_bias: Optional[torch.Tensor] = None
    ) -> None:
        start = self.tp_rank * self.in_per_rank
        end = start + self.in_per_rank
        with torch.no_grad():
            self.weight.copy_(full_weight[:, start:end])
            if self.bias is not None and full_bias is not None:
                self.bias.copy_(full_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_partial = F.linear(x, self.weight, None)
        if self.tp_size > 1 and self.adapter is not None:
            out = _ReduceFromTPRegion.apply(out_partial, self.tp_group, self.adapter, self.tp_size)
        else:
            out = out_partial
        if self.bias is not None:
            out = out + self.bias
        return out


@dataclass(frozen=True)
class TransformerStackConfig:
    """Architecture descriptor for the executable multi-device Transformer stack."""

    num_layers: int = 8
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 16
    vocab_size: int = 32000
    dtype: str = "bfloat16"
    rms_norm_eps: float = 1e-6
    seed: int = 2026
    name: str = "ParallelTransformerStack"

    @property
    def dtype_bytes(self) -> int:
        return 4 if self.dtype.lower() in ("float32", "fp32") else 2

    def to_model_architecture_spec(self) -> ModelArchitectureSpec:
        return ModelArchitectureSpec(
            name=self.name,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_heads,
            vocab_size=self.vocab_size,
            dtype_bytes=self.dtype_bytes,
        )

    @classmethod
    def from_hf_config(
        cls,
        hf_config: Any,
        *,
        num_layers_override: Optional[int] = None,
        dtype: str = "bfloat16",
        seed: int = 2026,
        name: Optional[str] = None,
    ) -> "TransformerStackConfig":
        """Build a ``TransformerStackConfig`` from a HuggingFace model config."""
        n_layers = int(
            num_layers_override
            if num_layers_override is not None
            else getattr(hf_config, "num_hidden_layers", 12)
        )
        h_size = int(getattr(hf_config, "hidden_size", 1024))
        ffn_size = int(getattr(hf_config, "intermediate_size", h_size * 4))
        n_heads = int(getattr(hf_config, "num_attention_heads", 16))
        v_size = int(getattr(hf_config, "vocab_size", 32000))
        eps = float(getattr(hf_config, "rms_norm_eps", getattr(hf_config, "layer_norm_eps", 1e-6)))
        model_tag = name or str(getattr(hf_config, "model_type", "hf_transformer"))
        return cls(
            num_layers=n_layers,
            hidden_size=h_size,
            ffn_hidden_size=ffn_size,
            num_attention_heads=n_heads,
            vocab_size=v_size,
            dtype=dtype,
            rms_norm_eps=eps,
            seed=seed,
            name=model_tag,
        )


def _init_deterministic_weight(
    shape: Tuple[int, int],
    gen: torch.Generator,
    scale: float,
) -> torch.Tensor:
    return torch.randn(shape, generator=gen, dtype=torch.float32) * scale


class ParallelTransformerBlock(nn.Module):
    """Tensor-parallel Transformer block (RMSNorm + MHA + SwiGLU MLP)."""

    def __init__(
        self,
        cfg: TransformerStackConfig,
        *,
        layer_idx: int = 0,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group: Any = None,
        adapter: Optional[_CollectiveAdapter] = None,
    ) -> None:
        super().__init__()
        if cfg.num_attention_heads % tp_size != 0:
            raise ValueError(
                f"num_attention_heads ({cfg.num_attention_heads}) not divisible by tp_size ({tp_size})"
            )
        self.layer_idx = int(layer_idx)
        self.hidden_size = cfg.hidden_size
        self.num_heads_local = cfg.num_attention_heads // tp_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.tp_group = tp_group
        self.adapter = adapter

        self.norm1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.q_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.hidden_size, tp_rank=tp_rank, tp_size=tp_size
        )
        self.k_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.hidden_size, tp_rank=tp_rank, tp_size=tp_size
        )
        self.v_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.hidden_size, tp_rank=tp_rank, tp_size=tp_size
        )
        self.o_proj = RowParallelLinear(
            cfg.hidden_size,
            cfg.hidden_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            adapter=adapter,
        )

        self.norm2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.gate_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.ffn_hidden_size, tp_rank=tp_rank, tp_size=tp_size
        )
        self.up_proj = ColumnParallelLinear(
            cfg.hidden_size, cfg.ffn_hidden_size, tp_rank=tp_rank, tp_size=tp_size
        )
        self.down_proj = RowParallelLinear(
            cfg.ffn_hidden_size,
            cfg.hidden_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            adapter=adapter,
        )
        self._init_from_deterministic_seed(cfg)

    def _init_from_deterministic_seed(self, cfg: TransformerStackConfig) -> None:
        """Initialize identical deterministic full weights then slice per-rank shard."""
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 10007 * self.layer_idx)
        h = cfg.hidden_size
        ffn = cfg.ffn_hidden_size
        attn_scale = 1.0 / math.sqrt(float(h))
        ffn_scale = 1.0 / math.sqrt(float(ffn))
        resid_scale = 0.25 / math.sqrt(float(max(1, cfg.num_layers)))

        self.q_proj.load_sharded_weight(_init_deterministic_weight((h, h), gen, attn_scale))
        self.k_proj.load_sharded_weight(_init_deterministic_weight((h, h), gen, attn_scale))
        self.v_proj.load_sharded_weight(_init_deterministic_weight((h, h), gen, attn_scale))
        self.o_proj.load_sharded_weight(
            _init_deterministic_weight((h, h), gen, attn_scale * resid_scale)
        )
        self.gate_proj.load_sharded_weight(_init_deterministic_weight((ffn, h), gen, attn_scale))
        self.up_proj.load_sharded_weight(_init_deterministic_weight((ffn, h), gen, attn_scale))
        self.down_proj.load_sharded_weight(
            _init_deterministic_weight((h, ffn), gen, ffn_scale * resid_scale)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h1 = self.norm1(x)
        if self.tp_size > 1 and self.adapter is not None:
            h1 = _CopyToTPRegion.apply(h1, self.tp_group, self.adapter, self.tp_size)

        q = self.q_proj(h1).view(bsz, seq_len, self.num_heads_local, self.head_dim).transpose(1, 2)
        k = self.k_proj(h1).view(bsz, seq_len, self.num_heads_local, self.head_dim).transpose(1, 2)
        v = self.v_proj(h1).view(bsz, seq_len, self.num_heads_local, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        x = x + self.o_proj(attn)

        h2 = self.norm2(x)
        if self.tp_size > 1 and self.adapter is not None:
            h2 = _CopyToTPRegion.apply(h2, self.tp_group, self.adapter, self.tp_size)
        mlp = F.silu(self.gate_proj(h2)) * self.up_proj(h2)
        x = x + self.down_proj(mlp)
        return x


class ParallelTransformerModel(nn.Module):
    """Pipeline-stage slice of a multi-layer ``ParallelTransformerBlock`` stack."""

    def __init__(
        self,
        cfg: TransformerStackConfig,
        *,
        layer_indices: Sequence[int],
        is_last_stage: bool = True,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_group: Any = None,
        adapter: Optional[_CollectiveAdapter] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.is_last_stage = bool(is_last_stage)
        self.layers = nn.ModuleList(
            [
                ParallelTransformerBlock(
                    cfg,
                    layer_idx=idx,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    tp_group=tp_group,
                    adapter=adapter,
                )
                for idx in self.layer_indices
            ]
        )
        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps) if is_last_stage else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        if self.final_norm is not None:
            h = self.final_norm(h)
        return h


def compute_stage_layer_splits(
    num_layers: int,
    pp_size: int,
    stage_layer_counts: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[int, ...], ...]:
    """Return tuple of layer-index tuples assigned to each pipeline stage."""
    p = max(1, int(pp_size))
    if stage_layer_counts is not None:
        counts = [int(c) for c in stage_layer_counts]
        if len(counts) != p or sum(counts) != num_layers or any(c <= 0 for c in counts):
            raise ValueError(
                f"stage_layer_counts={counts} must have length {p}, positive entries, "
                f"and sum to num_layers={num_layers}"
            )
    else:
        base, rem = divmod(num_layers, p)
        counts = [base + (1 if i < rem else 0) for i in range(p)]

    splits: List[Tuple[int, ...]] = []
    cursor = 0
    for c in counts:
        splits.append(tuple(range(cursor, cursor + c)))
        cursor += c
    return tuple(splits)


def _find_rank_group(groups: Tuple[Tuple[int, ...], ...], rank: int) -> Tuple[int, ...]:
    for grp in groups:
        if rank in grp:
            return grp
    return (rank,)


def _create_subgroups_deterministic(
    groups: Tuple[Tuple[int, ...], ...],
    rank: int,
) -> Tuple[Any, Tuple[int, ...]]:
    """Create process groups in identical order across all ranks to avoid deadlock."""
    my_group = _find_rank_group(groups, rank)
    if len(my_group) <= 1:
        return None, my_group
    my_pg = None
    for grp in groups:
        if len(grp) > 1:
            pg = dist.new_group(ranks=list(grp))
            if rank in grp:
                my_pg = pg
    return my_pg, my_group


def _apply_f09_shape_bucketing(enable: bool, mode: str = "inference") -> Optional[Any]:
    """Activate FlagTree F09 dynamic shape bucketing knobs and return an OnlineBucketPolicy."""
    if not enable:
        return None
    os.environ["FLAGTREE_SHAPE_BUCKETING"] = "1"
    os.environ["FLAGTREE_SHAPE_BUCKETING_MODE"] = mode
    try:
        from triton import knobs

        if hasattr(knobs, "runtime") and hasattr(knobs.runtime, "shape_bucketing"):
            knobs.runtime.shape_bucketing = True
    except Exception:
        pass
    try:
        from triton.runtime.shape_bucketing import BucketingMode, OnlineBucketPolicy

        bmode = (
            BucketingMode.TRAINING
            if str(mode).strip().lower() == "training"
            else BucketingMode.INFERENCE
        )
        return OnlineBucketPolicy(mode=bmode, dynamic_axes=["S"])
    except Exception:
        return None


class ParallelPlanExecutor:
    """Runtime executor materializing ``ExecutableParallelPlan`` across devices."""

    def __init__(
        self,
        *,
        plan: ExecutableParallelPlan,
        model: nn.Module,
        rank: int,
        world_size: int,
        device: torch.device,
        adapter: _CollectiveAdapter,
        dp_group: Any,
        tp_group: Any,
        dp_ranks: Tuple[int, ...],
        tp_ranks: Tuple[int, ...],
        pp_ranks: Tuple[int, ...],
        bucket_policy: Optional[Any] = None,
    ) -> None:
        self.plan = plan
        self.model = model
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.adapter = adapter
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.dp_ranks = dp_ranks
        self.tp_ranks = tp_ranks
        self.pp_ranks = pp_ranks
        self.dp_size = len(dp_ranks)
        self.tp_size = len(tp_ranks)
        self.pp_size = len(pp_ranks)
        self.dp_rank = dp_ranks.index(self.rank)
        self.tp_rank = tp_ranks.index(self.rank)
        self.pp_rank = pp_ranks.index(self.rank)
        self.bucket_policy = bucket_policy
        self.observed_raw_seq_lens: List[int] = []
        self.observed_bucket_keys: List[Any] = []

    @classmethod
    def from_plan(
        cls,
        plan: ExecutableParallelPlan,
        model_builder: Callable[..., nn.Module],
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        backend: str = "gloo",
        device: Optional[torch.device] = None,
        device_type: Optional[str] = None,
        num_layers: int = 8,
        stage_layer_counts: Optional[Sequence[int]] = None,
        enable_shape_bucketing: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "ParallelPlanExecutor":
        """Instantiate an executor for ``plan`` inside an initialized process group."""
        if dist.is_initialized():
            r = dist.get_rank() if rank is None else int(rank)
            w = dist.get_world_size() if world_size is None else int(world_size)
        else:
            r = 0 if rank is None else int(rank)
            w = 1 if world_size is None else int(world_size)

        if device is None:
            resolved_dt = device_type or _detect_default_device_type()
            device = _set_device_for_rank(resolved_dt, r)

        mesh: DeviceMeshSpec = plan.mesh
        if dist.is_initialized() and w > 1:
            dp_pg, dp_ranks = _create_subgroups_deterministic(mesh.dp_groups, r)
            tp_pg, tp_ranks = _create_subgroups_deterministic(mesh.tp_groups, r)
            pp_ranks = _find_rank_group(mesh.pp_groups, r)
        else:
            dp_pg, dp_ranks = None, (r,)
            tp_pg, tp_ranks = None, (r,)
            pp_ranks = (r,)

        tp_size = len(tp_ranks)
        tp_rank = tp_ranks.index(r)
        pp_size = len(pp_ranks)
        pp_rank = pp_ranks.index(r)

        adapter = _CollectiveAdapter(backend=backend, device=device)
        splits = compute_stage_layer_splits(num_layers, pp_size, stage_layer_counts)
        my_layers = splits[pp_rank]

        use_bucket = (
            bool(plan.runtime_knobs.get("FLAGTREE_SHAPE_BUCKETING", True))
            if enable_shape_bucketing is None
            else bool(enable_shape_bucketing)
        )
        bucket_policy = _apply_f09_shape_bucketing(use_bucket, mode=plan.workload_kind)

        model = model_builder(
            layer_indices=my_layers,
            is_last_stage=(pp_rank == pp_size - 1),
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_pg,
            pp_rank=pp_rank,
            pp_size=pp_size,
            adapter=adapter,
        )
        if dtype is not None:
            model = model.to(device=device, dtype=dtype)
        else:
            model = model.to(device=device)

        return cls(
            plan=plan,
            model=model,
            rank=r,
            world_size=w,
            device=device,
            adapter=adapter,
            dp_group=dp_pg,
            tp_group=tp_pg,
            dp_ranks=dp_ranks,
            tp_ranks=tp_ranks,
            pp_ranks=pp_ranks,
            bucket_policy=bucket_policy,
        )

    def _record_seq_bucket(self, seq_len: int) -> None:
        self.observed_raw_seq_lens.append(int(seq_len))
        if self.bucket_policy is not None:
            try:
                bkey = self.bucket_policy.bucket_value("S", int(seq_len))
                self.observed_bucket_keys.append(bkey)
            except Exception:
                self.observed_bucket_keys.append(int(seq_len))

    def _slice_for_dp(self, full_tensor: torch.Tensor) -> torch.Tensor:
        if self.dp_size <= 1:
            return full_tensor
        chunks = torch.chunk(full_tensor, self.dp_size, dim=0)
        return chunks[self.dp_rank].contiguous()

    def _microbatch_shapes(
        self,
        local_batch: int,
        seq_len: int,
        hidden_size: int,
        num_microbatches: int,
    ) -> List[Tuple[int, int, int]]:
        m = max(1, min(int(num_microbatches), int(local_batch)))
        base, rem = divmod(local_batch, m)
        return [(base + (1 if i < rem else 0), seq_len, hidden_size) for i in range(m)]

    def forward_inference(
        self,
        x_full: torch.Tensor,
        *,
        num_microbatches: int = 1,
    ) -> Optional[torch.Tensor]:
        """Run inference forward pass across DP / TP / PP / Hybrid mesh.

        ``x_full`` is the global batch ``(B, S, H)`` on device (or its shape/dtype
        metadata on non-first PP stages). Returns the DP-local output tensor on
        last pipeline stages (``pp_rank == pp_size - 1``) and ``None`` on
        intermediate stages.
        """
        self._record_seq_bucket(int(x_full.shape[1]))
        x_local = self._slice_for_dp(x_full)
        local_b, seq_len, hidden = x_local.shape
        dtype = x_local.dtype

        if self.pp_size <= 1:
            with torch.no_grad():
                out = self.model(x_local)
            _sync_device(self.device)
            return out

        shapes = self._microbatch_shapes(local_b, seq_len, hidden, num_microbatches)
        m_count = len(shapes)
        prev_rank = self.pp_ranks[self.pp_rank - 1] if self.pp_rank > 0 else None
        next_rank = self.pp_ranks[self.pp_rank + 1] if self.pp_rank < self.pp_size - 1 else None
        micro_inputs = (
            list(torch.split(x_local, [s[0] for s in shapes], dim=0))
            if self.pp_rank == 0
            else None
        )

        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for m_idx in range(m_count):
                tag = 1000 + self.dp_rank * 256 + m_idx
                if prev_rank is None:
                    h = micro_inputs[m_idx]
                else:
                    h = self.adapter.recv(shapes[m_idx], dtype, src=prev_rank, tag=tag)
                h_out = self.model(h)
                if next_rank is not None:
                    self.adapter.send(h_out, dst=next_rank, tag=tag, async_op=True)
                else:
                    outputs.append(h_out)
        self.adapter.wait_pending()
        _sync_device(self.device)
        if next_rank is None:
            return torch.cat(outputs, dim=0)
        return None

    def _sync_dp_gradients(self, bucket_cap_bytes: int = 25 * 1024 * 1024) -> None:
        if self.dp_size <= 1 or self.dp_group is None:
            return
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return
        bucket: List[torch.Tensor] = []
        bucket_bytes = 0

        def _flush(items: List[torch.Tensor]) -> None:
            flat = torch.cat([g.reshape(-1) for g in items], dim=0)
            reduced = self.adapter.all_reduce(flat, group=self.dp_group, average_by=self.dp_size)
            offset = 0
            for g in items:
                n = g.numel()
                g.copy_(reduced[offset : offset + n].view_as(g))
                offset += n

        for g in grads:
            nbytes = g.numel() * g.element_size()
            if bucket and bucket_bytes + nbytes > bucket_cap_bytes:
                _flush(bucket)
                bucket = []
                bucket_bytes = 0
            bucket.append(g)
            bucket_bytes += nbytes
        if bucket:
            _flush(bucket)

    def train_step(
        self,
        x_full: torch.Tensor,
        target_full: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        *,
        num_microbatches: int = 1,
    ) -> Dict[str, float]:
        """Run one forward + backward + gradient-sync + optimizer training step."""
        self._record_seq_bucket(int(x_full.shape[1]))
        self.adapter.reset_stats()
        _sync_device(self.device)
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        x_local = self._slice_for_dp(x_full)
        t_local = self._slice_for_dp(target_full)
        local_b, seq_len, hidden = x_local.shape
        dtype = x_local.dtype

        loss_val = 0.0
        if self.pp_size <= 1:
            out = self.model(x_local)
            loss = F.mse_loss(out.float(), t_local.float())
            loss.backward()
            loss_val = float(loss.detach().item())
        else:
            shapes = self._microbatch_shapes(local_b, seq_len, hidden, num_microbatches)
            m_count = len(shapes)
            prev_rank = self.pp_ranks[self.pp_rank - 1] if self.pp_rank > 0 else None
            next_rank = self.pp_ranks[self.pp_rank + 1] if self.pp_rank < self.pp_size - 1 else None
            sizes_b = [s[0] for s in shapes]
            micro_inputs = list(torch.split(x_local, sizes_b, dim=0)) if prev_rank is None else None
            micro_targets = list(torch.split(t_local, sizes_b, dim=0)) if next_rank is None else None

            saved_in: List[torch.Tensor] = []
            saved_out: List[torch.Tensor] = []
            saved_losses: List[torch.Tensor] = []

            # GPipe forward pass across microbatches
            for m_idx in range(m_count):
                fwd_tag = 2000 + self.dp_rank * 256 + m_idx
                if prev_rank is None:
                    h_in = micro_inputs[m_idx].detach()
                else:
                    h_in = self.adapter.recv(shapes[m_idx], dtype, src=prev_rank, tag=fwd_tag)
                    h_in.requires_grad_(True)
                h_out = self.model(h_in)
                saved_in.append(h_in)
                saved_out.append(h_out)
                if next_rank is not None:
                    self.adapter.send(h_out.detach(), dst=next_rank, tag=fwd_tag, async_op=True)
                else:
                    loss_m = F.mse_loss(h_out.float(), micro_targets[m_idx].float()) / float(m_count)
                    saved_losses.append(loss_m)
                    loss_val += float(loss_m.detach().item())
            self.adapter.wait_pending()

            # GPipe backward pass in reverse microbatch order
            for m_idx in range(m_count - 1, -1, -1):
                bwd_tag = 3000 + self.dp_rank * 256 + m_idx
                if next_rank is None:
                    saved_losses[m_idx].backward()
                else:
                    grad_out = self.adapter.recv(shapes[m_idx], dtype, src=next_rank, tag=bwd_tag)
                    torch.autograd.backward(saved_out[m_idx], grad_out)
                if prev_rank is not None:
                    self.adapter.send(saved_in[m_idx].grad.detach(), dst=prev_rank, tag=bwd_tag, async_op=True)
            self.adapter.wait_pending()

        self._sync_dp_gradients()
        optimizer.step()
        _sync_device(self.device)

        step_ms = (time.perf_counter() - t0) * 1e3
        comm_ms = self.adapter.comm_time_ms
        return {
            "loss": loss_val,
            "step_ms": step_ms,
            "comm_ms": comm_ms,
            "compute_ms": max(0.0, step_ms - comm_ms),
            "collective_calls": float(self.adapter.collective_calls),
        }


def _gather_full_output_on_rank0(
    executor: ParallelPlanExecutor,
    local_out: Optional[torch.Tensor],
    full_shape: Tuple[int, int, int],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Assemble full global-batch output on rank 0 for numerical parity verification."""
    # Identify last-stage rank for each DP replica (choosing tp_rank == 0)
    if executor.world_size <= 1:
        return local_out.detach().cpu() if local_out is not None else None

    dp_size = executor.dp_size
    is_emitter = (executor.pp_rank == executor.pp_size - 1) and (executor.tp_rank == 0)

    # Determine global ranks that emit each DP shard in DP order
    emitter_ranks: List[int] = []
    for d_idx in range(dp_size):
        # Pick the rank belonging to dp_idx, last pp stage, tp_rank 0
        for r in range(executor.world_size):
            dp_grp = _find_rank_group(executor.plan.mesh.dp_groups, r)
            pp_grp = _find_rank_group(executor.plan.mesh.pp_groups, r)
            tp_grp = _find_rank_group(executor.plan.mesh.tp_groups, r)
            if dp_grp.index(r) == d_idx and pp_grp.index(r) == len(pp_grp) - 1 and tp_grp.index(r) == 0:
                emitter_ranks.append(r)
                break

    base, rem = divmod(full_shape[0], dp_size)
    dp_shapes = [(base + (1 if i < rem else 0), full_shape[1], full_shape[2]) for i in range(dp_size)]

    if executor.rank == 0:
        shards: List[torch.Tensor] = []
        for d_idx, src_r in enumerate(emitter_ranks):
            if src_r == 0:
                shards.append(local_out.detach().to("cpu").contiguous())
            else:
                buf = torch.empty(dp_shapes[d_idx], dtype=dtype, device="cpu")
                dist.recv(buf, src=src_r, tag=9000 + d_idx)
                shards.append(buf)
        return torch.cat(shards, dim=0)

    if is_emitter:
        d_idx = executor.dp_rank
        buf = local_out.detach().to("cpu").contiguous()
        dist.send(buf, dst=0, tag=9000 + d_idx)
    return None


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    cos = float(F.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item())
    max_diff = float((af - bf).abs().max().item())
    return cos, max_diff


def _parallel_exec_worker(
    rank: int,
    world_size: int,
    plan_dict: Dict[str, Any],
    cfg_dict: Dict[str, Any],
    batch_size: int,
    seq_lengths: Tuple[int, ...],
    num_microbatches: int,
    warmup_steps: int,
    measure_steps: int,
    mode: str,
    backend: str,
    device_type: str,
    verify_parity: bool,
    stage_layer_counts: Optional[Tuple[int, ...]],
    enable_shape_bucketing: bool,
    master_addr: str,
    master_port: int,
    out_path: str,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    device = _set_device_for_rank(device_type, rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        mesh_d = plan_dict["mesh"]
        mesh = DeviceMeshSpec(
            mesh_shape=tuple(mesh_d["mesh_shape"]),
            mesh_dim_names=tuple(mesh_d["mesh_dim_names"]),
            dp_groups=tuple(tuple(g) for g in mesh_d["dp_groups"]),
            tp_groups=tuple(tuple(g) for g in mesh_d["tp_groups"]),
            pp_groups=tuple(tuple(g) for g in mesh_d["pp_groups"]),
            ep_groups=tuple(tuple(g) for g in mesh_d["ep_groups"]),
        )
        plan = ExecutableParallelPlan(
            model_name=plan_dict["model_name"],
            plan_id=plan_dict["plan_id"],
            strategy=plan_dict["strategy"],
            workload_kind=plan_dict["workload_kind"],
            backend=plan_dict["backend"],
            num_devices=int(plan_dict["num_devices"]),
            mesh=mesh,
            runtime_knobs=dict(plan_dict["runtime_knobs"]),
            predicted_metrics=dict(plan_dict["predicted_metrics"]),
        )
        cfg = TransformerStackConfig(**cfg_dict)
        torch_dtype = _resolve_torch_dtype(cfg.dtype)

        def _builder(
            *,
            layer_indices,
            is_last_stage,
            tp_rank,
            tp_size,
            tp_group,
            pp_rank,
            pp_size,
            adapter,
        ):
            return ParallelTransformerModel(
                cfg,
                layer_indices=layer_indices,
                is_last_stage=is_last_stage,
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_group=tp_group,
                adapter=adapter,
            )

        executor = ParallelPlanExecutor.from_plan(
            plan,
            _builder,
            rank=rank,
            world_size=world_size,
            backend=backend,
            device=device,
            num_layers=cfg.num_layers,
            stage_layer_counts=stage_layer_counts,
            enable_shape_bucketing=enable_shape_bucketing,
            dtype=torch_dtype,
        )

        # Optional parity check BEFORE optimizer mutates weights
        parity_cos = 1.0
        parity_max_diff = 0.0
        if verify_parity:
            s_ref = int(seq_lengths[0])
            gen_in = torch.Generator(device="cpu").manual_seed(cfg.seed + 999)
            x_ref_cpu = torch.randn((batch_size, s_ref, cfg.hidden_size), generator=gen_in).to(
                torch_dtype
            )
            x_ref_dev = x_ref_cpu.to(device)
            dist.barrier()
            local_out = executor.forward_inference(x_ref_dev, num_microbatches=num_microbatches)
            dist.barrier()
            gathered = _gather_full_output_on_rank0(
                executor, local_out, (batch_size, s_ref, cfg.hidden_size), torch_dtype
            )
            if rank == 0 and gathered is not None:
                # Reference single-device model computed layer-by-layer to save HBM
                with torch.no_grad():
                    h_ref = x_ref_cpu.to(device)
                    for l_idx in range(cfg.num_layers):
                        blk = ParallelTransformerBlock(cfg, layer_idx=l_idx, tp_rank=0, tp_size=1)
                        blk = blk.to(device=device, dtype=torch_dtype)
                        h_ref = blk(h_ref)
                        del blk
                    fnorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(
                        device=device, dtype=torch_dtype
                    )
                    h_ref = fnorm(h_ref).to("cpu")
                    del fnorm
                parity_cos, parity_max_diff = _cosine_similarity(gathered, h_ref)
                del h_ref, gathered
            dist.barrier()

        optimizer = (
            torch.optim.SGD(executor.model.parameters(), lr=1e-4)
            if mode == "training"
            else None
        )

        # Pre-generate variable-length inputs cleanly on device
        step_inputs: List[Tuple[int, torch.Tensor, Optional[torch.Tensor]]] = []
        for idx, s_len in enumerate(seq_lengths):
            g_step = torch.Generator(device="cpu").manual_seed(cfg.seed + 500 + idx)
            x_s = torch.randn((batch_size, int(s_len), cfg.hidden_size), generator=g_step).to(
                device=device, dtype=torch_dtype
            )
            t_s = (
                torch.randn((batch_size, int(s_len), cfg.hidden_size), generator=g_step).to(
                    device=device, dtype=torch_dtype
                )
                if mode == "training"
                else None
            )
            step_inputs.append((int(s_len), x_s, t_s))

        # Warmup
        for w_idx in range(warmup_steps):
            _, x_s, t_s = step_inputs[w_idx % len(step_inputs)]
            dist.barrier()
            if mode == "training":
                executor.train_step(x_s, t_s, optimizer, num_microbatches=num_microbatches)
            else:
                executor.forward_inference(x_s, num_microbatches=num_microbatches)
            dist.barrier()

        _reset_peak_memory(device)
        executor.observed_raw_seq_lens.clear()
        executor.observed_bucket_keys.clear()
        step_latencies_ms: List[float] = []
        comm_latencies_ms: List[float] = []
        total_tokens = 0
        last_loss = 0.0

        for m_idx in range(measure_steps):
            s_len, x_s, t_s = step_inputs[m_idx % len(step_inputs)]
            executor.adapter.reset_stats()
            dist.barrier()
            _sync_device(device)
            t0 = time.perf_counter()
            if mode == "training":
                info = executor.train_step(x_s, t_s, optimizer, num_microbatches=num_microbatches)
                last_loss = info["loss"]
            else:
                executor.forward_inference(x_s, num_microbatches=num_microbatches)
            _sync_device(device)
            dist.barrier()
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            step_latencies_ms.append(elapsed_ms)
            comm_latencies_ms.append(executor.adapter.comm_time_ms)
            total_tokens += batch_size * s_len

        peak_mb = _peak_memory_mb(device)
        # Gather max peak memory across ranks to rank 0 via CPU all_reduce
        peak_t = torch.tensor([peak_mb], dtype=torch.float64, device="cpu")
        dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
        max_peak_mb = float(peak_t.item())

        mean_comm_t = torch.tensor(
            [sum(comm_latencies_ms) / max(1, len(comm_latencies_ms))],
            dtype=torch.float64,
            device="cpu",
        )
        dist.all_reduce(mean_comm_t, op=dist.ReduceOp.MAX)
        max_rank_comm_ms = float(mean_comm_t.item())

        loss_t = torch.tensor([last_loss], dtype=torch.float64, device="cpu")
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        global_last_loss = float(loss_t.item()) / float(max(1, executor.dp_size * executor.tp_size))

        if rank == 0:
            mean_step_ms = sum(step_latencies_ms) / max(1, len(step_latencies_ms))
            total_sec = sum(step_latencies_ms) * 1e-3
            tps = float(total_tokens) / max(1e-6, total_sec)
            payload = {
                "plan_id": plan.plan_id,
                "strategy": plan.strategy,
                "mode": mode,
                "world_size": world_size,
                "dp_size": executor.dp_size,
                "tp_size": executor.tp_size,
                "pp_size": executor.pp_size,
                "stage_layer_counts": (
                    list(stage_layer_counts)
                    if stage_layer_counts is not None
                    else [
                        len(sp)
                        for sp in compute_stage_layer_splits(cfg.num_layers, executor.pp_size)
                    ]
                ),
                "mean_step_ms": round(mean_step_ms, 4),
                "min_step_ms": round(min(step_latencies_ms), 4),
                "max_step_ms": round(max(step_latencies_ms), 4),
                "max_rank_comm_ms": round(max_rank_comm_ms, 4),
                "throughput_tokens_per_sec": round(tps, 2),
                "peak_memory_mb": round(max_peak_mb, 2),
                "parity_cos": round(parity_cos, 7),
                "parity_max_abs_diff": round(parity_max_diff, 6),
                "all_finite": bool(math.isfinite(parity_cos) and math.isfinite(global_last_loss)),
                "last_loss": round(global_last_loss, 6),
                "shape_bucketing_enabled": bool(enable_shape_bucketing),
                "unique_raw_seq_lens": len(set(executor.observed_raw_seq_lens)),
                "unique_bucketed_seq_lens": len(set(executor.observed_bucket_keys)),
            }
            Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def run_parallel_plan_benchmark(
    *,
    plan: ExecutableParallelPlan,
    model_config: TransformerStackConfig,
    batch_size: int = 8,
    seq_lengths: Sequence[int] = (128, 192, 256),
    num_microbatches: int = 1,
    warmup_steps: int = 1,
    measure_steps: int = 3,
    mode: Optional[str] = None,
    backend: str = "gloo",
    device_type: Optional[str] = None,
    verify_parity: bool = True,
    stage_layer_counts: Optional[Sequence[int]] = None,
    enable_shape_bucketing: bool = True,
    master_addr: str = "127.0.0.1",
    master_port: Optional[int] = None,
) -> Dict[str, Any]:
    """Spawn ``plan.num_devices`` ranks to execute and benchmark ``plan`` end-to-end."""
    import torch.multiprocessing as mp

    resolved_mode = (mode or plan.workload_kind or "inference").strip().lower()
    resolved_dt = (device_type or _detect_default_device_type()).strip().lower()
    world_size = max(1, int(plan.num_devices))
    port = int(master_port) if master_port is not None else _find_free_port()
    counts_tuple = tuple(int(c) for c in stage_layer_counts) if stage_layer_counts is not None else None

    with tempfile.TemporaryDirectory() as tmpd:
        out_path = str(Path(tmpd) / "parallel_bench_result.json")
        with _prefer_loaded_triton_in_spawn():
            mp.spawn(
                _parallel_exec_worker,
                args=(
                    world_size,
                    plan.to_dict(),
                    asdict(model_config),
                    int(batch_size),
                    tuple(int(s) for s in seq_lengths),
                    int(num_microbatches),
                    int(warmup_steps),
                    int(measure_steps),
                    resolved_mode,
                    str(backend),
                    resolved_dt,
                    bool(verify_parity),
                    counts_tuple,
                    bool(enable_shape_bucketing),
                    str(master_addr),
                    port,
                    out_path,
                ),
                nprocs=world_size,
                join=True,
            )
        return json.loads(Path(out_path).read_text(encoding="utf-8"))


__all__ = [
    "ColumnParallelLinear",
    "ParallelPlanExecutor",
    "ParallelTransformerBlock",
    "ParallelTransformerModel",
    "RMSNorm",
    "RowParallelLinear",
    "TransformerStackConfig",
    "compute_stage_layer_splits",
    "run_parallel_plan_benchmark",
]
