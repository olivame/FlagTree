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

"""Backend-agnostic multi-device collective communication topology profiler.

Measures real point-to-point (P2P) and collective (``all_reduce``,
``all_gather``, ``reduce_scatter``) alpha-beta latency and effective bus
bandwidth across multi-device process groups. Supports direct accelerator
collectives (``nccl`` / ``hccl`` / ``xccl``) as well as automatic
device<->host staging when running over ``gloo`` so compute stays on
accelerators while collectives remain portable across heterogeneous backends.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from triton.flagtune.runtime.partition_planner import (
    AlphaBetaSegment,
    CollectiveCostModel,
    CollectiveKind,
    PiecewiseAlphaBetaCurve,
)

DEFAULT_PROFILE_SIZES_BYTES: Tuple[int, ...] = (
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
)

SUPPORTED_COLLECTIVES: Tuple[str, ...] = (
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "p2p",
)

_DTYPE_BYTES: Dict[str, int] = {
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "fp16": 2,
    "float32": 4,
    "fp32": 4,
}


def needs_host_staging(backend: str, device_type: str) -> bool:
    """Return True if collectives require device<->CPU staging for ``backend``."""
    bk = (backend or "").strip().lower()
    dt = (device_type or "cpu").strip().lower()
    if dt == "cpu":
        return False
    if bk == "gloo":
        return True
    try:
        from triton.runtime.backend_caps import resolve_collective_device

        return resolve_collective_device(bk) == "cpu"
    except Exception:
        return False


def _resolve_torch_dtype(dtype_name: str):
    import torch

    key = (dtype_name or "bfloat16").strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported profiler dtype: {dtype_name!r}")
    return mapping[key]


def _detect_default_device_type() -> str:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
        for ext in ("npu", "xpu", "mlu", "musa"):
            mod = getattr(torch, ext, None)
            if mod is not None and callable(getattr(mod, "is_available", None)):
                if mod.is_available():
                    return ext
    except Exception:
        pass
    return "cpu"


def _set_device_for_rank(device_type: str, rank: int):
    import torch

    dt = (device_type or "cpu").strip().lower()
    if dt == "cpu":
        return torch.device("cpu")
    mod = getattr(torch, dt, None)
    if mod is not None:
        count = mod.device_count() if callable(getattr(mod, "device_count", None)) else 1
        local_idx = rank % max(1, int(count))
        if callable(getattr(mod, "set_device", None)):
            mod.set_device(local_idx)
        return torch.device(f"{dt}:{local_idx}")
    return torch.device("cpu")


def _sync_device(device) -> None:
    import torch

    dt = device.type
    if dt == "cpu":
        return
    mod = getattr(torch, dt, None)
    if mod is not None and callable(getattr(mod, "synchronize", None)):
        mod.synchronize(device)


def _query_device_info(device_type: str, rank: int = 0) -> Tuple[str, int]:
    import torch

    dt = (device_type or "cpu").strip().lower()
    if dt == "cpu":
        return "cpu", 16 * (1024 ** 3)
    mod = getattr(torch, dt, None)
    if mod is None:
        return dt, 16 * (1024 ** 3)
    count = mod.device_count() if callable(getattr(mod, "device_count", None)) else 1
    idx = rank % max(1, int(count))
    name = dt
    mem_bytes = 32 * (1024 ** 3)
    try:
        if callable(getattr(mod, "get_device_name", None)):
            name = str(mod.get_device_name(idx))
    except Exception:
        pass
    try:
        if callable(getattr(mod, "get_device_properties", None)):
            props = mod.get_device_properties(idx)
            total_mem = getattr(props, "total_memory", None)
            if total_mem:
                mem_bytes = int(total_mem)
    except Exception:
        pass
    return name, mem_bytes


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _prefer_loaded_triton_in_spawn():
    """Ensure spawned child processes resolve the same loaded ``triton`` root."""
    triton_mod = sys.modules.get("triton")
    tfile = getattr(triton_mod, "__file__", None) if triton_mod is not None else None
    if not tfile:
        yield
        return
    triton_root = str(Path(tfile).resolve().parent.parent)
    orig_sys_path = list(sys.path)
    orig_pypath = os.environ.get("PYTHONPATH")
    try:
        sys.path = [triton_root] + [p for p in sys.path if p != triton_root]
        if orig_pypath:
            parts = [p for p in orig_pypath.split(os.pathsep) if p and p != triton_root]
            os.environ["PYTHONPATH"] = os.pathsep.join([triton_root] + parts)
        else:
            os.environ["PYTHONPATH"] = triton_root
        yield
    finally:
        sys.path[:] = orig_sys_path
        if orig_pypath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = orig_pypath


def _effective_bus_bytes(collective: str, message_bytes: int, world_size: int) -> float:
    w = max(2, int(world_size))
    ring_ratio = (w - 1.0) / float(w)
    if collective == "all_reduce":
        return 2.0 * ring_ratio * float(message_bytes)
    if collective in ("all_gather", "reduce_scatter"):
        return ring_ratio * float(message_bytes)
    return float(message_bytes)


def _hop_alpha_scale(collective: str, world_size: int) -> float:
    w = max(2, int(world_size))
    if collective == "all_reduce":
        return 1.0 + 0.18 * math.log2(w)
    if collective in ("all_gather", "reduce_scatter"):
        return 1.0 + 0.10 * math.log2(w)
    if collective == "all_to_all":
        return 1.0 + 0.25 * math.log2(w)
    return 1.0


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


@dataclass(frozen=True)
class CommSample:
    """Single message-size measurement for one collective primitive."""

    message_bytes: int
    latency_us: float
    p50_latency_us: float
    p95_latency_us: float
    effective_bandwidth_gbps: float
    alg_bandwidth_gbps: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CommSample":
        return cls(
            message_bytes=int(payload["message_bytes"]),
            latency_us=float(payload["latency_us"]),
            p50_latency_us=float(payload["p50_latency_us"]),
            p95_latency_us=float(payload["p95_latency_us"]),
            effective_bandwidth_gbps=float(payload["effective_bandwidth_gbps"]),
            alg_bandwidth_gbps=float(payload["alg_bandwidth_gbps"]),
        )


@dataclass(frozen=True)
class CollectiveProfile:
    """Alpha-beta summary and raw samples for one collective primitive."""

    collective: str
    world_size: int
    samples: Tuple[CommSample, ...]
    alpha_us: float
    beta_us_per_byte: float
    peak_bandwidth_gbps: float
    min_latency_us: float
    emulated: bool = False

    def to_piecewise_curve(self) -> PiecewiseAlphaBetaCurve:
        """Fit a ``PiecewiseAlphaBetaCurve`` calibrated for ``CollectiveCostModel``.

        Converts raw ``(message_bytes, latency_us)`` into effective bus bytes
        and normalizes ``alpha_us`` by ``_hop_alpha_scale`` so
        ``CollectiveCostModel.estimate_ms`` reproduces measured latencies at
        ``world_size`` while scaling cleanly across sub-groups.
        """
        hop_scale = _hop_alpha_scale(self.collective, self.world_size)
        eff_samples: List[Tuple[int, float]] = []
        for s in self.samples:
            eff_b = max(
                1,
                int(round(_effective_bus_bytes(self.collective, s.message_bytes, self.world_size))),
            )
            eff_samples.append((eff_b, max(0.05, float(s.latency_us))))
        raw_curve = PiecewiseAlphaBetaCurve.fit_from_samples(eff_samples)
        adj_segments = tuple(
            AlphaBetaSegment(
                max_bytes=seg.max_bytes,
                alpha_us=max(0.05, seg.alpha_us / max(1e-6, hop_scale)),
                bandwidth_gbps=max(0.01, seg.bandwidth_gbps),
            )
            for seg in raw_curve.segments
        )
        return PiecewiseAlphaBetaCurve(adj_segments)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collective": self.collective,
            "world_size": self.world_size,
            "samples": [s.to_dict() for s in self.samples],
            "alpha_us": self.alpha_us,
            "beta_us_per_byte": self.beta_us_per_byte,
            "peak_bandwidth_gbps": self.peak_bandwidth_gbps,
            "min_latency_us": self.min_latency_us,
            "emulated": self.emulated,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CollectiveProfile":
        return cls(
            collective=str(payload["collective"]),
            world_size=int(payload["world_size"]),
            samples=tuple(CommSample.from_dict(s) for s in payload["samples"]),
            alpha_us=float(payload["alpha_us"]),
            beta_us_per_byte=float(payload["beta_us_per_byte"]),
            peak_bandwidth_gbps=float(payload["peak_bandwidth_gbps"]),
            min_latency_us=float(payload["min_latency_us"]),
            emulated=bool(payload.get("emulated", False)),
        )


def _build_collective_profile(
    collective: str,
    world_size: int,
    samples: Sequence[CommSample],
    *,
    emulated: bool = False,
) -> CollectiveProfile:
    if not samples:
        raise ValueError(f"CollectiveProfile({collective}) requires at least one sample")
    ordered = tuple(sorted(samples, key=lambda s: s.message_bytes))
    eff_pts = [
        (_effective_bus_bytes(collective, s.message_bytes, world_size), s.latency_us)
        for s in ordered
    ]
    if len(eff_pts) >= 2:
        mean_b = sum(b for b, _ in eff_pts) / len(eff_pts)
        mean_t = sum(t for _, t in eff_pts) / len(eff_pts)
        var_b = sum((b - mean_b) ** 2 for b, _ in eff_pts)
        if var_b > 0:
            cov_bt = sum((b - mean_b) * (t - mean_t) for b, t in eff_pts)
            beta = max(1e-7, cov_bt / var_b)
            alpha = max(0.05, min(ordered[0].latency_us, mean_t - beta * mean_b))
        else:
            alpha = max(0.05, ordered[0].latency_us * 0.5)
            beta = max(1e-7, (ordered[-1].latency_us - alpha) / max(1.0, eff_pts[-1][0]))
    else:
        b0, t0 = eff_pts[0]
        alpha = max(0.05, t0 * 0.35)
        beta = max(1e-7, (t0 - alpha) / max(1.0, b0))

    peak_bw = max(s.effective_bandwidth_gbps for s in ordered)
    min_lat = min(s.latency_us for s in ordered)
    return CollectiveProfile(
        collective=collective,
        world_size=int(world_size),
        samples=ordered,
        alpha_us=float(alpha),
        beta_us_per_byte=float(beta),
        peak_bandwidth_gbps=float(peak_bw),
        min_latency_us=float(min_lat),
        emulated=emulated,
    )


@dataclass
class CommProfileResult:
    """Complete multi-device communication topology profile."""

    world_size: int
    backend: str
    device_type: str
    device_name: str
    device_memory_bytes: int
    host_staged: bool
    dtype: str
    collectives: Dict[str, CollectiveProfile]
    intra_node_bandwidth_gbps: float
    base_latency_us: float
    schema: str = "flagtree.comm_profile.v1"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_collective_cost_model(self) -> CollectiveCostModel:
        """Build a hardware-calibrated ``CollectiveCostModel`` from measurements."""
        curves: Dict[CollectiveKind, PiecewiseAlphaBetaCurve] = {}
        mapping = {
            "all_reduce": CollectiveKind.ALL_REDUCE,
            "all_gather": CollectiveKind.ALL_GATHER,
            "reduce_scatter": CollectiveKind.REDUCE_SCATTER,
            "p2p": CollectiveKind.P2P,
            "all_to_all": CollectiveKind.ALL_TO_ALL,
        }
        for key, ckind in mapping.items():
            if key in self.collectives:
                curves[ckind] = self.collectives[key].to_piecewise_curve()

        if CollectiveKind.ALL_TO_ALL not in curves:
            fallback_src = curves.get(CollectiveKind.ALL_GATHER) or curves.get(
                CollectiveKind.ALL_REDUCE
            )
            if fallback_src is not None:
                curves[CollectiveKind.ALL_TO_ALL] = fallback_src

        return CollectiveCostModel(
            curves=curves,
            intra_node_bandwidth_gbps=max(0.05, self.intra_node_bandwidth_gbps),
            base_latency_us=max(0.1, self.base_latency_us),
        )

    def to_hardware_topology(
        self,
        *,
        device_memory_bytes: Optional[int] = None,
        memory_safety_margin: float = 0.90,
        supports_tle: bool = False,
        backend: Optional[str] = None,
        devices_per_node: Optional[int] = None,
        inter_node_bandwidth_gbps: float = 25.0,
        inter_node_latency_us: float = 45.0,
    ):
        """Construct a calibrated ``HardwareTopologySpec`` / ``HardwareClusterSpec``."""
        from triton.flagtune.runtime.partition_planner import HardwareClusterSpec

        return HardwareClusterSpec.from_comm_profile(
            self,
            device_memory_bytes=device_memory_bytes,
            memory_safety_margin=memory_safety_margin,
            supports_tle=supports_tle,
            backend=backend,
            devices_per_node=devices_per_node,
            inter_node_bandwidth_gbps=inter_node_bandwidth_gbps,
            inter_node_latency_us=inter_node_latency_us,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "world_size": self.world_size,
            "backend": self.backend,
            "device_type": self.device_type,
            "device_name": self.device_name,
            "device_memory_bytes": self.device_memory_bytes,
            "host_staged": self.host_staged,
            "dtype": self.dtype,
            "intra_node_bandwidth_gbps": self.intra_node_bandwidth_gbps,
            "base_latency_us": self.base_latency_us,
            "collectives": {k: v.to_dict() for k, v in self.collectives.items()},
            "metadata": dict(self.metadata),
        }

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CommProfileResult":
        colls = {
            str(k): CollectiveProfile.from_dict(v)
            for k, v in payload["collectives"].items()
        }
        return cls(
            schema=str(payload.get("schema", "flagtree.comm_profile.v1")),
            world_size=int(payload["world_size"]),
            backend=str(payload["backend"]),
            device_type=str(payload["device_type"]),
            device_name=str(payload.get("device_name", payload["device_type"])),
            device_memory_bytes=int(payload["device_memory_bytes"]),
            host_staged=bool(payload.get("host_staged", False)),
            dtype=str(payload.get("dtype", "bfloat16")),
            collectives=colls,
            intra_node_bandwidth_gbps=float(payload["intra_node_bandwidth_gbps"]),
            base_latency_us=float(payload["base_latency_us"]),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "CommProfileResult":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _measure_all_reduce(
    nbytes: int,
    dtype,
    dtype_bytes: int,
    device,
    host_staged: bool,
    warmup_iters: int,
    measure_iters: int,
    pg,
) -> Tuple[List[float], bool]:
    import torch
    import torch.distributed as dist

    numel = max(1, nbytes // dtype_bytes)
    t_dev = torch.ones(numel, dtype=dtype, device=device)
    cpu_buf = torch.empty(numel, dtype=dtype, device="cpu") if host_staged else None

    def _op() -> None:
        if host_staged:
            cpu_buf.copy_(t_dev)
            dist.all_reduce(cpu_buf, op=dist.ReduceOp.SUM, group=pg)
            t_dev.copy_(cpu_buf)
            _sync_device(device)
        else:
            dist.all_reduce(t_dev, op=dist.ReduceOp.SUM, group=pg)
            _sync_device(device)

    for _ in range(warmup_iters):
        _op()
    dist.barrier(group=pg)
    times_us: List[float] = []
    for _ in range(measure_iters):
        dist.barrier(group=pg)
        t0 = time.perf_counter()
        _op()
        times_us.append((time.perf_counter() - t0) * 1e6)
    return times_us, False


def _measure_all_gather(
    nbytes: int,
    world_size: int,
    dtype,
    dtype_bytes: int,
    device,
    host_staged: bool,
    warmup_iters: int,
    measure_iters: int,
    pg,
) -> Tuple[List[float], bool]:
    import torch
    import torch.distributed as dist

    total_numel = max(world_size, (nbytes // dtype_bytes // world_size) * world_size)
    shard_numel = max(1, total_numel // world_size)
    shard_dev = torch.ones(shard_numel, dtype=dtype, device=device)
    out_dev = torch.empty(shard_numel * world_size, dtype=dtype, device=device)
    if host_staged:
        shard_cpu = torch.empty(shard_numel, dtype=dtype, device="cpu")
        out_list_cpu = [
            torch.empty(shard_numel, dtype=dtype, device="cpu") for _ in range(world_size)
        ]
    else:
        shard_cpu = None
        out_list_cpu = None

    def _op() -> None:
        if host_staged:
            shard_cpu.copy_(shard_dev)
            dist.all_gather(out_list_cpu, shard_cpu, group=pg)
            out_dev.copy_(torch.cat(out_list_cpu, dim=0))
            _sync_device(device)
        else:
            if hasattr(dist, "all_gather_into_tensor"):
                dist.all_gather_into_tensor(out_dev, shard_dev, group=pg)
            else:
                chunks = list(out_dev.chunk(world_size, dim=0))
                dist.all_gather(chunks, shard_dev, group=pg)
            _sync_device(device)

    for _ in range(warmup_iters):
        _op()
    dist.barrier(group=pg)
    times_us: List[float] = []
    for _ in range(measure_iters):
        dist.barrier(group=pg)
        t0 = time.perf_counter()
        _op()
        times_us.append((time.perf_counter() - t0) * 1e6)
    return times_us, False


def _measure_reduce_scatter(
    nbytes: int,
    rank: int,
    world_size: int,
    dtype,
    dtype_bytes: int,
    device,
    host_staged: bool,
    warmup_iters: int,
    measure_iters: int,
    pg,
) -> Tuple[List[float], bool]:
    import torch
    import torch.distributed as dist

    total_numel = max(world_size, (nbytes // dtype_bytes // world_size) * world_size)
    shard_numel = max(1, total_numel // world_size)
    in_dev = torch.ones(shard_numel * world_size, dtype=dtype, device=device)
    out_dev = torch.empty(shard_numel, dtype=dtype, device=device)

    emulated = False
    if host_staged:
        in_cpu = torch.empty(shard_numel * world_size, dtype=dtype, device="cpu")
        out_cpu = torch.empty(shard_numel, dtype=dtype, device="cpu")
        try:
            in_chunks = list(in_cpu.chunk(world_size, dim=0))
            dist.reduce_scatter(out_cpu, in_chunks, op=dist.ReduceOp.SUM, group=pg)
        except Exception:
            emulated = True
    else:
        in_cpu = None
        out_cpu = None
        try:
            if hasattr(dist, "reduce_scatter_tensor"):
                dist.reduce_scatter_tensor(out_dev, in_dev, op=dist.ReduceOp.SUM, group=pg)
            else:
                in_chunks = list(in_dev.chunk(world_size, dim=0))
                dist.reduce_scatter(out_dev, in_chunks, op=dist.ReduceOp.SUM, group=pg)
            _sync_device(device)
        except Exception:
            emulated = True

    start = rank * shard_numel
    end = start + shard_numel

    def _op() -> None:
        if host_staged:
            in_cpu.copy_(in_dev)
            if emulated:
                dist.all_reduce(in_cpu, op=dist.ReduceOp.SUM, group=pg)
                out_dev.copy_(in_cpu[start:end])
            else:
                in_chunks = list(in_cpu.chunk(world_size, dim=0))
                dist.reduce_scatter(out_cpu, in_chunks, op=dist.ReduceOp.SUM, group=pg)
                out_dev.copy_(out_cpu)
            _sync_device(device)
        else:
            if emulated:
                dist.all_reduce(in_dev, op=dist.ReduceOp.SUM, group=pg)
                out_dev.copy_(in_dev[start:end])
            elif hasattr(dist, "reduce_scatter_tensor"):
                dist.reduce_scatter_tensor(out_dev, in_dev, op=dist.ReduceOp.SUM, group=pg)
            else:
                in_chunks = list(in_dev.chunk(world_size, dim=0))
                dist.reduce_scatter(out_dev, in_chunks, op=dist.ReduceOp.SUM, group=pg)
            _sync_device(device)

    for _ in range(warmup_iters):
        _op()
    dist.barrier(group=pg)
    times_us: List[float] = []
    for _ in range(measure_iters):
        dist.barrier(group=pg)
        t0 = time.perf_counter()
        _op()
        times_us.append((time.perf_counter() - t0) * 1e6)
    return times_us, emulated


def _measure_p2p(
    nbytes: int,
    rank: int,
    dtype,
    dtype_bytes: int,
    device,
    host_staged: bool,
    warmup_iters: int,
    measure_iters: int,
    pg,
) -> Tuple[List[float], bool]:
    """Measure one-way P2P activation transfer via rank 0 <-> rank 1 ping-pong / 2."""
    import torch
    import torch.distributed as dist

    numel = max(1, nbytes // dtype_bytes)
    send_dev = torch.ones(numel, dtype=dtype, device=device)
    recv_dev = torch.empty(numel, dtype=dtype, device=device)
    send_cpu = torch.empty(numel, dtype=dtype, device="cpu") if host_staged else None
    recv_cpu = torch.empty(numel, dtype=dtype, device="cpu") if host_staged else None

    def _ping_pong() -> None:
        if rank == 0:
            if host_staged:
                send_cpu.copy_(send_dev)
                dist.send(send_cpu, dst=1, group=pg, tag=101)
                dist.recv(recv_cpu, src=1, group=pg, tag=102)
                recv_dev.copy_(recv_cpu)
            else:
                dist.send(send_dev, dst=1, group=pg, tag=101)
                dist.recv(recv_dev, src=1, group=pg, tag=102)
            _sync_device(device)
        elif rank == 1:
            if host_staged:
                dist.recv(recv_cpu, src=0, group=pg, tag=101)
                recv_dev.copy_(recv_cpu)
                _sync_device(device)
                send_cpu.copy_(recv_dev)
                dist.send(send_cpu, dst=0, group=pg, tag=102)
            else:
                dist.recv(recv_dev, src=0, group=pg, tag=101)
                _sync_device(device)
                dist.send(recv_dev, dst=0, group=pg, tag=102)
            _sync_device(device)

    for _ in range(warmup_iters):
        _ping_pong()
    dist.barrier(group=pg)
    times_us: List[float] = []
    for _ in range(measure_iters):
        dist.barrier(group=pg)
        t0 = time.perf_counter()
        _ping_pong()
        one_way_us = (time.perf_counter() - t0) * 1e6 * 0.5
        times_us.append(one_way_us)
    return times_us, False


def _profile_in_initialized_pg(
    *,
    backend: str,
    device_type: str,
    sizes_bytes: Sequence[int],
    dtype: str,
    warmup_iters: int,
    measure_iters: int,
    collectives: Sequence[str],
    pg,
) -> CommProfileResult:
    import torch.distributed as dist

    rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    device = _set_device_for_rank(device_type, rank)
    host_staged = needs_host_staging(backend, device_type)
    torch_dtype = _resolve_torch_dtype(dtype)
    dtype_bytes = _DTYPE_BYTES.get(dtype.lower(), 2)
    dev_name, dev_mem = _query_device_info(device_type, rank)

    cleaned_sizes = sorted(
        {max(dtype_bytes * max(1, world_size), int(sz)) for sz in sizes_bytes}
    )
    profiles: Dict[str, CollectiveProfile] = {}

    for coll in collectives:
        cname = str(coll).strip().lower()
        if cname not in SUPPORTED_COLLECTIVES:
            raise ValueError(
                f"Unsupported collective {coll!r}; expected one of {SUPPORTED_COLLECTIVES}"
            )
        if world_size < 2:
            raise ValueError("Communication profiling requires world_size >= 2")

        samples: List[CommSample] = []
        emulated_flag = False
        for nbytes in cleaned_sizes:
            if cname == "all_reduce":
                raw_us, emu = _measure_all_reduce(
                    nbytes, torch_dtype, dtype_bytes, device, host_staged,
                    warmup_iters, measure_iters, pg,
                )
            elif cname == "all_gather":
                raw_us, emu = _measure_all_gather(
                    nbytes, world_size, torch_dtype, dtype_bytes, device, host_staged,
                    warmup_iters, measure_iters, pg,
                )
            elif cname == "reduce_scatter":
                raw_us, emu = _measure_reduce_scatter(
                    nbytes, rank, world_size, torch_dtype, dtype_bytes, device, host_staged,
                    warmup_iters, measure_iters, pg,
                )
            else:
                raw_us, emu = _measure_p2p(
                    nbytes, rank, torch_dtype, dtype_bytes, device, host_staged,
                    warmup_iters, measure_iters, pg,
                )
            emulated_flag = emulated_flag or emu
            ordered_us = sorted(max(0.05, float(v)) for v in raw_us)
            mean_us = sum(ordered_us) / len(ordered_us)
            p50_us = _percentile(ordered_us, 50.0)
            p95_us = _percentile(ordered_us, 95.0)
            eff_b = _effective_bus_bytes(cname, nbytes, world_size)
            eff_bw_gbps = (eff_b / max(1e-6, mean_us)) * 1e-3
            alg_bw_gbps = (float(nbytes) / max(1e-6, mean_us)) * 1e-3
            samples.append(
                CommSample(
                    message_bytes=int(nbytes),
                    latency_us=float(mean_us),
                    p50_latency_us=float(p50_us),
                    p95_latency_us=float(p95_us),
                    effective_bandwidth_gbps=float(eff_bw_gbps),
                    alg_bandwidth_gbps=float(alg_bw_gbps),
                )
            )
        profiles[cname] = _build_collective_profile(
            cname, world_size, samples, emulated=emulated_flag
        )

    if "all_reduce" in profiles:
        intra_bw = profiles["all_reduce"].peak_bandwidth_gbps
        base_lat = profiles["all_reduce"].min_latency_us
    else:
        first_prof = next(iter(profiles.values()))
        intra_bw = first_prof.peak_bandwidth_gbps
        base_lat = first_prof.min_latency_us

    return CommProfileResult(
        world_size=int(world_size),
        backend=str(backend),
        device_type=str(device_type),
        device_name=dev_name,
        device_memory_bytes=int(dev_mem),
        host_staged=bool(host_staged),
        dtype=str(dtype),
        collectives=profiles,
        intra_node_bandwidth_gbps=float(intra_bw),
        base_latency_us=float(base_lat),
        metadata={
            "warmup_iters": int(warmup_iters),
            "measure_iters": int(measure_iters),
            "sizes_bytes": list(cleaned_sizes),
        },
    )


def _comm_profile_worker(
    rank: int,
    world_size: int,
    backend: str,
    device_type: str,
    sizes_bytes: Tuple[int, ...],
    dtype: str,
    warmup_iters: int,
    measure_iters: int,
    collectives: Tuple[str, ...],
    master_addr: str,
    master_port: int,
    out_path: str,
) -> None:
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    _set_device_for_rank(device_type, rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    try:
        result = _profile_in_initialized_pg(
            backend=backend,
            device_type=device_type,
            sizes_bytes=sizes_bytes,
            dtype=dtype,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            collectives=collectives,
            pg=dist.group.WORLD,
        )
        dist.barrier()
        if rank == 0:
            result.save_json(out_path)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def profile_comm_topology(
    world_size: Optional[int] = None,
    backend: str = "gloo",
    device_type: Optional[str] = None,
    sizes_bytes: Sequence[int] = DEFAULT_PROFILE_SIZES_BYTES,
    *,
    dtype: str = "bfloat16",
    warmup_iters: int = 2,
    measure_iters: int = 5,
    collectives: Sequence[str] = SUPPORTED_COLLECTIVES,
    process_group: Any = None,
    master_addr: str = "127.0.0.1",
    master_port: Optional[int] = None,
) -> CommProfileResult:
    """Profile real multi-device collective latency and effective bandwidth.

    If ``torch.distributed`` is already initialized, profiles directly inside
    ``process_group`` (defaulting to ``WORLD``). Otherwise spawns ``world_size``
    local workers using ``torch.multiprocessing.spawn``, collects rank-0
    synchronized results, and returns the ``CommProfileResult``.
    """
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    resolved_dev = (device_type or _detect_default_device_type()).strip().lower()
    if dist.is_initialized():
        pg = process_group if process_group is not None else dist.group.WORLD
        active_backend = backend or str(dist.get_backend(pg))
        return _profile_in_initialized_pg(
            backend=active_backend,
            device_type=resolved_dev,
            sizes_bytes=sizes_bytes,
            dtype=dtype,
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            collectives=collectives,
            pg=pg,
        )

    if world_size is None:
        if resolved_dev == "cuda" and torch.cuda.is_available():
            world_size = max(2, torch.cuda.device_count())
        else:
            world_size = 2
    w = int(world_size)
    if w < 2:
        raise ValueError(f"profile_comm_topology requires world_size >= 2, got {w}")

    port = int(master_port) if master_port is not None else _find_free_port()
    with tempfile.TemporaryDirectory() as tmpd:
        out_path = str(Path(tmpd) / "comm_profile_result.json")
        with _prefer_loaded_triton_in_spawn():
            mp.spawn(
                _comm_profile_worker,
                args=(
                    w,
                    str(backend),
                    resolved_dev,
                    tuple(int(s) for s in sizes_bytes),
                    str(dtype),
                    int(warmup_iters),
                    int(measure_iters),
                    tuple(str(c) for c in collectives),
                    str(master_addr),
                    port,
                    out_path,
                ),
                nprocs=w,
                join=True,
            )
        return CommProfileResult.load_json(out_path)


__all__ = [
    "CollectiveProfile",
    "CommProfileResult",
    "CommSample",
    "DEFAULT_PROFILE_SIZES_BYTES",
    "SUPPORTED_COLLECTIVES",
    "needs_host_staging",
    "profile_comm_topology",
]
