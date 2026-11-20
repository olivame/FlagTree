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

"""Compiler- and model-graph-level shape bucketing front-end.

Bridges F09 shape bucketing with PyTorch ``torch.compile`` / TorchInductor,
Core ATen ``ExportedProgram`` / FX ``GraphModule``, and framework-imported
(ONNX/TensorFlow -> FX/Core ATen) computation graphs. Provides unified
model-level cache keys, segmented symbolic dynamic-shape guards, automatic
profiling, and OOM recovery integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .backend_caps import detect_active_backend_name, get_backend_capabilities
from .bucket_guard import BucketSegmentPlan, SymbolicBucketGuard
from ._distributed import GLOBAL_FALLBACK_REGISTRY, OOMRecoveryController
from .shape_bucketing import BucketingMode, OnlineBucketPolicy, warmup_buckets
from .shape_profiler import ProfileMode, ShapeProfiler


@dataclass(frozen=True)
class ModelCacheKey:
    """Unified model-level cache key bridging F09 shape buckets and F12 cache governance."""

    model_id: str
    graph_hash: str
    bucket_intervals: Tuple[Tuple[str, int, int], ...]
    dtypes: Tuple[str, ...]
    backend: str
    mode: str
    triton_version: str
    pass_signature: str = "default"

    def digest(self) -> str:
        payload = json.dumps(
            {
                "model_id": self.model_id,
                "graph_hash": self.graph_hash,
                "bucket_intervals": list(self.bucket_intervals),
                "dtypes": list(self.dtypes),
                "backend": self.backend,
                "mode": self.mode,
                "triton_version": self.triton_version,
                "pass_signature": self.pass_signature,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_graph_hash(model_or_graph: Any) -> str:
    """Compute a stable structural hash for nn.Module, FX GraphModule, ExportedProgram, or IR dict."""
    parts: List[str] = [type(model_or_graph).__qualname__]
    # Core ATen ExportedProgram
    gm = getattr(model_or_graph, "graph_module", None)
    if gm is not None and hasattr(gm, "graph"):
        parts.append(str(gm.graph))
    elif hasattr(model_or_graph, "graph") and hasattr(model_or_graph.graph, "nodes"):
        parts.append(str(model_or_graph.graph))
    elif isinstance(model_or_graph, Mapping):
        parts.append(json.dumps(model_or_graph, sort_keys=True, default=str))
    elif hasattr(model_or_graph, "named_modules"):
        try:
            for name, mod in model_or_graph.named_modules():
                parts.append(f"{name}:{type(mod).__name__}")
        except Exception:
            parts.append(repr(model_or_graph)[:512])
    else:
        parts.append(getattr(model_or_graph, "__name__", repr(model_or_graph)[:256]))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def build_model_cache_key(
    model_or_graph: Any,
    plan: BucketSegmentPlan,
    dtypes: Sequence[str],
    *,
    model_id: Optional[str] = None,
    backend: Optional[str] = None,
    mode: BucketingMode = BucketingMode.AUTO,
    pass_signature: str = "default",
) -> ModelCacheKey:
    """Construct a deterministic ModelCacheKey for one bucketed graph invocation."""
    try:
        import triton
        t_ver = str(getattr(triton, "__version__", "unknown"))
    except Exception:
        t_ver = "unknown"
    intervals = tuple(
        (ax, int(lo), int(hi))
        for ax, (lo, hi) in sorted(plan.axis_intervals.items())
    )
    return ModelCacheKey(
        model_id=model_id or getattr(model_or_graph, "__name__", type(model_or_graph).__name__),
        graph_hash=_compute_graph_hash(model_or_graph),
        bucket_intervals=intervals,
        dtypes=tuple(str(d) for d in dtypes),
        backend=backend or detect_active_backend_name(),
        mode=mode.value if hasattr(mode, "value") else str(mode),
        triton_version=t_ver,
        pass_signature=pass_signature,
    )


class GraphBucketCompiler:
    """Model/graph-level bucketed compilation & execution wrapper.

    Supports:
      - PyTorch ``nn.Module`` with optional ``torch.compile`` + ``SymbolicBucketGuard``
      - Core ATen ``ExportedProgram`` / FX ``GraphModule`` (from PyTorch/ONNX/TF frontends)
      - Arbitrary callable models with dynamic tensor dimensions
    """

    def __init__(
        self,
        model_or_graph: Any,
        *,
        dynamic_dims: Optional[Mapping[int, Mapping[int, str]]] = None,
        mode: BucketingMode = BucketingMode.AUTO,
        policy: Optional[OnlineBucketPolicy] = None,
        use_torch_compile: bool = False,
        compile_backend: str = "inductor",
        compile_kwargs: Optional[Mapping[str, Any]] = None,
        enable_profiler: bool = False,
        profiler_mode: ProfileMode = ProfileMode.GENERIC_GRAPH,
        activation_budget_bytes: Optional[int] = None,
        bytes_per_token_activation: int = 0,
        enable_oom_recovery: bool = True,
        model_id: Optional[str] = None,
    ) -> None:
        self.raw_model = model_or_graph
        self.model_id = model_id or getattr(model_or_graph, "__name__", type(model_or_graph).__name__)
        # dynamic_dims: {arg_index: {tensor_dim_index: axis_name}}
        self.dynamic_dims: Dict[int, Dict[int, str]] = {
            int(arg_i): {int(d): str(ax) for d, ax in dim_map.items()}
            for arg_i, dim_map in (dynamic_dims or {}).items()
        }
        self.mode = BucketingMode(mode) if isinstance(mode, str) else mode
        caps = get_backend_capabilities()
        all_dyn_axes = {
            ax for dim_map in self.dynamic_dims.values() for ax in dim_map.values()
        }
        self.policy = policy or OnlineBucketPolicy(
            mode=mode,
            divisibility=caps.divisibility,
            backend=caps.backend,
            op_name=self.model_id,
            dynamic_axes=all_dyn_axes,
        )
        if all_dyn_axes:
            self.policy.mark_dynamic_axes(all_dyn_axes, unaligned=True)
        self.guard = SymbolicBucketGuard(
            policy=self.policy,
            mode=mode,
            activation_budget_bytes=activation_budget_bytes,
            bytes_per_token_activation=bytes_per_token_activation,
        )
        self.use_torch_compile = bool(use_torch_compile)
        self.compile_backend = compile_backend
        self.compile_kwargs = dict(compile_kwargs or {})
        self.profiler = ShapeProfiler(mode=profiler_mode) if enable_profiler else None
        self.enable_oom_recovery = bool(enable_oom_recovery)
        self.oom_controller = OOMRecoveryController(registry=GLOBAL_FALLBACK_REGISTRY)
        self._compiled_fn: Any = None
        self._seen_model_keys: Dict[str, ModelCacheKey] = {}
        self.last_plan: Optional[BucketSegmentPlan] = None
        self.last_cache_key: Optional[ModelCacheKey] = None
        self.step_count: int = 0

        # Resolve callable from ExportedProgram if provided
        if hasattr(model_or_graph, "module") and callable(model_or_graph.module):
            self._base_callable = model_or_graph.module()
        else:
            self._base_callable = model_or_graph

    @classmethod
    def from_exported_program(cls, exported_program: Any, **kwargs: Any) -> "GraphBucketCompiler":
        """Wrap a Core ATen ``torch.export.ExportedProgram`` (PyTorch/ONNX/TF unified IR)."""
        return cls(exported_program, **kwargs)

    @classmethod
    def from_fx_graph(cls, graph_module: Any, **kwargs: Any) -> "GraphBucketCompiler":
        """Wrap an FX ``GraphModule`` produced by a frontend converter."""
        return cls(graph_module, **kwargs)

    def _get_executor(self) -> Any:
        if not self.use_torch_compile:
            return self._base_callable
        if self._compiled_fn is not None:
            return self._compiled_fn
        try:
            import torch
            if hasattr(torch, "compile"):
                self._compiled_fn = torch.compile(
                    self._base_callable,
                    backend=self.compile_backend,
                    dynamic=True,
                    **self.compile_kwargs,
                )
                return self._compiled_fn
        except Exception:
            pass
        self._compiled_fn = self._base_callable
        return self._compiled_fn

    def _extract_dims_and_dtypes(
        self, args: Tuple[Any, ...]
    ) -> Tuple[Dict[str, int], List[str], int]:
        dims: Dict[str, int] = {}
        dtypes: List[str] = []
        batch_size = 1
        for idx, arg in enumerate(args):
            if hasattr(arg, "dtype") and hasattr(arg, "shape"):
                dtypes.append(str(arg.dtype))
                shape = tuple(int(d) for d in arg.shape)
                if idx == 0 and shape:
                    batch_size = max(1, shape[0])
                dim_map = self.dynamic_dims.get(idx)
                if dim_map:
                    for dim_i, ax_name in dim_map.items():
                        if 0 <= dim_i < len(shape):
                            dims[ax_name] = shape[dim_i]
        return dims, dtypes, batch_size

    def warmup(
        self,
        axis_ranges: Mapping[str, Tuple[int, int]],
        sample_builder: Callable[[Dict[str, int]], Tuple[Tuple[Any, ...], Dict[str, Any]]],
        *,
        include_unaligned_tail: bool = True,
    ) -> List[Dict[str, int]]:
        """Pre-warm all bucket intervals for this model/graph before serving traffic."""
        return warmup_buckets(
            self.__call__,
            axis_ranges,
            sample_builder,
            policy=self.policy,
            mode=self.mode,
            include_unaligned_tail=include_unaligned_tail,
            divisibility=self.policy.divisibility,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from .. import knobs

        dims, dtypes, batch_size = self._extract_dims_and_dtypes(args)
        self.step_count += 1
        plan = self.guard.plan_step(dims, batch_size=batch_size, step_id=self.step_count)
        self.last_plan = plan

        # Apply zero-padding segmented mark_dynamic bounds to dynamic tensors
        if self.use_torch_compile:
            for arg_i, dim_map in self.dynamic_dims.items():
                if 0 <= arg_i < len(args):
                    self.guard.mark_tensor_dynamic(args[arg_i], dim_map, plan=plan)

        mkey = build_model_cache_key(
            self.raw_model,
            plan,
            dtypes,
            model_id=self.model_id,
            backend=self.policy.backend,
            mode=self.mode,
            pass_signature=self.compile_backend if self.use_torch_compile else "eager",
        )
        self.last_cache_key = mkey
        self._seen_model_keys[mkey.digest()] = mkey

        executor = self._get_executor()

        def _invoke() -> Any:
            with knobs.autotuning.scope():
                knobs.autotuning.shape_bucketing = True
                knobs.autotuning.shape_bucketing_mode = self.mode.value
                if self.profiler is not None:
                    self.profiler.step()
                    self.profiler.record_op(
                        self.model_id,
                        [tuple(int(d) for d in a.shape) for a in args if hasattr(a, "shape")],
                        dtypes=dtypes,
                    )
                return executor(*args, **kwargs)

        if not self.enable_oom_recovery:
            return _invoke()

        phase_name = "training" if self.mode is BucketingMode.TRAINING else "inference"
        return self.oom_controller.execute(
            self.model_id,
            phase_name,
            work_units=max(1, batch_size),
            primary_fn=_invoke,
            fallback_fn=(lambda: self._base_callable(*args, **kwargs)) if self.use_torch_compile else None,
            bucket_key=(mkey.digest(),),
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend": self.policy.backend,
            "mode": self.mode.value,
            "use_torch_compile": self.use_torch_compile,
            "steps_executed": self.step_count,
            "unique_model_cache_keys": len(self._seen_model_keys),
            "profiler": self.profiler.summary() if self.profiler is not None else None,
        }


def compile_with_bucketing(
    model_or_graph: Any,
    dynamic_dims: Optional[Mapping[int, Mapping[int, str]]] = None,
    **kwargs: Any,
) -> GraphBucketCompiler:
    """Convenience constructor wrapping a model/graph with segmented shape bucketing."""
    return GraphBucketCompiler(model_or_graph, dynamic_dims=dynamic_dims, **kwargs)


__all__ = [
    "GraphBucketCompiler",
    "ModelCacheKey",
    "build_model_cache_key",
    "compile_with_bucketing",
]


def recommended_warmup_lengths(
    bucket_bounds: Sequence[int],
    *,
    divisibility: Optional[int] = None,
    unaligned_offset: int = 4,
) -> List[int]:
    """Return one representative length per bucket that triggers the universal
    non-divisible JIT specialization during warmup so subsequent aligned and
    unaligned traffic both reuse the same compiled binary with zero recompiles.
    """
    if divisibility is None:
        from .backend_caps import get_backend_capabilities
        divisibility = max(1, int(get_backend_capabilities().divisibility))
    reps: List[int] = []
    prev_hi = 0
    step = max(1, min(int(unaligned_offset), max(1, divisibility - 1)))
    for raw_hi in sorted(set(int(b) for b in bucket_bounds if int(b) > 0)):
        rep = raw_hi
        if divisibility > 1 and (rep % divisibility) == 0 and (rep - step) > prev_hi:
            rep = rep - step
        reps.append(rep)
        prev_hi = raw_hi
    return reps
