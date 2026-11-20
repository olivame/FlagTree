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

"""Unified backend capability registry for portable shape bucketing and runtime guards.

Centralizes hardware/compiler-specific parameters (alignment divisibility, minimum
dot/load tile sizes, static grid baking, default pipeline stages, collective device
mapping, and Device Graph capture APIs) so core runtime modules and FlagGems
wrappers avoid hardcoded vendor branches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
import os
import threading
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class BackendBucketingCapabilities:
    """Declarative hardware and compiler capabilities for one Triton backend."""

    backend: str = "default"
    divisibility: int = 16
    min_dot_size: Tuple[int, int, int] = (16, 16, 16)
    min_load_size: int = 1
    bakes_static_grid: bool = False
    default_num_stages: int = 3
    candidate_generator_module: Optional[str] = None
    candidate_generator_attr: str = "block_size_candidates"
    collective_device: Optional[str] = None
    collective_backends: Tuple[str, ...] = ()
    graph_capture_apis: Tuple[Tuple[str, str, str], ...] = (
        ("npu", "NPUGraph", "graph"),
        ("cuda", "CUDAGraph", "graph"),
        ("xpu", "XPUGraph", "graph"),
        ("mlu", "MLUGraph", "graph"),
        ("musa", "MUSAGraph", "graph"),
    )
    alignment_spec_tags: Tuple[str, ...] = ("D",)
    supports_event_timing: bool = True


_REGISTRY_LOCK = threading.Lock()

_DEFAULT_CAPABILITIES: Dict[str, BackendBucketingCapabilities] = {
    "default": BackendBucketingCapabilities(backend="default"),
    "nvidia": BackendBucketingCapabilities(
        backend="nvidia",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        collective_device="cuda",
        collective_backends=("nccl", "flagcx"),
        graph_capture_apis=(("cuda", "CUDAGraph", "graph"),),
    ),
    "cuda": BackendBucketingCapabilities(
        backend="cuda",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        collective_device="cuda",
        collective_backends=("nccl", "flagcx"),
        graph_capture_apis=(("cuda", "CUDAGraph", "graph"),),
    ),
    "amd": BackendBucketingCapabilities(
        backend="amd",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        collective_device="cuda",
        collective_backends=("nccl", "rccl", "flagcx"),
        graph_capture_apis=(("cuda", "CUDAGraph", "graph"),),
    ),
    "ppu": BackendBucketingCapabilities(
        backend="ppu",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        collective_device="cuda",
        collective_backends=("nccl", "pccl", "flagcx"),
        graph_capture_apis=(("cuda", "CUDAGraph", "graph"),),
    ),
    "thead": BackendBucketingCapabilities(
        backend="thead",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        collective_device="cuda",
        collective_backends=("nccl", "pccl", "flagcx"),
        graph_capture_apis=(("cuda", "CUDAGraph", "graph"),),
    ),
    "xpu": BackendBucketingCapabilities(
        backend="xpu",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        bakes_static_grid=True,
        default_num_stages=2,
        candidate_generator_module="triton.runtime._xpu_autotune_candidates",
        collective_device="xpu",
        collective_backends=("xccl", "bkcl", "nccl", "flagcx"),
        graph_capture_apis=(("xpu", "XPUGraph", "graph"), ("cuda", "CUDAGraph", "graph")),
        supports_event_timing=False,
    ),
    "kunlunxin": BackendBucketingCapabilities(
        backend="kunlunxin",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        bakes_static_grid=True,
        default_num_stages=2,
        candidate_generator_module="triton.runtime._xpu_autotune_candidates",
        collective_device="xpu",
        collective_backends=("xccl", "bkcl", "nccl", "flagcx"),
        graph_capture_apis=(("xpu", "XPUGraph", "graph"), ("cuda", "CUDAGraph", "graph")),
        supports_event_timing=False,
    ),
    "ascend": BackendBucketingCapabilities(
        backend="ascend",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        collective_device="npu",
        collective_backends=("hccl", "flagcx"),
        graph_capture_apis=(("npu", "NPUGraph", "graph"),),
    ),
    "cambricon": BackendBucketingCapabilities(
        backend="cambricon",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        collective_device="mlu",
        collective_backends=("cncl", "flagcx"),
        graph_capture_apis=(("mlu", "MLUGraph", "graph"),),
    ),
    "mlu": BackendBucketingCapabilities(
        backend="mlu",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        collective_device="mlu",
        collective_backends=("cncl", "flagcx"),
        graph_capture_apis=(("mlu", "MLUGraph", "graph"),),
    ),
    "mthreads": BackendBucketingCapabilities(
        backend="mthreads",
        divisibility=16,
        min_dot_size=(64, 16, 16),
        collective_device="musa",
        collective_backends=("mccl", "flagcx"),
        graph_capture_apis=(("musa", "MUSAGraph", "graph"),),
    ),
    "musa": BackendBucketingCapabilities(
        backend="musa",
        divisibility=16,
        min_dot_size=(64, 16, 16),
        collective_device="musa",
        collective_backends=("mccl", "flagcx"),
        graph_capture_apis=(("musa", "MUSAGraph", "graph"),),
    ),
    "enflame": BackendBucketingCapabilities(
        backend="enflame",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        min_load_size=4,
        collective_device="gcu",
        collective_backends=("eccl", "flagcx"),
    ),
    "sunrise": BackendBucketingCapabilities(
        backend="sunrise",
        divisibility=16,
        min_dot_size=(8, 8, 16),
    ),
    "metax": BackendBucketingCapabilities(
        backend="metax",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        collective_device="cuda",
        collective_backends=("nccl", "mccl", "flagcx"),
        graph_capture_apis=(("cuda", "CUDAGraph", "graph"),),
    ),
    "hygon": BackendBucketingCapabilities(
        backend="hygon",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        collective_device="cuda",
        collective_backends=("nccl", "rccl", "flagcx"),
        graph_capture_apis=(("cuda", "CUDAGraph", "graph"),),
    ),
    "iluvatar": BackendBucketingCapabilities(
        backend="iluvatar",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        collective_device="cuda",
        collective_backends=("nccl", "flagcx"),
        graph_capture_apis=(("cuda", "CUDAGraph", "graph"),),
    ),
}

_BACKEND_ALIASES: Dict[str, str] = {
    "cann": "ascend",
    "npu": "ascend",
    "hip": "amd",
    "rocm": "amd",
    "gcu": "enflame",
}


def canonicalize_backend_name(name: Optional[str]) -> str:
    """Normalize vendor/backend aliases into canonical registry keys."""
    if not name:
        return "default"
    norm = str(name).strip().lower()
    return _BACKEND_ALIASES.get(norm, norm)


def detect_active_backend_name() -> str:
    """Detect the active Triton/FlagTree backend without hardcoding a vendor."""
    env_bk = os.environ.get("FLAGTREE_BACKEND", "").strip().lower()
    if env_bk:
        return canonicalize_backend_name(env_bk)
    env_triton_bk = os.environ.get("TRITON_BACKEND", "").strip().lower()
    if env_triton_bk:
        return canonicalize_backend_name(env_triton_bk)
    try:
        from triton._flagtree_backend import FLAGTREE_BACKEND
        if FLAGTREE_BACKEND:
            return canonicalize_backend_name(FLAGTREE_BACKEND)
    except Exception:
        pass
    try:
        from .driver import driver
        target = driver.active.get_current_target()
        bk = getattr(target, "backend", None)
        if bk:
            return canonicalize_backend_name(bk)
    except Exception:
        pass
    return "default"


_ACTIVE_CAPS_CACHE: Dict[Tuple[str, str, str, str], BackendBucketingCapabilities] = {}
_SORTED_TAGS_CACHE: Dict[Tuple[str, ...], Tuple[str, ...]] = {}


def register_backend_capabilities(
    backend: str,
    capabilities: BackendBucketingCapabilities,
) -> None:
    """Register or override bucketing capabilities for a backend plugin."""
    key = canonicalize_backend_name(backend)
    with _REGISTRY_LOCK:
        _DEFAULT_CAPABILITIES[key] = capabilities
        _ACTIVE_CAPS_CACHE.clear()


_COMPILER_CAPS_CACHE: Dict[Any, Optional[BackendBucketingCapabilities]] = {}


def _probe_compiler_backend_capabilities() -> Optional[BackendBucketingCapabilities]:
    """Query the active BaseBackend subclass (cached per target) if it exposes get_bucketing_capabilities."""
    try:
        from .driver import driver
        target = driver.active.get_current_target()
        if target is None:
            return None
        if target in _COMPILER_CAPS_CACHE:
            return _COMPILER_CAPS_CACHE[target]
        from ..compiler.compiler import make_backend
        backend_obj = make_backend(target)
        getter = getattr(backend_obj, "get_bucketing_capabilities", None)
        res = None
        if callable(getter):
            caps = getter()
            if isinstance(caps, BackendBucketingCapabilities):
                res = caps
        _COMPILER_CAPS_CACHE[target] = res
        return res
    except Exception:
        return None


def get_backend_capabilities(
    backend: Optional[str] = None,
    *,
    vendor_name: Optional[str] = None,
) -> BackendBucketingCapabilities:
    """Resolve effective BackendBucketingCapabilities for the active or requested backend."""
    if backend is None and vendor_name is None:
        env = os.environ
        env_key = (
            env.get("FLAGTREE_BACKEND", ""),
            env.get("TRITON_BACKEND", ""),
            env.get("FLAGTREE_BUCKET_DIVISIBILITY", ""),
            env.get("FLAGTREE_BAKES_STATIC_GRID", ""),
        )
        cached = _ACTIVE_CAPS_CACHE.get(env_key)
        if cached is not None:
            return cached
    else:
        env_key = None

    explicit = canonicalize_backend_name(backend) if backend is not None else None
    if explicit is None:
        compiler_caps = _probe_compiler_backend_capabilities()
        if compiler_caps is not None:
            base = compiler_caps
        else:
            active = detect_active_backend_name()
            if active == "default" and vendor_name:
                active = canonicalize_backend_name(vendor_name)
            with _REGISTRY_LOCK:
                base = _DEFAULT_CAPABILITIES.get(
                    active,
                    replace(_DEFAULT_CAPABILITIES["default"], backend=active),
                )
    else:
        with _REGISTRY_LOCK:
            base = _DEFAULT_CAPABILITIES.get(
                explicit,
                replace(_DEFAULT_CAPABILITIES["default"], backend=explicit),
            )

    env_div = os.environ.get("FLAGTREE_BUCKET_DIVISIBILITY", "").strip()
    if env_div:
        try:
            div_val = max(1, int(env_div))
            base = replace(base, divisibility=div_val)
        except ValueError:
            pass

    env_static_grid = os.environ.get("FLAGTREE_BAKES_STATIC_GRID", "").strip().lower()
    if env_static_grid in ("1", "true", "on", "yes"):
        base = replace(base, bakes_static_grid=True)
    elif env_static_grid in ("0", "false", "off", "no"):
        base = replace(base, bakes_static_grid=False)

    if env_key is not None:
        _ACTIVE_CAPS_CACHE[env_key] = base
    return base


def resolve_candidate_generator(
    caps: Optional[BackendBucketingCapabilities] = None,
) -> Optional[Callable[..., Any]]:
    """Dynamically load an optional backend candidate generator hook."""
    effective = caps or get_backend_capabilities()
    mod_name = effective.candidate_generator_module
    if not mod_name:
        return None
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, effective.candidate_generator_attr, None)
    except Exception:
        return None


_INT_SPEC_TYPES = frozenset(("i32", "i64", "u32", "u64"))


def strip_int_alignment_spec(
    spec: Any,
    caps: Optional[BackendBucketingCapabilities] = None,
) -> Tuple[Any, bool, bool]:
    """Normalize a JIT specialization tuple without hardcoding backend tag strings.

    Returns ``(normalized_spec, was_aligned_tag, is_unaligned_int_scalar)``.
    """
    if type(spec) is not tuple or len(spec) != 2:
        return spec, False, False
    ty, sp = spec
    if ty not in _INT_SPEC_TYPES or type(sp) is not str:
        return spec, False, False
    if sp == "":
        return spec, False, True
    effective = caps if caps is not None else get_backend_capabilities()
    tags = effective.alignment_spec_tags
    if tags == ("D",):
        if sp == "D":
            return (ty, ""), True, False
        if "D" in sp:
            return (ty, sp.replace("D", "")), True, False
        return spec, False, False
    sorted_tags = _SORTED_TAGS_CACHE.get(tags)
    if sorted_tags is None:
        sorted_tags = tuple(sorted(tags, key=len, reverse=True))
        _SORTED_TAGS_CACHE[tags] = sorted_tags
    stripped = sp
    changed = False
    for tag in sorted_tags:
        if tag in stripped:
            stripped = stripped.replace(tag, "")
            changed = True
    if changed:
        return (ty, stripped), True, False
    return spec, False, False


def resolve_collective_device(
    pg_backend: str = "",
    caps: Optional[BackendBucketingCapabilities] = None,
) -> str:
    """Resolve the torch device string suitable for collective all_reduce."""
    pg = (pg_backend or "").strip().lower()
    if pg == "gloo":
        return "cpu"
    try:
        import torch
    except Exception:
        return "cpu"

    effective = caps or get_backend_capabilities()
    preferred = effective.collective_device
    if preferred and preferred != "cpu":
        dev_mod = getattr(torch, preferred, None)
        if dev_mod is not None and callable(getattr(dev_mod, "is_available", None)):
            try:
                if dev_mod.is_available():
                    return preferred
            except Exception:
                pass

    probe_order = (
        ("hccl", "npu"),
        ("xccl", "xpu"),
        ("bkcl", "xpu"),
        ("cncl", "mlu"),
        ("mccl", "musa"),
        ("pccl", "cuda"),
        ("nccl", "cuda"),
        ("rccl", "cuda"),
    )
    for backend_key, dev_name in probe_order:
        if pg == backend_key:
            dev_mod = getattr(torch, dev_name, None)
            if dev_mod is not None and callable(getattr(dev_mod, "is_available", None)):
                try:
                    if dev_mod.is_available():
                        return dev_name
                except Exception:
                    pass

    for dev_name in ("npu", "xpu", "mlu", "musa", "cuda"):
        dev_mod = getattr(torch, dev_name, None)
        if dev_mod is not None and callable(getattr(dev_mod, "is_available", None)):
            try:
                if dev_mod.is_available():
                    return dev_name
            except Exception:
                pass
    return "cpu"


__all__ = [
    "BackendBucketingCapabilities",
    "canonicalize_backend_name",
    "detect_active_backend_name",
    "get_backend_capabilities",
    "register_backend_capabilities",
    "resolve_candidate_generator",
    "resolve_collective_device",
    "strip_int_alignment_spec",
]
