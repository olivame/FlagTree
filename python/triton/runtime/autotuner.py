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

from __future__ import annotations

import builtins
import time
import inspect
import hashlib
import json
from functools import cached_property
from typing import Dict, Tuple, List, Optional

from .. import knobs
from .jit import KernelInterface, JITFunction
from .errors import BackendCompilerError, CompileTimeoutError, OutOfResources, PTXASError, AutotunerError
from .driver import driver
from .cache import get_cache_manager, triton_key
from .adjust_kernel_param import auto_adjust_block_sizes
from triton._C.libtriton import get_cache_invalidating_env_vars


def _resolve_xpu_block_size_candidates():
    try:
        from ._xpu_autotune_candidates import block_size_candidates
        return block_size_candidates
    except ImportError:
        return None


def _default_num_stages() -> int:
    import os
    if os.environ.get("FLAGTREE_BACKEND") == "xpu" or _resolve_xpu_block_size_candidates() is not None:
        return 2
    return 3


_EVENT_TIMER_ZERO_DIV: bool = False


def _wallclock_bench_fallback(kernel_call) -> List[float]:
    """Synchronized wall-clock fallback when device events return 0.0ms (e.g. XPU without KL3 event timers)."""
    di = driver.active.get_device_interface()
    kernel_call()
    di.synchronize()
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(8):
            kernel_call()
        di.synchronize()
        samples.append(((time.perf_counter() - t0) * 1e3) / 8.0)
    samples.sort()
    return [samples[1], samples[0], samples[2]]


class Autotuner(KernelInterface):

    def __init__(self, fn, arg_names, configs, key, reset_to_zero, restore_value, pre_hook=None, post_hook=None,
                 prune_configs_by: Optional[Dict] = None, warmup=None, rep=None, use_cuda_graph=False, do_bench=None,
                 cache_results=False, shape_policy=None, generate_configs=None, op_affiliation="", row_sign="",
                 col_sign="", n_elem_sign=""):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'early_config_prune': a function used to prune configs. It should have the signature
                `prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
                and return pruned configs. It should return at least one config.
        """
        default_stages = 2 if generate_configs is not None else 3
        if not configs:
            self.configs = [Config({}, num_warps=4, num_stages=default_stages, num_ctas=1)]
        else:
            self.configs = configs
        self.no_configs = (not configs)
        self.generate_configs = generate_configs
        self.op_affiliation = op_affiliation
        self.row_sign = row_sign
        self.col_sign = col_sign
        self.n_elem_sign = n_elem_sign
        if self.configs and (len(self.configs) > 0):
            self.shared_config_pre_hook = self.configs[0].pre_hook  # flagtree aabs
        self.keys = key
        self.cache: Dict[Tuple, Config] = {}
        self.arg_names = arg_names
        self.cache_results = (cache_results or knobs.autotuning.cache) and not knobs.runtime.interpret

        # Reset to zero or restore values
        self.reset_to_zero = []
        if reset_to_zero is not None:
            self.reset_to_zero = list(reset_to_zero)
        self.restore_value = []
        if restore_value is not None:
            self.restore_value = list(restore_value)

        # Hook to reset or restore for required tensors
        self.pre_hook = lambda kwargs, reset_only=False: 0
        self.post_hook = lambda kwargs, exception: 0
        self.user_defined_pre_hook = False
        self.user_defined_post_hook = False
        if pre_hook:
            self.pre_hook = pre_hook
            self.user_defined_pre_hook = True
        elif (len(self.reset_to_zero) > 0 or len(self.restore_value) > 0):

            def _pre_hook(kwargs, reset_only=False):
                for name in self.reset_to_zero:
                    kwargs[name].zero_()
                if not reset_only:
                    self.restore_copies = {name: kwargs[name].clone() for name in self.restore_value}

            self.pre_hook = _pre_hook

        if post_hook:
            self.post_hook = post_hook
            self.user_defined_post_hook = True
        elif len(self.restore_value) > 0:

            def _post_hook(kwargs, exception):
                for name in self.restore_value:
                    kwargs[name].copy_(self.restore_copies[name])
                self.restore_copies = {}

            self.post_hook = _post_hook

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get("early_config_prune", self.early_config_prune)

        self.fn = fn
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn

        self._do_bench = do_bench
        self.num_warmups = warmup
        self.num_reps = rep
        self.use_cuda_graph = use_cuda_graph
        self.seen_tuned_metas = {}  # flagtree aabs: deduplicate tuned meta
        self.shape_policy = shape_policy
        self._bucket_budget = None
        self.last_dispatch_result = None

        # If we got explicitly called via the old interface, raise a warning
        # and proceed with the old behavior.
        if warmup is not None or rep is not None or use_cuda_graph:
            import warnings
            warnings.warn(("warmup, rep, and use_cuda_graph parameters are deprecated. See "
                           "https://github.com/triton-lang/triton/pull/4496 for details."), DeprecationWarning,
                          stacklevel=1)
            if use_cuda_graph:
                from ..testing import do_bench_cudagraph
                self._do_bench = lambda kernel_call, quantiles: do_bench_cudagraph(
                    kernel_call,
                    rep=rep if rep is not None else 100,
                    quantiles=quantiles,
                )
                return

            import triton.testing
            self._do_bench = lambda kernel_call, quantiles: triton.testing.do_bench(
                kernel_call,
                warmup=warmup if warmup is not None else 25,
                rep=rep if rep is not None else 100,
                quantiles=quantiles,
            )
            return

    @cached_property
    def do_bench(self):
        if self._do_bench is None:
            return driver.active.get_benchmarker()
        return self._do_bench

    def _bench(self, *args, config, **meta):
        from ..compiler.errors import CompileTimeAssertionFailure

        verbose = knobs.autotuning.print
        if verbose:
            print(f"Autotuning kernel {self.base_fn.__name__} with config {config}")

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(f"Conflicting meta-parameters: {', '.join(conflicts)}."
                             " Make sure that you don't re-define auto-tuned symbols.")
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        # flagtree aabs: auto_adjust_block_sizes
        if knobs.autotuning.adjust_block_size:

            def _unwrap_to_jitfunction(fn):
                from triton.runtime.jit import JITFunction
                while not isinstance(fn, JITFunction):
                    if not hasattr(fn, 'fn'):
                        return None
                    fn = fn.fn
                return fn

            jit_fn = _unwrap_to_jitfunction(self.fn)
            if jit_fn is not None:
                auto_adjust_block_sizes(self.nargs, jit_fn, self.configs, current, config)
        meta_key = tuple(sorted(current.items()))
        if meta_key in self.seen_tuned_metas:
            return self.seen_tuned_metas[meta_key]  # flagtree aabs: deduplicate tuned meta
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(full_nargs)
            try:
                self.fn.run(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    self.post_hook(full_nargs, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            self.post_hook(full_nargs, exception=None)

        from ._async_compile import GLOBAL_COMPILE_BLACKLIST, run_with_compile_timeout
        cfg_sig = str(config)
        if GLOBAL_COMPILE_BLACKLIST.is_blacklisted(self.base_fn.__name__, cfg_sig):
            return [float("inf"), float("inf"), float("inf")]

        timeout_sec = knobs.autotuning.compile_timeout_sec

        def _invoke_bench():
            global _EVENT_TIMER_ZERO_DIV
            if _EVENT_TIMER_ZERO_DIV and self._do_bench is None:
                return _wallclock_bench_fallback(kernel_call)
            try:
                return self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
            except ZeroDivisionError:
                if self._do_bench is None:
                    _EVENT_TIMER_ZERO_DIV = True
                return _wallclock_bench_fallback(kernel_call)

        try:
            if timeout_sec is not None and timeout_sec > 0:
                rett = run_with_compile_timeout(
                    _invoke_bench,
                    timeout_sec,
                    kernel_name=self.base_fn.__name__,
                    config_desc=cfg_sig,
                )
            else:
                rett = _invoke_bench()
        # except (OutOfResources, CompileTimeAssertionFailure, PTXASError) as e:
        # flagtree tileir: prune backend compiler failures during autotuning.
        except (
            OutOfResources,
            CompileTimeAssertionFailure,
            PTXASError,
            BackendCompilerError,
            CompileTimeoutError,
            RuntimeError,
        ) as e:
            if isinstance(e, RuntimeError) and not isinstance(
                e,
                (
                    OutOfResources,
                    CompileTimeAssertionFailure,
                    PTXASError,
                    BackendCompilerError,
                    CompileTimeoutError,
                ),
            ):
                err_msg = str(e)
                if not any(
                    marker in err_msg
                    for marker in (
                        "PassManager::run failed",
                        "out of resource",
                        "OutOfResources",
                        "shared memory",
                        "exceed",
                    )
                ):
                    raise
            if isinstance(e, (BackendCompilerError, CompileTimeoutError)):
                GLOBAL_COMPILE_BLACKLIST.add(self.base_fn.__name__, cfg_sig, str(e))
            if isinstance(e, CompileTimeoutError):
                from ._distributed import GLOBAL_FALLBACK_REGISTRY, PerfRiskLevel
                GLOBAL_FALLBACK_REGISTRY.record(
                    self.base_fn.__name__,
                    "autotune",
                    (),
                    (),
                    reason=str(e),
                    recovery_action="blacklist_config_and_select_next_candidate",
                    perf_risk_level=PerfRiskLevel.LOW,
                )
            if verbose:
                print(f"Autotuning failed with {e}")
            rett = [float("inf"), float("inf"), float("inf")]

        self.seen_tuned_metas[meta_key] = rett  # flagtree aabs: deduplicate tuned meta
        return rett

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        # We can't serialize prehooks, so just give up and run the benchmarks.
        if not tuning_key or any(cfg.pre_hook for cfg in configs):
            bench_fn()
            return False

        from triton.compiler.compiler import make_backend

        fn = self.fn
        while not isinstance(fn, JITFunction):
            fn = fn.fn

        env_vars = get_cache_invalidating_env_vars()
        cache_key = [
            triton_key(),
            make_backend(driver.active.get_current_target()).hash(),
            fn.cache_key,
            str(sorted(env_vars.items())),
            str(tuning_key),
        ] + [str(c) for c in configs]
        cache_key = hashlib.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
        cache = get_cache_manager(cache_key)
        file_name = f"{fn.__name__[:150]}.autotune.json"
        path = cache.get_file(file_name)
        if path:
            with open(path, "r") as cached_configs:
                timings = json.load(cached_configs)["configs_timings"]
                timings = {Config(**config): timing for config, timing in timings}
                self.cache[tuning_key] = builtins.min(timings, key=timings.get)
                self.configs_timings = timings
            return True

        bench_fn()
        cache.put(
            json.dumps({
                "key":
                tuning_key,
                "configs_timings":
                [(config.__dict__, timings) for config, timings in self.configs_timings.items() if not config.pre_hook],
            }), file_name, binary=False)
        max_artifacts = knobs.autotuning.max_bucket_artifacts
        if max_artifacts is not None and max_artifacts > 0:
            from .shape_bucketing import enforce_disk_cache_budget
            enforce_disk_cache_budget(knobs.cache.dir, max_artifacts)
        return False

    def run(self, *args, **kwargs):
        self.seen_tuned_metas = {}  # flagtree aabs: deduplicate tuned meta
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        bucketing_enabled = (self.shape_policy is not None) or bool(knobs.autotuning.shape_bucketing)
        has_dynamic_cand_gen = self.no_configs and self.generate_configs is not None
        if len(self.configs) > 1 or has_dynamic_cand_gen:
            all_args = {**self.nargs, **kwargs}
            _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            raw_key = [_args[k] for k in self.keys if k in _args]
            extra_dtypes = [str(arg.dtype) for _, arg in _args.items() if hasattr(arg, "dtype")]
            raw_key.extend(extra_dtypes)
            raw_key = tuple(raw_key)

            lookup = None
            if bucketing_enabled:
                from .shape_bucketing import (
                    BucketCacheBudget,
                    BucketLookupResult,
                    DispatchStatus,
                    GLOBAL_BUCKET_TELEMETRY,
                    resolve_tuner_policy,
                )
                policy = resolve_tuner_policy(
                    self,
                    explicit_policy=self.shape_policy,
                    policy_path=knobs.autotuning.shape_policy_path,
                    mode_str=knobs.autotuning.shape_bucketing_mode,
                )
                lookup = policy.lookup_autotune_key(_args, self.keys, extra_dtypes=extra_dtypes)
                key = lookup.bucket_key
            else:
                key = raw_key

            if has_dynamic_cand_gen and not (bucketing_enabled and key in self.cache):
                cand_fn = _resolve_xpu_block_size_candidates()
                if cand_fn is not None:
                    cand_nargs = (
                        {**self.nargs, **lookup.representative_dims}
                        if (bucketing_enabled and lookup is not None)
                        else self.nargs
                    )
                    cands = cand_fn(
                        cand_nargs,
                        self.generate_configs,
                        self.op_affiliation,
                        self.row_sign,
                        self.col_sign,
                        self.n_elem_sign,
                    )
                    import os as _os
                    _max_cands = _os.environ.get("FLAGTREE_MAX_AUTOTUNE_CANDIDATES")
                    if _max_cands is not None and cands and int(_max_cands) > 0 and len(cands) > int(_max_cands):
                        _k = int(_max_cands)
                        if _k == 1:
                            cands = [cands[0]]
                        else:
                            _idxs = [round(i * (len(cands) - 1) / (_k - 1)) for i in range(_k)]
                            cands = [cands[idx] for idx in dict.fromkeys(_idxs)]
                    self.configs = cands if cands else [Config({}, num_warps=4, num_stages=2, num_ctas=1)]

            guard_mismatch = False
            if bucketing_enabled and key in self.cache and self.early_config_prune is not None:
                valid_cached = self.early_config_prune([self.cache[key]], self.nargs, **kwargs)
                if not valid_cached:
                    guard_mismatch = True
                    key = key + ("guard_override",) + tuple(
                        _args[k] for k in self.keys if k in _args and isinstance(_args[k], (int, str, float, bool))
                    )

            if key not in self.cache:
                if len(self.configs) <= 1:
                    self.cache[key] = self.configs[0]
                else:
                    used_cached_result = False
                    pruned_configs = self.prune_configs(kwargs)

                    def benchmark():
                        bench_start = time.time()
                        timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                        bench_end = time.time()
                        self.bench_time = bench_end - bench_start
                        self.cache[key] = builtins.min(timings, key=timings.get)
                        full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                        self.pre_hook(full_nargs, reset_only=True)
                        self.configs_timings = timings

                    if self.cache_results:
                        used_cached_result = self.check_disk_cache(key, pruned_configs, benchmark)
                    else:
                        benchmark()

            config = self.cache[key]
            if bucketing_enabled and lookup is not None:
                if guard_mismatch:
                    eff_status = DispatchStatus.GUARD_MISMATCH
                elif lookup.status in (DispatchStatus.UNDERFLOW, DispatchStatus.OVERFLOW, DispatchStatus.GUARD_MISMATCH):
                    eff_status = lookup.status
                elif not used_cached_result:
                    eff_status = DispatchStatus.COLD_TUNED
                else:
                    eff_status = DispatchStatus.HIT
                self.last_dispatch_result = BucketLookupResult(
                    bucket_key=key,
                    status=eff_status,
                    representative_dims=lookup.representative_dims,
                    raw_dims=lookup.raw_dims,
                    clamped_axes=lookup.clamped_axes,
                )
                GLOBAL_BUCKET_TELEMETRY.record(self.base_fn.__name__, eff_status, raw_key, key)
                max_artifacts = knobs.autotuning.max_bucket_artifacts
                if max_artifacts is not None and max_artifacts > 0:
                    if self._bucket_budget is None:
                        self._bucket_budget = BucketCacheBudget(max_artifacts)
                    self._bucket_budget.touch_and_evict(self.cache, key, max_entries=max_artifacts)
        else:
            config = self.configs[0]
        self.best_config = config
        if knobs.autotuning.print and not used_cached_result:
            print(f"Triton autotuning for function {self.base_fn.__name__},\nwith key as {key},\n"
                  f"finished after {self.bench_time:.2f}s,\nbest config selected: {self.best_config};")
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret

    def prune_configs(self, kwargs: Dict) -> List[Config]:
        # flagtree aabs: use deepcopy to prevent modification of the original configs
        import copy
        pruned_configs = copy.deepcopy(self.configs)
        # pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(pruned_configs, self.nargs, **kwargs)
            if not pruned_configs:
                raise AutotunerError(
                    "No valid autotuner configs after pruning. `early_config_prune` should return at least one config.")
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            elif not isinstance(top_k, int):
                # Slice index must be an integer
                raise TypeError("Error while pruning configs, top_k must be either 1) a float <= 1.0 or 2) an int")

            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for autotune_config in self.prune_configs(kwargs):
            ret.append(self.fn.warmup(
                *args,
                **kwargs,
                **autotune_config.all_kwargs(),
            ))
        self.nargs = None
        return ret


class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_stages: int
    :ivar num_ctas: number of blocks in a block cluster. SM90+ only.
    :type num_ctas: int
    :type maxnreg: Optional[int]
    :ivar maxnreg: maximum number of registers one thread can use.  Corresponds
                       to ptx .maxnreg directive.  Not supported on all platforms.
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    :ivar ir_override: filename of a user-defined IR (*.{ttgir|llir|ptx|amdgcn}).
    """

    def __init__(self, kwargs, num_warps=4, num_stages=None, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.num_stages = _default_num_stages() if num_stages is None else num_stages
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook
        self.ir_override = ir_override

    def __setstate__(self, state):
        self.kwargs = state.get("kwargs", {})
        self.num_warps = state.get("num_warps", 4)
        self.num_stages = state.get("num_stages", _default_num_stages())
        self.num_ctas = state.get("num_ctas", 1)
        self.maxnreg = state.get("maxnreg", None)
        self.pre_hook = state.get("pre_hook", None)
        self.ir_override = state.get("ir_override", None)

    def all_kwargs(self):
        return {
            **self.kwargs, **{
                k: v
                for (k, v) in (
                    ("num_warps", self.num_warps),
                    ("num_ctas", self.num_ctas),
                    ("num_stages", self.num_stages),
                    ("maxnreg", self.maxnreg),
                    ("ir_override", self.ir_override),
                ) if v is not None
            }
        }

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        res.append(f"num_warps: {self.num_warps}")
        res.append(f"num_ctas: {self.num_ctas}")
        res.append(f"num_stages: {self.num_stages}")
        res.append(f"maxnreg: {self.maxnreg}")
        return ", ".join(res)

    def __hash__(self):
        return hash((*self.all_kwargs().items(), self.pre_hook))

    def __eq__(self, other):
        self_tuple = tuple((
            *self.all_kwargs().items(),
            self.pre_hook,
        ))
        other_tuple = tuple((
            *other.all_kwargs().items(),
            other.pre_hook,
        ))
        return self_tuple == other_tuple


def autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, pre_hook=None, post_hook=None,
             warmup=None, rep=None, use_cuda_graph=False, do_bench=None, cache_results=False, shape_policy=None,
             generate_configs=None, op_affiliation="sdnn", row_sign=None, col_sign=None, n_elem_sign=None):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune': a function used to prune configs. It should have the signature
                `prune_configs_by( configs: List[triton.Config], named_args: Dict[str, Any], **kwargs: Dict[str, Any]) -> List[triton.Config]:`
                and return pruned configs. It should return at least one config.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: warmup time (in ms) to pass to benchmarking (deprecated).
    :type warmup: int
    :param rep: repetition time (in ms) to pass to benchmarking (deprecated).
    :type rep: int
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    :param cache_results: whether to cache autotune timings to disk.  Defaults to False.
    "type cache_results: bool
    """

    def decorator(fn):
        return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
                         post_hook=post_hook, prune_configs_by=prune_configs_by, warmup=warmup, rep=rep,
                         use_cuda_graph=use_cuda_graph, do_bench=do_bench, cache_results=cache_results,
                         shape_policy=shape_policy, generate_configs=generate_configs,
                         op_affiliation=op_affiliation or "", row_sign=row_sign or "",
                         col_sign=col_sign or "", n_elem_sign=n_elem_sign or "")

    return decorator


class Heuristics(KernelInterface):

    def __init__(self, fn, arg_names, values) -> None:
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)


def heuristics(values):
    """
    Decorator for specifying how the values of certain meta-parameters may be computed.
    This is useful for cases where auto-tuning is prohibitively expensive, or just not applicable.

    .. highlight:: python
    .. code-block:: python

        # smallest power-of-two >= x_size
        @triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['x_size'])})
        @triton.jit
        def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
            ...
    :param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
                   each such function takes a list of positional arguments as input.
    :type values: dict[str, Callable[[dict[str, Any]], Any]]
    """

    def decorator(fn):
        return Heuristics(fn, fn.arg_names, values)

    return decorator
