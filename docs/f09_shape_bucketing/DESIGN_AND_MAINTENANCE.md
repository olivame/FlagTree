# FlagTree F09: Design, Architecture & Maintenance Guide / 架构设计与维护升级指南

> **Related Docs**: [README & Quick Start](./README.md) | [Known Issues](./KNOWN_ISSUES.md) | [Changelog](./CHANGELOG.md) | [Security Policy](../../SECURITY.md)

---

## 1. System Architecture / 系统架构

FlagTree F09 organizes dynamic shape bucketing into four decoupled runtime layers so hardware/vendor traits remain declarative and upstream Triton core diffs stay localized and auditable:

```text
+-----------------------------------------------------------------------------+
| Layer 4: Graph Bucketing, Profiling & Offline Planning (PR-4)               |
|  - graph_bucketing.py   : GraphBucketCompiler, ModelCacheKey                |
|  - bucket_guard.py      : SymbolicBucketGuard, DecodeGraphBucketRunner,     |
|                           TokenBudgetBatcher                                |
|  - shape_profiler.py    : ShapeProfiler (LLM / Retrieval / Training / Graph)|
|  - partition_planner.py : Offline analytical PartitionPlanner (DDP/TP/PP/EP)|
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Layer 3: Distributed Resilience & Circuit Breakers (PR-3 + Fix-1 + Fix-3)   |
|  - _distributed.py      : DistributedDispatchGuard (non-collective default, |
|                           eager init_default_dispatch_guard, re-entrant     |
|                           dispatch_guard_scope/coordinate),                 |
|                           OOMRecoveryController, FallbackRegistry           |
|  - _async_compile.py    : run_with_compile_timeout, CompileBlacklist        |
|  - errors.py            : CompileTimeoutError                               |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Layer 2: Core Runtime Integration & Cache Budgeting (PR-2 + Fix-2)          |
|  - jit.py               : Bucket-scoped alignment stripping, deterministic  |
|                           _stable_constexpr_token() for callable constexprs |
|  - autotuner.py         : Bucket key lookup, dispatch guard scope, OOM &    |
|                           timeout wiring, in-memory BucketCacheBudget       |
|  - adjust_kernel_param.py: AST bucket_axes inference & AABS integration     |
|  - cache.py             : enforce_binary_cache_budget() (entries / bytes)   |
|  - knobs.py             : FLAGTREE_* runtime environment knobs              |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Layer 1: Backend Capability Contract & Bucketing Policies (PR-1 + Fix-1/2)  |
|  - backends/compiler.py : BaseBackend.get_bucketing_capabilities() hook     |
|  - backend_caps.py      : BackendBucketingCapabilities registry & resolvers |
|  - shape_bucketing.py   : OnlineBucketPolicy, IntervalPolicy, AST heuristic |
|                           classifier, config-constexpr filter, telemetry    |
+-----------------------------------------------------------------------------+
```

---

## 2. Module Inventory / 模块清单（`e86aa7c..61534d97`）

### 2.1 Runtime Modules (17 files)
| File | Origin | Responsibility |
| --- | --- | --- |
| `python/triton/backends/compiler.py` | PR-1 | Adds optional `BaseBackend.get_bucketing_capabilities()` hook |
| `python/triton/runtime/backend_caps.py` | PR-1 | Declarative `BackendBucketingCapabilities` registry, plugin registration, collective device & graph API resolvers |
| `python/triton/runtime/shape_bucketing.py` | PR-1, Fix-1, Fix-2 | `OnlineBucketPolicy`, `IntervalPolicy`, geometric/PGO ladders, AST heuristic evaluator, `_is_config_constexpr_key()` tile-constant filter, `BucketCacheBudget`, `GLOBAL_BUCKET_TELEMETRY`, `warmup_buckets` |
| `python/triton/knobs.py` | PR-2, PR-3 | `env_opt_int` helper and `FLAGTREE_*` cache/autotuning knobs |
| `python/triton/runtime/jit.py` | PR-2, Fix-2 | Bucket-scoped integer alignment stripping and `_stable_constexpr_token()` cross-process deterministic callable hashing |
| `python/triton/runtime/autotuner.py` | PR-2, PR-3, Fix-3 | Bucketing lookup, guard mismatch fallback, re-entrant `dispatch_guard_scope()` integration, OOM retry, compile timeout circuit breaker, LRU cache budget |
| `python/triton/runtime/adjust_kernel_param.py` | PR-2, Fix-3 | AST-based dynamic axis inference (`bucket_axes`) and AABS cooperation |
| `python/triton/runtime/cache.py` | PR-2 | Disk binary cache entry/byte budget eviction (`enforce_binary_cache_budget`) |
| `python/triton/runtime/_distributed.py` | PR-3, Fix-1, Fix-3 | `DistributedDispatchGuard`, `init_default_dispatch_guard`, `get_default_dispatch_guard`, `reset_default_dispatch_guard`, `dispatch_guard_scope`, `is_in_dispatch_guard_scope`, `OOMRecoveryController`, `FallbackRegistry` |
| `python/triton/runtime/_async_compile.py` | PR-3 | Subprocess/thread timeout execution (`run_with_compile_timeout`) and `GLOBAL_COMPILE_BLACKLIST` |
| `python/triton/runtime/errors.py` | PR-3, Fix-3 | `CompileTimeoutError` exception class |
| `python/triton/runtime/shape_profiler.py` | PR-4, Fix-3 | Multi-mode `ShapeProfiler` (`ProfileMode`: LLM serving, retrieval, training, generic graph) |
| `python/triton/runtime/bucket_guard.py` | PR-4 | `SymbolicBucketGuard`, `DecodeGraphBucketRunner`, `TokenBudgetBatcher` |
| `python/triton/runtime/graph_bucketing.py` | PR-4 | `GraphBucketCompiler`, `ModelCacheKey`, `build_model_cache_key`, `compile_with_bucketing` |
| `python/triton/flagtune/runtime/partition_planner.py` | PR-4, Fix-3 | Offline analytical `PartitionPlanner`, piecewise alpha-beta `CollectiveCostModel`, `BucketedOperatorCostModel`, `build_executable_plan` |
| `python/triton/runtime/__init__.py` | PR-1..4, Fix-3 | Public runtime API re-exports |
| `python/triton/flagtune/__init__.py` | PR-4 | Public `PartitionPlanner` & cost-model re-exports |

### 2.2 Unit Test Suites (6 files, 48 tests)
| Test File | Coverage |
| --- | --- |
| `python/test/unit/runtime/test_backend_caps_and_shape_bucketing.py` | Capability registry, AST heuristics, `OnlineBucketPolicy`, `IntervalPolicy`, deterministic callable `cache_key`, tile `constexpr` exclusion |
| `python/test/unit/runtime/test_runtime_bucketing_and_cache_budget.py` | `JITFunction` bucket-scoped alignment stripping, multi-tag alignment normalization, cache budget |
| `python/test/unit/runtime/test_distributed_dispatch_guard.py` | 8-rank skew negotiation, compile timeout circuit breaker, infer/train OOM recovery, lazy non-collective vs eager isolated `gloo` PG, re-entrant `dispatch_guard_scope()` / `coordinate()`, real 2-process `gloo` regression |
| `python/test/unit/runtime/test_shape_profiler_and_guard.py` | Three-mode `ShapeProfiler`, `SymbolicBucketGuard`, `DecodeGraphBucketRunner`, `TokenBudgetBatcher` |
| `python/test/unit/runtime/test_f09_portability_and_generality.py` | Cross-backend plugin portability, disk cache budget, explicit `bucket_axes`, `Autotuner` guard + OOM integration, `GraphBucketCompiler`, 3D `PartitionPlanner` |
| `python/test/flagtune/test_partition_planner.py` | Calibrated inference/training/MoE/retrieval parallel strategy selection and piecewise collective curve fitting |

---

## 3. Backend Capability Contract / 后端能力契约

To onboard a new hardware backend without editing core runtime files, implement `get_bucketing_capabilities()` on the backend's `BaseBackend` subclass OR call `register_backend_capabilities()` at plugin import time:

```python
from triton.runtime import BackendBucketingCapabilities, register_backend_capabilities

register_backend_capabilities(
    "my_accelerator",
    BackendBucketingCapabilities(
        backend="my_accelerator",
        divisibility=16,
        min_dot_size=(16, 16, 16),
        min_load_size=1,
        bakes_static_grid=False,
        default_num_stages=3,
        collective_device="cuda",          # device string used by torch.distributed all_reduce
        collective_backends=("nccl",),     # matching PG backends
        alignment_spec_tags=("D",),        # Triton specialization tag(s) stripped on dynamic axes
    ),
)
```

Lookup precedence in `get_backend_capabilities(backend)`:
1. Explicit plugin registration (`register_backend_capabilities`).
2. Active `BaseBackend.get_bucketing_capabilities()` hook.
3. Built-in defaults (`_DEFAULT_CAPABILITIES`) + environment overrides (`FLAGTREE_BACKEND_DIVISIBILITY`, etc.).

---

## 4. Key Public API Contracts / 核心接口契约

### 4.1 Shape Bucketing Policies (`triton.runtime.shape_bucketing`)
- **`OnlineBucketPolicy(mode, backend, axis_specs, ...)`**:
  - `lookup_autotune_key(all_args, keys)` -> `BucketLookupResult(bucket_key, status, representative_dims, raw_dims, clamped_axes)`.
  - `observe_args(all_args)` promotes axes seen with multiple distinct sizes across buckets into the dynamic axis set while strictly excluding Triton tile/config constexpr keys (`BLOCK_*`, `TILE_*`, `GROUP_*`, `SPLIT_K`, `NUM_*`, etc.).
  - `normalize_entry_scalar(name, val, ...)` normalizes scalar shape arguments for `LibEntry` / `JITFunction` cache keys.
- **`IntervalPolicy`**:
  - Loads offline PGO interval tables (`schema = flagtree.shape_policy.v1`) via `FLAGTREE_SHAPE_POLICY_PATH` and delegates online helper hooks (`observe_args`, `normalize_entry_scalar`) safely to its fallback policy.

### 4.2 Distributed Dispatch Guard (`triton.runtime._distributed` & `triton.runtime`)
- **`get_default_dispatch_guard(*, create_pg: bool = False) -> Optional[DistributedDispatchGuard]`**:
  - **Strictly non-collective by default (`create_pg=False`)**: never invokes `dist.new_group()` lazily. Returns the guard registered by `init_default_dispatch_guard()` (validated against `(world_size, rank, dist.group.WORLD)`), or binds locally to `process_group=None` (default PG).
- **`init_default_dispatch_guard(process_group=None, create_gloo_group=True, *, backend=None, timeout=None)`**:
  - Collective setup helper intended to run across all ranks immediately after `dist.init_process_group()` to isolate guard collectives onto a dedicated `gloo` PG (or user-provided PG).
- **`reset_default_dispatch_guard()`**: Clears the cached singleton (call before/after `dist.destroy_process_group()` or in tests).
- **`dispatch_guard_scope()` & `is_in_dispatch_guard_scope()`**:
  - Thread-local re-entrant depth counter allowing outer dispatchers (e.g., FlagGems `LibEntry.run()`) to own a single `negotiate()` + `post_compile_sync()` round across L0 hit, L1 hit, and L1 miss paths while nested `Autotuner.run()` / `LibTuner.run()` skip duplicate collectives.
- **`DistributedDispatchGuard.coordinate(local_state)`**:
  - Context manager combining `is_in_dispatch_guard_scope()`, `negotiate()`, `dispatch_guard_scope()`, and guaranteed `finally: post_compile_sync(decision)` so peer ranks never hang if a compiling rank raises an exception.

---

## 5. Key Design Trade-offs / 关键设计取舍

1. **Opt-in by default (`FLAGTREE_SHAPE_BUCKETING=False`, `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD=False`)**:
   Guarantees zero behavioral regression and zero hot-path overhead for existing FlagTree/Triton users unless explicitly opted in.
2. **Non-collective lazy guard vs. eager isolated `gloo` PG**:
   Calling `dist.new_group()` lazily on first cache miss caused immediate deadlocks whenever Rank 0 hit warm cache and skipped initialization while Rank 1 missed. Making `get_default_dispatch_guard()` strictly non-collective fixes out-of-the-box safety, while `init_default_dispatch_guard()` preserves isolated CPU `gloo` side-channel execution for NCCL/FlagCX training clusters.
3. **Deterministic callable `constexpr` hashing (`_stable_constexpr_token`)**:
   Upstream `JITFunction.cache_key` stringified callable `@tl.constexpr` arguments as `<function foo at 0x7f...>`, embedding process-random memory addresses and breaking cross-process disk cache reuse. Hashing `(module, qualname, inspect.getsource)` restores deterministic `.hgbin` reuse across restarts.
4. **Exclusion of tile/config `constexpr` keys from dynamic-axis tracking**:
   During autotuning, `BLOCK_M`/`BLOCK_N`/`BLOCK_K` vary across candidate trials. Tracking them as dynamic shape axes caused heuristic divisibility checks such as `FULL_M_TILES = (M % BLOCK_M == 0)` to evaluate `1 % 1 == 0 -> True`, corrupting boundary masks. Filtering config constexpr names prevents silent accuracy corruption.
5. **Offline analytical `PartitionPlanner`**:
   `PartitionPlanner` intentionally separates analytical strategy search and `ExecutableParallelPlan` JSON generation from framework-specific runtime orchestrators (TorchTitan, Megatron, vLLM) to keep FlagTree dependency-free; live runtime execution bridging is tracked as a follow-up item (see [KNOWN_ISSUES.md](./KNOWN_ISSUES.md#ki-04)).

---

## 6. Patch Maintenance & Upstream Upgrade Guide / 补丁维护与上游升级步骤

### 6.1 Upstream Baseline & Ordered Commit Stack
- **Upstream Base**: `FlagTree@e86aa7cf` (Triton 3.6.0 / FlagTree 0.7.0)
- **Ordered F09 Stack (`v0.2.0-f09`)** (9 code commits + 2 docs commits = 11 commits):
  1. `b99e75cb` — PR-1: `backend_caps.py` + `shape_bucketing.py` + `BaseBackend` hook
  2. `e42e6213` — PR-2: `jit.py`, `autotuner.py`, `adjust_kernel_param.py`, `cache.py`, `knobs.py` integration
  3. `28a520ce` — PR-3: `_distributed.py`, `_async_compile.py`, `errors.py`
  4. `18445a06` — PR-4: `graph_bucketing.py`, `bucket_guard.py`, `shape_profiler.py`, `partition_planner.py`
  5. `afa43e63` — Fix-1: `IntervalPolicy` delegation & guard PG isolation
  6. `74843e00` — Fix-2: deterministic callable `constexpr` `cache_key` & tile `constexpr` protection
  7. `61534d97` — Fix-3: non-collective `get_default_dispatch_guard()`, eager `init_default_dispatch_guard()`, re-entrant `dispatch_guard_scope()`, and 0 `>120`-char CI lint compliance
  8. `b04da1ef` — Docs (`v0.1.0-f09`): bilingual F09 guide, design/maintenance docs, known issues, changelog, and `SECURITY.md`
  9. `9b92bd75` — FT-P1-A: Auto-PGO `IntervalPolicy` / `group_frontier()` fixes, `GraphBucketCompiler` LRU cache & `seq_axis`, and `DecodeGraphBucketRunner` eager fallback / static device Tensor `static_kv_lens`
  10. `e4522a5f` — FT-P1-B: POSIX `fcntl.flock` binary cache eviction lock, FP8 `min_dot_size` (`K >= 32`), and core vendor dehardcoding
  11. `v0.2.0-f09` Docs: updated `CHANGELOG.md`, `KNOWN_ISSUES.md` (KI-01/KI-07/KI-08 RESOLVED), `README.md`, and `DESIGN_AND_MAINTENANCE.md`

### 6.2 Rebasing onto a Newer Upstream FlagTree / Triton Release
1. **Primary conflict hotspots** (only 6 upstream-modified files contain F09 hooks; the other 11 runtime files are self-contained F09 additions):
   - `python/triton/runtime/jit.py` (`cache_key` and alignment specialization stripping)
   - `python/triton/runtime/autotuner.py` (`run()` bucket lookup, dispatch guard scope, timeout/OOM hooks)
   - `python/triton/runtime/adjust_kernel_param.py` (`bucket_axes` AST helper)
   - `python/triton/runtime/cache.py` (`put_group()` budget eviction call)
   - `python/triton/knobs.py` (`cache_knobs` / `autotuning_knobs` fields)
   - `python/triton/runtime/_distributed.py` (appended below upstream `validate_node_launch_bindings`)
2. **Step-by-step upgrade procedure**:
   ```bash
   # 1. Fetch new upstream base and rebase F09 commits onto it
   git fetch upstream
   git checkout -b feat/f09-rebase-<new_version> feat/f09-pr4-graph-bucketing-profiler-and-partition-planner
   git rebase --onto upstream/main e86aa7cf

   # 2. Verify line-length, whitespace, and syntax compliance
   git diff --check upstream/main..HEAD
   python3 -m py_compile $(git diff --name-only upstream/main..HEAD -- '*.py')

   # 3. Run isolated --target install + all 6 FlagTree F09 unit test files
   bash <FlagGems_repo>/tools/install_f09_editable.sh \
       --flagtree "$(pwd)" --flaggems <FlagGems_repo> --target /tmp/f09_upgrade_check
   USE_FLAGTUNE_COST_MODEL=0 PYTHONPATH=/tmp/f09_upgrade_check pytest -v \
       python/test/unit/runtime/test_backend_caps_and_shape_bucketing.py \
       python/test/unit/runtime/test_runtime_bucketing_and_cache_budget.py \
       python/test/unit/runtime/test_distributed_dispatch_guard.py \
       python/test/unit/runtime/test_shape_profiler_and_guard.py \
       python/test/unit/runtime/test_f09_portability_and_generality.py \
       python/test/flagtune/test_partition_planner.py
   ```
3. **Sync with FlagGems**: If `triton.runtime._distributed` or `shape_bucketing` helper signatures change during rebase, re-verify FlagGems `tests/test_libentry_shape_bucketing.py` and `tests/test_training_ops_ppu.py` together against the same `--target` directory.
