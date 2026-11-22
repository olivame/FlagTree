# FlagTree F09 Changelog / 变更日志

All notable changes for the FlagOS **F09 (Universal & Portable Dynamic Shape Bucketing & Parallel Runtime)** stack on `FlagTree` (upstream baseline `e86aa7cf`, Triton 3.6.0 / FlagTree 0.7.0) are documented here.

---

## [F09 Release v0.9.0-f09] — 2026-11-22

### Documentation, Governance & Security (`docs/f09-docs-and-release-notes`)
- Added bilingual CN + EN Quick Start & Overview guide (`docs/f09_shape_bucketing/README.md`) covering supported environments, environment knobs, isolated installation, runnable minimal example with expected output, unit test commands, and troubleshooting (§9.1).
- Added Architecture, Backend Capability Contract, Public APIs, Design Trade-offs & Upstream Upgrade Guide (`docs/f09_shape_bucketing/DESIGN_AND_MAINTENANCE.md`) (§9.2).
- Added structured 5-field Known Issues & Boundaries register (`docs/f09_shape_bucketing/KNOWN_ISSUES.md`) covering KI-01..KI-08 (§9.2).
- Added F09 Release Changelog (`docs/f09_shape_bucketing/CHANGELOG.md`) (§9.3).
- Added repository vulnerability disclosure policy (`SECURITY.md`) (§8.4) and minimal F09 documentation pointers in root `README.md`.

### Fix-3 (`61534d97a2b57120099fe140792b7be5c1641307`) — 2026-11-22
- **Deadlock Fix (`_distributed.py`, `autotuner.py`, `__init__.py`)**:
  - Eliminated lazy collective `dist.new_group(backend="gloo")` inside `get_default_dispatch_guard()`; default calls (`create_pg=False`) are now strictly non-collective (`~60 us`), binding to the active default `ProcessGroup` (`process_group=None`) or returning an eagerly registered guard.
  - Added `init_default_dispatch_guard()` (eager collective isolated `gloo` PG setup), `reset_default_dispatch_guard()`, thread-local re-entrant `dispatch_guard_scope()` / `is_in_dispatch_guard_scope()`, and exception-safe `DistributedDispatchGuard.coordinate(local_state)`.
  - Updated `Autotuner.run()` to respect outer `dispatch_guard_scope()` (preventing duplicate collectives when invoked under FlagGems `LibEntry.run()`) and always execute `post_compile_sync()` in `finally:`.
- **CI Style & Lint Compliance**:
  - Wrapped all 35 `>120`-char lines across 11 F09 changed files to **0 violations** across all 23 changed Python files (`<=120` chars per `pyproject.toml`), and fixed EOF trailing blank line so `git diff --check e86aa7c..HEAD` passes cleanly.
- **Tests (`test_distributed_dispatch_guard.py`)**:
  - Fixed `_ensure_triton_importable()` bootstrap so standalone pytest runs respect installed/`PYTHONPATH` `triton`.
  - Added re-entrancy/Autotuner integration tests and a real 2-process `gloo` asymmetric deadlock regression test (total FlagTree F09 suite: **39/39 PASSED**).

### Fix-2 (`74843e00d7e7c66bab86fe8f868aca158f794586`) — 2026-11-22
- **Deterministic Cross-Process Cache Keys (`jit.py`)**: Added `_stable_constexpr_token()` to hash callable `@tl.constexpr` arguments by `(module, qualname, source)` rather than memory address (`str(val)`), fixing cross-process `.hgbin` cache misses on restart.
- **Heuristic Accuracy Protection (`shape_bucketing.py`)**: Promoted dynamic axes across buckets in `OnlineBucketPolicy.observe_args()` and introduced `_is_config_constexpr_key()` to exclude tile/config constants (`BLOCK_*`, `TILE_*`, `GROUP_*`, `SPLIT_K`, `NUM_*`, etc.) from dynamic-axis tracking and heuristic divisibility substitution (preventing `1 % 1 == 0` false positives).
- **Tests (`test_backend_caps_and_shape_bucketing.py`)**: Added 3 regression tests verifying cross-instance callable cache-key determinism, dynamic-axis promotion, and tile-constexpr protection.

### Fix-1 (`afa43e63945e817e8125303138d6bf34bf9469cc`) — 2026-11-21
- **PGO Policy Compatibility (`shape_bucketing.py`)**: Delegated `IntervalPolicy` online helper methods (`observe_args`, `normalize_entry_scalar`) cleanly to its fallback policy so offline PGO policies interoperate with `LibEntry` / `JITFunction` bucketing hooks.
- **Guard ProcessGroup Isolation (`_distributed.py`)**: Added isolated `gloo` PG support for `DistributedDispatchGuard` (later upgraded in Fix-3 to explicit eager initialization via `init_default_dispatch_guard()` to prevent lazy collective deadlocks).

### PR-4 (`18445a0664c930b966f87968f07a60b68b7f769f`) — 2026-11-20
- Added `GraphBucketCompiler`, `ModelCacheKey`, `build_model_cache_key`, and `compile_with_bucketing` (`python/triton/runtime/graph_bucketing.py`).
- Added three-mode `ShapeProfiler` (`python/triton/runtime/shape_profiler.py`) and `SymbolicBucketGuard`, `DecodeGraphBucketRunner`, and `TokenBudgetBatcher` (`python/triton/runtime/bucket_guard.py`).
- Added offline analytical `PartitionPlanner`, piecewise alpha-beta `CollectiveCostModel`, `BucketedOperatorCostModel`, and `build_executable_plan` (`python/triton/flagtune/runtime/partition_planner.py`, re-exported via `python/triton/flagtune/__init__.py`).
- Added unit tests in `test_shape_profiler_and_guard.py`, `test_f09_portability_and_generality.py`, and `test_partition_planner.py`.

### PR-3 (`28a520ce7fb061b9a865f6cbe36a782698e6c589`) — 2026-11-20
- Added `DistributedDispatchGuard`, `RankDispatchState`, `CoordinatedDispatchDecision`, `OOMRecoveryController`, `FallbackRegistry`, and `PerfRiskLevel` (`python/triton/runtime/_distributed.py`).
- Added compile timeout circuit breaker (`run_with_compile_timeout`, `GLOBAL_COMPILE_BLACKLIST` in `_async_compile.py`; `CompileTimeoutError` in `errors.py`) and integrated resilience knobs (`FLAGTREE_COMPILE_TIMEOUT_SEC`, `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD`, `FLAGTREE_OOM_RECOVERY`).
- Added `test_distributed_dispatch_guard.py`.

### PR-2 (`e42e62137859bea59a2c4006fe5b2037b68a2f6b`) — 2026-11-20
- Integrated shape bucketing into `JITFunction` (`jit.py`), `Autotuner` (`autotuner.py`), and AABS (`adjust_kernel_param.py` with `bucket_axes` AST inference).
- Added bounded disk binary cache eviction (`enforce_binary_cache_budget` in `cache.py`) and F09 environment knobs (`FLAGTREE_SHAPE_BUCKETING`, `FLAGTREE_SHAPE_BUCKETING_MODE`, `FLAGTREE_SHAPE_POLICY_PATH`, `FLAGTREE_MAX_BUCKET_ARTIFACTS`, `FLAGTREE_MAX_CACHE_ENTRIES`, `FLAGTREE_MAX_CACHE_BYTES` in `knobs.py`).
- Added `test_runtime_bucketing_and_cache_budget.py`.

### PR-1 (`b99e75cb9881c32da1903cb12446e204b8fde0cb`) — 2026-11-20
- Added `BaseBackend.get_bucketing_capabilities()` hook (`backends/compiler.py`) and declarative `BackendBucketingCapabilities` registry (`backend_caps.py`).
- Added `OnlineBucketPolicy`, `IntervalPolicy`, AST heuristic classifier, `BucketCacheBudget`, `GLOBAL_BUCKET_TELEMETRY`, and `warmup_buckets` (`shape_bucketing.py`).
- Added `test_backend_caps_and_shape_bucketing.py`.
