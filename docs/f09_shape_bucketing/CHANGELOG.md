# FlagTree F09 Changelog / 变更日志

All notable changes for the FlagOS **F09 (Universal & Portable Dynamic Shape Bucketing & Parallel Runtime)** stack on `FlagTree` (upstream baseline `e86aa7cf`, Triton 3.6.0 / FlagTree 0.7.0) are documented here.

---

## [v0.2.0-f09] — 2026-11-22

### Documentation & Release Notes (`docs/f09-v020-release-notes`)
- Updated `CHANGELOG.md`, `KNOWN_ISSUES.md`, `README.md`, and `DESIGN_AND_MAINTENANCE.md` for immutable release **`v0.2.0-f09`** (0 Python runtime/test files modified).
- Marked **KI-01** (Jina `aiu_load_mask`), **KI-07** (`IntervalPolicy` Auto-PGO & `GraphBucketCompiler`/`DecodeGraphBucketRunner`), and **KI-08** (`cache.py` concurrent eviction lock) as **RESOLVED in `v0.2.0-f09`** with real-model PPU-ZW810E verification numbers (`USE_FLAGTUNE_COST_MODEL=0`).

### FT-P1-B (`e4522a5fc1a83bd35d2f2a8f6f04cd757287dc91`) — 2026-11-22
- **Inter-Process Binary Cache Lock (`cache.py`)**:
  - Added POSIX `fcntl.flock` + in-process `threading.RLock` hybrid lock (`_cache_root_lock` on `.flagtree_cache.lock`, graceful thread-lock fallback on non-POSIX): `FileCacheManager.put()` holds shared `LOCK_SH` with atomic `os.replace()` and `shutil.rmtree(ignore_errors=True)`, while `enforce_binary_cache_budget()` holds exclusive `LOCK_EX`.
  - Preserved **100% lock-free warm reads** (`has_file()`, `get_file()`, `get_group()`). 6-process × 30-iter × 3-round stress repro (`max_cache_entries=3`): **15 `FileNotFoundError` crashes -> 0 exceptions, 0 budget overshoots, `warm_read_lockfree=True`** (resolves KI-08).
- **Core Vendor Dehardcoding & FP8 `min_dot_size` Contract (`backend_caps.py`, `shape_bucketing.py`, `_distributed.py`, `autotuner.py`, `adjust_kernel_param.py`)**:
  - Added `fp8_min_dot_size` (`(16, 16, 32)` default / `(64, 16, 32)` MUSA), `min_dot_size_for_dtype(dtype)` / `get_min_dot_size()`, `supports_cudagraph_bench` (`False` for `ppu`/`thead`, `True` default), `extra_config_constexpr_keys` (`("aiu_load_mask",)` for `ppu`/`thead`), and `torch_extension_module` (`"torch_npu"` for `ascend`, etc.) to `BackendBucketingCapabilities`.
  - Removed all hardcoded `"aiu_load_mask"` from `shape_bucketing.py`, `import torch_npu` from `_distributed.py`, and 30 source / 10 signature XPU leaks from `autotuner.py` (replaced with generic `**backend_kwargs` forwarding).
- **Tests**: Expanded FlagTree F09 6-file unit suite from 45 to **48/48 PASSED** (0 failed, 0 skipped; 0 lines `>120` chars).

### FT-P1-A (`9b92bd75dc7e01b7ada6cb15196f8ac69e83f863`) — 2026-11-22
- **Auto-PGO `IntervalPolicy` & `group_frontier()` Fixes (`shape_bucketing.py`)**:
  - Unified 1D `IntervalPolicy.lookup_autotune_key()` by removing redundant `(val % divisibility) == 0` while preserving `val == 1` singleton isolation (eliminating 2× redundant cold tuning between aligned and unaligned shapes in the same interval).
  - Fixed `group_frontier()` DP interval construction so consecutive segments satisfy `lo = prev_hi + 1` with zero uncovered gaps and zero overlap, plus `_select_interval_segment()` safe routing for legacy policy JSONs.
  - Fixed `IntervalPolicy.normalize_entry_scalar()` to honor explicit `group_id`, match `all_args` static dimensions + Tensor dtypes across both 1D `groups` and N-D `axis_groups`, and fall back cleanly on unseen groups.
  - Added `IntervalPolicy.bind_op()` so `resolve_tuner_policy()` isolates per-op `OnlineBucketPolicy` fallbacks with AST-inferred `axis_specs`, preventing cross-op dynamic-axis pollution.
- **Graph Bucketing & Decode Runner Fixes (`graph_bucketing.py`, `bucket_guard.py`)**:
  - Bounded `SymbolicBucketGuard._paired_step_plans` with an `OrderedDict` LRU (`max_paired_step_plans=128`), wired configurable `seq_axis` into memory-cap enforcement, and added real `_compiled_by_bucket` LRU caching (`compiled_buckets`, hit/miss counters) in `GraphBucketCompiler`.
  - Hardened `DecodeGraphBucketRunner` with safe eager fallback when `graph_adapter.capture()` returns `None` (`capture_fallback_count`), static device `torch.Tensor` `static_kv_lens` updated in-place via `.copy_()` (stable `data_ptr()`), and trailing padding-row zeroing on batch shrink.
- **Real-Model Auto-PGO V2 Verification (PPU-10/11, `USE_FLAGTUNE_COST_MODEL=0`)**:
  - `bge_reranker_v2_m3`: 11,109 real `ShapeProfiler` samples, `4/4` groups, `7/7` `ft_pgo_v1` hits; `COLD_TUNED` `33 -> 14 (-57.6%)`, cold `60,108.1 -> 54,073.0 ms (1.11×)`, unseen `13,400.7 -> 274.6 ms (48.80×, 0 new_hgbin, 0 cold_tuned)`, steady `1.115×`, `cos_vs_aten = 0.999983`.
  - `qwen3_embedding_0_6b`: 43,659 real `ShapeProfiler` samples, `5/5` groups, `10/10` `ft_pgo_v1` hits; 6-len `COLD_TUNED` `65 -> 38 (-41.5%)`, cold `104,355.2 -> 91,892.7 ms (1.14×, 3.9s faster than ON)`, unseen `2,582.3 -> 1,667.9 ms (19.47×)`, steady `1.105×` (beats ON `1.092×`), `min_cos_pgo_vs_off = 1.000000`; policy-covered 7-len A/B `1,995.0 ms vs 1,994.6 ms (1.000× exact parity, 0 cold_tuned)` (resolves KI-07).
- **Tests**: 6/6 repros BEFORE FAIL -> AFTER PASS; FlagTree 6-file unit suite expanded 39 -> **45/45 PASSED**.

---

## [v0.1.0-f09] (F09 Release v0.9.0-f09) — 2026-11-22

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
