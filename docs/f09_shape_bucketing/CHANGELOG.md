# FlagTree F09 Changelog / 变更日志

All notable changes for the FlagOS **F09 (Universal & Portable Dynamic Shape Bucketing & Parallel Runtime)** stack on `FlagTree` (upstream baseline `e86aa7cf`, Triton 3.6.0 / FlagTree 0.7.0) are documented here.

---

## [v0.3.2-f09] — 2026-11-24

### Declarative Second-Chip XPU Port & Hot-Path Env Cache (`feat/f09-v032-xpu-port-and-hotpath-cache`)
- **Declarative Backend Capabilities (`triton/runtime/backend_caps.py`, `jit.py`, `autotuner.py`)**:
  - Added `binary_ext`, `static_grid_specializations`, and `canonicalize_compile_grid()` to
    `BackendBucketingCapabilities` so backends with static baked launch grids collapse dynamic grids to
    compiler-distinguishable equivalence classes without vendor branching in core runtime.
  - Added `supports_event_timing` wall-clock benchmark fallback in `Autotuner` and `TensorWrapper.numel()`.
  - Added fast pre-encoded `os.environ._data` dict lookup in `get_backend_capabilities()` (with automatic
    invalidation in `register_backend_capabilities()` and fallback to `os.environ.get`), plus a fast tuple path
    in `canonicalize_compile_grid()`, eliminating per-launch env-key encoding overhead on all backends.
- **Verification (Dual-Chip: T-Head PPU-ZW810E + Kunlunxin P800)**:
  - Unit tests: **`62/62 PASSED`** (including `test_canonicalize_compile_grid_and_tensor_wrapper_numel` and
    `test_fast_env_cache_invalidation_on_env_and_register`).
  - Kunlunxin P800 L1 pure FlagTree (zero FlagGems): `6/6` core LLM kernels PASS (`cold 1.36–1.91x`, unseen
    `102–403x` with `+0 .xpubin`, `.xpubin -88.0% ~ -92.6%`, LRU `max_xpubin=2<=4`, restart `6.55x` `new=0`).
  - Kunlunxin P800 L2 62 unique local models (`40/48 = 83.3%` ATen-loadable PASS, `0` ON-only regressions,
    `cold_gmean=1.765x`, `rest_gmean=2.016x`, `.xpubin -45.93%`, `cos(ON,OFF) >= 0.999989`); after hot-path
    env-cache optimization, 6 worst-tail steady models flipped from `0.951x` -> **`1.015x` (batched) / `1.028x`
    (per-step)** with `max_abs=0.0`.
  - Native `libbkcl.so` single-node (`4x`/`7x` XPU-Link, `AR 37.93–46.89 GB/s`) and 2-node cross-node (`2x2=4x`
    200GbE RoCE v2 + XPU-Link, `AR 55.09 GB/s`, `P2P 31.76 GB/s`): `F09-G04` hardware-adaptive Top-1 flip from
    PPU `dp_pp(2x2)` (`5.34x` over `dp_tp`) to P800 `dp_tp(2x2)` (`1.69–1.71x` over `dp_pp`, `rho=1.0`);
    `F09-G05` **`12/12 PASS`** across `4x`, `7x`, and cross-node `2x2` (`max_rank_param_diff=0.0`, closing
    `EX-004`/`KI-04`).

---

## [v0.3.1-f09] — 2026-11-23

### Runtime Alignment Contract & Cache-First Fix (`fix/f09-v031-alignment-contract-and-cache-first`)
- **Cross-Layer Alignment Contract (`triton/runtime/jit.py`)**:
  - Added `JITFunction._strip_alignment_on_indices()` (via generic `backend_caps` alignment tags) and
    `_bkt_force_unaligned_indices` so an outer bucket cache (such as FlagGems `LibEntry`) can require
    alignment-agnostic compilation whenever an aligned runtime scalar maps into an unaligned-compatible cache slot.
  - Established 3-branch cache-first lookup (`norm_kernel` unaligned reuse > `exact_kernel` aligned reuse
    signaling `_bkt_last_run_used_exact_aligned=True` > cold compile with `norm_spec`), preventing both
    cross-call static-axis cache invalidation and aligned-binary leakage into unaligned cache entries.
- **Heuristics Cache-First Preservation (`triton/runtime/autotuner.py`)**:
  - Added cache-first check in `Heuristics.run` so online dynamic-axis promotion never orphans already-compiled
    kernels in `JITFunction.kernel_cache`, while recording `_ft_last_eval` and `_ft_last_used_raw` so outer caches
    always bind exact heuristic constexprs.
- **Stride Unaligned Consistency (`triton/runtime/shape_bucketing.py`)**:
  - Unified `OnlineBucketPolicy._normalize_stride_value` with scalar dimension normalization by checking
    `axis_name not in self._unaligned_axes`, eliminating cross-bucket stride recompile churn.
- **Verification**:
  - Unit tests: `50/50 PASSED` (including 2 new v0.3.1 regression tests in
    `test_runtime_bucketing_and_cache_budget.py`).
  - Real PPU-ZW810E `torch.bmm` varlen micro-repro (`S=126, 512, 53, 91`): `v0.3.0-f09` `141.10s / 176 .hgbin`
    (`OFF 86.29s / 124 .hgbin`) -> **`58.49s / 63 .hgbin`** (`1.48x` faster than `OFF`, `-49.2%` `.hgbin`, 0
    recompiles after step 0).
  - Large-scale 656-model `x-benchmark v4` evaluation on 16x PPU-ZW810E: eliminated `59/59` hardware alignment
    launch failures (`HGGC: invalid global mem access alignment`), `13/13` cross-bucket numerical corruptions
    (restored to `cos=0.999998–1.000000`), and `2/2` abnormal OOMs (`75/75` clean sweep; merged 656-model run:
    `500/656` ON success, `450 PASS` `cold_gmean=1.305x`, `rest_gmean=1.739x`, `steady_gmean=1.092x`,
    `.hgbin -26.2%`; pure V3 69-model subset: `cold_gmean=1.666x`, `rest_gmean=3.295x`, `.hgbin -50.3%`).

---

## [v0.3.0-f09] — 2026-11-23

### Documentation & Release Notes (`docs/f09-v030-release-notes`)
- Updated `CHANGELOG.md`, `KNOWN_ISSUES.md`, `README.md`, and `DESIGN_AND_MAINTENANCE.md` for immutable release
  **`v0.3.0-f09`** (0 Python/Shell files modified in this docs commit).
- Updated **KI-04** (`PartitionPlanner` comm-profile calibration & single-node 4× PPU live execution) to
  **RESOLVED (Single-Node 4× PPU) in `v0.3.0-f09`** and **KI-06** (`DistributedDispatchGuard` bounded timeout,
  rank-fault isolation & `CoordinatedStepGuard`) to **Hardened in `v0.3.0-f09 (Usage Contract)`** while keeping
  **KI-02** (`libpccl` `ncclInternalError` env boundary) and **KI-05** (non-PPU physical HW incl. Kunlunxin XPU)
  transparently documented.

### F09-G05 (`ca4d466c87c72485d5263dc14f8eef4ddf1dace4`) — 2026-11-23
*(Stacked cleanly via zero-conflict cherry-pick on `F09-G04` `025903bf12`; bit-identical 5-file diff from isolated
4× PPU verified commit `ef5c5c0d51b3913866e8d3220be16bdbe5269227`.)*
- **Bounded Comm Timeout, Rank-Fault Isolation & `CoordinatedStepGuard` (`errors.py`, `_distributed.py`,
  `__init__.py`)**:
  - Added pickleable structured exceptions `DistributedCommTimeoutError` and `DistributedRankFaultError` carrying
    `rank`, `world_size`, `op`, `phase`, `timeout_s`, and `cause`.
  - Extended `RankDispatchState` with `ANY_RANK_FAULT = 16`, `CoordinatedDispatchDecision` with `degraded_solo`,
    `fault_kind`, `fault_reason`, and `DistributedDispatchGuard` with `comm_timeout_s` (`FLAGTREE_COMM_TIMEOUT_SEC`),
    `raise_on_fault`, `fallback_registry`, `release_memory_on_fault`, `_compute_work_wait_timeout()` (fixing the
    PyTorch `ProcessGroupGloo` C++ teardown race), `cleanup_after_fault()`, and `reset_fault_state()`.
  - Added `CoordinatedStepGuard` (`DistributedFaultCoordinator`) and `StepHealthDecision` providing cross-rank
    OK/OOM/ERROR consensus (`check_step_health`), synchronized multi-rank OOM recovery (`recover_coordinated_oom` /
    `run_coordinated_step`), and deterministic step-skip consensus on non-OOM local errors.
- **Real 4× PPU-ZW810E Fault-Injection & Long-Tail Soak Verification (`CUDA_VISIBLE_DEVICES=12..15`, `gloo`,
  `USE_FLAGTUNE_COST_MODEL=0`, `benchmarks/f09_g05_multicard_fault_injection.py`)**:
  - **Fault 1 (`SIGKILL` Rank 3)**: BEFORE silent swallow + `rc=11` crash (`fallback_count=0`) -> AFTER surviving
    3 ranks detect `DistributedRankFaultError` in **`0.508–0.538s`**, record `abort_rank_fault` (`HIGH`), and exit
    cleanly (wall `3.70–3.71s`, `hung=[]`, `lingering=0`).
  - **Fault 2 (`12.0s` stall vs `comm_timeout_s=3.0s`)**: BEFORE indefinite hang killed by `9.01s` watchdog ->
    AFTER surviving 3 ranks trip `DistributedCommTimeoutError` in **`3.008–3.028s`**, record `abort_comm_timeout`
    (`HIGH`), and clean up PG cleanly (wall `6.21s`, 0 `std::terminate`, `hung=[]`).
  - **Fault 3 (`S=2047` asymmetric Rank 1 OOM)**: BEFORE cross-rank `all_reduce` deadlock killed by `8.01s`
    watchdog -> AFTER `CoordinatedStepGuard` + 2-microbatch `OOMRecoveryController` recovers all 4 ranks in
    **`0.232–0.275s`** (`all_finite=True`, **`max_rank_param_diff=0.0`**).
  - **Scenario 4 (18-step long-tail `S=1..2047` × 2 rounds soak, `FLAGTREE_SHAPE_BUCKETING=1`)**: 17 committed
    steps + 1 coordinated skip + 4 cross-rank OOM recoveries + 2 cold-compile window extensions in **`10.58s`**
    (`all_finite=True`, **`max_rank_param_diff=0.0`**, 0 hangs, 0 lingering processes).
- **Tests**: Added 4 multi-process & fault-coordination tests in `test_distributed_dispatch_guard.py`; combined
  stacked FlagTree 7-file suite achieves **58/58 PASSED** (`63.50s` on PPU-15, 0 lines `>120` chars).

### F09-G04 (`025903bf1233749559899094999f98ba0a5d405d`) — 2026-11-23
- **Multi-Device Comm Profiler, Calibrated/Explainable Planner & Live Parallel Executor (`comm_profiler.py`,
  `partition_planner.py`, `parallel_executor.py`, `flagtune/__init__.py`)**:
  - Added backend-agnostic `triton.flagtune.runtime.comm_profiler` (`profile_comm_topology`, `CommProfileResult`,
    `needs_host_staging`, JSON serialization, `to_collective_cost_model`, `to_hardware_topology`).
  - Extended `partition_planner.py` with `ParallelStrategyKind.HYBRID_DP_PP`, `HardwareTopologySpec`,
    `PartitionPlanner.from_comm_profile()`, structured 5-key `explanation` with quantified baseline comparisons,
    and corrected non-decode PP micro-batch & Megatron TP `4 * AllReduce` cost modeling.
  - Added `triton.flagtune.runtime.parallel_executor` (`ParallelPlanExecutor`) materializing `ExecutableParallelPlan`
    (`DeviceMeshSpec`) into live DP / TP / PP / 2D Hybrid (`DP×PP`, `DP×TP`) sub-process-groups and sharded
    inference/training execution with numerical parity verification.
- **Real 4× PPU-ZW810E Comm Profile & 3-Workload Parallel Benchmark (`CUDA_VISIBLE_DEVICES=8..11`, `gloo`
  host-staged, `bfloat16`, `USE_FLAGTUNE_COST_MODEL=0`, `benchmarks/f09_g04_parallel_benchmark.py`)**:
  - **Comm Profile**: `all_reduce` `459.2 us` / `2.475 GB/s`, `all_gather` `296.3 us` / `1.015 GB/s`,
    `reduce_scatter` `2153.8 us` / `0.895 GB/s`, `p2p` `115.9 us` / `2.657 GB/s` (all `emulated=False`).
  - **Workload A (`Transformer-1B-16L`, 2 GiB/card infer)**: calibrated Top-1 `hybrid_dp2_pp2` (`28.28 ms`,
    `95,063.4 tok/s`, `rho=+1.000`) beats uncalibrated Top-1 `dp2_tp2` (`151.09 ms`) by **`5.34×`** and naive
    `tp4` (`322.46 ms`) by **`11.40×`**.
  - **Workload B (`Transformer-1B-16L`, distributed train)**: calibrated Top-1 `balanced_pp4` (`134.75 ms`,
    `22,798.6 tok/s`, `rho=+0.700`) beats unbalanced `pp4` by **`1.63×`**, `tp4` by **`5.56×`**, and uncalibrated
    Top-1 `ddp4` (`3,615.75 ms`) by **`26.83×`**.
  - **Workload C (`Qwen3-1.7B-16L`, full-mem infer)**: calibrated Top-1 `replica_dp4` (`13.96 ms`,
    `183,359.3 tok/s`, `rho=+1.000`) beats `dp2_pp2` by **`2.02×`**, `pp4` by **`3.16×`**, and `tp4` by
    **`22.42×`**.
  - **Aggregate**: **3/3 (100%) Top-1 match**, mean `pairwise_concordance 0.3778 -> 0.9333`, mean `spearman_rho
    -0.4000 -> +0.9000`, all 16 runs `all_finite=True`, `min_parity_cos = 0.999844 >= 0.999`.
- **Tests**: Added 6 unit & live multi-process tests in `python/test/flagtune/test_comm_profiler_and_parallel_exec.py`.

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
