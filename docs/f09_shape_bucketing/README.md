# FlagTree F09: Dynamic Shape Bucketing & Parallel Runtime / 动态 Shape 分桶与并行运行时

> **Navigation / 文档导航**:
> [Overview & Quick Start (README)](./README.md) |
> [Design & Maintenance Guide](./DESIGN_AND_MAINTENANCE.md) |
> [Known Issues](./KNOWN_ISSUES.md) |
> [Changelog](./CHANGELOG.md) |
> [Security Policy](../../SECURITY.md)

---

## 1. Overview / 项目概述

**English**:
FlagTree F09 introduces a portable, backend-agnostic **Dynamic Shape Bucketing & Resilient Runtime** layer for Triton/FlagTree kernels. Without requiring per-operator vendor branches, it eliminates JIT recompilation stalls on variable sequence lengths, dynamic batch sizes, and ragged inputs across inference and training workloads while preserving full compatibility with existing kernels (opt-in via `FLAGTREE_SHAPE_BUCKETING=1`, default `False`).

Core capabilities include:
- **Backend Capability Registry (`backend_caps.py`)**: Centralizes alignment divisibility, minimum `tl.dot`/load tile sizes, static grid baking, collective device mapping, and Device Graph APIs behind `BaseBackend.get_bucketing_capabilities()`.
- **Online & Offline Shape Bucketing (`shape_bucketing.py`, `jit.py`, `autotuner.py`, `adjust_kernel_param.py`, `cache.py`)**: AST-driven dynamic-axis inference, geometric/piecewise bucket ladders, bucket-scoped alignment normalization, deterministic `@tl.constexpr` callable cache keys, and bounded binary/memory cache eviction.
- **Distributed Resilience (`_distributed.py`, `_async_compile.py`, `errors.py`)**: Cross-rank `DistributedDispatchGuard` (non-collective default lookup, eager isolated `gloo` PG setup, re-entrant `dispatch_guard_scope()`/`coordinate()`), differentiated inference/training `OOMRecoveryController`, and compile timeout circuit breaker with blacklist.
- **Graph Bucketing, Profiling & Offline Partition Planning (`graph_bucketing.py`, `bucket_guard.py`, `shape_profiler.py`, `partition_planner.py`)**: Multi-mode `ShapeProfiler`, `SymbolicBucketGuard`, `TokenBudgetBatcher`, `GraphBucketCompiler`, and analytical `PartitionPlanner`.

**中文简介**：
FlagTree F09 为 Triton/FlagTree 编译器运行时提供**架构无关、可移植的动态 Shape 分桶与容错调度能力**。在不向通用运行时硬编码厂商分支的前提下，显著消除变长序列、动态 Batch 等场景下的重复 JIT 编译与自动调优抖动，默认关闭（`FLAGTREE_SHAPE_BUCKETING=0`）以保持与上游 100% 兼容，按需通过环境变量或 Python `triton.knobs` 启用。

核心能力包括：
- **后端能力注册表（`backend_caps.py`）**：统一描述对齐整除性、最小 `tl.dot`/访存分块、静态 Grid 烘焙、集合通信设备映射与 Device Graph API，彻底解耦厂商特化判断。
- **在线/离线动态分桶与缓存预算（`shape_bucketing.py`、`jit.py`、`autotuner.py`、`cache.py`）**：基于 AST 静态分析自动识别动态轴与启发式谓词、几何/分段分桶阶梯、跨进程确定性 `@tl.constexpr` 函数哈希及内存/磁盘二进制缓存预算淘汰。
- **多卡防死锁守卫、OOM 自愈与编译超时熔断（`_distributed.py`、`_async_compile.py`）**：提供默认非集合获取 + 显式独立 `gloo` 侧通道初始化 + 可重入作用域的 `DistributedDispatchGuard`、推理/训练差异化 `OOMRecoveryController` 与超时熔断黑名单。
- **图级分桶、离线画像与并行切分规划（`graph_bucketing.py`、`bucket_guard.py`、`shape_profiler.py`、`partition_planner.py`）**：支持三模式 `ShapeProfiler`、符号边界守卫、Token 预算合批器与离线解析式 `PartitionPlanner`。

---

## 2. Supported Environments & Verification Boundary / 支持环境与验证边界

| Component / Backend | Status | Notes / Verification Scope |
| --- | --- | --- |
| **Python / PyTorch / Triton** | **Verified** | Python 3.12.3, PyTorch 2.10.0, Triton 3.6.0, FlagTree 0.7.0 base (`e86aa7cf`), pytest 7.2.0 |
| **Container Runtime (Reference)** | **Verified** | `flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64` (`sha256:1ba7ead15378160ba9aa1dafcfb77686de6b16732590ffab062bab19c60c32b8`) |
| **T-Head PPU-ZW810E (1 / 3 / 4 cards)** | **Verified (HW)** | Full FlagTree F09 unit suite (39/39), FlagGems op/training suite (21/21), multi-model E2E & DDP (`gloo`) |
| **NVIDIA CUDA / AMD ROCm / Ascend NPU / XPU / MLU / MUSA** | **Portable / Pending HW Sign-off** | Backend capability profiles & portable unit tests pass; dedicated physical hardware CI verification pending (see [KNOWN_ISSUES.md](./KNOWN_ISSUES.md#ki-05)) |

### Dependencies / 依赖说明
- Core FlagTree F09 runtime modules require **zero new third-party dependencies** beyond standard Python (`ast`, `contextlib`, `dataclasses`, `json`, `threading`) and existing Triton/PyTorch (`torch`, optional `torch.distributed` with `gloo`/backend PG).
- For air-gapped or offline environments without external FlagTune XGBoost model tarballs, set `USE_FLAGTUNE_COST_MODEL=0` (see [KNOWN_ISSUES.md](./KNOWN_ISSUES.md#ki-03)).

---

## 3. Configuration Knobs / 核心环境变量配置

| Environment Variable | `triton.knobs` Path | Default | Description / 说明 |
| --- | --- | --- | --- |
| `FLAGTREE_SHAPE_BUCKETING` | `knobs.autotuning.shape_bucketing` | `False` (`0`) | Enable/disable F09 dynamic shape bucketing / 启用动态 Shape 分桶总开关 |
| `FLAGTREE_SHAPE_BUCKETING_MODE` | `knobs.autotuning.shape_bucketing_mode` | `"auto"` | Policy mode: `auto`, `inference`, `training` / 分桶模式 |
| `FLAGTREE_SHAPE_POLICY_PATH` | `knobs.autotuning.shape_policy_path` | `None` | Optional JSON path for offline PGO `IntervalPolicy` / 离线 PGO 分桶策略路径 |
| `FLAGTREE_MAX_BUCKET_ARTIFACTS` | `knobs.autotuning.max_bucket_artifacts` | `None` | In-memory bucket cache entry cap (LRU eviction) / 内存分桶缓存条目上限 |
| `FLAGTREE_MAX_CACHE_ENTRIES` | `knobs.cache.max_cache_entries` | `None` | Disk binary cache entry cap (`TRITON_CACHE_DIR`) / 磁盘编译缓存条目上限 |
| `FLAGTREE_MAX_CACHE_BYTES` | `knobs.cache.max_cache_bytes` | `None` | Disk binary cache byte budget / 磁盘编译缓存字节预算上限 |
| `FLAGTREE_COMPILE_TIMEOUT_SEC` | `knobs.autotuning.compile_timeout_sec` | `None` | Per-config compile timeout circuit breaker (seconds) / 单配置编译超时熔断秒数 |
| `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD` | `knobs.autotuning.distributed_dispatch_guard` | `False` (`0`) | Enable cross-rank dispatch guard negotiation / 启用多卡调度防死锁守卫 |
| `FLAGTREE_OOM_RECOVERY` | `knobs.autotuning.oom_recovery` | `True` (`1`) | Enable OOM config retry & fallback audit / 启用 OOM 自动恢复与审计记录 |

---

## 4. Installation / 安装方式

### Option A: Isolated `--target` Overlay (Recommended, Non-Destructive / 推荐隔离安装)
Use the portable installer shipped with FlagGems to overlay F09 Python runtime files onto a prebuilt `triton` wheel in an isolated directory without modifying system `site-packages`:

```bash
bash <FlagGems_repo>/tools/install_f09_editable.sh \
    --flagtree <FlagTree_repo> \
    --flaggems <FlagGems_repo> \
    --target /tmp/f09_isolated

export PYTHONPATH="/tmp/f09_isolated:${PYTHONPATH:-}"
export FLAGTREE_SHAPE_BUCKETING=1
export USE_FLAGTUNE_COST_MODEL=0
```

### Option B: Build from Source / 源码完整编译安装
Follow the standard FlagTree build instructions in root [README.md](../../README.md#install-from-source):

```bash
export FLAGTREE_BACKEND=${backend_name}  # omit on nvidia/amd/triton-shared
python3 -m pip install . --no-build-isolation -v
```

---

## 5. Quick Start & Minimal Reproducible Example / 快速上手与最小可复现示例

The following self-contained example runs on **CPU-only or any accelerator** with zero external model downloads, demonstrating `BackendBucketingCapabilities`, `OnlineBucketPolicy`, and `DistributedDispatchGuard`:

```bash
PYTHONPATH=/tmp/f09_isolated python3 - <<'PY'
from triton.runtime import (
    BucketingMode,
    DistributedDispatchGuard,
    OnlineBucketPolicy,
    RankDispatchState,
    get_backend_capabilities,
    is_in_dispatch_guard_scope,
)

# 1. Query declarative backend capabilities (portable across backends)
caps = get_backend_capabilities("cuda")
print(f"caps: backend={caps.backend} div={caps.divisibility} min_dot={caps.min_dot_size}")

# 2. Online shape bucketing collapses distinct variable sequence lengths into shared buckets
policy = OnlineBucketPolicy(mode=BucketingMode.INFERENCE, backend=caps.backend)
seen_keys = []
for seq_len in (45, 50, 61, 130, 180, 250):
    res = policy.lookup_autotune_key({"M": seq_len, "N": 1024}, keys=["M", "N"])
    seen_keys.append((seq_len, res.representative_dims.get("M")))
unique_buckets = sorted({b for _, b in seen_keys})
print(f"mapped: {seen_keys}")
print(f"unique_buckets ({len(unique_buckets)} from {len(seen_keys)} shapes): {unique_buckets}")

# 3. Re-entrant DistributedDispatchGuard coordination (2-rank simulation: rank 0 hot, rank 1 cold)
barriers = []
guard = DistributedDispatchGuard(
    world_size=2,
    rank=0,
    all_reduce_or_fn=lambda local: local | int(RankDispatchState.ANY_COLD_COMPILE),
    barrier_fn=lambda: barriers.append(1),
)
with guard.coordinate(RankDispatchState.ALL_HOT_HIT) as decision:
    assert is_in_dispatch_guard_scope()
    with guard.coordinate(RankDispatchState.ALL_HOT_HIT) as nested:
        assert nested is None  # re-entrant no-op: zero duplicate collectives
print(
    f"guard: extend_window={decision.extend_collective_window} "
    f"negotiations={guard.negotiation_count} barriers={len(barriers)}"
)
PY
```

### Expected Output / 预期输出
```text
caps: backend=cuda div=16 min_dot=(16, 16, 16)
mapped: [(45, 48), (50, 64), (61, 64), (130, 192), (180, 192), (250, 256)]
unique_buckets (4 from 6 shapes): [48, 64, 192, 256]
guard: extend_window=True negotiations=1 barriers=2
```

### Multi-Rank Distributed Usage Pattern / 多卡分布式推荐用法
When enabling `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD=1` under DDP / FSDP / TP / EP, call `init_default_dispatch_guard()` **once collectively** right after `torch.distributed.init_process_group()` to create a dedicated `gloo` side-channel ProcessGroup without interfering with GPU collectives:

```python
import torch.distributed as dist
from triton.runtime import init_default_dispatch_guard, reset_default_dispatch_guard

dist.init_process_group(backend="gloo")  # or "nccl" / "flagcx"
init_default_dispatch_guard(create_gloo_group=True)
# ... run training / inference with FLAGTREE_DISTRIBUTED_DISPATCH_GUARD=1 ...
reset_default_dispatch_guard()
dist.destroy_process_group()
```

---

## 6. Running Unit Tests / 运行单元测试

Run all 6 FlagTree F09 unit test suites (39 tests total) against an isolated `--target` installation:

```bash
USE_FLAGTUNE_COST_MODEL=0 PYTHONPATH=/tmp/f09_isolated pytest -v \
    python/test/unit/runtime/test_backend_caps_and_shape_bucketing.py \
    python/test/unit/runtime/test_runtime_bucketing_and_cache_budget.py \
    python/test/unit/runtime/test_distributed_dispatch_guard.py \
    python/test/unit/runtime/test_shape_profiler_and_guard.py \
    python/test/unit/runtime/test_f09_portability_and_generality.py \
    python/test/flagtune/test_partition_planner.py
```

Expected result: **`39 passed, 0 failed`**.

---

## 7. Troubleshooting / 常见问题排查

1. **Unexpected 39–260s stalls during cold kernel runs on offline machines**:
   FlagGems' FlagTune cost model attempts to download vendor XGBoost archives by default. Always export `USE_FLAGTUNE_COST_MODEL=0` in offline or air-gapped environments (see [KNOWN_ISSUES.md](./KNOWN_ISSUES.md#ki-03)).
2. **Distributed ranks hang when `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD=1`**:
   - Ensure you are on FlagTree Fix-3 (`61534d97a2`) or later and FlagGems Fix-2 (`3d15211c90`) or later, which eliminated both lazy `dist.new_group()` creation and `LibEntry` L0/L1 bypass deadlocks.
   - If running over `nccl`/`flagcx`, call `triton.runtime.init_default_dispatch_guard()` collectively after `dist.init_process_group()` so guard collectives run on a dedicated CPU `gloo` group.
3. **Cross-process binary cache misses after process restart**:
   - Ensure `TRITON_CACHE_DIR` points to a persistent directory and that you include Fix-2 (`74843e00d7`), which hashes `@tl.constexpr` callable objects deterministically instead of using memory addresses.
   - In multi-rank jobs with `FLAGTREE_MAX_CACHE_ENTRIES` / `FLAGTREE_MAX_CACHE_BYTES` enabled, use per-rank cache subdirectories to avoid concurrent eviction races (see [KNOWN_ISSUES.md](./KNOWN_ISSUES.md#ki-08)).
4. **Standalone `pytest` reports `No module named 'triton.backends.amd'`**:
   - Ensure `PYTHONPATH` points to your isolated `--target` install (containing compiled `triton._C` and backends) rather than an unbuilt `FlagTree/python` source tree alone, and include Fix-3 (`61534d97a2`) which fixed the test bootstrap priority.
