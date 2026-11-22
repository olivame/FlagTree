# FlagTree F09: Known Issues & Boundaries / 已知问题与限制清单

> **Related Docs**: [README & Quick Start](./README.md) | [Design & Maintenance Guide](./DESIGN_AND_MAINTENANCE.md) | [Changelog](./CHANGELOG.md) | [Security Policy](../../SECURITY.md)

每项已知问题均按交付规范 §9.2 完整记录 **影响范围**、**复现方式**、**规避方法**、**责任人** 与 **整改计划** 五个必填字段。

---

## Summary Table / 问题总览

| ID | 标题 / Summary | 严重级别 | 状态 |
| --- | --- | :---: | :---: |
| [KI-01](#ki-01) | Jina 系列模型因 `aiu_load_mask` 对齐翻转触发 Warmup 后额外重编译 | Medium | **RESOLVED in `v0.2.0-f09`** |
| [KI-02](#ki-02) | T-Head PPU 多卡环境 `nccl` 后端出现 `ncclInternalError`（当前多卡验证基于 `gloo`） | Medium | Open (Vendor Runtime Boundary) |
| [KI-03](#ki-03) | 离线/受限网络环境下 FlagTune Cost Model 预加载阻塞（需设置 `USE_FLAGTUNE_COST_MODEL=0`） | High (Ops) | Mitigated via Env |
| [KI-04](#ki-04) | `PartitionPlanner` 当前为离线解析规划器，尚未直连在线分布式执行框架 | Low (Scope) | Planned (Phase 2) |
| [KI-05](#ki-05) | 真机硬件验证边界：当前仅在单芯片平台（T-Head PPU-ZW810E）完成实机验证 | Medium (Scope) | Planned (Multi-Chip CI) |
| [KI-06](#ki-06) | `DistributedDispatchGuard` 需对称进入受保护算子，NCCL/FlagCX 下建议显式初始化隔离 PG | Low (Usage) | Documented Contract |
| [KI-07](#ki-07) | `IntervalPolicy` 全局策略单文件映射与 `GraphBucketCompiler` / `DecodeGraphBucketRunner` 图捕获边界 | Medium | **RESOLVED in `v0.2.0-f09`** |
| [KI-08](#ki-08) | 多进程共享 `TRITON_CACHE_DIR` 开启磁盘预算淘汰时缺少跨进程文件锁 | Low (Ops) | **RESOLVED in `v0.2.0-f09`** |

---

## Detailed Known Issues / 详细说明（五要素完整记录）

### KI-01

**Jina 系列模型 `aiu_load_mask` 对齐翻转导致 Warmup 后冷编译**

- **影响范围**：使用自定义 eager attention（通过 `torch.einsum` 下沉至 `FlagGems` `_thead/ops/bmm.py`）且序列长度在 `S % 16 == 0` 与 `S % 16 != 0` 间切换的模型（如 `jina-reranker-v2-base-multilingual`、`jina-embeddings-v3` 在 T-Head PPU 后端上的变长推理）。
- **复现方式**：在 `FLAGTREE_SHAPE_BUCKETING=1` 下，仅使用非 16 对齐长度集合（如 `[264, 380, 524, 764]`）预热 `jina-reranker-v2-base` 或 `jina-embeddings-v3`，随后输入 16 对齐长度（如 `L=288`、`L=576`）。`_thead/ops/bmm.py` 的 `@libtuner(key=[..., "aiu_load_mask", ...])` 中 `aiu_load_mask` 从 `1/2` 翻转为 `3`，导致 2 个 `bmm` 算子 × 2 个桶触发 `COLD_TUNED=4` 与 `+62 .hgbin` 编译抖动。
- **规避方法**：在预热阶段为目标模型同时覆盖 16 对齐与非 16 对齐代表长度（例如每个桶额外包含一个 `S % 16 == 0` 样本），或对固定长度服务将序列显式 pad 至 16 的倍数。
- **责任人**：FlagTree / FlagGems F09 算子分桶维护小组（`@olivame` / FlagGems `_thead` 后端维护者）。
- **整改计划**：**已于 `v0.2.0-f09` 解决（RESOLVED）**。FlagGems `FG-P1-A` (`13590357eb`) 已将 `aiu_load_mask` 从 `_thead/ops/bmm.py` 的 `@libtuner(key=...)` 与 4 个核函数签名彻底解耦，FlagTree `FT-P1-B` (`e4522a5fc1`) 已将 `"aiu_load_mask"` 移出核心 `shape_bucketing.py` 并收敛至 `BackendBucketingCapabilities.extra_config_constexpr_keys`。PPU-8/9 真机复测（`USE_FLAGTUNE_COST_MODEL=0`）：`jina_reranker_v2_base` R0 预热后 `COLD_TUNED 4 -> 0`、`post_warm_new_hgbin +62 -> 0`、`hit_rate 99.70% -> 100.00%`、`post_warm_first 42,961.15 ms -> 428.34 ms`（`100.3×`）、`max_step 41,598.52 ms -> 40.26 ms`（`stall_reduction 1.41× -> 2,019.79×`）、稳态 `1.092×`（`cos_vs_aten=0.999968`）、R2-B Zipf P99 `24,706.07 ms -> 4,890.91 ms`（原 4 档 `5.05×`）及 `-> 42.50 ms`（推荐 7 档 `581.4×`，`cold_tuned=0`）、重启 `new_hgbin=0`（`9.14×`）；`jina_embeddings_v3` R0 `COLD_TUNED 4 -> 0`、`post_warm_new_hgbin +62 -> 0`、`hit_rate 99.89% -> 100.00%`、`post_warm_first 45,220.22 ms -> 1,483.83 ms`（`30.5×`，`1.068×` 超越 OFF）、`max_step 42,877.15 ms -> 128.80 ms`（`stall_reduction 1.74× -> 747.39×`）、稳态 `1.119×`（`cos_vs_aten=0.999314`, `cos_vs_off=0.999845`）、R2-B Zipf P99 `25,684.37 ms -> 4,035.08 ms`（`6.37×`）及 `-> 133.99 ms`（`191.7×`，`cold_tuned=0`）、重启 `new_hgbin=0`（`8.14×`），达成全量 **18/18（100%）真实模型零重编译**里程碑。

### KI-02

**T-Head PPU 多卡环境 `nccl` 后端 `ncclInternalError` 限制**

- **影响范围**：在当前参考容器（`flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64`）内于 T-Head PPU-ZW810E 上使用 `torch.distributed.init_process_group(backend="nccl")` 启动多卡 DDP/TP 训练。
- **复现方式**：在参考容器中设置 `CUDA_VISIBLE_DEVICES=0,1` 并运行 2-rank PyTorch 分布式脚本调用 `dist.init_process_group(backend="nccl")` 与 `dist.all_reduce(...)`，底层厂商通信库抛出 `ncclInternalError`。
- **规避方法**：在当前 PPU 参考镜像中使用 `backend="gloo"` 执行多卡分布式通信；若业务侧已部署支持设备间高速互联的新版 PCCL / FlagCX 通信库，可通过 `init_default_dispatch_guard(create_gloo_group=True)` 将分桶守卫侧通道隔离至 `gloo` PG，不影响训练主通信组。
- **责任人**：硬件运行环境与通信栈对接负责人（厂商 SDK 支持团队 + F09 分布式维护者 `@olivame`）。
- **整改计划**：在正式交付《例外记录》中登记本运行环境边界；待 T-Head 厂商发布修复 `ncclInternalError` 的新版 `thead-ppu-runtime` / PCCL / FlagCX 镜像后，补跑多卡 `nccl`/`flagcx` 实机 DDP/TP 验收矩阵。

### KI-03

**离线/受限网络环境下 FlagTune Cost Model 预加载阻塞**

- **影响范围**：在无公网访问、代理受限或本地模型缓存缺失的环境中，同时运行 FlagTree F09 与 FlagGems 算子（尤其是触发 `LibTuner` 冷调优的 Decoder/Embedding 模型）。
- **复现方式**：未设置 `USE_FLAGTUNE_COST_MODEL=0` 时，在受限网络容器内运行冷启动推理/训练，`flag_gems/flagtune/inference/cost_model.py` 会尝试拉取远端 `flagtune-xgb-*.tar.gz`，触发单次 39–264 秒阻塞，并在 `_AUTO_RETRY_SECONDS=60` 后静默重试，严重污染冷启/稳态时延。
- **规避方法**：在所有离线评估、基准测试及生产容器启动脚本中显式设置环境变量 `export USE_FLAGTUNE_COST_MODEL=0`（FlagGems 配套基准工具 `benchmarks/f09_acceptance_runner.py` 已默认注入该开关）。
- **责任人**：FlagTree / FlagGems F09 评测与运行时维护者（`@olivame`）。
- **整改计划**：已在当前交付文档与基准入口中强制说明并默认关闭；后续拟推动 FlagGems 上游在首次预加载网络失败后增加快速失败（fail-fast）开关，避免 60s 周期静默重试。

### KI-04

**`PartitionPlanner` 离线解析规划与在线执行框架解耦边界**

- **影响范围**：期望直接由 `triton.flagtune.PartitionPlanner` 自动接管 PyTorch `init_device_mesh`、TorchTitan、Megatron-LM、vLLM/SGLang 或 FlagCX 实时集群通信拓扑切分的用户。
- **复现方式**：检索仓库中 `PartitionPlanner` / `build_executable_plan()` 的调用点，可见其输出结构化策略报告与 `ExecutableParallelPlan` JSON 字典，当前未内嵌调用分布式训练/推理框架的进程组创建 API；此外 `HardwareClusterSpec.inter_node_bandwidth_gbps` 与 `inter_node_latency_us` 当前作为多节点扩展字段预留，尚未接入跨节点分层代价曲线。
- **规避方法**：在离线容量规划与并行策略选型阶段调用 `PartitionPlanner.plan_training()` / `plan_inference()` 生成 `ExecutableParallelPlan` JSON，再将输出的 `(dp, tp, pp, ep)` 维度映射为上层框架（TorchTitan / Megatron / vLLM）的启动参数。
- **责任人**：FlagTune 并行规划模块维护者（`@olivame`）。
- **整改计划**：二期路线图将补充跨节点分层通信代价曲线（启用 `inter_node_bandwidth_gbps` / `inter_node_latency_us`）、混合推理 `TP+PP`/`TP+EP` 搜索以及面向 `torch.distributed.device_mesh` 的薄适配层（目标版本：`v0.10-f09.2`）。

### KI-05

**单芯片实机验证边界（多芯片硬件待实机复验）**

- **影响范围**：部署在 NVIDIA CUDA、AMD ROCm、Ascend NPU、Kunlunxin XPU、Cambricon MLU、Moore Threads MUSA 等非 T-Head PPU 物理硬件上的用户。
- **复现方式**：检查本交付包全部实机测试记录可见，所有真实硬件算子精度、多模型 E2E 及多卡测试均在 T-Head PPU-ZW810E（1/3/4 卡）完成；其余后端仅通过 `BackendBucketingCapabilities` 契约与架构无关单元测试验证。
- **规避方法**：在目标新硬件上线前，使用隔离安装命令（`install_f09_editable.sh --target`）运行 FlagTree 全部 6 个单元测试文件及 FlagGems 算子回归，并通过 `register_backend_capabilities()` 核对目标芯片的 `divisibility`、`min_dot_size`（例如 FP8 `tl.dot` 的 `K >= 32` 约束）与 `alignment_spec_tags`。
- **责任人**：FlagOS F09 跨后端验收与 CI 维护团队（`@olivame` + 各硬件后端 Maintainer）。
- **整改计划**：`v0.2.0-f09` (`FT-P1-B` `e4522a5fc1`) 已补齐 `BackendBucketingCapabilities.fp8_min_dot_size`（`K >= 32`）与核心运行时去硬编码（`extra_config_constexpr_keys`、`torch_extension_module`、`supports_cudagraph_bench`、`autotuner.py` 通用 `**backend_kwargs`），但在交付《兼容性矩阵》与《例外记录》中仍严格标注非 PPU 后端为“待实机验证”；待多芯片 CI Runner 就绪后，优先补齐 NVIDIA CUDA 与 Ascend NPU 的端到端实机验证流水线。

### KI-06

**`DistributedDispatchGuard` 对称进入与隔离 PG 初始化约定**

- **影响范围**：启用 `FLAGTREE_DISTRIBUTED_DISPATCH_GUARD=1` 的多卡分布式训练或推理作业。
- **复现方式**：(1) 若在 NCCL/FlagCX 环境中未调用 `init_default_dispatch_guard()`，守卫会使用默认 PG（`process_group=None`）执行协商，可能在 GPU 通信流上产生同步竞争；(2) 若某动态控制流分支（如稀疏 MoE 专家路由）仅在部分 Rank 上进入受守卫保护的内核入口，未进入的 Rank 不会参与该次 `negotiate()`。
- **规避方法**：在 `dist.init_process_group()` 后由全部 Rank 统一调用一次 `triton.runtime.init_default_dispatch_guard(create_gloo_group=True)`；对于非对称条件触发的局部算子，关闭 `distributed_dispatch_guard` 或在算子外层使用对称控制流。
- **责任人**：FlagTree 运行时分布式模块维护者（`@olivame`）。
- **整改计划**：已通过 Fix-3 (`61534d97a2`) 彻底消除惰性 `dist.new_group()` 死锁并提供可重入 `dispatch_guard_scope()`；后续将在多卡框架集成中支持按算子粒度白名单/黑名单控制守卫启用。

### KI-07

**`IntervalPolicy` 全局路径复用与 `GraphBucketCompiler` / `DecodeGraphBucketRunner` 图捕获边界**

- **影响范围**：(1) 通过 `FLAGTREE_SHAPE_POLICY_PATH` 加载离线 PGO 策略运行包含多算子/异构形状的模型；(2) 直接在生产环境使用 `GraphBucketCompiler`（Inductor 模式）或 `DecodeGraphBucketRunner`（真实 CUDA/PPU Graph 重放）。
- **复现方式**：(1) `IntervalPolicy` Schema `flagtree.shape_policy.v1` 当前为单策略文件共享，未按算子名分表且 1D PGO 键包含对齐标志位，未见非对齐长度可能触发额外冷调优；(2) `DecodeGraphBucketRunner` 当前使用 Python `list` 更新 `static_kv_lens` 并在捕获失败返回 `None` 时未降级 eager；`GraphBucketCompiler` 当前主要验证 `backend="eager"` 路径，且 `plan_step` 配对历史需手动清理。
- **规避方法**：生产变长推理/训练优先使用默认在线策略 `OnlineBucketPolicy`（`FLAGTREE_SHAPE_BUCKETING=1`，不设置 `FLAGTREE_SHAPE_POLICY_PATH`）；`GraphBucketCompiler` 与 `DecodeGraphBucketRunner` 当前仅作为实验性图级分桶组件使用。
- **责任人**：FlagTree F09 运行时维护者（`@olivame`）。
- **整改计划**：**已于 `v0.2.0-f09` 解决（RESOLVED）**。FlagTree `FT-P1-A` (`9b92bd75dc`) 已修复全部 6 项核心缺陷：(1) 移除 1D `IntervalPolicy` 键中冗余 `align_sig[1]`（保留 `M=1` 单例隔离）；(2) `group_frontier()` 生成零空洞连续覆盖区间；(3) `normalize_entry_scalar()` 按 `group_id` 与 `all_args` 静态维度/dtype 精确匹配 1D/N-D 策略组；(4) 新增 `IntervalPolicy.bind_op()` 隔离每算子回退策略；(5) `GraphBucketCompiler` 增加 `_paired_step_plans` LRU 上界（`<=128`）、自定义 `seq_axis` 显存上限与 `_compiled_by_bucket` 编译缓存；(6) `DecodeGraphBucketRunner` 支持 `graph_obj is None` 安全回退 eager、静态设备 Tensor `static_kv_lens` 原地 `.copy_()` 更新及缩批尾行清零。PPU-10/11 真实模型 Auto-PGO V2 闭环验证（`USE_FLAGTUNE_COST_MODEL=0`）：`bge_reranker_v2_m3`（`11,109` 真实样本、`4/4` 组、`7/7` `ft_pgo_v1` 命中）`COLD_TUNED 33 -> 14 (-57.6%)`、冷启 `60,108.1 -> 54,073.0 ms (1.11×)`、未见长度 `13,400.7 -> 274.6 ms (48.80×, 0 new_hgbin, 0 cold_tuned)`、稳态 `1.115×`（`cos_vs_aten=0.999983`）；`qwen3_embedding_0_6b`（`43,659` 真实样本、`5/5` 组、`10/10` `ft_pgo_v1` 命中）6 档 `COLD_TUNED 65 -> 38 (-41.5%)`、冷启 `104,355.2 -> 91,892.7 ms (1.14×，比 ON 快 3.9s)`、未见长度 `2,582.3 -> 1,667.9 ms (19.47×)`、稳态 `1.105×`（超越 ON `1.092×`）、`min_cos_pgo_vs_off=1.000000`，策略全覆盖 7 档 A/B 实现 `1,995.0 ms vs 1,994.6 ms`（`1.000×` 精确持平、`0` 调优，彻底消除倒挂）。注：生产级 `torch.compile(backend="inductor")` 与硬件 CUDA/PPU Graph 端到端重放仍保持为既有范围边界，待后续框架集成专项扩展。

### KI-08

**多进程共享 `TRITON_CACHE_DIR` 的磁盘缓存淘汰并发竞态**

- **影响范围**：多卡多进程 DDP/TP 作业在共享同一 `TRITON_CACHE_DIR` 目录的同时开启 `FLAGTREE_MAX_CACHE_ENTRIES` 或 `FLAGTREE_MAX_CACHE_BYTES`。
- **复现方式**：多个进程同时调用 `cache.py` 中 `put_group()` 触发 `enforce_binary_cache_budget()`，因当前淘汰删除（`shutil.rmtree`）未加跨进程文件锁（`fcntl.flock`），存在一个进程删除另一进程正在加载的缓存条目的竞态窗口。
- **规避方法**：多卡运行开启磁盘缓存预算时，为每个 Rank 指定独立缓存目录（例如 `TRITON_CACHE_DIR=/tmp/triton_cache/rank_${RANK}`），或在共享缓存目录下保持磁盘预算关闭（默认 `None`）。
- **责任人**：FlagTree 缓存子系统维护者（`@olivame`）。
- **整改计划**：**已于 `v0.2.0-f09` 解决（RESOLVED）**。FlagTree `FT-P1-B` (`e4522a5fc1`) 在 `python/triton/runtime/cache.py` 中新增 POSIX `fcntl.flock` + 进程内 `threading.RLock` 混合锁 `_cache_root_lock`（冷写 `put()` 持有共享锁 `LOCK_SH` 并采用原子 `os.replace()` 与 `shutil.rmtree(ignore_errors=True)`，预算淘汰 `enforce_binary_cache_budget()` 持有排他锁 `LOCK_EX`，同时保持 `has_file()`/`get_file()`/`get_group()` **100% 无锁热读**）。6 进程 × 30 轮 × 3 次并发淘汰压测（`max_cache_entries=3`）：BEFORE **15 次 `FileNotFoundError`** → AFTER **0 异常、0 预算超限、`warm_read_lockfree=True`**，并已纳入 `test_runtime_bucketing_and_cache_budget.py`（FlagTree 单测总计 **48/48 PASS**）。注：无 `lockd` 的旧版 NFS 仍自动降级为进程内线程锁，此类网络文件系统建议继续按 Rank 隔离 `TRITON_CACHE_DIR`。
