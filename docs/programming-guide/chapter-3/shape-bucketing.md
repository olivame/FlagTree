# Online Shape Bucketing and Bucketed Autotune Cache

FlagTree provides an AST-guided online shape bucketing subsystem in `triton.runtime.shape_bucketing` and `triton.runtime.adjust_kernel_param` that eliminates per-shape autotuning and cache fragmentation for both dynamic inference and dynamic training workloads without requiring offline per-chip sweep tables.

## Overview

Dynamic workloads exercise different regions of the shape space:

- **Autoregressive LLM & Generative Recommendation Inference**:
  - **Decode**: small token batch sizes (`M in [1, 64]`) where memory-bound tiles differ sharply from compute-bound tiles, while `KV_CTX` grows by `+1` on every generated token (`16 -> 131072`).
  - **Prefill & MoE Routing**: variable sequence lengths (`M in [128, 32768]`) and per-expert token counts (`EM`).
- **Multi-Batch Retrieval, Reranker, Multimodal, and CTR Inference**:
  - Dynamic 2D `(Batch, SeqLen)` combinations requiring instant cold-start deployment without per-model offline sweeps.
- **Dynamic-Sequence Pretraining, SFT, and RL**:
  - Coupled forward (`M = B * S`) and backward weight-gradient contractions (`dW = dY^T @ X`, where contraction dimension `K = B * S` and contiguous tensor strides vary simultaneously).

## Environment Knobs

| Environment Variable | Knob Attribute | Default | Description |
| :--- | :--- | :--- | :--- |
| `FLAGTREE_SHAPE_BUCKETING` | `knobs.autotuning.shape_bucketing` | `False` | Enable AST-guided online shape bucketing in `Autotuner`, `Flagtuner`, and downstream operator libraries. |
| `FLAGTREE_SHAPE_BUCKETING_MODE` | `knobs.autotuning.shape_bucketing_mode` | `"auto"` | Phase ladder selection: `"auto"`, `"inference"`, or `"training"`. |
| `FLAGTREE_SHAPE_POLICY_PATH` | `knobs.autotuning.shape_policy_path` | `None` | Optional path to a profile-guided `IntervalPolicy` JSON file. |
| `FLAGTREE_MAX_BUCKET_ARTIFACTS` | `knobs.autotuning.max_bucket_artifacts` | `None` | Optional LRU capacity cap for in-memory bucket entries and on-disk `.autotune.json` artifacts. |
| `FLAGTREE_COMPILE_TIMEOUT_SEC` | `knobs.autotuning.compile_timeout_sec` | `None` | Optional subprocess compilation timeout threshold in seconds. |

## Architecture

1. **AST Axis Role Inference (`infer_bucketable_axes`)**:
   - Inspects kernel ASTs (including `@triton.jit` inner helpers, multi-`tl.dot` attention loops, `tl.make_block_ptr`, TMA descriptors, and reductions) to classify each parameter as `DYNAMIC_TILE_DIM`, `DYNAMIC_STRIDE`, or `INVARIANT_CONST`.
   - Parameters appearing in `tl.static_assert`, `tl.constexpr` annotations, or spatial convolution parameters (`stride_h`, `stride_w`, `groups`) remain exact invariants.
2. **Dual-Domain Bucket Ladder (`OnlineBucketPolicy`)**:
   - **Fine Decode Domain (`<= 64`)**: `(1, 2, 4, 8, 16, 24, 32, 48, 64)` preserves optimal small-batch tiles for Decode and online single-query serving.
   - **Geometric Throughput Domain (`> 64`)**: half-power-of-two progression (`64, 96, 128, 192, 256, ..., 131072, ...`) bounds worst-case tile padding while compressing `131,072` exact shapes into `<= 28` buckets.
   - **Contiguous Stride Pairing**: binds dynamic row-major strides (`stride_am == K`) to their paired dimension so backward `dW` contractions do not multiply bucket counts.
   - **Strict Specialization & Dtype Isolation**: bucket keys always preserve `(backend, tensor_dtypes, value == 1, value % 16 == 0)` while launching kernels with the exact unpadded runtime dimensions.
3. **Five-State Dispatch & Pre-Warming**:
   - Every dispatch records `HIT`, `COLD_TUNED`, `UNDERFLOW`, `OVERFLOW`, or `GUARD_MISMATCH` in `GLOBAL_BUCKET_TELEMETRY` and `Autotuner.last_dispatch_result`.
   - `warmup_buckets` pre-populates aligned and unaligned bucket representatives ahead of online serving or training Step 0.
