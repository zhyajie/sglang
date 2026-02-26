# SGLang Diffusion/多模态 Attention 模块技术报告

> 版本: v1.1 | 日期: 2026-02-26
> 目标: 分析 SGLang 多模态代码中 6 种高级 Attention 算法的原理、适用场景、精度影响、性能特征及 NVIDIA 实现现状
> 更新: 补充完整论文链接，基于论文 abstract 修正技术描述

---

## 目录

1. [总体架构概览](#1-总体架构概览)
2. [Sliding Tile Attention (STA)](#2-sliding-tile-attention-sta)
3. [SageAttention (Sage Attn)](#3-sageattention-sage-attn)
4. [Video Sparse Attention (VSA)](#4-video-sparse-attention-vsa)
5. [Sparse Linear Attention (SLA)](#5-sparse-linear-attention-sla)
6. [Sage Sparse Linear Attention (SageSLA)](#6-sage-sparse-linear-attention-sagesla)
7. [Video MoBA Attention (VMoBA)](#7-video-moba-attention-vmoba--svg2)
8. [术语说明与上游兼容矩阵](#8-术语说明与上游兼容矩阵) — T2V/TI2V/I2V 含义、TeaCache、SVG2、完整兼容表格
9. [六种 Attention 横向对比](#9-六种-attention-横向对比)
10. [NVIDIA 实现现状与 AMD 适配需求](#10-nvidia-实现现状与-amd-适配需求)
11. [Benchmark 测试方案](#11-benchmark-测试方案)

---

## 1. 总体架构概览

### 1.1 Attention Backend 注册体系

SGLang 的 Diffusion Attention 模块采用 **Backend 注册 + 工厂模式**。所有 Backend 定义在统一枚举中:

```
AttentionBackendEnum (interface.py:26-38)
├── FA / FA2          — FlashAttention 3/4 / FlashAttention 2
├── SLIDING_TILE_ATTN — Sliding Tile Attention
├── SAGE_ATTN         — SageAttention v2
├── SAGE_ATTN_3       — SageAttention 3 (Blackwell)
├── VIDEO_SPARSE_ATTN — Video Sparse Attention
├── VMOBA_ATTN        — Video MoBA Attention
├── SLA_ATTN          — Sparse Linear Attention
├── SAGE_SLA_ATTN     — Sage Sparse Linear Attention
├── AITER             — AMD AITer
├── TORCH_SDPA        — PyTorch SDPA
└── NO_ATTENTION      — No-op
```

### 1.2 统一接口

每个 Backend 都实现以下四个组件:

```
AttentionBackend (abstract)
├── get_impl_cls()     → AttentionImpl       # 具体计算实现
├── get_metadata_cls() → AttentionMetadata    # 元数据结构
├── get_builder_cls()  → AttentionMetadataBuilder  # 元数据构建器
└── get_enum()         → AttentionBackendEnum # 枚举标识
```

`AttentionImpl` 提供三个核心方法:
- `preprocess_qkv()` — 前处理 (tile/重排)
- `forward()` — 注意力计算
- `postprocess_output()` — 后处理 (untile/逆重排)

### 1.3 分布式 Attention 包装层

```
分布式 Attention 层
├── UlyssesAttention      — Ulysses SP: all-to-all scatter heads → attn → gather
├── UlyssesAttention_VSA  — VSA 专用: 额外传递 gate_compress
├── USPAttention          — Ulysses + Ring Attention 混合
├── LocalAttention        — 单设备直接调用
├── DistributedAttention  — A2A context-parallel (TurboDiffusion)
└── MinimalA2AAttnOp      — TurboWan 专用: 仅 SLA/SageSLA
```

### 1.4 整体数据流

```
                     ┌─────────────────────────────────────────┐
                     │         Distributed Wrapper             │
                     │  (Ulysses / USP / DistributedAttention) │
                     └─────────────┬───────────────────────────┘
                                   │
                     ┌─────────────▼───────────────────────────┐
                     │       preprocess_qkv()                  │
                     │  (tile / rearrange / partition)          │
                     └─────────────┬───────────────────────────┘
                                   │
               ┌───────────────────▼──────────────────────────┐
               │            AttentionImpl.forward()            │
               │                                               │
               │  ┌─────┐ ┌──────┐ ┌─────┐ ┌─────┐ ┌──────┐ │
               │  │ STA │ │ Sage │ │ VSA │ │ SLA │ │VMoBA │ │
               │  └─────┘ └──────┘ └─────┘ └─────┘ └──────┘ │
               └───────────────────┬──────────────────────────┘
                                   │
                     ┌─────────────▼───────────────────────────┐
                     │       postprocess_output()               │
                     │  (untile / rearrange / reverse)          │
                     └─────────────────────────────────────────┘
```

---

## 2. Sliding Tile Attention (STA)

### 2.1 基本信息

| 属性 | 详情 |
|------|------|
| **源码文件** | `backends/sliding_tile_attn.py`, `STA_configuration.py` |
| **枚举值** | `SLIDING_TILE_ATTN` |
| **底层算子** | `st_attn.sliding_tile_attention` (外部 CUDA 包) |
| **支持 head_size** | 32, 64, 96, 128, 160, 192, 224, 256 |
| **论文** | [Fast Video Generation with Sliding Tile Attention](https://arxiv.org/abs/2502.04507) (Zhang et al., 2025) |
| **来源** | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) |

### 2.2 算法原理

STA 利用了预训练视频扩散模型中 attention score 主要集中在局部 3D 窗口的特性。与传统 token 级滑动窗口注意力 (SWA) 不同，STA 以 **tile-by-tile** 的方式操作，采用硬件感知的滑动窗口设计，在保持表达能力的同时实现高效计算。论文报告 STA 在 HunyuanVideo 上将端到端延迟从 945s (FA3) 降至 685s (无需训练)，微调后进一步降至 268s，VBench 仅下降 0.09%。STA 实现了 58.79% MFU，是首个高效的 2D/3D 滑动窗口注意力实现。

#### 核心流程

```
Step 1: Tile 分割
──────────────────
将序列从 (B, T*H*W, Heads, D) 重排为 tile 结构

  原始序列: (n_t*ts_t, n_h*ts_h, n_w*ts_w)
                    ↓ rearrange
  Tile 序列: (n_t*n_h*n_w, ts_t*ts_h*ts_w)

  Base tile size = [6, 8, 8] (T, H, W)
  每个 tile 包含 6*8*8 = 384 个 token


Step 2: Per-Head Window Mask 策略
─────────────────────────────────
每个 attention head 可以有不同的 sliding window 大小

  Full window mapping:
  ┌────────────────────────────────────────┐
  │ 分辨率      │ dit_seq_shape │ full_win │
  │─────────────┼──────────────┼─────────│
  │ 115200~115456 │  30x48x80   │ [5,6,10]│
  │ 82944        │  36x48x48   │ [6,6,6] │
  │ 69120        │  18x48x80   │ [3,6,10]│
  └────────────────────────────────────────┘


Step 3: 稀疏 Attention 计算
───────────────────────────
Q_tile only attend to K/V within its sliding window

  ┌───┬───┬───┬───┬───┐
  │ T │ T │ T │   │   │  ← K/V tiles
  │ 1 │ 2 │ 3 │   │   │
  ├───┼───┼───┼───┼───┤
  │   │ T │ T │ T │   │  ← window shifts
  │   │ 2 │ 3 │ 4 │   │
  ├───┼───┼───┼───┼───┤
  │   │   │ T │ T │ T │
  │   │   │ 3 │ 4 │ 5 │
  └───┴───┴───┴───┴───┘
       Q tiles →

Step 4: Untile 还原
───────────────────
逆重排回原始序列结构
```

#### 四种工作模式

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ STA_searching│ ──→ │  STA_tuning  │ ──→ │STA_tuning_cfg│ ──→ │STA_inference │
│              │     │              │     │              │     │              │
│ 生成候选      │     │ 基于L2/L1    │     │ pos+neg CFG  │     │ 加载预计算的  │
│ window mask  │     │ 选最优per-   │     │ 联合调优      │     │ mask strategy│
│              │     │ head mask    │     │              │     │ (JSON)       │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### 2.3 适用场景

- **视频生成**: 长序列视频 DiT 的推理加速 (如 WanVideo, HunyuanVideo)
- **图像生成**: 高分辨率图像的 DiT 加速 (如 Flux)
- **特点**: 需要 **预搜索** 最优 mask 策略，不适合动态分辨率场景

### 2.4 对模型精度的影响

| 指标 | 影响 |
|------|------|
| **VBench** | 微调后仅 0.09% 下降 (论文数据) |
| **无训练场景** | 无质量退化即可实现 1.4x 端到端加速 |
| **L2 Loss** | per-head 精度搜索确保每个 head 使用最优窗口 |
| **视觉感知** | 高稀疏度下边缘细节可能模糊 |
| **CFG 敏感性** | 正/负 prompt 的最优 mask 可能不同，需 STA_tuning_cfg 模式 |

**精度控制机制**: STA 通过 `skip_time_steps` 参数控制前 N 步使用 Full Attention (默认 12 步)，后续步骤才使用稀疏 mask，因为扩散过程前期步骤对精度更敏感。

### 2.5 性能分析

```
性能收益模型:

  Full Attention FLOPs ∝ L^2 * H * D
  STA FLOPs ∝ L * W * H * D   (W = window_size << L)

  论文报告加速比:
  - Attention 单独: 2.8-17x vs FA2, 1.6-10x vs FA3
  - 端到端 (HunyuanVideo):
    * 无训练: 945s → 685s (1.38x)
    * 微调后: 945s → 268s (3.53x)
```

| 场景 | Attention 加速 | 端到端加速 | 备注 |
|------|-------------|-----------|------|
| vs FA2 | 2.8-17x | - | kernel 级别 |
| vs FA3 | 1.6-10x | - | kernel 级别 |
| HunyuanVideo (无训练) | - | 1.38x | 945s → 685s |
| HunyuanVideo (微调) | - | 3.53x | 945s → 268s, VBench -0.09% |

### 2.6 NVIDIA 实现现状

| 项目 | 状态 |
|------|------|
| **CUDA Kernel** | `st_attn` 外部包, 需要单独安装 |
| **SM 架构支持** | SM80+ (A100, H100, B200) |
| **torch.compile** | 不支持 (使用 `@torch.compiler.disable`) |
| **Tensor Core 利用** | 使用 FlashAttention 风格 tile kernel |
| **B200 适配** | 需要验证, Blackwell 架构可能需要调整 tile 参数 |

---

## 3. SageAttention (Sage Attn)

### 3.1 基本信息

| 属性 | 详情 |
|------|------|
| **源码文件** | `backends/sage_attn.py` (v2), `backends/sage_attn3.py` (v3) |
| **枚举值** | `SAGE_ATTN` (v2), `SAGE_ATTN_3` (v3) |
| **底层算子** | `sageattention.sageattn` (v2), `sageattn3.sageattn3_blackwell` (v3) |
| **支持 head_size** | v2: 32-256; v3: 64, 128, 256 |
| **论文 (v1)** | [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367) (Zhang et al., ICLR 2025) |
| **论文 (v2)** | [SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization](https://arxiv.org/abs/2411.10958) (Zhang et al., ICML 2025) |
| **论文 (v3)** | [SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training](https://arxiv.org/abs/2505.11594) (Zhang et al., NeurIPS 2025 Spotlight) |
| **来源** | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) |
| **安装** | `pip install sageattention==2.2.0 --no-build-isolation` |

### 3.2 算法原理

SageAttention 系列的核心思想是利用低精度量化加速 attention 的矩阵乘法，同时通过精度补偿技术维持数值精度。该系列论文分三个版本迭代:

- **v1** (ICLR 2025): INT8 量化 Q/K，OPS 比 FlashAttention2 快 ~2.1x，比 xformers 快 ~2.7x
- **v2** (ICML 2025): **INT4** 量化 Q/K + FP8 量化 PV，OPS 比 FlashAttention2 快 ~3x (RTX4090)，与 FlashAttention3-FP8 速度持平但精度更高 (Hopper)
- **v3** (NeurIPS 2025 Spotlight): **FP4** Microscaling 量化，利用 Blackwell FP4 Tensor Core，在 RTX5090 上达 1038 TOPS，比最快 FA 快 5x

#### SageAttention v2 原理

```
Standard Attention:
  Score = Q @ K^T / sqrt(d)     ← FP16/BF16 matmul
  Attn  = softmax(Score) @ V    ← FP16/BF16 matmul

SageAttention v2:
  Q_int4, q_scale = quantize_int4(Q)    ← per-thread INT4 量化 (更细粒度)
  K_int4, k_scale = quantize_int4(K)    ← per-thread INT4 量化
  Score_int = Q_int4 @ K_int4^T          ← INT4 matmul (Tensor Core)
  Score_fp  = dequantize(Score_int, q_scale, k_scale) / sqrt(d)
  P         = softmax(Score_fp)
  P_fp8     = quantize_fp8(P)            ← FP8 量化
  V_fp8     = quantize_fp8(V)            ← FP8 量化
  Attn      = P_fp8 @ V_fp8             ← FP8 matmul + two-level 累加

  ┌─────────────────────────────────────────────────┐
  │  关键精度增强技术:                                │
  │                                                   │
  │  1. Smooth-Q: 量化前平滑 Q，降低 INT4 clipping    │
  │     Q_smooth = Q - mean(Q, dim=-1)               │
  │                                                   │
  │  2. Per-thread 量化: 比 per-channel 更细的粒度    │
  │     每个 warp thread 独立 scale                   │
  │                                                   │
  │  3. Two-level PV 累加: 增强 FP8 PV 精度          │
  │     先在小 tile 内 FP32 累加,再跨 tile 合并       │
  │                                                   │
  │  Throughput:                                       │
  │  INT4 matmul ≈ 3x+ throughput vs FP16 matmul     │
  │  FP8  matmul ≈ 2x throughput vs FP16 matmul      │
  └─────────────────────────────────────────────────┘
```

> **注**: SGLang 代码中 `sage_attn.py` 使用的是 `sageattention.sageattn` API，包含 v1+v2 的统一接口。代码中使用 `tensor_layout="NHD"` 输入格式。

#### SageAttention 3 (Blackwell) 原理

```
SageAttention 3:
  Q, K → FP4 Microscaling 量化    ← 利用 Blackwell FP4 Tensor Core
  V → FP4 Microscaling 量化
  在 RTX5090 上达 1038 TOPS (5x faster than fastest FA)
  仅支持 MHA (Hq == Hkv)
  GQA 场景自动回退到 torch SDPA

  论文还探索了 8-bit attention 用于训练:
  - Fine-tuning: 无损
  - Pre-training: 收敛较慢
```

### 3.3 适用场景

- **通用 DiT 推理加速**: 无需修改模型，即插即用
- **低精度容忍场景**: 扩散模型 denoise 过程对量化误差有天然容忍度
- **显存敏感场景**: 降低 Q/K 精度减少约 40-50% 的 attention 显存
- **v3 仅适用于 Blackwell** (B200/B100)

### 3.4 对模型精度的影响

| 指标 | SageAttn v1 | SageAttn v2 | SageAttn 3 |
|------|-----------|------------|-----------|
| **端到端精度** | 几乎无损 (论文验证 LLM/图像/视频) | 几乎无损 (与 v1 一致) | 推理无损; 训练 fine-tuning 无损 |
| **QK 量化** | INT8 per-channel | INT4 per-thread (更细粒度) | FP4 Microscaling |
| **PV 量化** | FP16 | FP8 + two-level 累加 | FP4 |
| **数值稳定性** | smooth-K 技术 | smooth-Q 技术 (更有效) | Blackwell 专有指令 |
| **GQA 兼容** | 支持 | 支持 | 不支持, 回退 SDPA |
| **CogVideoX1.5 H20** | 25'34'' (FA2 基线) → 12'07'' | 同左 | N/A |

**关键精度技术**:
- **v1 smooth-K**: 量化前减去 K 的均值 (`K - mean(K, dim=-2)`)，降低 outlier 导致的 INT8 clipping 误差
- **v2 smooth-Q**: 对 Q 做平滑处理，比 smooth-K 更有效；per-thread INT4 量化提供更细粒度，减少信息损失
- **v2 two-level 累加**: FP8 PV 乘法使用两级 FP32 累加策略，先在小 tile 内累加再跨 tile 合并

### 3.5 性能分析

```
论文报告 OPS (operations per second) 对比:

  SageAttn v1:
    vs FlashAttention2: ~2.1x
    vs xformers:        ~2.7x

  SageAttn v2:
    vs FlashAttention2: ~3x   (RTX4090)
    vs xformers:        ~4.5x (RTX4090)
    vs FlashAttention3-FP8: 速度持平但精度更高 (Hopper)

  SageAttn 2++ (最新):
    RTX5090: SageAttention 达 560 TOPS, 2.7x faster than FlashAttention2

  SageAttn 3:
    RTX5090: 1038 TOPS, 5x faster than fastest FlashAttention
```

| 配置 | 版本 | vs FlashAttention | 备注 |
|------|-----|------------------|------|
| RTX4090 | v2 | ~3x vs FA2 | INT4 QK + FP8 PV |
| H100 | v2 | ≈ FA3-FP8 速度 | 更高精度 |
| H20 | v1/v2 | CogVideoX 25'34''→12'07'' | 端到端实测 |
| RTX5090 | v2++ | 2.7x vs FA2 | 560 TOPS |
| RTX5090 | v3 | 5x vs fastest FA | 1038 TOPS, FP4 |
| B200 | v2/v3 | TBD | 待测试 |

### 3.6 NVIDIA 实现现状

| 项目 | 状态 |
|------|------|
| **SM80 (A100)** | INT8/INT4 QK + FP8 PV kernel (v1/v2) |
| **SM86 (A10/A6000)** | 同上, v2 实测有加速 |
| **SM89 (L40/RTX4090)** | v2 最优: ~3x vs FA2 |
| **SM90 (H100/H20/H800)** | v2 ≈ FA3-FP8 速度但更高精度 |
| **SM12x (RTX5090)** | v2++: 560T, 2.7x vs FA2; v3: 1038T |
| **SM10x (B200)** | v3 Blackwell 专有 FP4 kernel |
| **安装** | `pip install sageattention==2.2.0 --no-build-isolation` |
| **torch.compile** | v2 支持 non-cudagraphs mode |
| **分布式推理** | v2 支持 |

---

## 4. Video Sparse Attention (VSA)

### 4.1 基本信息

| 属性 | 详情 |
|------|------|
| **源码文件** | `backends/video_sparse_attn.py` |
| **枚举值** | `VIDEO_SPARSE_ATTN` |
| **底层算子** | `vsa.video_sparse_attn` (外部 CUDA 包) |
| **支持 head_size** | 64, 128 |
| **Tile 大小** | `(4, 4, 4)` — T x H x W |
| **论文** | [VSA: Faster Video Diffusion with Trainable Sparse Attention](https://arxiv.org/abs/2505.13389) (Zhang et al., 2025) |
| **来源** | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) |

### 4.2 算法原理

VSA 是一种 **可训练的硬件高效稀疏 Attention**，可在训练和推理中替代 Full Attention。论文提出两阶段方案: **粗阶段** 将 token 池化为 tile 并识别高权重的 critical token; **细阶段** 仅在这些 tile 内计算 token 级 attention，采用 block computing layout 确保硬件效率。VSA 是一个单一可微 kernel，端到端可训练，无需后验 profiling，维持 FlashAttention3 85% 的 MFU。论文在 60M-1.4B 参数 DiT 上做了大规模消融和 scaling law 实验，VSA 在训练 FLOPS 降低 2.53x 的情况下扩散 loss 无下降。在 Wan-2.1 上 attention 加速 6x，端到端从 31s 降至 18s。

#### 核心流程

```
Step 1: 3D Tile 分区
────────────────────
将 (T, H, W) 空间分割成 4x4x4 的 tile

  Input:  (B, T*H*W, Heads, D)
              ↓ tile partition
  Tiled:  (B, num_tiles * tile_size, Heads, D)

  Variable block sizes 处理边界:
  ┌────┬────┬────┬──┐
  │4x4x│4x4x│4x4x│不│  ← 最后一列 tile 可能不满
  │ 4  │ 4  │ 4  │足│
  ├────┼────┼────┼──┤
  │4x4x│4x4x│4x4x│不│
  │ 4  │ 4  │ 4  │足│
  └────┴────┴────┴──┘


Step 2: Gate 路由
─────────────────
gate_compress 计算每个 tile pair 的重要性

  gate_compress: (B, L, Heads, D)  — 与 Q/K/V 同形状
          ↓
  score = Q_tile @ gate_compress_tile^T
          ↓
  topk tiles = select top-k by score


Step 3: TopK 稀疏 Attention
───────────────────────────
只计算选中的 tile pair 之间的 attention

  Sparsity 控制:
  cur_topk = ceil((1 - VSA_sparsity) * (total_seq / tile_size))

  Example: total_seq=115200, tile_size=64, sparsity=0.8
  cur_topk = ceil(0.2 * 1800) = 360 tiles


Step 4: Untile 还原
───────────────────
  reverse_tile_partition_indices 逆映射回原序列
```

#### 示意图

```
    Q tiles                    K/V tiles
  ┌─────────┐              ┌─────────────────┐
  │ Tile 0  │──gate──→     │ Tile 2  (top-1) │
  │         │──score──→    │ Tile 5  (top-2) │
  │         │──topk───→    │ Tile 0  (self)  │
  ├─────────┤              ├─────────────────┤
  │ Tile 1  │──gate──→     │ Tile 1  (self)  │
  │         │──score──→    │ Tile 3  (top-1) │
  │         │──topk───→    │ Tile 7  (top-2) │
  └─────────┘              └─────────────────┘

  每个 Q tile 只 attend 到 topk 个 K/V tile
  而非所有 tile → 稀疏化
```

### 4.3 适用场景

- **训练 + 推理加速**: 唯一同时用于训练和推理的稀疏注意力 (论文在 60M-1.4B 模型预训练验证)
- **视频生成**: 专为 3D 视频 latent 设计
- **Self-Attention Only**: 不支持 cross-attention (text tokens 不经过 VSA)
- **动态稀疏**: 每次 forward 都重新路由，适应内容变化
- **配合 WanVideo / HunyuanVideo** 使用
- **稀疏蒸馏**: FastVideo 支持 VSA + sparse distillation 实现 >50x denoising 加速

### 4.4 对模型精度的影响

| 指标 | 影响 |
|------|------|
| **训练 FLOPS** | 2.53x 降低, 扩散 loss 无下降 (论文 Pareto 最优点) |
| **推理质量** | Wan-2.1 上与 Full Attention 质量可比 |
| **Scaling Law** | 60M-1.4B 参数规模均验证有效 |
| **运动连贯性** | 两阶段 coarse-to-fine 保留关键时空关联 |
| **稀疏度 trade-off** | sparsity 越高加速越大但精度越差 |

### 4.5 性能分析

```
论文报告实测数据:

  训练: 2.53x FLOPs 降低 (Pareto 最优, diffusion loss 无下降)
  推理 (Wan-2.1):
    Attention 加速: 6x
    端到端: 31s → 18s (1.72x)
  MFU: 85% of FlashAttention3 (硬件利用率)
```

| 场景 | 加速 | 精度影响 | 备注 |
|------|------|---------|------|
| 训练 (预训练 60M-1.4B) | 2.53x FLOPs | diffusion loss 无下降 | Pareto optimal |
| 推理 Wan-2.1 | Attn 6x, E2E 1.72x | 质量可比 | 31s → 18s |
| 硬件效率 | 85% MFU of FA3 | - | 单一可微 kernel |

### 4.6 NVIDIA 实现现状

| 项目 | 状态 |
|------|------|
| **CUDA Kernel** | `vsa` 外部包, 需要单独安装 |
| **Variable Block Size** | 支持不等长 tile |
| **SM 架构** | SM80+ 预期支持 |
| **B200 适配** | 需要验证 |
| **torch.compile** | 不支持 (`@torch.compiler.disable`) |

---

## 5. Sparse Linear Attention (SLA)

### 5.1 基本信息

| 属性 | 详情 |
|------|------|
| **源码文件** | `backends/sparse_linear_attn.py` (前半部分, L1-L384) |
| **枚举值** | `SLA_ATTN` |
| **底层算子** | **自带 Triton JIT kernel** (`_attn_fwd`, `compress_kernel`) |
| **支持 head_size** | 64, 128 |
| **论文** | [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention](https://arxiv.org/abs/2509.24006) (Zhang et al., 2025) |
| **论文 (v2)** | [SLA2: Sparse-Linear Attention with Learnable Routing and QAT](https://arxiv.org/abs/2602.12675) (Zhang et al., 2026) |
| **框架论文** | [TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times](https://arxiv.org/abs/2512.16093) (2025) |
| **来源** | [thu-ml/SLA](https://github.com/thu-ml/SLA), [thu-ml/TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) |

### 5.2 算法原理

SLA 的核心观察是: attention weights 可以分为两部分——**少量大权重 (高秩)** 和 **大量小权重 (极低秩)**。这自然地引出了对大权重部分使用稀疏加速 (O(N^2))，对小权重部分使用低秩/线性加速 (O(N)) 的策略。SLA 将 attention 权重分为 critical、marginal、negligible 三类: 对 critical 权重用 O(N^2) attention，对 marginal 权重用 O(N) linear attention，跳过 negligible 权重。论文报告仅需少量微调步骤，即可实现 attention 计算量降低 20x，在 Wan2.1-1.3B 上实现 attention 13.7x 加速和端到端 2.2x 加速。

#### 核心流程

```
                         Input: Q, K, V (B, H, L, D)
                                  │
                    ┌─────────────┴─────────────┐
                    │                            │
           ┌────────▼────────┐          ┌────────▼────────┐
           │  Block-Sparse   │          │    Linear        │
           │  Attention      │          │    Attention     │
           └────────┬────────┘          └────────┬────────┘
                    │                            │
                    ▼                            ▼
               o_sparse                      o_linear
                    │                            │
                    │                    ┌───────▼───────┐
                    │                    │   proj_l()    │
                    │                    │ (learnable   │
                    │                    │  linear proj) │
                    │                    └───────┬───────┘
                    │                            │
                    └──────────┬─────────────────┘
                               │
                          output = o_sparse + proj_l(o_linear)
```

#### Block-Sparse Attention 详解

```
Step 1: Block Mean Pooling
──────────────────────────
  Q_blocks = mean_pool(Q, BLKQ=128)    ← Triton compress_kernel
  K_blocks = mean_pool(K-mean(K), BLKK=64)  ← smooth-K

Step 2: Block Score 计算
────────────────────────
  block_score = Q_blocks @ K_blocks^T   ← (M_BLOCKS, N_BLOCKS)

Step 3: TopK Block 选择
───────────────────────
  topk_ratio = 0.1 (默认 10%)
  lut = topk(block_score, k=topk_ratio * N_BLOCKS)

  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │   │ ■ │   │ ■ │   │   │ ■ │   │  ← N blocks
  ├───┼───┼───┼───┼───┼───┼───┼───┤
  │ ■ │   │ ■ │   │   │ ■ │   │   │
  ├───┼───┼───┼───┼───┼───┼───┼───┤
  │   │   │ ■ │   │ ■ │   │   │ ■ │  ← topk 选中
  └───┴───┴───┴───┴───┴───┴───┴───┘
    M blocks →

Step 4: Sparse Flash Attention
──────────────────────────────
  仅在选中的 block pair 上做标准 FlashAttention
  使用 Triton _attn_fwd kernel
  online softmax + LUT indexed access
```

#### Linear Attention 详解

```
Feature Map:
  φ_q = softmax(Q, dim=-1)   (或 elu+1, relu)
  φ_k = softmax(K, dim=-1)

Linear Attention:
  KV = φ_k^T @ V              ← (D, D) 矩阵
  K_sum = sum(φ_k, dim=-2)    ← (1, D) 向量
  output = (φ_q @ KV) / (φ_q @ K_sum^T + eps)

  复杂度: O(L * D^2) vs 标准 O(L^2 * D)
```

#### Triton Kernel 规格

```
compress_kernel:
  Grid: (L_BLOCKS, B*H)
  功能: block mean pooling, D 维 reduce

_attn_fwd:
  Grid: (M_BLOCKS, B*H)
  BLOCK_M: 64 或 128
  BLOCK_N: 64
  num_warps: 4 (D=64) 或 8 (D=128)
  num_stages: 3
  功能: block-sparse flash attention with LUT
  使用 log2 scale 优化 exp 计算
```

### 5.3 适用场景

- **TurboWan 推理**: 与 WanVideo 模型配合使用
- **长序列视频生成**: 结合稀疏和线性两种优势
- **可训练场景**: `proj_l` 是可学习参数，可微调适应特定模型
- **Sequence Parallel**: 支持 A2A context-parallel (DistributedAttention)

### 5.4 对模型精度的影响

| 指标 | 影响 |
|------|------|
| **端到端质量** | 减少 95% attention 计算量, 无生成质量退化 (论文) |
| **Linear Attn 补偿** | 通过 proj_l 学习残差, 补偿稀疏损失; marginal weights 由 linear attention 近似 |
| **微调需求** | 仅需少量微调步骤 (few-shot fine-tuning) 即可适配 |
| **初始化** | proj_l 初始化为 0, 保证初始行为 = sparse only |
| **Feature Map 选择** | softmax > elu > relu (精度从高到低) |
| **三级分类** | critical → O(N^2) sparse attn; marginal → O(N) linear attn; negligible → skip |

### 5.5 性能分析

```
FLOPs 分析:

  Sparse Branch:
    Block pooling:  O(L * D)
    Block scoring:  O(M * N * D)  (M, N << L)
    Sparse attn:    O(L * topk * BLKK * D / BLKQ)
    ≈ O(L * topk_ratio * L * D)

  Linear Branch:
    Feature map:    O(L * D)
    KV compute:     O(L * D^2)
    Output:         O(L * D^2)
    ≈ O(L * D^2)

  Total: O(L * topk_ratio * L * D + L * D^2)
  当 topk_ratio=0.1, D=128:
    ≈ O(0.1 * L^2 * D + L * D^2)
    相比 Full Attn O(L^2 * D), 约 0.1x + overhead
```

| 场景 | Attention 加速 | 端到端加速 | 备注 |
|------|--------------|-----------|------|
| Wan2.1-1.3B | 13.7x | 2.2x | 论文实测 |
| Attention 计算量 | 20x 降低 | - | 95% 计算跳过 |
| TurboDiffusion E2E | - | 100-200x | 含 SLA + rCM 蒸馏 + SageAttn |

### 5.6 NVIDIA 实现现状

| 项目 | 状态 |
|------|------|
| **Triton Kernel** | 自带, 不依赖外部包 |
| **SM 架构** | Triton 支持的所有 CUDA 架构 |
| **可训练** | `proj_l` 为 nn.Linear, 支持梯度 |
| **autograd** | `_attention` 实现了 `torch.autograd.Function` (仅 forward) |
| **B200 适配** | Triton 理论支持, 可能需要调 num_warps/num_stages |

---

## 6. Sage Sparse Linear Attention (SageSLA)

### 6.1 基本信息

| 属性 | 详情 |
|------|------|
| **源码文件** | `backends/sparse_linear_attn.py` (后半部分, L387-L695) |
| **枚举值** | `SAGE_SLA_ATTN` |
| **底层算子** | `spas_sage_attn._qattn` (INT8 QK + FP8 V 量化 kernel) |
| **支持 head_size** | 64, 128 |
| **论文 (SLA)** | [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention](https://arxiv.org/abs/2509.24006) (Zhang et al., 2025) |
| **论文 (SpargeAttn)** | [SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference](https://arxiv.org/abs/2502.18137) (Zhang et al., ICML 2025) |
| **来源** | [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn) + [thu-ml/SLA](https://github.com/thu-ml/SLA) |
| **安装** | `pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation` |

### 6.2 算法原理

SageSLA = SLA + **SpargeAttn 量化 kernel**。在 SLA 的 Block-Sparse 分支上，用 SpargeAttn (基于 SageAttention2) 提供的量化 block-sparse attention kernel 替代原始 Triton kernel。SpargeAttn 是一种通用的无需训练的稀疏注意力加速方案，使用两阶段在线过滤: 第一阶段快速预测 attention map 跳过部分矩阵乘法; 第二阶段使用 online softmax-aware filter 进一步跳过计算。SpargeAttn 已被 ICML 2025 接收。

#### 与 SLA 的区别

```
                    SLA                          SageSLA
              ┌──────────────┐            ┌──────────────┐
  Block-Sparse│  Triton FP16 │            │  INT8 QK +   │  ← 量化加速
  Branch      │  _attn_fwd   │            │  FP8/FP16 V  │
              └──────────────┘            └──────────────┘
              ┌──────────────┐            ┌──────────────┐
  Linear      │  torch FP16  │            │  torch FP16  │  ← 相同
  Branch      │  matmul      │            │  matmul      │
              └──────────────┘            └──────────────┘
```

#### 分架构 Kernel 选择

```
┌──────────────────────────────────────────────────────────────────┐
│ GPU 架构    │ Q/K 精度 │ V 精度  │ Kernel 函数                   │
│─────────────┼─────────┼────────┼──────────────────────────────│
│ SM80/86/87  │ INT8    │ FP16   │ qk_int8_sv_f16_accum_f16_   │
│ (A100/A10)  │         │        │ block_sparse_attn_inst_buf  │
│             │         │        │ _with_pv_threshold           │
│─────────────┼─────────┼────────┼──────────────────────────────│
│ SM90        │ INT8    │ FP8    │ qk_int8_sv_f8_accum_f32_    │
│ (H100)      │         │(e4m3fn)│ block_sparse_attn_inst_buf  │
│             │         │        │ _fuse_v_scale_sm90           │
│─────────────┼─────────┼────────┼──────────────────────────────│
│ SM10x+      │ INT8    │ FP8    │ qk_int8_sv_f8_accum_f16_    │
│ (B200)      │         │(e4m3fn)│ block_sparse_attn_inst_buf  │
│             │         │        │ _fuse_v_scale_with_pv_thresh │
└──────────────────────────────────────────────────────────────────┘
```

#### 量化流程

```
Step 1: Smooth-K + INT8 量化
────────────────────────────
  km = K.mean(dim=-2)
  Q_int8, q_scale, K_int8, k_scale = get_vanilla_qk_quant(Q, K, km, BLKQ, BLKK)

Step 2: Block Map + LUT
───────────────────────
  sparse_map, lut, topk = get_block_map(Q, K, topk_ratio)
  lut, valid_block_num = block_map_lut_triton(sparse_map)

Step 3: V 量化 (SM90+)
──────────────────────
  V → transpose + pad → FP8 (e4m3fn) + v_scale
  fused.transpose_pad_permute_cuda(V, V_transposed)
  fused.scale_fuse_quant_cuda(V_transposed, V_fp8, v_scale)

Step 4: 量化 Sparse Attention
────────────────────────────
  qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90(
      Q_int8, K_int8, V_fp8, output, lut, valid_block_num,
      q_scale, k_scale, v_scale, ...)
```

### 6.3 适用场景

- **TurboWan 推理**: 与 SLA 互为替代，量化加速版
- **SM90 (H100) 最优**: FP8 V 量化在 Hopper 架构上有最大收益
- **SM10x (B200) 支持**: 有专门的 Blackwell kernel
- **可训练**: 与 SLA 共享 `proj_l` 可学习参数

### 6.4 对模型精度的影响

| 指标 | 影响 |
|------|------|
| **量化误差** | INT8 QK + FP8 V 双重量化, 比 SageAttn v2 略大 |
| **Block-Sparse 补偿** | 仅在重要 block 上计算, 减少量化误差累积 |
| **pv_threshold** | SM80/SM10x 使用 PV threshold 过滤低贡献项 |
| **Linear Attn 缓冲** | 线性分支使用全精度, 补偿量化损失 |

### 6.5 性能分析

```
相比 SLA 的额外收益:

  INT8 QK matmul: ~2x throughput vs FP16
  FP8 V matmul:   ~2x throughput vs FP16 (SM90+)
  V transpose+pad+quant: 融合 kernel, 开销低

  Overall vs Full Attention:
    SLA 加速 ~3-5x
    SageSLA 加速 ~5-10x (SM90)
```

### 6.6 NVIDIA 实现现状

| 项目 | 状态 |
|------|------|
| **SM80 Kernel** | INT8 QK + FP16 V, 完整实现 |
| **SM90 Kernel** | INT8 QK + FP8 V, 使用 FP32 累加, 最优实现 |
| **SM10x Kernel** | INT8 QK + FP8 V + PV threshold, 已有 |
| **依赖** | `spas_sage_attn` 外部包 |
| **Block Map** | 使用 Triton `block_map_lut_triton` |
| **B200 适配** | 已有专用 kernel (`SAGE2PP_ENABLED` 检测) |

---

## 7. Video MoBA Attention (VMoBA / SVG2)

### 7.1 基本信息

| 属性 | 详情 |
|------|------|
| **源码文件** | `backends/vmoba.py`, `csrc/attn/vmoba_attn/vmoba/vmoba.py` |
| **枚举值** | `VMOBA_ATTN` |
| **底层算子** | `moba_attn_varlen` (仓库自带), 依赖 `flash_attn` |
| **支持 head_size** | 不限 (由底层 FlashAttention 决定) |
| **论文** | [VMoBA: Mixture-of-Block Attention for Video Diffusion Models](https://arxiv.org/abs/2506.23858) (Wu et al., KwaiVGI, 2025) |
| **来源** | [KwaiVGI/VMoBA](https://github.com/KwaiVGI/VMoBA) |

### 7.2 算法原理

VMoBA (Video Mixture of Block Attention) 是 MoBA 在视频扩散模型的专用扩展，由快手 KwaiVGI 团队提出。论文通过分析预训练视频 transformer 的 attention pattern，发现了强时空局部性、不同 query 重要性差异以及 head 级别的注意力集中度差异。基于此，VMoBA 在原始 MoBA 框架上做了三个关键改进:
1. **Layer-wise Recurrent Block Partition (1D-2D-3D)**: 逐层交替使用时间/空间/时空三种分块方式
2. **Global Block Selection**: 在整个 attention head 范围内全局选择最重要的 Q-K block 交互
3. **Threshold-based Block Selection**: 基于累积相似度动态决定每个 head attend 的 block 数量

论文报告训练加速 2.92x FLOPs, 1.48x 延迟 (576p, 55K tokens)；无训练推理加速 2.40x FLOPs, 1.35x 延迟。

#### 三种 Chunk 模式

```
按 layer 索引循环切换:

  Layer 0: temporal chunk   ← 按时间维度分块
  Layer 1: spatial chunk    ← 按空间维度分块
  Layer 2: st chunk         ← 时空联合分块
  Layer 3: temporal chunk   (循环)
  ...

  ┌──────────────────────────────────────────────────┐
  │  Temporal Chunk          Spatial Chunk            │
  │  ┌──┬──┬──┐              ┌────────────┐          │
  │  │t1│t2│t3│              │  ┌──┬──┬──┐│          │
  │  │  │  │  │              │  │h1│h2│h3││          │
  │  │  │  │  │              │  │w1│w2│w3││          │
  │  └──┴──┴──┘              │  └──┴──┴──┘│          │
  │  chunk = t * H * W       │  chunk = T * h * w    │
  │                          └────────────┘          │
  │                                                   │
  │  ST Chunk                                         │
  │  ┌────────┐                                       │
  │  │ct*ch*cw│  ← 3D block                          │
  │  └────────┘                                       │
  └──────────────────────────────────────────────────┘
```

#### MoBA 核心算法

```
Step 1: KV Chunking
───────────────────
  将 KV 按 chunk_size 分割
  cu_chunk = cumulative chunk boundaries

Step 2: Gate Score 计算
─────────────────────
  key_gate_weight = mean(K_chunk, dim=seq)  ← 每个 chunk 的均值
  gate = key_gate_weight @ Q^T              ← (num_chunk, H, L)

Step 3: Self-Chunk 保留
─────────────────────
  gate_self_chunk_mask: 每个 query 属于哪个 chunk
  self-chunk 始终被选中

Step 4: 选择策略 (topk 或 threshold)
───────────────────────────────────

  TopK 模式:
    amplify self-chunk gate score (+1e9)
    topk(gate, k=moba_topk, dim=chunk)

  Threshold 模式:
    normalize gate → cumsum → 选到 cumsum > threshold
    四种粒度: query_head / block / overall / head_global

Step 5: Mixed Attention
──────────────────────
  Self-Attention Branch:
    flash_attn(Q, K, V, cu_chunk)  ← 每个 chunk 内部 attention

  MoBA Branch:
    flash_attn(selected_Q, selected_KV)  ← 选中的跨 chunk attention

  LSE Merge:
    output = (self_output * exp(self_lse - mixed_lse)
            + moba_output * exp(moba_lse - mixed_lse))
```

#### 示意图

```
  KV Chunks:  [C1] [C2] [C3] [C4] [C5]
                │    │    │    │    │
  Gate Scores:  ┌────┼────┼────┼────┼────┐
  Q token i:    │0.8 │0.1 │0.9 │0.3 │0.2 │
                └────┼────┼────┼────┼────┘
                     │         │
                   [self]   [selected]    ← topk=2
                     ↓         ↓
              ┌──────────────────────┐
              │  Self Branch (C2)    │  → self_out, self_lse
              │  MoBA Branch (C3)    │  → moba_out, moba_lse
              │                      │
              │  LSE Merge ─────────→│  → final_output
              └──────────────────────┘
```

### 7.3 适用场景

- **视频生成**: WanVideo, HunyuanVideo 等 DiT 视频模型
- **自适应稀疏**: gate score 动态决定，不需要预搜索
- **多粒度 chunk**: 时间/空间/时空 三种 chunk 循环使用
- **threshold 模式**: 可以实现自适应稀疏度 (不同 head/layer 不同)

### 7.4 对模型精度的影响

| 指标 | 影响 |
|------|------|
| **生成质量** | 训练时与 Full Attention 可比甚至更优 (论文实验) |
| **LSE Merge** | 数学上等价于 softmax 重加权，无损合并 |
| **moba_threshold** | 默认 0.25, 基于累积相似度动态决定 attend block 数 |
| **first_full_step** | 前 12 步使用 Full Attention (精度敏感阶段) |
| **Layer-wise 分块** | 不同层交替使用 temporal/spatial/st 分块, 适应多样的时空模式 |
| **Chunk 粒度 trade-off** | 小 chunk → 更精确路由但更多 overhead |
| **vs 其他稀疏注意力** | VMoBA 是专门为 VDM 训练设计的 (不仅仅是推理加速) |

### 7.5 性能分析

```
复杂度:
  Self Branch: O(chunk_size^2 * H * D) per chunk
  MoBA Branch: O(selected_Q * chunk_size * H * D) per selected pair
  Gate Score:  O(num_chunk * H * L)

  Total ≈ O(L * chunk_size * H * D + topk * selected_Q * chunk_size * H * D)

  threshold=0.25 时约保留 25-40% 的 attention 计算量
```

| 场景 | FLOPs 加速 | 延迟加速 | 备注 |
|------|-----------|---------|------|
| 训练 576p (93x576x1024, 55K tokens) | 2.92x | 1.48x | 论文报告 |
| 无训练推理 (高分辨率) | 2.40x | 1.35x | 论文报告 |
| 更长序列 | 更高 | 更高 | 序列越长加速越明显 |
| 短序列 (<33K tokens) | 有限 | 可能无加速 | FlashAttn overhead 主导 |

### 7.6 NVIDIA 实现现状

| 项目 | 状态 |
|------|------|
| **核心依赖** | `flash_attn` (FlashAttention v2) |
| **MixedAttention** | 自定义 `torch.autograd.Function`, 支持前向+反向 |
| **Gate 计算** | PyTorch 原生 (bmm + arange) |
| **B200 适配** | 依赖 FlashAttention 对 Blackwell 的支持 |
| **训练支持** | 完整 backward 实现 |

---

## 8. 术语说明与上游兼容矩阵

### 8.1 模型任务类型缩写说明

在上游 sglang 兼容矩阵和模型命名中，频繁出现以下缩写:

| 缩写 | 全称 | 含义 | 示例模型 |
|------|------|------|---------|
| **T2V** | Text-to-Video | 纯文本输入 → 视频输出 | `Wan2.1-T2V-1.3B`, `Wan2.2-T2V-A14B` |
| **I2V** | Image-to-Video | 图像输入 → 视频输出 (以一张图作为首帧) | `Wan2.1-I2V-14B-480P`, `Wan2.2-I2V-A14B` |
| **TI2V** | Text-and-Image-to-Video | 文本 + 图像 → 视频输出 (同时接受文本描述和参考图) | `Wan2.2-TI2V-5B` |
| **T2I** | Text-to-Image | 纯文本输入 → 图像输出 | `FLUX.1-dev`, `Qwen-Image` |
| **I2I** | Image-to-Image | 图像输入 → 图像输出 (如风格转换) | - |
| **TI2I** | Text-and-Image-to-Image | 文本 + 图像 → 图像输出 (如图像编辑) | `Qwen-Image-Edit` |

代码中的枚举定义 (`configs/pipeline_configs/base.py`):
```python
class ModelTaskType(Enum):
    I2V  = auto()   # Image to Video
    T2V  = auto()   # Text to Video
    TI2V = auto()   # Text and Image to Video
    T2I  = auto()   # Text to Image
    I2I  = auto()   # Image to Image
    TI2I = auto()   # Text-Image to Image
```

### 8.2 TeaCache 说明

TeaCache (Timestep Embedding Aware Cache) 不是 Attention 算法，而是一种**时间步缓存加速技术**，通过检测连续去噪步骤之间的相似性来跳过冗余计算。

| 属性 | 详情 |
|------|------|
| **全称** | Timestep Embedding Aware Cache |
| **论文** | [Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://arxiv.org/abs/2411.19108) (2024) |
| **类型** | 训练无关的推理加速 (非 Attention 算法) |
| **原理** | 跟踪相邻时间步的 modulated input 的 L1 距离，距离小于阈值时复用上一步缓存的残差 |
| **加速** | 最高 4.41x (Open-Sora-Plan)，VBench 仅 -0.07% |

**工作流程**:
```
每一步 denoise:
  1. 计算 modulated_input = timestep_embedding 作用后的输入
  2. rel_l1 = |current - previous|.mean() / |previous|.mean()
  3. accumulated += poly(coefficients)(rel_l1)
  4. 如果 accumulated < threshold → 跳过计算, 复用缓存残差
     如果 accumulated >= threshold → 正常计算, 重置累加器
```

**CFG 支持**: Wan/HunyuanVideo/Z-Image 支持正负分支分别缓存; Flux/Qwen 不支持 CFG 分离。

### 8.3 SVG2 (Sparse Video Gen 2) 说明

SVG2 在上游 sglang 兼容矩阵中是一个独立列，对应枚举 `SPARSE_VIDEO_GEN_2_ATTN`。

> **注意**: 当前本地代码 (`Qwen-Image-v0.5.8` 分支) 的 `AttentionBackendEnum` 中**尚未包含** `SPARSE_VIDEO_GEN_2_ATTN`，该枚举仅存在于上游社区最新代码中。SVG2 的具体实现应参考上游 `main` 分支的最新代码。

从兼容矩阵来看，SVG2 支持以下模型:
- HunyuanVideo / FastHunyuan ✅
- Wan2.1 全系列 (T2V 1.3B/14B, I2V 480P/720P) ✅
- TurboWan 系列 ⭕ (不适用)
- FastWan / Wan2.2 系列 ❌

### 8.4 上游兼容矩阵 (完整)

> 来源: [sgl-project/sglang compatibility_matrix.md](https://github.com/sgl-project/sglang/blob/main/docs/diffusion/compatibility_matrix.md)

符号含义: ✅ = 完整兼容 | ❌ = 不兼容 | ⭕ = 不适用于该模型

#### 视频生成模型

| 模型名称 | HuggingFace Model ID | 分辨率 | TeaCache | STA | SageAttn | VSA | SLA | SageSLA | SVG2 |
|:---------|:---------------------|:-------|:--------:|:---:|:--------:|:---:|:---:|:-------:|:----:|
| FastWan2.1 T2V 1.3B | `FastVideo/FastWan2.1-T2V-1.3B-Diffusers` | 480p | ⭕ | ⭕ | ⭕ | ✅ | ❌ | ❌ | ❌ |
| FastWan2.2 TI2V 5B Full Attn | `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` | 720p | ⭕ | ⭕ | ⭕ | ✅ | ❌ | ❌ | ❌ |
| Wan2.2 TI2V 5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 720p | ⭕ | ⭕ | ✅ | ⭕ | ❌ | ❌ | ❌ |
| Wan2.2 T2V A14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 480p/720p | ❌ | ❌ | ✅ | ⭕ | ❌ | ❌ | ❌ |
| Wan2.2 I2V A14B | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 480p/720p | ❌ | ❌ | ✅ | ⭕ | ❌ | ❌ | ❌ |
| HunyuanVideo | `hunyuanvideo-community/HunyuanVideo` | 720x1280 / 544x960 | ❌ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| FastHunyuan | `FastVideo/FastHunyuan-diffusers` | 720x1280 / 544x960 | ❌ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| Wan2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 480p | ✅ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| Wan2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 480p/720p | ✅ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| Wan2.1 I2V 480P | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | 480p | ✅ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| Wan2.1 I2V 720P | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | 720p | ✅ | ✅ | ✅ | ⭕ | ❌ | ❌ | ✅ |
| TurboWan2.1 T2V 1.3B | `IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers` | 480p | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ⭕ |
| TurboWan2.1 T2V 14B | `IPostYellow/TurboWan2.1-T2V-14B-Diffusers` | 480p | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ⭕ |
| TurboWan2.1 T2V 14B 720P | `IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers` | 720p | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ⭕ |
| TurboWan2.2 I2V A14B | `IPostYellow/TurboWan2.2-I2V-A14B-Diffusers` | 720p | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ⭕ |

**备注**:
1. FastWan 系列模型的 STA/SageAttn 标记为 ⭕ (不适用) 是因为这些模型已经内置了 VSA 稀疏注意力
2. TurboWan 系列仅支持 SLA/SageSLA，因为 TurboDiffusion 框架专门使用这两种注意力
3. SageSLA 依赖 SpargeAttn: `pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation`
4. STA 目前仅支持 Hopper GPU (H100)

#### 图像生成模型

| 模型名称 | HuggingFace Model ID | 分辨率 |
|:---------|:---------------------|:-------|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | 任意分辨率 |
| FLUX.2-dev | `black-forest-labs/FLUX.2-dev` | 任意分辨率 |
| FLUX.2-Klein | `black-forest-labs/FLUX.2-klein-4B` | 任意分辨率 |
| Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` | 任意分辨率 |
| GLM-Image | `zai-org/GLM-Image` | 任意分辨率 |
| Qwen Image | `Qwen/Qwen-Image` | 任意分辨率 |
| Qwen Image 2512 | `Qwen/Qwen-Image-2512` | 任意分辨率 |
| Qwen Image Edit | `Qwen/Qwen-Image-Edit` | 任意分辨率 |

### 8.5 兼容矩阵关键解读

从上表可以得出以下分组规律:

```
┌─────────────────────────────────────────────────────────────────┐
│  模型分组          │ 优化路线                                   │
│───────────────────┼───────────────────────────────────────────│
│  FastWan 系列      │ VSA (内置稀疏蒸馏模型)                     │
│  (FastVideo出品)   │ 不需要额外 attention 优化                  │
│───────────────────┼───────────────────────────────────────────│
│  TurboWan 系列     │ SLA + SageSLA + TeaCache                  │
│  (TurboDiffusion) │ 与 rCM 蒸馏配合实现 100-200x 加速          │
│───────────────────┼───────────────────────────────────────────│
│  原版 Wan2.1       │ TeaCache + STA + SageAttn + SVG2          │
│                   │ 多种优化可组合                              │
│───────────────────┼───────────────────────────────────────────│
│  原版 Wan2.2       │ SageAttn (仅)                             │
│                   │ 较新模型, 更多优化还在开发中                 │
│───────────────────┼───────────────────────────────────────────│
│  HunyuanVideo     │ STA + SageAttn + SVG2                     │
│                   │ 无 TeaCache (待支持)                       │
│───────────────────┼───────────────────────────────────────────│
│  图像模型          │ SageAttn / FlashAttention                 │
│  (Flux/Qwen/etc.) │ 序列较短, 稀疏优化收益有限                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 六种 Attention 横向对比

### 9.1 特性对比矩阵

| 特性 | STA | SageAttn | VSA | SLA | SageSLA | VMoBA |
|------|-----|---------|-----|-----|---------|-------|
| **稀疏类型** | 结构化 sliding window | 量化 (非稀疏) | 动态 topk tile | block-sparse + linear | 量化 block-sparse + linear | 动态 chunk 路由 |
| **预搜索** | 需要 | 不需要 | 不需要 | 不需要 | 不需要 | 不需要 |
| **可训练** | 是 (微调后更优) | 否 | 是 (端到端可训练) | 是 (proj_l) | 是 (proj_l) | 是 (为训练设计) |
| **head_size** | 32-256 | 32-256 | 64, 128 | 64, 128 | 64, 128 | 不限 |
| **Cross-Attn** | 支持 | 支持 | 不支持 | 支持 | 支持 | 支持 |
| **GQA** | 支持 | 支持 | 支持 | 支持 | 支持 | 支持 |
| **训练支持** | 是 (微调) | 否 (推理专用) | 是 (预训练+推理) | 部分 (仅 forward autograd) | 否 (推理专用) | 是 (forward+backward) |
| **torch.compile** | 否 | 否 | 否 | 否 | 否 | 否 |
| **外部依赖** | st_attn | sageattention | vsa | 无 (Triton JIT) | spas_sage_attn | flash_attn |

### 9.2 加速比对比 (估算)

```
Attention 加速比 (论文报告数据汇总)

         ┌───────────────────────────────────────────┐
   14x   │                                      ■   │  SLA (attn 13.7x, Wan2.1-1.3B)
         │                                           │
   10x   │                                           │
         │                                           │
    6x   │                         ■                 │  VSA (attn 6x, Wan-2.1)
    5x   │                    ■                      │  SageAttn v3 (5x, RTX5090)
         │                                           │
    3x   │          ■    ■                           │  STA (2.8-17x vs FA2, 1.6-10x vs FA3)
         │     ■                                     │  VMoBA (2.92x FLOPs, 1.48x latency)
    2x   │ ■                                         │  SageAttn v2 (2.1-3x)
         │                                           │
    1x   │ baseline (FlashAttention)                 │
         └───────────────────────────────────────────┘

注: 各论文测试条件不同, 仅供定性参考
```

### 9.3 精度 vs 速度 Trade-off

```
  高精度 ──────────────────────────────────── 低精度
    │                                           │
    │  FlashAttn  SageAttn  STA  VMoBA  VSA  SageSLA
    │     ■         ■       ■     ■     ■      ■
    │                                           │
    │  1x         1.8x     3x   2.5x  4x    8x
    │                                           │
  低速度 ──────────────────────────────────── 高速度
```

### 9.4 适用模型对比

| 模型 | 推荐 Attention | 备注 |
|------|---------------|------|
| **WanVideo** | STA / VMoBA / SLA | TurboWan 仅支持 SLA/SageSLA |
| **HunyuanVideo** | VMoBA / VSA / STA | 支持多种稀疏策略 |
| **Flux** | SageAttn / STA | 图像生成, 序列较短 |
| **CausalWanVideo** | FA / SDPA | 因果 attention, 稀疏策略受限 |
| **QwenImage** | SageAttn / FA | VL 模型, 需要精确 attention |

---

## 10. NVIDIA 实现现状与 AMD 适配需求

### 10.1 NVIDIA 各架构支持矩阵

| Attention | SM80 (A100) | SM86 (A10) | SM89 (L40) | SM90 (H100) | SM10x (B200) |
|-----------|:-----------:|:----------:|:----------:|:-----------:|:------------:|
| **STA** | CUDA kernel | CUDA kernel | CUDA kernel | CUDA kernel | 需验证 |
| **SageAttn v2** | INT8 QK | INT8 QK | INT8 QK | INT8 QK | INT8 QK |
| **SageAttn 3** | - | - | - | - | FP4 kernel |
| **VSA** | CUDA kernel | CUDA kernel | CUDA kernel | CUDA kernel | 需验证 |
| **SLA** | Triton JIT | Triton JIT | Triton JIT | Triton JIT | Triton JIT |
| **SageSLA** | INT8+FP16 | INT8+FP16 | INT8+FP16 | INT8+FP8 | INT8+FP8+PV |
| **VMoBA** | FlashAttn v2 | FlashAttn v2 | FlashAttn v2 | FlashAttn v2 | 需验证 FA |

### 10.2 AMD 适配优先级

```
优先级排序 (基于使用频率 + 开发复杂度):

  P0 (必须):  SageAttn v2    — 最通用, 即插即用
  P0 (必须):  FlashAttention — VMoBA 和 FA 的基础依赖
  P1 (高):    SLA            — Triton 可能原生支持 ROCm
  P1 (高):    STA            — 需要完整 kernel 开发
  P2 (中):    VSA            — 需要 variable block attention kernel
  P2 (中):    SageSLA        — 需要完整量化 kernel 套件
  P3 (低):    VMoBA          — 依赖 FA 完成后可快速适配
  P4 (最低):  SageAttn 3     — Blackwell 专有, AMD 无对应架构
```

---

## 11. Benchmark 测试方案

### 11.1 测试环境

- **GPU**: NVIDIA B200 (Blackwell)
- **CUDA**: 12.x+
- **PyTorch**: 2.5+
- **精度**: BF16

### 11.2 测试矩阵

| 测试维度 | 值 |
|---------|---|
| **序列长度** | 4096, 16384, 32768, 69120, 115200 |
| **Batch Size** | 1 |
| **Heads** | 12, 24, 40 |
| **Head Dim** | 64, 128 |
| **Attention** | FA, SageAttn, STA, VSA, SLA, SageSLA, VMoBA |

### 11.3 测试指标

1. **延迟** (Latency): 单次 forward 时间 (ms)
2. **吞吐量** (Throughput): tokens/s
3. **显存** (Memory): peak GPU memory (GB)
4. **精度** (Accuracy): 与 Full Attention 的 L2/cosine 误差

### 11.4 Benchmark 脚本

参见附件: `benchmark_attention_backends.py`

---

## 附录

### A. 论文列表

| # | 名称 | 论文链接 | 代码仓库 | 发表 |
|---|------|---------|---------|------|
| 1 | **SageAttention v1** | [arxiv:2410.02367](https://arxiv.org/abs/2410.02367) | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) | ICLR 2025 |
| 2 | **SageAttention v2** | [arxiv:2411.10958](https://arxiv.org/abs/2411.10958) | 同上 | ICML 2025 |
| 3 | **SageAttention 2++** | [arxiv:2505.21136](https://arxiv.org/abs/2505.21136) | 同上 | - |
| 4 | **SageAttention 3** | [arxiv:2505.11594](https://arxiv.org/abs/2505.11594) | 同上 | NeurIPS 2025 Spotlight |
| 5 | **SpargeAttention** | [arxiv:2502.18137](https://arxiv.org/abs/2502.18137) | [thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn) | ICML 2025 |
| 6 | **Sliding Tile Attention** | [arxiv:2502.04507](https://arxiv.org/abs/2502.04507) | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) | - |
| 7 | **VSA** | [arxiv:2505.13389](https://arxiv.org/abs/2505.13389) | [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) | - |
| 8 | **SLA** | [arxiv:2509.24006](https://arxiv.org/abs/2509.24006) | [thu-ml/SLA](https://github.com/thu-ml/SLA) | - |
| 9 | **SLA2** | [arxiv:2602.12675](https://arxiv.org/abs/2602.12675) | 同上 | - |
| 10 | **VMoBA** | [arxiv:2506.23858](https://arxiv.org/abs/2506.23858) | [KwaiVGI/VMoBA](https://github.com/KwaiVGI/VMoBA) | - |
| 11 | **TurboDiffusion** | [arxiv:2512.16093](https://arxiv.org/abs/2512.16093) | [thu-ml/TurboDiffusion](https://github.com/thu-ml/TurboDiffusion) | - |

### B. 代码路径速查

```
sglang/multimodal_gen/runtime/
├── platforms/interface.py            # AttentionBackendEnum 定义
├── platforms/cuda.py                 # CUDA 平台 backend 选择逻辑
├── layers/attention/
│   ├── backends/
│   │   ├── attention_backend.py      # 抽象基类
│   │   ├── flash_attn.py             # FlashAttention 3/4
│   │   ├── sliding_tile_attn.py      # STA
│   │   ├── sage_attn.py              # SageAttention v2
│   │   ├── sage_attn3.py             # SageAttention 3
│   │   ├── video_sparse_attn.py      # VSA
│   │   ├── sparse_linear_attn.py     # SLA + SageSLA
│   │   ├── vmoba.py                  # VMoBA
│   │   ├── sdpa.py                   # PyTorch SDPA
│   │   └── aiter.py                  # AMD AITer
│   ├── layer.py                      # Ulysses/USP/Local wrappers
│   ├── turbo_layer.py                # TurboWan A2A wrappers
│   └── STA_configuration.py          # STA 搜索/调优工具
├── csrc/attn/vmoba_attn/vmoba/
│   └── vmoba.py                      # VMoBA 核心实现
```
