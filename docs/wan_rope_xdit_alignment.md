# SGLang WAN 模型 RoPE 实现对齐 xDiT/Diffusers 文档

## 概述

本文档记录了 SGLang 中 WAN 视频生成模型的 RoPE（Rotary Position Embedding）实现改造，使其与 xDiT/Diffusers 的实现完全对齐，并解决了 `torch.compile` 兼容性问题。

## 背景问题

### 原始问题
1. **精度问题**：生成的视频全是噪点
2. **帧数不一致**：输入 81 帧，输出 93 帧
3. **`torch.compile` 不兼容**：启用编译后出现各种错误

### 根本原因
1. SGLang 原有的 RoPE 实现（`NDRotaryEmbedding`）与 Diffusers 的 `WanRotaryPosEmbed` 在数学计算上存在差异
2. FSDP 的 meta device 初始化与 `torch.compile` 的动态 tensor 创建冲突
3. 序列并行（SP）的 padding 策略在 latent frames 层级而非 hidden states 层级

## 解决方案

### 1. RoPE 实现替换

将 SGLang 的 RoPE 实现完全替换为 xDiT/Diffusers 风格。

#### 核心函数 `apply_rotary_emb_wan`

```python
def apply_rotary_emb_wan(
    hidden_states: torch.Tensor,  # [B, S, H, D]
    freqs_cos: torch.Tensor,      # [1, S, 1, D]
    freqs_sin: torch.Tensor,      # [1, S, 1, D]
) -> torch.Tensor:
    """
    Apply rotary embeddings using xDiT/Diffusers style (interleaved).
    """
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)
```

#### `WanRotaryPosEmbed` 类

支持两种初始化模式：

| 模式 | 参数 | 适用场景 | torch.compile |
|------|------|----------|---------------|
| 直接初始化 | `use_meta_device=False` | 显存充足 | ✅ 完全兼容 |
| 延迟初始化 | `use_meta_device=True` | FSDP/显存不足 | 需要额外处理 |

### 2. 新增 `use_meta_device` 参数

#### 修改文件

**`server_args.py`**
```python
use_meta_device: bool = True  # 默认保持兼容
```

命令行参数：
```bash
--use-meta-device False  # 禁用 meta device，使用 xDiT 风格
```

**`fsdp_load.py`**
```python
def maybe_load_fsdp_model(..., use_meta_device: bool = True):
    if use_meta_device:
        # 原有逻辑：meta device 初始化（省显存）
        with torch.device("meta"):
            model = model_cls(**init_params)
    else:
        # 新增：直接在 GPU 初始化（xDiT 风格）
        model = model_cls(**init_params)
```

**`component_loader.py`**
```python
model = maybe_load_fsdp_model(
    ...
    use_meta_device=server_args.use_meta_device,
)
```

### 3. 序列并行 Padding 策略

从 latent frames 层级改为 hidden states 层级（与 xDiT 一致）：

```python
# 在 hidden_states 层级 padding（而非 latent frames）
if seq_pad_amount > 0:
    hidden_states = torch.cat([
        hidden_states,
        torch.zeros(batch_size, seq_pad_amount, hidden_dim, ...)
    ], dim=1)

# Chunk for SP
if sp_world_size > 1:
    hidden_states = torch.chunk(hidden_states, sp_world_size, dim=1)[sp_rank]
```

### 4. RoPE Padding 策略（与 xDiT 一致）

RoPE 的 padding 使用 zeros 填充 cos 和 sin（与 xDiT 完全一致）：

```python
# xDiT style: use zeros for both cos and sin padding
if seq_pad_amount > 0:
    freqs_cos = torch.cat([
        freqs_cos,
        torch.zeros(1, seq_pad_amount, 1, freqs_cos.shape[3], ...)
    ], dim=1)
    freqs_sin = torch.cat([
        freqs_sin,
        torch.zeros(1, seq_pad_amount, 1, freqs_sin.shape[3], ...)
    ], dim=1)
```

### 5. USPAttention torch.compile 兼容性

由于分布式通信操作（all-to-all）与 torch.compile 可能不兼容，USPAttention 的 forward 方法需要禁用 torch.compile：

```python
class USPAttention(nn.Module):
    @torch.compiler.disable
    def forward(self, q, k, v, ...):
        # 分布式 all-to-all 通信
        ...
```

## 配置对比

### RoPE 参数

| 参数 | SGLang | xDiT/Diffusers | 对齐状态 |
|------|--------|----------------|----------|
| `rope_max_seq_len` | 从 config 读取 | 从 config 读取 | ✅ |
| 默认值 | 1024 | 1024 | ✅ |
| `theta` | 10000.0 | 10000.0 | ✅ |
| `repeat_interleave` | ✅ | ✅ | ✅ |

### 初始化模式对比

| 特性 | `use_meta_device=True` | `use_meta_device=False` |
|------|------------------------|-------------------------|
| 显存需求 | 较低（分片加载） | 较高（完整加载） |
| FSDP 兼容 | ✅ | ❌ |
| torch.compile | 需要 `@torch.compiler.disable` | ✅ 完全兼容 |
| 加载速度 | 较慢 | 较快 |

## 使用方法

### 基本用法（xDiT 风格，推荐显存充足时使用）

```python
from sglang.multimodal_gen import DiffGenerator

generator = DiffGenerator.from_pretrained(
    model_path="/path/to/Wan2.2-T2V-A14B-Diffusers",
    num_gpus=8,
    ulysses_degree=8,
    enable_torch_compile=True,
    attention_backend="fa",
    use_meta_device=False,  # xDiT 风格：直接 GPU 初始化
)
```

### FSDP 模式（显存不足时使用）

```python
generator = DiffGenerator.from_pretrained(
    model_path="/path/to/Wan2.2-T2V-A14B-Diffusers",
    num_gpus=8,
    ulysses_degree=8,
    enable_torch_compile=True,
    attention_backend="fa",
    use_meta_device=True,   # 默认，使用 FSDP meta device
    dit_cpu_offload=True,   # 可选：启用 CPU offload
)
```

## 支持的视频尺寸

`rope_max_seq_len=1024` 时每个维度最多支持 1024 个 patch 位置：

| 维度 | patch_size | 最大支持 | 示例 (81帧, 720x1280) |
|------|------------|----------|----------------------|
| 时间 | 1 | 1024 帧 | 81 ✅ |
| 高度 | 2 | 2048 像素 | 720 ✅ |
| 宽度 | 2 | 2048 像素 | 1280 ✅ |

如需支持更大视频，可在模型配置中增加 `rope_max_seq_len`。

## 修改的文件列表

| 文件 | 修改内容 |
|------|----------|
| `runtime/models/dits/wanvideo.py` | RoPE 实现替换、`WanRotaryPosEmbed` 类、`apply_rotary_emb_wan` 函数（切片赋值）、RoPE padding 使用 zeros |
| `runtime/layers/attention/layer.py` | `USPAttention.forward` 添加 `@torch.compiler.disable` 装饰器 |
| `runtime/server_args.py` | 新增 `use_meta_device` 参数 |
| `runtime/loader/fsdp_load.py` | 支持非 meta device 初始化 |
| `runtime/loader/component_loader.py` | 传递 `use_meta_device` 参数 |

## 已修复的问题

1. ✅ 精度问题（噪点视频）
2. ✅ 帧数不一致（81→93）
3. ✅ `torch.compile` 兼容性
4. ✅ FSDP meta device 与动态 tensor 创建冲突
5. ✅ RoPE tensor 设备不匹配（CPU vs GPU）
6. ✅ `apply_rotary_emb_wan` 函数与 xDiT 完全一致（使用切片赋值）
7. ✅ RoPE padding 策略与 xDiT 一致（都使用 zeros 填充）
8. ✅ USPAttention 添加 `@torch.compiler.disable` 避免分布式通信与 compile 冲突

## 注意事项

1. `use_meta_device=False` 需要足够的 GPU 显存来完整加载模型
2. 使用 `use_meta_device=False` 时不要同时启用 `dit_cpu_offload`
3. 如遇到 `torch.compile` 问题，可尝试清除缓存：
   ```bash
   rm -rf ~/.cache/torch_compile
   find /path/to/sglang -name '__pycache__' -type d -exec rm -rf {} +
   ```
