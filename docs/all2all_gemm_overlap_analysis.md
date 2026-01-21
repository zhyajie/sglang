# Qwen Image Edit vs Wan: All-to-All 与 GEMM Overlap 分析

## 问题现象

在 timeline 分析中发现：
- **Qwen Image Edit**: All-to-All 通信能与 GEMM 计算 overlap
- **Wan**: All-to-All 通信无法与 GEMM 计算 overlap

本文档详细分析两者的差异和原因。

---

## 1. 核心差异总结

| 对比项 | Qwen Image Edit | Wan |
|--------|-----------------|-----|
| Attention 类 | `USPAttention` | `UlyssesAttention_VSA` |
| All-to-All API | `ft_c.all_to_all_single` | `dist.all_to_all_single` |
| `@torch.compiler.disable` | ❌ 没有 | ✅ 有 |
| torch.compile 优化 | ✅ 可以被优化 | ❌ 被禁用 |
| 能否 Overlap | ✅ 可以 | ❌ 不行 |

---

## 2. 代码调用链对比

### 2.1 Qwen Image Edit 调用链

```
QwenImageCrossAttention (qwen_image.py:520)
    └── self.attn = USPAttention(...)
            │
            ▼
USPAttention.forward (layer.py:341)  # 没有 @torch.compiler.disable
    └── _usp_input_all_to_all(q, head_dim=2)  (layer.py:373)
            │
            ▼
_usp_input_all_to_all (usp.py:49)
    └── _usp_all_to_all_single(x_c)  (usp.py:89)
            │
            ▼
_usp_all_to_all_single (usp.py:36)
    └── ft_c.all_to_all_single(x, ...)  # functional_collectives API
    └── _maybe_wait(x)
```

### 2.2 Wan 调用链

```
WanTransformerBlock_VSA (wanvideo.py:439)
    └── self.attn1 = UlyssesAttention_VSA(...)  (wanvideo.py:463)
            │
            ▼
UlyssesAttention_VSA.forward (layer.py:160)  # 有 @torch.compiler.disable !!!
    └── sequence_model_parallel_all_to_all_4D(qkvg, ...)  (layer.py:203)
            │
            ▼
sequence_model_parallel_all_to_all_4D (communication_op.py:29)
    └── get_sp_group().all_to_all_4D(input_, ...)
            │
            ▼
DeviceCommunicatorBase.all_to_all_4D (base_device_communicator.py:238)
    └── dist.all_to_all_single(output, input_, group=group)  # torch.distributed API
```

---

## 3. 关键差异分析

### 3.1 All-to-All API 差异

#### `ft_c.all_to_all_single` (Qwen Image 使用)

```python
# 文件: usp.py:36-46
import torch.distributed._functional_collectives as ft_c

def _usp_all_to_all_single(x: torch.Tensor) -> torch.Tensor:
    ulysses_pg = get_sp_group().ulysses_group
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single(
        x, output_split_sizes=None, input_split_sizes=None, group=ulysses_pg
    )
    x = _maybe_wait(x)  # 等待完成
    x = x.reshape(x_shape)
    return x
```

**特点**：
- `torch.distributed._functional_collectives` 是专门为 `torch.compile` 设计的
- 返回 `AsyncCollectiveTensor`，可以被编译器追踪和优化
- 编译器可以识别通信模式，自动调度 overlap

#### `dist.all_to_all_single` (Wan 使用)

```python
# 文件: base_device_communicator.py:142
import torch.distributed as dist

dist.all_to_all_single(
    output, input_, group=group
)  # 同步阻塞
```

**特点**：
- 传统的 `torch.distributed` API
- 默认是同步阻塞的（虽然有 `async_op=True` 选项，但代码未使用）
- 不易被 `torch.compile` 追踪

### 3.2 `@torch.compiler.disable` 的影响

#### UlyssesAttention_VSA (Wan)

```python
# 文件: layer.py:160-161
class UlyssesAttention_VSA(UlyssesAttention):
    @torch.compiler.disable  # <-- 禁用编译器优化！
    def forward(self, q, k, v, ...):
        ...
```

**影响**：
- 整个 forward 函数被编译器跳过
- 即使使用 `torch.compile`，这部分代码仍以 eager 模式执行
- 无法获得编译器的通信优化

#### USPAttention (Qwen Image)

```python
# 文件: layer.py:341
class USPAttention(nn.Module):
    # 没有 @torch.compiler.disable
    def forward(self, q, k, v, ...):
        ...
```

**影响**：
- 可以被 `torch.compile` 完整追踪
- 编译器识别 `ft_c.all_to_all_single` 返回的 `AsyncCollectiveTensor`
- 自动延迟 wait，实现通信与计算 overlap

---

## 4. Overlap 实现原理

### 4.1 torch.compile + functional_collectives 的魔法

当 `torch.compile` 遇到 `ft_c.all_to_all_single` 时：

```
Step 1: 发起异步通信
        ft_c.all_to_all_single(x) → 返回 AsyncCollectiveTensor
        
Step 2: 编译器分析数据依赖
        识别到后续的 GEMM 操作不依赖通信结果
        
Step 3: 调度优化
        ┌─────────────────────────────────────────────────────┐
        │ GPU Stream 1 (Compute)    GPU Stream 2 (NCCL Comm) │
        │                                                     │
        │ GEMM_1 ─────────────┐     All2All_1 ───────────┐   │
        │                     │                          │   │
        │ GEMM_2 ─────────────┤     All2All_2 ───────────┤   │
        │         overlap ────┼──────────────────────────┤   │
        │                     ▼                          ▼   │
        │                   sync (仅在需要数据时)            │
        └─────────────────────────────────────────────────────┘

Step 4: 延迟同步
        只有当真正需要通信结果时才调用 wait()
```

### 4.2 Wan 为什么无法 Overlap

```
┌─────────────────────────────────────────────────────┐
│ @torch.compiler.disable → Eager 模式执行           │
│                                                     │
│ dist.all_to_all_single() ──── 同步阻塞 ────────────│
│                               │                     │
│                               ▼ (必须等待完成)      │
│ GEMM ─────────────────────────────────────────────│
│                                                     │
│ 无法 overlap，串行执行                              │
└─────────────────────────────────────────────────────┘
```

---

## 5. 为什么 Wan 要禁用编译器？

`UlyssesAttention_VSA` 使用 `@torch.compiler.disable` 的可能原因：

### 5.1 VSA (Video Sparse Attention) 后端不兼容

```python
# video_sparse_attn.py
from vsa import video_sparse_attn  # 外部库
```

- VSA 使用外部的 Triton kernel
- `torch.compile` 难以追踪外部 kernel

### 5.2 动态缓存不兼容

```python
# video_sparse_attn.py
@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(...):
    ...
```

- `@lru_cache` 是 Python 层面的缓存
- 与 `torch.compile` 的图捕获机制冲突

### 5.3 dist.all_to_all_single 的图捕获问题

```python
# 使用 dist API 而非 ft_c API
dist.all_to_all_single(output, input_, group=group)
```

- `dist.all_to_all_single` 在图追踪时可能产生问题
- 涉及跨进程通信，编译器难以正确处理

---

## 6. 解决方案

### 方案 1: 让 Wan 使用 USPAttention

修改 `wanvideo.py`，将 `UlyssesAttention_VSA` 替换为 `USPAttention`：

```python
# 修改前
self.attn1 = UlyssesAttention_VSA(...)

# 修改后
self.attn1 = USPAttention(...)
```

**优点**：最小改动，复用现有代码
**缺点**：可能失去 VSA 的稀疏注意力优化

### 方案 2: 使用异步 All-to-All + 手动 Overlap

参考 `turbo_layer.py` 的实现：

```python
# turbo_layer.py:81-88
a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
    a2a_outputs[i - 1], a2a_inputs[i - 1], 
    group=cp_group, 
    async_op=True  # 异步!
)

# 在另一个 stream 中等待
with torch.cuda.stream(cp_stream):
    a2a_reqs[i - 2].wait()
```

**优点**：不依赖 `torch.compile`，在 eager 模式下也能 overlap
**缺点**：需要较多代码改动，手动管理 stream

### 方案 3: 将 UlyssesAttention_VSA 改用 ft_c API

修改 `layer.py` 中的 `UlyssesAttention_VSA`：

```python
# 修改前
qkvg = sequence_model_parallel_all_to_all_4D(qkvg, scatter_dim=2, gather_dim=1)

# 修改后
qkvg = _usp_input_all_to_all(qkvg, head_dim=2)  # 使用 ft_c 版本
```

同时移除 `@torch.compiler.disable`（如果 VSA 后端兼容）

**优点**：保留 VSA 功能，获得 overlap 能力
**缺点**：需要验证 VSA 后端与 torch.compile 的兼容性

---

## 7. 附录：代码位置索引

| 文件 | 路径 | 关键代码 |
|------|------|----------|
| USPAttention | `runtime/layers/attention/layer.py:289` | 无 `@torch.compiler.disable` |
| UlyssesAttention_VSA | `runtime/layers/attention/layer.py:157` | 有 `@torch.compiler.disable` |
| ft_c.all_to_all_single | `runtime/layers/usp.py:41` | functional_collectives API |
| dist.all_to_all_single | `runtime/distributed/device_communicators/base_device_communicator.py:142` | torch.distributed API |
| 异步 All-to-All 示例 | `runtime/layers/attention/turbo_layer.py:81` | `async_op=True` |
| Qwen Image Attention | `runtime/models/dits/qwen_image.py:520` | 使用 USPAttention |
| Wan Attention | `runtime/models/dits/wanvideo.py:463` | 使用 UlyssesAttention_VSA |
