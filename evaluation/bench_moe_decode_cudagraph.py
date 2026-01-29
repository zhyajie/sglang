#!/usr/bin/env python3
"""
Decode 性能对比: sglang triton moe vs aiter moe (batch=1)
使用 CUDA Graph 减少 kernel 启动开销
正确的 CUDA Graph 用法：捕获 num_runs 次完整迭代，只 replay 一次
"""

import os
import time
import torch

# 配置 (Qwen3-235B-A22B-FP8-dynamic with TP8, batch=1 decode)
num_tokens = 1              # batch=1 for decode
hidden_dim = 4096           # hidden_size
inter_dim_total = 1536      # moe_intermediate_size (total)
inter_dim = 192             # TP8: 1536 / 8 = 192 per rank
num_experts = 128           # num_experts (全部 experts，不切分)
top_k = 8                   # num_experts_per_tok

hidden_dtype = torch.bfloat16
weight_dtype = torch.float8_e4m3fnuz

num_warmup = 10
num_runs = 1000  # 捕获 1000 次迭代，只下发一次

print("="*80)
print("MoE Decode 性能对比测试 (CUDA Graph)")
print("="*80)
print(f"配置 (Qwen3-235B-A22B-FP8-dynamic with TP8, Decode):")
print(f"  num_tokens={num_tokens} (batch=1 decode)")
print(f"  hidden_dim={hidden_dim}")
print(f"  inter_dim={inter_dim} (TP8: {inter_dim_total}/8)")
print(f"  num_experts={num_experts}")
print(f"  top_k={top_k}")
print(f"  hidden_dtype={hidden_dtype}")
print(f"  weight_dtype={weight_dtype}")
print(f"  warmup={num_warmup}, runs={num_runs}")
print(f"  CUDA Graph: 捕获 {num_runs} 次迭代，只 replay 一次")
print()


def prepare_inputs():
    """准备测试输入"""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    hidden = torch.randn(num_tokens, hidden_dim, dtype=hidden_dtype, device="cuda")
    
    # TP8: intermediate_size 被切分
    w1 = (torch.randn(num_experts, inter_dim * 2, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.1).to(weight_dtype)
    w2 = (torch.randn(num_experts, hidden_dim, inter_dim, dtype=torch.bfloat16, device="cuda") * 0.1).to(weight_dtype)
    
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    weights = torch.softmax(scores, dim=-1)
    topk_weights, topk_ids = torch.topk(weights, k=top_k, dim=-1)
    
    # sglang triton moe 使用 per-channel quantization (PTPC)
    w1_scale = torch.ones(num_experts, inter_dim * 2, dtype=torch.float32, device="cuda")
    w2_scale = torch.ones(num_experts, hidden_dim, dtype=torch.float32, device="cuda")
    
    # aiter moe 使用 per-token quantization
    w1_scale_aiter = w1_scale.unsqueeze(-1)
    w2_scale_aiter = w2_scale.unsqueeze(-1)
    
    return hidden, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale, w1_scale_aiter, w2_scale_aiter


def benchmark_aiter_moe_with_cudagraph():
    """测试 aiter moe 性能 (使用 CUDA Graph)"""
    print("\n" + "="*80)
    print("测试 1: aiter MoE (ASM kernel) with CUDA Graph")
    print("="*80)
    
    os.environ["SGLANG_USE_AITER"] = "1"
    os.environ["SGLANG_USE_TRITON_MOE"] = "0"
    os.environ["AITER_LOG_LEVEL"] = "WARNING"
    os.environ["AITER_BYPASS_TUNE_CONFIG"] = "1"
    os.environ["AITER_MOE_SMALL_BATCH"] = "1"  # 启用 small batch 优化
    
    from aiter.fused_moe import fused_moe
    from aiter import ActivationType, QuantType
    
    hidden, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale, w1_scale_aiter, w2_scale_aiter = prepare_inputs()
    
    # 输入输出 buffer (固定内存位置用于 CUDA Graph)
    hidden_buf = hidden.clone()
    output_buf = torch.empty_like(hidden)
    
    # Warmup (不使用 CUDA Graph)
    print("Warmup (普通模式)...")
    for _ in range(num_warmup):
        output = fused_moe(
            hidden_buf,
            w1,
            w2,
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            quant_type=QuantType.per_Token,
            w1_scale=w1_scale_aiter,
            w2_scale=w2_scale_aiter,
            activation=ActivationType.Silu,
            expert_mask=None,
        )
    torch.cuda.synchronize()
    
    # 测试普通模式性能
    print(f"运行 {num_runs} 次测试 (普通模式)...")
    start = time.perf_counter()
    for _ in range(num_runs):
        output = fused_moe(
            hidden_buf,
            w1,
            w2,
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            quant_type=QuantType.per_Token,
            w1_scale=w1_scale_aiter,
            w2_scale=w2_scale_aiter,
            activation=ActivationType.Silu,
            expert_mask=None,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_time_normal_ms = (end - start) / num_runs * 1000
    print(f"普通模式平均耗时: {avg_time_normal_ms:.4f} ms")
    
    # 捕获 CUDA Graph (捕获完整的 num_runs 次迭代)
    print(f"\n捕获 CUDA Graph ({num_runs} 次迭代)...")
    graph = torch.cuda.CUDAGraph()
    
    # Warmup for graph capture
    for _ in range(3):
        output_buf = fused_moe(
            hidden_buf,
            w1,
            w2,
            topk_weights.to(torch.float32),
            topk_ids.to(torch.int32),
            quant_type=QuantType.per_Token,
            w1_scale=w1_scale_aiter,
            w2_scale=w2_scale_aiter,
            activation=ActivationType.Silu,
            expert_mask=None,
        )
    torch.cuda.synchronize()
    
    # Capture num_runs 次完整迭代
    with torch.cuda.graph(graph):
        for _ in range(num_runs):
            output_buf = fused_moe(
                hidden_buf,
                w1,
                w2,
                topk_weights.to(torch.float32),
                topk_ids.to(torch.int32),
                quant_type=QuantType.per_Token,
                w1_scale=w1_scale_aiter,
                w2_scale=w2_scale_aiter,
                activation=ActivationType.Silu,
                expert_mask=None,
            )
    
    print("CUDA Graph 捕获完成")
    
    # 测试 CUDA Graph 性能 (只 replay 一次，但包含 num_runs 次迭代)
    print(f"Replay CUDA Graph (包含 {num_runs} 次迭代)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    graph.replay()  # 只下发一次
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_time_graph_ms = (end - start) / num_runs * 1000
    print(f"CUDA Graph 模式平均耗时: {avg_time_graph_ms:.4f} ms (单次迭代)")
    
    speedup = avg_time_normal_ms / avg_time_graph_ms
    print(f"CUDA Graph 加速比: {speedup:.2f}x")
    
    return avg_time_normal_ms, avg_time_graph_ms, output_buf


def benchmark_sglang_triton_moe_with_cudagraph():
    """测试 sglang triton moe 性能 (使用 CUDA Graph)"""
    print("\n" + "="*80)
    print("测试 2: sglang Triton MoE with CUDA Graph")
    print("="*80)
    
    os.environ["SGLANG_USE_AITER"] = "0"
    os.environ["SGLANG_USE_TRITON_MOE"] = "1"
    os.environ["SGLANG_MOE_PADDING"] = "0"
    
    # 初始化 sglang 全局配置
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    server_args = ServerArgs(
        model_path="dummy",
        enable_deterministic_inference=False,
    )
    set_global_server_args_for_scheduler(server_args)
    
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
    from sglang.srt.layers.moe import MoeRunnerConfig
    
    hidden, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale, w1_scale_aiter, w2_scale_aiter = prepare_inputs()
    
    # 输入输出 buffer
    hidden_buf = hidden.clone()
    
    topk_output = (topk_weights, topk_ids, None)
    moe_config = MoeRunnerConfig(
        activation="silu",
        is_gated=True,
        inplace=False,
        num_experts=num_experts,
        num_local_experts=num_experts,
    )
    
    # Warmup (不使用 CUDA Graph)
    print("Warmup (普通模式)...")
    for _ in range(num_warmup):
        output = fused_moe(
            hidden_buf.clone(),
            w1,
            w2,
            topk_output,
            moe_config,
            use_fp8_w8a8=True,
            per_channel_quant=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
    torch.cuda.synchronize()
    
    # 测试普通模式性能
    print(f"运行 {num_runs} 次测试 (普通模式)...")
    start = time.perf_counter()
    for _ in range(num_runs):
        output = fused_moe(
            hidden_buf.clone(),
            w1,
            w2,
            topk_output,
            moe_config,
            use_fp8_w8a8=True,
            per_channel_quant=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_time_normal_ms = (end - start) / num_runs * 1000
    print(f"普通模式平均耗时: {avg_time_normal_ms:.4f} ms")
    
    # 捕获 CUDA Graph (捕获完整的 num_runs 次迭代)
    print(f"\n捕获 CUDA Graph ({num_runs} 次迭代)...")
    graph = torch.cuda.CUDAGraph()
    output_buf = torch.empty_like(hidden)
    
    # Warmup for graph capture
    for _ in range(3):
        output_buf = fused_moe(
            hidden_buf,
            w1,
            w2,
            topk_output,
            moe_config,
            use_fp8_w8a8=True,
            per_channel_quant=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
    torch.cuda.synchronize()
    
    # Capture num_runs 次完整迭代
    with torch.cuda.graph(graph):
        for _ in range(num_runs):
            output_buf = fused_moe(
                hidden_buf,
                w1,
                w2,
                topk_output,
                moe_config,
                use_fp8_w8a8=True,
                per_channel_quant=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )
    
    print("CUDA Graph 捕获完成")
    
    # 测试 CUDA Graph 性能 (只 replay 一次，但包含 num_runs 次迭代)
    print(f"Replay CUDA Graph (包含 {num_runs} 次迭代)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    graph.replay()  # 只下发一次
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_time_graph_ms = (end - start) / num_runs * 1000
    print(f"CUDA Graph 模式平均耗时: {avg_time_graph_ms:.4f} ms (单次迭代)")
    
    speedup = avg_time_normal_ms / avg_time_graph_ms
    print(f"CUDA Graph 加速比: {speedup:.2f}x")
    
    return avg_time_normal_ms, avg_time_graph_ms, output_buf


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # 测试 aiter moe
    aiter_time_normal, aiter_time_graph, aiter_output = benchmark_aiter_moe_with_cudagraph()
    
    # 测试 sglang triton moe
    triton_time_normal, triton_time_graph, triton_output = benchmark_sglang_triton_moe_with_cudagraph()
    
    # 性能对比总结
    print("\n" + "="*80)
    print("性能对比结果总结")
    print("="*80)
    print("\n【普通模式 (含 kernel 启动开销)】")
    print(f"aiter MoE:          {aiter_time_normal:.4f} ms")
    print(f"sglang Triton MoE:  {triton_time_normal:.4f} ms")
    
    if aiter_time_normal < triton_time_normal:
        speedup = triton_time_normal / aiter_time_normal
        print(f"✅ aiter MoE 更快: {speedup:.2f}x")
    else:
        speedup = aiter_time_normal / triton_time_normal
        print(f"✅ sglang Triton MoE 更快: {speedup:.2f}x")
    
    print("\n【CUDA Graph 模式 (纯计算性能，消除下发开销)】")
    print(f"aiter MoE:          {aiter_time_graph:.4f} ms")
    print(f"sglang Triton MoE:  {triton_time_graph:.4f} ms")
    
    if aiter_time_graph < triton_time_graph:
        speedup = triton_time_graph / aiter_time_graph
        print(f"✅ aiter MoE 更快: {speedup:.2f}x")
    else:
        speedup = aiter_time_graph / triton_time_graph
        print(f"✅ sglang Triton MoE 更快: {speedup:.2f}x")
    
    print("\n【Kernel 启动开销占比】")
    aiter_overhead = (aiter_time_normal - aiter_time_graph) / aiter_time_normal * 100
    triton_overhead = (triton_time_normal - triton_time_graph) / triton_time_normal * 100
    print(f"aiter MoE:          {aiter_overhead:.2f}%")
    print(f"sglang Triton MoE:  {triton_overhead:.2f}%")
    
    print("="*80)
