#!/usr/bin/env python3
"""
简单测试: 验证 aiter MoE 算子多次运行输出是否 bit 一致
使用 fused_moe (和 sglang 相同的接口)
"""

import os
os.environ["SGLANG_USE_AITER"] = "1"
os.environ["SGLANG_USE_TRITON_MOE"] = "0"
# 启用 aiter 详细日志
os.environ["AITER_LOG_LEVEL"] = "INFO"  # 可选: DEBUG, INFO, WARNING, ERROR
os.environ["AITER_LOG_MORE"] = "0"      # 设为 "1" 显示更详细的日志信息
# 绕过 tune config，使用默认配置（会选择 1stage ASM kernel）
os.environ["AITER_BYPASS_TUNE_CONFIG"] = "1"

import torch
# 使用 aiter.fused_moe (和 sglang 相同的函数)
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType
import aiter


def test_moe_bit_consistency(num_runs=10):
    """测试 MoE 算子的 bit 一致性"""
    
    # Qwen3-235B-A22B-FP8-dynamic 实际配置
    # 来自: /mnt/raid0/models/Qwen3-235B-A22B-FP8-dynamic/config.json
    num_tokens = 256
    hidden_dim = 4096           # hidden_size
    inter_dim = 1536            # moe_intermediate_size
    num_experts = 128           # num_experts
    top_k = 8                   # num_experts_per_tok
    
    # FP8 (W8A8, dynamic quantization)
    # hidden_states 是 bfloat16，weights 是 fp8
    hidden_dtype = torch.bfloat16
    weight_dtype = torch.float8_e4m3fnuz
    
    print(f"\n配置 (Qwen3-235B-A22B-FP8-dynamic):")
    print(f"  模型路径: /mnt/raid0/models/Qwen3-235B-A22B-FP8-dynamic")
    print(f"  tokens={num_tokens}, hidden={hidden_dim}, inter={inter_dim}")
    print(f"  experts={num_experts}, top_k={top_k}")
    print(f"  hidden_dtype={hidden_dtype}, weight_dtype={weight_dtype}")
    print(f"  W8A8 dynamic quantization, MI300X")
    print(f"\n运行 {num_runs} 次测试...")
    print(f"使用 fused_moe with QuantType.per_Token (和 sglang 相同的接口)")
    print(f"AITER_BYPASS_TUNE_CONFIG=1 (使用默认配置，token>16 会选择 1stage ASM kernel)")
    print(f"注意: 第一次运行时会打印 hipModuleLoad 加载 .co kernel 文件的日志\n")
    
    # 固定随机种子创建输入
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # 创建输入: hidden_states 是 bf16, weights 是 fp8
    hidden = torch.randn(num_tokens, hidden_dim, dtype=hidden_dtype, device="cuda")
    w1 = (torch.randn(num_experts, inter_dim * 2, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.1).to(weight_dtype)
    w2 = (torch.randn(num_experts, hidden_dim, inter_dim, dtype=torch.bfloat16, device="cuda") * 0.1).to(weight_dtype)
    
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    weights = torch.softmax(scores, dim=-1)
    topk_weights, topk_ids = torch.topk(weights, k=top_k, dim=-1)
    
    # FP8 需要 scale 参数 (per-tensor quantization)
    # 为了测试确定性，使用固定的 scale
    w1_scale = torch.ones(num_experts, inter_dim * 2, 1, dtype=torch.float32, device="cuda")
    w2_scale = torch.ones(num_experts, hidden_dim, 1, dtype=torch.float32, device="cuda")
    
    # 保存输入用于后续验证
    inputs = (
        hidden.clone(),
        w1.clone(),
        w2.clone(),
        topk_weights.to(torch.float32).clone(),
        topk_ids.to(torch.int32).clone(),
        w1_scale.clone(),
        w2_scale.clone(),
    )
    
    # 多次运行 (使用和 sglang 相同的 fused_moe)
    outputs = []
    for i in range(num_runs):
        output = fused_moe(
            inputs[0].clone(),
            inputs[1].clone(),
            inputs[2].clone(),
            inputs[3].clone(),
            inputs[4].clone(),
            quant_type=QuantType.per_Token,  # per-token quantization
            w1_scale=inputs[5].clone(),
            w2_scale=inputs[6].clone(),
            activation=ActivationType.Silu,
            expert_mask=None,
        )
        outputs.append(output.clone())
        print(f"完成第 {i+1}/{num_runs} 次运行")
    
    # 检查所有输出是否完全一致
    print("\n检查结果:")
    reference = outputs[0]
    all_consistent = True
    
    for i in range(1, num_runs):
        if torch.equal(outputs[i], reference):
            print(f"  运行 {i+1}: ✓ 一致")
        else:
            diff = (outputs[i] - reference).abs()
            max_diff = diff.max().item()
            num_diff = (outputs[i] != reference).sum().item()
            total = outputs[i].numel()
            print(f"  运行 {i+1}: ✗ 不一致 (不同元素: {num_diff}/{total}, 最大差异: {max_diff})")
            all_consistent = False
    
    print("\n" + "="*80)
    if all_consistent:
        print("✅ 测试通过! 所有运行输出完全一致 (bit-level)")
    else:
        print("❌ 测试失败! 检测到非确定性行为")
    print("="*80)
    
    return all_consistent


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("aiter MoE 算子 bit 一致性测试")
    print("="*80)
    print(f"环境变量:")
    print(f"  SGLANG_USE_AITER={os.environ['SGLANG_USE_AITER']}")
    print(f"  SGLANG_USE_TRITON_MOE={os.environ['SGLANG_USE_TRITON_MOE']}")
    print(f"  AITER_LOG_LEVEL={os.environ['AITER_LOG_LEVEL']}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 显示 aiter logger 状态
    aiter_logger = aiter.getLogger()
    print(f"aiter logger 级别: {aiter_logger.level} ({aiter_logger.name})")
    print()
    
    try:
        success = test_moe_bit_consistency(num_runs=10)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
