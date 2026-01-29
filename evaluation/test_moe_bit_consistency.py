#!/usr/bin/env python3

import os
os.environ["SGLANG_USE_AITER"] = "1"
os.environ["SGLANG_USE_TRITON_MOE"] = "0"
os.environ["AITER_LOG_LEVEL"] = "INFO"
os.environ["AITER_LOG_MORE"] = "0"
os.environ["AITER_BYPASS_TUNE_CONFIG"] = "1"

import torch
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType


def test_moe_bit_consistency(num_runs=10):
    num_tokens = 256
    hidden_dim = 4096
    inter_dim = 1536
    num_experts = 128
    top_k = 8
    
    hidden_dtype = torch.bfloat16
    weight_dtype = torch.float8_e4m3fnuz
    
    print(f"\nConfig: tokens={num_tokens}, hidden={hidden_dim}, inter={inter_dim}")
    print(f"experts={num_experts}, top_k={top_k}")
    print(f"Running {num_runs} tests...\n")
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    hidden = torch.randn(num_tokens, hidden_dim, dtype=hidden_dtype, device="cuda")
    w1 = (torch.randn(num_experts, inter_dim * 2, hidden_dim, dtype=torch.bfloat16, device="cuda") * 0.1).to(weight_dtype)
    w2 = (torch.randn(num_experts, hidden_dim, inter_dim, dtype=torch.bfloat16, device="cuda") * 0.1).to(weight_dtype)
    
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")
    weights = torch.softmax(scores, dim=-1)
    topk_weights, topk_ids = torch.topk(weights, k=top_k, dim=-1)
    
    w1_scale = torch.ones(num_experts, inter_dim * 2, 1, dtype=torch.float32, device="cuda")
    w2_scale = torch.ones(num_experts, hidden_dim, 1, dtype=torch.float32, device="cuda")
    
    inputs = (
        hidden.clone(),
        w1.clone(),
        w2.clone(),
        topk_weights.to(torch.float32).clone(),
        topk_ids.to(torch.int32).clone(),
        w1_scale.clone(),
        w2_scale.clone(),
    )
    
    outputs = []
    for i in range(num_runs):
        output = fused_moe(
            inputs[0].clone(),
            inputs[1].clone(),
            inputs[2].clone(),
            inputs[3].clone(),
            inputs[4].clone(),
            quant_type=QuantType.per_Token,
            w1_scale=inputs[5].clone(),
            w2_scale=inputs[6].clone(),
            activation=ActivationType.Silu,
            expert_mask=None,
        )
        outputs.append(output.clone())
        print(f"Run {i+1}/{num_runs} completed")
    
    print("\nResults:")
    reference = outputs[0]
    all_consistent = True
    
    for i in range(1, num_runs):
        if torch.equal(outputs[i], reference):
            print(f"  Run {i+1}: consistent")
        else:
            diff = (outputs[i] - reference).abs()
            max_diff = diff.max().item()
            num_diff = (outputs[i] != reference).sum().item()
            total = outputs[i].numel()
            print(f"  Run {i+1}: inconsistent (diff elements: {num_diff}/{total}, max diff: {max_diff})")
            all_consistent = False
    
    print("\n" + "="*80)
    if all_consistent:
        print("PASSED: All outputs are bit-level consistent")
    else:
        print("FAILED: Non-deterministic behavior detected")
    print("="*80)
    
    return all_consistent


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("MoE bit consistency test")
    print("="*80)
    print(f"SGLANG_USE_AITER={os.environ['SGLANG_USE_AITER']}")
    print(f"SGLANG_USE_TRITON_MOE={os.environ['SGLANG_USE_TRITON_MOE']}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    try:
        success = test_moe_bit_consistency(num_runs=10)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
