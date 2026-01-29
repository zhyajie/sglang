import torch
import sys
sys.path.insert(0, '/home/yajizhan/qwen_code/sglang/python')
sys.path.insert(0, '/home/yajizhan/qwen_code/aiter')

from aiter import rmsnorm2d_fwd as rms_norm
from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm
from aiter import rmsnorm2d_fwd_with_dynamicquant as fused_rms_norm_dy_quant
from aiter import rmsnorm2d_fwd_with_add_dynamicquant as fused_add_rms_norm_dy_quant
from aiter import dtypes, QuantType, get_torch_quant

from sglang.srt.layers.layernorm import RMSNorm


def baseline_forward(input, weight, eps, residual=None):
    layer = RMSNorm(hidden_size=input.shape[-1], eps=eps)
    layer.weight.data = weight
    layer = layer.cuda()
    with torch.no_grad():
        return layer.forward_native(input, residual)


def test_rmsnorm2d_fwd():
    print("\n=== Test rmsnorm2d_fwd ===")
    m, n = 256, 4096
    dtype = torch.bfloat16
    eps = 1e-5
    
    input = torch.randn(m, n, dtype=dtype, device='cuda')
    weight = torch.randn(n, dtype=dtype, device='cuda')
    
    baseline_out = baseline_forward(input, weight, eps)
    
    aiter_out = rms_norm(input, weight, eps)
    
    diff = (baseline_out - aiter_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Shape: ({m}, {n}), dtype: {dtype}")
    print(f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    
    atol, rtol = 1e-3, 1e-2
    assert torch.allclose(baseline_out, aiter_out, atol=atol, rtol=rtol), \
        f"Test failed: max_diff={max_diff}, mean_diff={mean_diff}"
    print("PASSED")


def test_rmsnorm2d_fwd_with_add():
    print("\n=== Test rmsnorm2d_fwd_with_add ===")
    m, n = 256, 4096
    dtype = torch.bfloat16
    eps = 1e-5
    
    input = torch.randn(m, n, dtype=dtype, device='cuda')
    weight = torch.randn(n, dtype=dtype, device='cuda')
    residual = torch.randn(m, n, dtype=dtype, device='cuda')
    
    baseline_out, baseline_residual_out = baseline_forward(input, weight, eps, residual)
    
    aiter_out = torch.empty_like(input)
    aiter_residual_out = torch.empty_like(input)
    fused_add_rms_norm(aiter_out, input, residual, aiter_residual_out, weight, eps)
    
    out_diff = (baseline_out - aiter_out).abs()
    res_diff = (baseline_residual_out - aiter_residual_out).abs()
    
    out_max_diff = out_diff.max().item()
    out_mean_diff = out_diff.mean().item()
    res_max_diff = res_diff.max().item()
    res_mean_diff = res_diff.mean().item()
    
    print(f"Shape: ({m}, {n}), dtype: {dtype}")
    print(f"Output - Max diff: {out_max_diff:.6e}, Mean diff: {out_mean_diff:.6e}")
    print(f"Residual - Max diff: {res_max_diff:.6e}, Mean diff: {res_mean_diff:.6e}")
    
    atol, rtol = 1e-3, 1e-2
    assert torch.allclose(baseline_out, aiter_out, atol=atol, rtol=rtol), \
        f"Output test failed"
    assert torch.allclose(baseline_residual_out, aiter_residual_out, atol=atol, rtol=rtol), \
        f"Residual test failed"
    print("PASSED")




def test_rmsnorm2d_fwd_with_dynamicquant():
    print("\n=== Test rmsnorm2d_fwd_with_dynamicquant ===")
    m, n = 256, 4096
    dtype = torch.bfloat16
    eps = 1e-5
    q_dtype = dtypes.fp8
    
    #input = torch.randn(m, n, dtype=dtype, device='cuda')
    #weight = torch.randn(n, dtype=dtype, device='cuda')

    input = torch.floor(torch.distributions.Uniform(-5, 5).sample((m, n))).to(
        dtype=dtype, device="cuda"
    )
    weight = torch.floor(torch.distributions.Uniform(-3, 3).sample((n,))).to(
        dtype=dtype, device="cuda"
    )

    
    # Baseline: use baseline_forward then torch quant
    torch_out = baseline_forward(input, weight, eps)
    quant_func = get_torch_quant(QuantType.per_Token)
    torch_out_q, torch_scale = quant_func(torch_out, quant_dtype=q_dtype)
    
    aiter_out_q = torch.empty(m, n, dtype=q_dtype, device='cuda')
    aiter_scale = torch.empty(m, 1, dtype=torch.float32, device='cuda')
    fused_rms_norm_dy_quant(aiter_out_q, input, aiter_scale, weight, eps)
    
    # Direct comparison of quantized values and scales
    out_q_diff = (torch_out_q.to(torch.float32) - aiter_out_q.to(torch.float32)).abs()
    scale_diff = (torch_scale - aiter_scale).abs()
    
    out_q_max_diff = out_q_diff.max().item()
    out_q_mean_diff = out_q_diff.mean().item()
    scale_max_diff = scale_diff.max().item()
    scale_mean_diff = scale_diff.mean().item()
    

    torch_out_dequant = torch_out_q.to(torch.float32) * torch_scale
    torch_out_dequant = torch_out_dequant.to(dtype)
    aiter_out_dequant = aiter_out_q.to(torch.float32) * aiter_scale
    aiter_out_dequant = aiter_out_dequant.to(dtype)
    
    atol_out_q = 1e-3
    rtol_out_q = 1e-2
    atol_scale = 1e-4
    rtol_scale = 1e-3
    
    assert torch.allclose(torch_scale, aiter_scale, atol=atol_scale, rtol=rtol_scale), \
        f"Scale test failed: max_diff={scale_max_diff}, mean_diff={scale_mean_diff}"

    assert torch.allclose(torch_out_dequant, aiter_out_dequant, 
                         atol=atol_out_q, rtol=rtol_out_q), \
        f"Quantized output test failed: max_diff={out_q_max_diff}, mean_diff={out_q_mean_diff}"

    print("PASSED")


def test_rmsnorm2d_fwd_with_add_dynamicquant():
    print("\n=== Test rmsnorm2d_fwd_with_add_dynamicquant ===")
    m, n = 256, 4096
    dtype = torch.bfloat16
    eps = 1e-5
    q_dtype = dtypes.fp8

    input = torch.floor(torch.distributions.Uniform(-2, 2).sample((m, n))).to(
        dtype=dtype, device="cuda"
    )
    weight = torch.floor(torch.distributions.Uniform(-3, 3).sample((n,))).to(
        dtype=dtype, device="cuda"
    )
    residual = torch.floor(torch.distributions.Uniform(-1, 1).sample((m,n,))).to(
        dtype=dtype, device="cuda"
    )
    #input = torch.randn(m, n, dtype=dtype, device='cuda')
    #weight = torch.randn(n, dtype=dtype, device='cuda')
    #residual = torch.randn(m, n, dtype=dtype, device='cuda')
    
    # Baseline: use baseline_forward with residual then torch quant
    torch_out, torch_residual_out = baseline_forward(input, weight, eps, residual)
    
    quant_func = get_torch_quant(QuantType.per_Token)
    torch_out_q, torch_scale = quant_func(torch_out, quant_dtype=q_dtype)
    
    aiter_out_q = torch.empty(m, n, dtype=q_dtype, device='cuda')
    aiter_residual_out = torch.empty_like(input)
    aiter_scale = torch.empty(m, 1, dtype=torch.float32, device='cuda')
    fused_add_rms_norm_dy_quant(aiter_out_q, input, residual, 
                                 aiter_residual_out, aiter_scale, weight, eps)
    
    # Direct comparison of quantized values, residual, and scales
    out_q_diff = (torch_out_q.to(torch.float32) - aiter_out_q.to(torch.float32)).abs()
    res_diff = (torch_residual_out - aiter_residual_out).abs()
    scale_diff = (torch_scale - aiter_scale).abs()
    
    out_q_max_diff = out_q_diff.max().item()
    out_q_mean_diff = out_q_diff.mean().item()
    res_max_diff = res_diff.max().item()
    res_mean_diff = res_diff.mean().item()
    scale_max_diff = scale_diff.max().item()
    scale_mean_diff = scale_diff.mean().item()
    
    print(f"Shape: ({m}, {n}), dtype: {dtype}, quant_dtype: {q_dtype}")
    print(f"Quantized output - Max diff: {out_q_max_diff:.6e}, Mean diff: {out_q_mean_diff:.6e}")
    print(f"Residual - Max diff: {res_max_diff:.6e}, Mean diff: {res_mean_diff:.6e}")
    print(f"Scale - Max diff: {scale_max_diff:.6e}, Mean diff: {scale_mean_diff:.6e}")
    
    atol_out_q = 1e-3
    rtol_out_q = 1e-2
    atol_res = 1e-3
    rtol_res = 1e-2
    atol_scale = 1e-4
    rtol_scale = 1e-3
    
    torch_out_dequant = torch_out_q.to(torch.float32) * torch_scale
    torch_out_dequant = torch_out_dequant.to(dtype)
    aiter_out_dequant = aiter_out_q.to(torch.float32) * aiter_scale
    aiter_out_dequant = aiter_out_dequant.to(dtype)
    torch_residual_dequant = torch_residual_out.to(torch.float32) * torch_scale
    torch_residual_dequant = torch_residual_dequant.to(dtype)
    aiter_residual_dequant = aiter_residual_out.to(torch.float32) * aiter_scale
    aiter_residual_dequant = aiter_residual_dequant.to(dtype)
    
    assert torch.allclose(torch_out_dequant, aiter_out_dequant, 
                         atol=atol_out_q, rtol=rtol_out_q), \
        f"Quantized output test failed: max_diff={out_q_max_diff}, mean_diff={out_q_mean_diff}"
    assert torch.allclose(torch_residual_dequant, aiter_residual_dequant, atol=atol_res, rtol=rtol_res), \
        f"Residual test failed: max_diff={res_max_diff}, mean_diff={res_mean_diff}"
    assert torch.allclose(torch_scale, aiter_scale, atol=atol_scale, rtol=rtol_scale), \
        f"Scale test failed: max_diff={scale_max_diff}, mean_diff={scale_mean_diff}"
    print("PASSED")


def test_rmsnorm2d_fwd_deterministic():
    print("\n=== Test rmsnorm2d_fwd deterministic ===")
    m, n = 256, 4096
    dtype = torch.bfloat16
    eps = 1e-5
    num_runs = 10
    
    input = torch.randn(m, n, dtype=dtype, device='cuda')
    weight = torch.randn(n, dtype=dtype, device='cuda')
    
    results = []
    for i in range(num_runs):
        output = rms_norm(input, weight, eps)
        results.append(output.clone())
    
    print(f"Shape: ({m}, {n}), dtype: {dtype}, num_runs: {num_runs}")
    
    for i in range(1, num_runs):
        diff = (results[0] - results[i]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        if max_diff > 0 or mean_diff > 0:
            print(f"Run {i+1} vs Run 1: Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
            print("FAILED - Non-deterministic behavior detected!")
            return
    
    print("All runs produce identical results")
    print("PASSED")


def test_rmsnorm2d_fwd_with_add_deterministic():
    print("\n=== Test rmsnorm2d_fwd_with_add deterministic ===")
    m, n = 256, 4096
    dtype = torch.bfloat16
    eps = 1e-5
    num_runs = 10
    
    input = torch.randn(m, n, dtype=dtype, device='cuda')
    weight = torch.randn(n, dtype=dtype, device='cuda')
    residual = torch.randn(m, n, dtype=dtype, device='cuda')
    
    outputs = []
    residual_outs = []
    
    for i in range(num_runs):
        output = torch.empty_like(input)
        residual_out = torch.empty_like(input)
        fused_add_rms_norm(output, input, residual, residual_out, weight, eps)
        outputs.append(output.clone())
        residual_outs.append(residual_out.clone())
    
    print(f"Shape: ({m}, {n}), dtype: {dtype}, num_runs: {num_runs}")
    
    failed = False
    for i in range(1, num_runs):
        out_diff = (outputs[0] - outputs[i]).abs()
        res_diff = (residual_outs[0] - residual_outs[i]).abs()
        
        out_max_diff = out_diff.max().item()
        out_mean_diff = out_diff.mean().item()
        res_max_diff = res_diff.max().item()
        res_mean_diff = res_diff.mean().item()
        
        if out_max_diff > 0 or out_mean_diff > 0:
            print(f"Run {i+1} vs Run 1 (output): Max diff: {out_max_diff:.6e}, Mean diff: {out_mean_diff:.6e}")
            failed = True
        
        if res_max_diff > 0 or res_mean_diff > 0:
            print(f"Run {i+1} vs Run 1 (residual): Max diff: {res_max_diff:.6e}, Mean diff: {res_mean_diff:.6e}")
            failed = True
    
    if failed:
        print("FAILED - Non-deterministic behavior detected!")
        return
    
    print("All runs produce identical results")
    print("PASSED")


def test_rmsnorm2d_fwd_with_dynamicquant_deterministic():
    print("\n=== Test rmsnorm2d_fwd_with_dynamicquant deterministic ===")
    m, n = 256, 4096
    dtype = torch.bfloat16
    eps = 1e-5
    q_dtype = dtypes.fp8
    num_runs = 10
    
    input = torch.randn(m, n, dtype=dtype, device='cuda')
    weight = torch.randn(n, dtype=dtype, device='cuda')
    
    outputs = []
    scales = []
    
    for i in range(num_runs):
        output = torch.empty(m, n, dtype=q_dtype, device='cuda')
        scale = torch.empty(m, 1, dtype=torch.float32, device='cuda')
        fused_rms_norm_dy_quant(output, input, scale, weight, eps)
        outputs.append(output.clone())
        scales.append(scale.clone())
    
    print(f"Shape: ({m}, {n}), dtype: {dtype}, quant_dtype: {q_dtype}, num_runs: {num_runs}")
    
    failed = False
    for i in range(1, num_runs):
        out_diff = (outputs[0].to(torch.float32) - outputs[i].to(torch.float32)).abs()
        scale_diff = (scales[0] - scales[i]).abs()
        
        out_max_diff = out_diff.max().item()
        out_mean_diff = out_diff.mean().item()
        scale_max_diff = scale_diff.max().item()
        scale_mean_diff = scale_diff.mean().item()
        
        if out_max_diff > 0 or out_mean_diff > 0:
            print(f"Run {i+1} vs Run 1 (output): Max diff: {out_max_diff:.6e}, Mean diff: {out_mean_diff:.6e}")
            failed = True
        
        if scale_max_diff > 0 or scale_mean_diff > 0:
            print(f"Run {i+1} vs Run 1 (scale): Max diff: {scale_max_diff:.6e}, Mean diff: {scale_mean_diff:.6e}")
            failed = True
    
    if failed:
        print("FAILED - Non-deterministic behavior detected!")
        return
    
    print("All runs produce identical results")
    print("PASSED")


def test_rmsnorm2d_fwd_with_add_dynamicquant_deterministic():
    print("\n=== Test rmsnorm2d_fwd_with_add_dynamicquant deterministic ===")
    m, n = 256, 4096
    dtype = torch.bfloat16
    eps = 1e-5
    q_dtype = dtypes.fp8
    num_runs = 10
    
    input = torch.randn(m, n, dtype=dtype, device='cuda')
    weight = torch.randn(n, dtype=dtype, device='cuda')
    residual = torch.randn(m, n, dtype=dtype, device='cuda')
    
    outputs = []
    residual_outs = []
    scales = []
    
    for i in range(num_runs):
        output = torch.empty(m, n, dtype=q_dtype, device='cuda')
        residual_out = torch.empty_like(input)
        scale = torch.empty(m, 1, dtype=torch.float32, device='cuda')
        fused_add_rms_norm_dy_quant(output, input, residual, residual_out, scale, weight, eps)
        outputs.append(output.clone())
        residual_outs.append(residual_out.clone())
        scales.append(scale.clone())
    
    print(f"Shape: ({m}, {n}), dtype: {dtype}, quant_dtype: {q_dtype}, num_runs: {num_runs}")
    
    failed = False
    for i in range(1, num_runs):
        out_diff = (outputs[0].to(torch.float32) - outputs[i].to(torch.float32)).abs()
        res_diff = (residual_outs[0] - residual_outs[i]).abs()
        scale_diff = (scales[0] - scales[i]).abs()
        
        out_max_diff = out_diff.max().item()
        out_mean_diff = out_diff.mean().item()
        res_max_diff = res_diff.max().item()
        res_mean_diff = res_diff.mean().item()
        scale_max_diff = scale_diff.max().item()
        scale_mean_diff = scale_diff.mean().item()
        
        if out_max_diff > 0 or out_mean_diff > 0:
            print(f"Run {i+1} vs Run 1 (output): Max diff: {out_max_diff:.6e}, Mean diff: {out_mean_diff:.6e}")
            failed = True
        
        if res_max_diff > 0 or res_mean_diff > 0:
            print(f"Run {i+1} vs Run 1 (residual): Max diff: {res_max_diff:.6e}, Mean diff: {res_mean_diff:.6e}")
            failed = True
        
        if scale_max_diff > 0 or scale_mean_diff > 0:
            print(f"Run {i+1} vs Run 1 (scale): Max diff: {scale_max_diff:.6e}, Mean diff: {scale_mean_diff:.6e}")
            failed = True
    
    if failed:
        print("FAILED - Non-deterministic behavior detected!")
        return
    
    print("All runs produce identical results")
    print("PASSED")


if __name__ == '__main__':
    test_rmsnorm2d_fwd()
    test_rmsnorm2d_fwd_with_add()
    test_rmsnorm2d_fwd_with_dynamicquant()
    test_rmsnorm2d_fwd_with_add_dynamicquant()
    test_rmsnorm2d_fwd_deterministic()
    test_rmsnorm2d_fwd_with_add_deterministic()
    test_rmsnorm2d_fwd_with_dynamicquant_deterministic()
    test_rmsnorm2d_fwd_with_add_dynamicquant_deterministic()
    print("\n=== All tests passed! ===")
