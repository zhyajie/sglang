"""
Performance benchmark: STA Triton vs FA4 vs FA3 vs SDPA on NVIDIA B200.

FLOPs formula: 4 * B * H * N^2 * d (Q@K^T + P@V matmuls)
For sparse STA: effective FLOPs = dense FLOPs * (1 - sparsity)

Usage:
    python tests/bench_sta_perf.py [--warmup 5] [--repeat 20] [--output results.json]
"""

import argparse
import gc
import json
import platform
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, "/workspace/sglang/tests")
from st_attn_triton import sliding_tile_attention_triton

# --- FA4 ---
try:
    from sglang.jit_kernel.flash_attention_v4 import (
        flash_attn_varlen_func as flash_attn_varlen_func_fa4,
    )
    FA4_AVAILABLE = True
except ImportError:
    FA4_AVAILABLE = False

# --- FA3 (sgl-kernel) ---
FA3_AVAILABLE = False
FA3_BLOCKED_REASON = ""
_fa3_varlen = None
try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func as _fa3_varlen
    from sgl_kernel.flash_attn import is_fa3_supported
    try:
        if is_fa3_supported():
            FA3_AVAILABLE = True
        else:
            FA3_BLOCKED_REASON = "is_fa3_supported() returns False (SM check blocks SM 10.0)"
    except Exception:
        FA3_BLOCKED_REASON = "is_fa3_supported() raised (CUDA not initialized yet)"
except ImportError as e:
    FA3_BLOCKED_REASON = f"ImportError: {e}"

if not torch.cuda.is_available():
    print("ERROR: No CUDA GPUs available. If in a container, try restarting it.")
    sys.exit(1)

BASE_TILE_SIZE = (6, 8, 8)

SHAPE_CONFIGS: dict[str, dict[str, Any]] = {
    "30x48x80": {
        "full_window": (5, 6, 10),
        "img_seq_len": 30 * 48 * 80,
        "has_text": True,
        "text_length": 256,
        "description": "HunyuanVideo 5s 720P",
    },
    "36x48x48": {
        "full_window": (6, 6, 6),
        "img_seq_len": 36 * 48 * 48,
        "has_text": False,
        "text_length": 0,
        "description": "StepVideo",
    },
    "18x48x80": {
        "full_window": (3, 6, 10),
        "img_seq_len": 18 * 48 * 80,
        "has_text": False,
        "text_length": 0,
        "description": "Wan 5s 480P",
    },
}


def get_device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
        info["sm_count"] = props.multi_processor_count
        cap = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{cap[0]}.{cap[1]}"
    return info


def flush_gpu():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def tile_tokens(x, full_window):
    n_t, n_h, n_w = full_window
    ts_t, ts_h, ts_w = BASE_TILE_SIZE
    return rearrange(
        x,
        "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
        n_t=n_t, n_h=n_h, n_w=n_w, ts_t=ts_t, ts_h=ts_h, ts_w=ts_w,
    )


def prepare_tiled_qkv(shape_str, batch, num_heads, head_dim, dtype=torch.bfloat16):
    cfg = SHAPE_CONFIGS[shape_str]
    fw = cfg["full_window"]
    img_seq = cfg["img_seq_len"]
    text_len = cfg["text_length"]
    total = img_seq + text_len

    q = torch.randn(batch, total, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch, total, num_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch, total, num_heads, head_dim, dtype=dtype, device="cuda")

    if text_len > 0:
        q_t = torch.cat([q[:, :text_len], tile_tokens(q[:, text_len:], fw)], dim=1)
        k_t = torch.cat([k[:, :text_len], tile_tokens(k[:, text_len:], fw)], dim=1)
        v_t = torch.cat([v[:, :text_len], tile_tokens(v[:, text_len:], fw)], dim=1)
    else:
        q_t = tile_tokens(q, fw)
        k_t = tile_tokens(k, fw)
        v_t = tile_tokens(v, fw)

    return (q_t.transpose(1, 2).contiguous(),
            k_t.transpose(1, 2).contiguous(),
            v_t.transpose(1, 2).contiguous(),
            text_len, cfg["has_text"], fw)


def sdpa_fn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def fa4_fn(q_bhsd, k_bhsd, v_bhsd):
    """FA4 expects [B, S, H, D]."""
    q = q_bhsd.transpose(1, 2).contiguous()
    k = k_bhsd.transpose(1, 2).contiguous()
    v = v_bhsd.transpose(1, 2).contiguous()
    out = flash_attn_varlen_func_fa4(q, k, v, max_seqlen_q=q.shape[1], max_seqlen_k=k.shape[1])
    return out.transpose(1, 2).contiguous()


def fa3_fn(q_bhsd, k_bhsd, v_bhsd):
    """FA3 (sgl-kernel) expects [total_tokens, H, D] with cu_seqlens."""
    b, h, s, d = q_bhsd.shape
    q = q_bhsd.transpose(1, 2).contiguous().view(b * s, h, d)
    k = k_bhsd.transpose(1, 2).contiguous().view(b * s, h, d)
    v = v_bhsd.transpose(1, 2).contiguous().view(b * s, h, d)
    cu = torch.arange(0, (b + 1) * s, step=s, device=q.device, dtype=torch.int32)
    out = _fa3_varlen(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                      max_seqlen_q=s, max_seqlen_k=s, ver=3)
    return out.view(b, s, h, d).transpose(1, 2).contiguous()


def fa3_bypass_fn(q_bhsd, k_bhsd, v_bhsd):
    """FA3 bypassing is_fa3_supported() check — directly call the CUDA op."""
    b, h, s, d = q_bhsd.shape
    q = q_bhsd.transpose(1, 2).contiguous().view(b * s, h, d)
    k = k_bhsd.transpose(1, 2).contiguous().view(b * s, h, d)
    v = v_bhsd.transpose(1, 2).contiguous().view(b * s, h, d)
    cu = torch.arange(0, (b + 1) * s, step=s, device=q.device, dtype=torch.int32)
    softmax_scale = d ** (-0.5)
    out, softmax_lse, *rest = torch.ops.sgl_kernel.fwd.default(
        q, k, v,
        None, None, None, None,  # k_new, v_new, qv, out
        cu, cu,                  # cu_seqlens_q, cu_seqlens_k
        None, None, None,        # cu_seqlens_k_new, seqused_q, seqused_k
        s, s,                    # max_seqlen_q, max_seqlen_k
        None, None, None,        # page_table, kv_batch_idx, leftpad_k
        None, None, None,        # rotary_cos, rotary_sin, seqlens_rotary
        None, None, None,        # q_descale, k_descale, v_descale
        softmax_scale,
        False,                   # causal
        -1, -1,                  # window_size
        0,                       # attention_chunk
        0.0,                     # softcap
        False,                   # is_rotary_interleaved
        None,                    # scheduler_metadata
        1,                       # num_splits
        None,                    # pack_gqa
        0,                       # sm_margin
        None,                    # sinks
    )
    return out.view(b, s, h, d).transpose(1, 2).contiguous()


def compute_metrics(output, reference):
    out_f = output.float().flatten()
    ref_f = reference.float().flatten()
    l2_rel = (torch.norm(out_f - ref_f) / torch.norm(ref_f)).item()
    max_abs = (out_f - ref_f).abs().max().item()
    cosine = F.cosine_similarity(out_f.unsqueeze(0), ref_f.unsqueeze(0)).item()
    return {"l2_rel_error": l2_rel, "max_abs_error": max_abs, "cosine_sim": cosine}


def measure_latency(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


def measure_peak_mem(fn):
    flush_gpu()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = fn()
    mem = torch.cuda.max_memory_allocated() / (1024**2)
    return out, mem


def clamp_window(window, full_window):
    return tuple(min(s, f) for s, f in zip(window, full_window))


def compute_dense_flops(batch, heads, seq_len, head_dim):
    """FLOPs = 4 * B * H * N^2 * d"""
    return 4 * batch * heads * seq_len * seq_len * head_dim


def test_fa3_on_b200():
    """Test whether FA3 kernel actually works on B200 by bypassing the SM check."""
    print("\n" + "=" * 80)
    print("FA3 COMPATIBILITY TEST ON B200 (SM 10.0)")
    print("=" * 80)
    print(f"  FA3 official check (is_fa3_supported): {'PASS' if FA3_AVAILABLE else 'BLOCKED'}")
    if FA3_BLOCKED_REASON:
        print(f"  Reason: {FA3_BLOCKED_REASON}")

    # Test with a small shape first
    b, s, h, d = 1, 1024, 4, 128
    q = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if FA3_AVAILABLE:
        print("  FA3 is officially supported — testing...")
        try:
            out = fa3_fn(q, k, v)
            sdpa_out = sdpa_fn(q, k, v)
            m = compute_metrics(out, sdpa_out)
            ok = torch.isfinite(out).all().item() and m["cosine_sim"] > 0.99
            print(f"  FA3 result: finite={torch.isfinite(out).all().item()}, "
                  f"cos={m['cosine_sim']:.6f}, L2={m['l2_rel_error']:.6f} => {'PASS' if ok else 'FAIL'}")
            return True
        except Exception as e:
            print(f"  FA3 failed: {e}")
            return False

    # Bypass the check and try the kernel directly
    print("  Attempting FA3 kernel bypass (calling torch.ops.sgl_kernel.fwd directly)...")
    try:
        out = fa3_bypass_fn(q, k, v)
        sdpa_out = sdpa_fn(q, k, v)
        m = compute_metrics(out, sdpa_out)
        ok = torch.isfinite(out).all().item() and m["cosine_sim"] > 0.99
        print(f"  FA3 bypass result: finite={torch.isfinite(out).all().item()}, "
              f"cos={m['cosine_sim']:.6f}, L2={m['l2_rel_error']:.6f} => {'PASS' if ok else 'FAIL'}")
        if ok:
            print("  FA3 kernel WORKS on B200 despite is_fa3_supported() returning False!")
            print("  The SM check in sgl-kernel should be updated to include SM 10.x.")
        return ok
    except Exception as e:
        print(f"  FA3 bypass FAILED: {e}")
        print("  FA3 kernel does NOT work on B200 — compiled only for SM 8.x/9.x.")
        return False


def run_benchmark(warmup=5, repeat=20):
    device_info = get_device_info()
    print(f"\nDevice: {device_info.get('gpu_name', 'Unknown')}")
    print(f"Torch: {device_info['torch']}, CC: {device_info.get('compute_capability', 'N/A')}")
    print(f"GPU Memory: {device_info.get('gpu_memory_gb', 'N/A')} GB")
    print(f"FA4: {'Available' if FA4_AVAILABLE else 'Not available'}")
    print(f"FA3: {'Available' if FA3_AVAILABLE else 'Blocked'}")

    # Test FA3 compatibility
    fa3_works = test_fa3_on_b200()

    results = []
    num_heads, head_dim = 24, 128

    for shape_str, cfg in SHAPE_CONFIGS.items():
        fw = cfg["full_window"]
        total_seq = cfg["img_seq_len"] + cfg["text_length"]
        full_tiles = fw[0] * fw[1] * fw[2]
        dense_flops = compute_dense_flops(1, num_heads, total_seq, head_dim)
        dense_tflop = dense_flops / 1e12

        print(f"\n{'=' * 110}")
        print(f"  {shape_str} ({cfg['description']})  seq_len={total_seq}  dense={dense_tflop:.2f} TFLOP")
        print(f"{'=' * 110}")
        print(f"  {'Method':<35} {'Latency':>10} {'Speedup':>8} {'Memory':>10} {'TFLOPS':>8} {'L2 Err':>10} {'CosSim':>10} {'Sparsity':>9}")
        print(f"  {'─' * 105}")

        q, k, v, text_len, has_text, _ = prepare_tiled_qkv(shape_str, 1, num_heads, head_dim)
        base = {"shape": shape_str, "description": cfg["description"],
                "seq_len": total_seq, "num_heads": num_heads, "head_dim": head_dim}

        # --- SDPA ---
        flush_gpu()
        sdpa_lat = measure_latency(lambda: sdpa_fn(q, k, v), warmup, repeat)
        sdpa_out, sdpa_mem = measure_peak_mem(lambda: sdpa_fn(q, k, v))
        sdpa_tflops = dense_tflop / (sdpa_lat / 1000)
        results.append({**base, "backend": "SDPA", "window": "full",
                        "latency_ms": round(sdpa_lat, 2), "peak_memory_mb": round(sdpa_mem, 1),
                        "speedup_vs_sdpa": 1.0, "tflops": round(sdpa_tflops, 1),
                        "dense_tflop": round(dense_tflop, 2)})
        print(f"  {'SDPA':<35} {sdpa_lat:>9.2f}ms {'1.00x':>8} {sdpa_mem:>9.0f}MB {sdpa_tflops:>7.0f}")

        # --- FA4 ---
        if FA4_AVAILABLE:
            flush_gpu()
            fa4_lat = measure_latency(lambda: fa4_fn(q, k, v), warmup, repeat)
            fa4_out, fa4_mem = measure_peak_mem(lambda: fa4_fn(q, k, v))
            fa4_m = compute_metrics(fa4_out, sdpa_out)
            fa4_sp = sdpa_lat / fa4_lat
            fa4_tflops = dense_tflop / (fa4_lat / 1000)
            results.append({**base, "backend": "FA4", "window": "full",
                            "latency_ms": round(fa4_lat, 2), "peak_memory_mb": round(fa4_mem, 1),
                            "speedup_vs_sdpa": round(fa4_sp, 2), "tflops": round(fa4_tflops, 1),
                            "l2_rel_error": round(fa4_m["l2_rel_error"], 6),
                            "cosine_sim": round(fa4_m["cosine_sim"], 6),
                            "max_abs_error": round(fa4_m["max_abs_error"], 6),
                            "dense_tflop": round(dense_tflop, 2)})
            print(f"  {'FA4 (FlashAttention v4)':<35} {fa4_lat:>9.2f}ms {fa4_sp:>7.2f}x {fa4_mem:>9.0f}MB {fa4_tflops:>7.0f} "
                  f"{fa4_m['l2_rel_error']:>10.6f} {fa4_m['cosine_sim']:>10.6f}")

        # --- FA3 (if works) ---
        if fa3_works:
            flush_gpu()
            fa3_call = fa3_fn if FA3_AVAILABLE else fa3_bypass_fn
            try:
                fa3_lat = measure_latency(lambda: fa3_call(q, k, v), warmup, repeat)
                fa3_out, fa3_mem = measure_peak_mem(lambda: fa3_call(q, k, v))
                fa3_m = compute_metrics(fa3_out, sdpa_out)
                fa3_sp = sdpa_lat / fa3_lat
                fa3_tflops = dense_tflop / (fa3_lat / 1000)
                results.append({**base, "backend": "FA3", "window": "full",
                                "latency_ms": round(fa3_lat, 2), "peak_memory_mb": round(fa3_mem, 1),
                                "speedup_vs_sdpa": round(fa3_sp, 2), "tflops": round(fa3_tflops, 1),
                                "l2_rel_error": round(fa3_m["l2_rel_error"], 6),
                                "cosine_sim": round(fa3_m["cosine_sim"], 6),
                                "max_abs_error": round(fa3_m["max_abs_error"], 6),
                                "dense_tflop": round(dense_tflop, 2)})
                label = "FA3 (bypass)" if not FA3_AVAILABLE else "FA3 (sgl-kernel)"
                print(f"  {label:<35} {fa3_lat:>9.2f}ms {fa3_sp:>7.2f}x {fa3_mem:>9.0f}MB {fa3_tflops:>7.0f} "
                      f"{fa3_m['l2_rel_error']:>10.6f} {fa3_m['cosine_sim']:>10.6f}")
            except Exception as e:
                print(f"  {'FA3':<35} ERROR on this shape: {e}")

        # --- STA Triton configs ---
        window_configs = [
            ("STA full " + str(fw), fw),
            ("STA sparse (3,3,3)", clamp_window((3, 3, 3), fw)),
            ("STA sparse (1,3,10)", clamp_window((1, 3, 10), fw)),
            ("STA sparse (3,1,10)", clamp_window((3, 1, 10), fw)),
            ("STA sparse (1,5,7)", clamp_window((1, 5, 7), fw)),
            ("STA sparse (3,6,1)", clamp_window((3, 6, 1), fw)),
        ]

        for name, window in window_configs:
            flush_gpu()
            try:
                lat = measure_latency(
                    lambda w=window: sliding_tile_attention_triton(
                        q, k, v, [w] * num_heads, text_len, has_text, shape_str),
                    warmup, repeat)
                out, mem = measure_peak_mem(
                    lambda w=window: sliding_tile_attention_triton(
                        q, k, v, [w] * num_heads, text_len, has_text, shape_str))
                m = compute_metrics(out, sdpa_out)
                sparse_tiles = window[0] * window[1] * window[2]
                sparsity = 1.0 - sparse_tiles / full_tiles
                speedup = sdpa_lat / lat
                eff_tflop = dense_tflop * (1 - sparsity)
                tflops = eff_tflop / (lat / 1000)

                results.append({
                    **base, "backend": f"STA_triton_{name}", "window": str(window),
                    "latency_ms": round(lat, 2), "peak_memory_mb": round(mem, 1),
                    "speedup_vs_sdpa": round(speedup, 2), "tflops": round(tflops, 1),
                    "l2_rel_error": round(m["l2_rel_error"], 6),
                    "cosine_sim": round(m["cosine_sim"], 6),
                    "max_abs_error": round(m["max_abs_error"], 6),
                    "sparsity": round(sparsity, 4),
                    "eff_tflop": round(eff_tflop, 2),
                    "dense_tflop": round(dense_tflop, 2)})
                sp_str = f"{sparsity*100:.0f}%" if sparsity > 0 else "0%"
                print(f"  {name:<35} {lat:>9.2f}ms {speedup:>7.2f}x {mem:>9.0f}MB {tflops:>7.0f} "
                      f"{m['l2_rel_error']:>10.6f} {m['cosine_sim']:>10.6f} {sp_str:>9}")
            except Exception as e:
                print(f"  {name:<35} ERROR: {e}")

        del q, k, v, sdpa_out
        flush_gpu()

    return results, fa3_works


def print_summary(results):
    print(f"\n{'=' * 140}")
    print("SUMMARY")
    print(f"{'=' * 140}")
    print(f"{'Backend':<30} {'Shape':<12} {'SeqLen':>8} {'Latency':>10} {'Speedup':>8} "
          f"{'Memory':>10} {'TFLOPS':>8} {'L2 Err':>10} {'CosSim':>10} {'Sparsity':>9}")
    print(f"{'─' * 140}")
    for r in results:
        l2 = f"{r['l2_rel_error']:.6f}" if 'l2_rel_error' in r else "—"
        cos = f"{r['cosine_sim']:.6f}" if 'cosine_sim' in r else "—"
        sp = f"{r['sparsity']*100:.0f}%" if 'sparsity' in r else "—"
        tfl = f"{r.get('tflops', 0):.0f}" if 'tflops' in r else "—"
        print(f"{r['backend']:<30} {r['shape']:<12} {r['seq_len']:>8} "
              f"{r['latency_ms']:>9.2f}ms {r.get('speedup_vs_sdpa', 1.0):>7.2f}x "
              f"{r['peak_memory_mb']:>9.0f}MB {tfl:>8} {l2:>10} {cos:>10} {sp:>9}")
    print(f"{'=' * 140}")


def main():
    parser = argparse.ArgumentParser(description="STA vs FA4 vs FA3 vs SDPA Benchmark")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device_info = get_device_info()
    results, fa3_works = run_benchmark(warmup=args.warmup, repeat=args.repeat)
    print_summary(results)

    output_path = args.output or f"sta_benchmark_{device_info.get('gpu_name', 'unknown').replace(' ', '_')}.json"
    report = {
        "device": device_info,
        "benchmark_config": {"warmup": args.warmup, "repeat": args.repeat},
        "fa3_works_on_b200": fa3_works,
        "fa4_available": FA4_AVAILABLE,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
