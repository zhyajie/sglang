"""
Benchmark script for SGLang Diffusion Attention Backends on NVIDIA B200.

This script benchmarks the following attention backends:
  1. FlashAttention (FA) — baseline
  2. SageAttention v2 (SageAttn)
  3. Sliding Tile Attention (STA) — requires st_attn + mask config
  4. Video Sparse Attention (VSA) — requires vsa
  5. Sparse Linear Attention (SLA) — Triton JIT
  6. Sage Sparse Linear Attention (SageSLA) — requires spas_sage_attn
  7. PyTorch SDPA — universal fallback

Usage:
    # Run all available backends
    CUDA_VISIBLE_DEVICES=0 python benchmark_attention_backends.py

    # Run specific backends
    CUDA_VISIBLE_DEVICES=0 python benchmark_attention_backends.py --backends fa sage_attn sla

    # Custom sequence lengths
    CUDA_VISIBLE_DEVICES=0 python benchmark_attention_backends.py --seq-lens 4096 16384

    # STA requires mask config
    SGLANG_DIFFUSION_ATTENTION_CONFIG=/path/to/mask_strategy.json \
    CUDA_VISIBLE_DEVICES=0 python benchmark_attention_backends.py --backends sliding_tile_attn
"""

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────
@dataclass
class BenchmarkResult:
    backend: str
    seq_len: int
    num_heads: int
    head_dim: int
    batch_size: int
    latency_ms: float
    peak_memory_mb: float
    l2_error: float | None = None
    cosine_sim: float | None = None


def get_device_info() -> dict[str, Any]:
    """Get GPU device information."""
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    return {
        "name": props.name,
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "total_memory_gb": props.total_mem / (1024**3),
        "sm_count": props.multi_processor_count,
    }


def flush_gpu():
    """Flush GPU memory and caches."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def generate_qkv(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random Q, K, V tensors."""
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=dtype)
    return q, k, v


def measure_latency(
    fn,
    warmup: int = 5,
    repeat: int = 20,
) -> float:
    """Measure average latency in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Measure
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat * 1000
    return elapsed


def compute_accuracy(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    """Compute L2 error and cosine similarity vs reference."""
    output_flat = output.float().flatten()
    reference_flat = reference.float().flatten()

    l2_error = torch.norm(output_flat - reference_flat).item() / torch.norm(reference_flat).item()
    cosine_sim = F.cosine_similarity(output_flat.unsqueeze(0), reference_flat.unsqueeze(0)).item()
    return l2_error, cosine_sim


# ─────────────────────────────────────────────────────────────────
# Backend implementations
# ─────────────────────────────────────────────────────────────────

def bench_sdpa(q, k, v) -> torch.Tensor:
    """PyTorch SDPA baseline."""
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=False)
    return out.transpose(1, 2)


def bench_flash_attn(q, k, v) -> torch.Tensor | None:
    """FlashAttention via sgl_kernel."""
    try:
        from sgl_kernel.flash_attn import flash_attn_varlen_func
    except ImportError:
        print("  [SKIP] sgl_kernel.flash_attn not available")
        return None

    B, L, H, D = q.shape
    out = flash_attn_varlen_func(
        q=q, k=k, v=v,
        cu_seqlens_q=None, cu_seqlens_k=None,
        max_seqlen_q=L, max_seqlen_k=L,
        softmax_scale=D**-0.5,
        causal=False,
        return_softmax_lse=False,
        ver=3,
    )
    return out


def bench_sage_attn(q, k, v) -> torch.Tensor | None:
    """SageAttention v2."""
    try:
        from sageattention import sageattn
    except ImportError:
        print("  [SKIP] sageattention not installed")
        return None

    out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
    return out


def bench_sage_attn3(q, k, v) -> torch.Tensor | None:
    """SageAttention 3 (Blackwell only)."""
    try:
        from sageattn3 import sageattn3_blackwell
    except ImportError:
        print("  [SKIP] sageattn3 not installed")
        return None

    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    if kt.shape[1] != qt.shape[1]:
        print("  [SKIP] SageAttn3 does not support GQA")
        return None
    out = sageattn3_blackwell(qt, kt, vt, is_causal=False)
    return out.transpose(1, 2)


def bench_sla(q, k, v, topk_ratio: float = 0.1) -> torch.Tensor | None:
    """Sparse Linear Attention (Triton JIT)."""
    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn import (
            SparseLinearAttentionImpl,
        )
    except ImportError:
        print("  [SKIP] SLA module not importable")
        return None

    B, L, H, D = q.shape
    impl = SparseLinearAttentionImpl(
        num_heads=H, head_size=D, topk_ratio=topk_ratio
    ).to("cuda")

    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()

    out = impl.forward(qt, kt, vt)
    return out


def bench_sage_sla(q, k, v, topk_ratio: float = 0.5) -> torch.Tensor | None:
    """Sage Sparse Linear Attention."""
    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn import (
            SAGESLA_ENABLED,
            SageSparseLinearAttentionImpl,
        )
        if not SAGESLA_ENABLED:
            print("  [SKIP] spas_sage_attn not installed")
            return None
    except ImportError:
        print("  [SKIP] SageSLA module not importable")
        return None

    B, L, H, D = q.shape
    impl = SageSparseLinearAttentionImpl(
        num_heads=H, head_size=D, topk_ratio=topk_ratio
    ).to("cuda")

    out = impl.forward(q, k, v, attn_metadata=None)
    return out


def bench_vsa(q, k, v, sparsity: float = 0.5) -> torch.Tensor | None:
    """Video Sparse Attention."""
    try:
        from vsa import video_sparse_attn
    except ImportError:
        print("  [SKIP] vsa not installed")
        return None

    B, L, H, D = q.shape

    # VSA requires gate_compress and specific 3D shapes
    # We create a synthetic setup
    gate_compress = torch.randn_like(q)

    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()
    gt = gate_compress.transpose(1, 2).contiguous()

    # Determine 3D shape from sequence length
    # Find T, H, W such that T*H*W = L and all divisible by 4
    T_dim = 4
    remaining = L // T_dim
    H_dim = 4
    W_dim = remaining // H_dim
    if T_dim * H_dim * W_dim != L:
        print(f"  [SKIP] VSA: seq_len={L} not compatible with tile (4,4,4)")
        return None

    tile_size = (4, 4, 4)
    num_tiles = (
        math.ceil(T_dim / tile_size[0]),
        math.ceil(H_dim / tile_size[1]),
        math.ceil(W_dim / tile_size[2]),
    )
    total_tiles = num_tiles[0] * num_tiles[1] * num_tiles[2]
    cur_topk = max(1, math.ceil((1 - sparsity) * total_tiles))

    variable_block_sizes = torch.full((total_tiles,), math.prod(tile_size), dtype=torch.long, device="cuda")

    out = video_sparse_attn(
        qt, kt, vt,
        variable_block_sizes=variable_block_sizes,
        topk=cur_topk,
        block_size=tile_size,
        compress_attn_weight=gt,
    ).transpose(1, 2)
    return out


# ─────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────

BACKEND_REGISTRY = {
    "sdpa": ("PyTorch SDPA", bench_sdpa),
    "fa": ("FlashAttention", bench_flash_attn),
    "sage_attn": ("SageAttention v2", bench_sage_attn),
    "sage_attn_3": ("SageAttention 3", bench_sage_attn3),
    "sla": ("Sparse Linear Attn", bench_sla),
    "sage_sla": ("Sage Sparse Linear Attn", bench_sage_sla),
    "vsa": ("Video Sparse Attn", bench_vsa),
}


def run_benchmark(
    backends: list[str],
    seq_lens: list[int],
    num_heads: int = 24,
    head_dim: int = 128,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> list[BenchmarkResult]:
    """Run the benchmark suite."""

    results: list[BenchmarkResult] = []
    device_info = get_device_info()
    print(f"\nDevice: {device_info['name']} (SM {device_info['compute_capability']})")
    print(f"Memory: {device_info['total_memory_gb']:.1f} GB")
    print(f"Config: batch={batch_size}, heads={num_heads}, head_dim={head_dim}, dtype={dtype}")
    print("=" * 90)

    for seq_len in seq_lens:
        print(f"\n--- Sequence Length: {seq_len:,} ---")

        flush_gpu()
        q, k, v = generate_qkv(batch_size, seq_len, num_heads, head_dim, dtype)

        # Compute reference output with SDPA
        with torch.no_grad():
            reference = bench_sdpa(q, k, v)

        for backend_key in backends:
            if backend_key not in BACKEND_REGISTRY:
                print(f"  Unknown backend: {backend_key}")
                continue

            name, fn = BACKEND_REGISTRY[backend_key]
            flush_gpu()
            torch.cuda.reset_peak_memory_stats()

            print(f"  {name:30s} ... ", end="", flush=True)

            try:
                # First run to check if backend works
                with torch.no_grad():
                    output = fn(q, k, v)

                if output is None:
                    continue

                # Measure accuracy
                l2_error, cosine_sim = compute_accuracy(output, reference)

                # Measure latency
                latency = measure_latency(lambda: fn(q, k, v))

                # Measure memory
                flush_gpu()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = fn(q, k, v)
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

                result = BenchmarkResult(
                    backend=name,
                    seq_len=seq_len,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    batch_size=batch_size,
                    latency_ms=latency,
                    peak_memory_mb=peak_mem,
                    l2_error=l2_error,
                    cosine_sim=cosine_sim,
                )
                results.append(result)

                print(
                    f"latency={latency:8.2f}ms  "
                    f"mem={peak_mem:8.1f}MB  "
                    f"L2_err={l2_error:.6f}  "
                    f"cos_sim={cosine_sim:.6f}"
                )

            except Exception as e:
                print(f"ERROR: {e}")
                continue

        # Cleanup
        del q, k, v, reference
        flush_gpu()

    return results


def print_summary_table(results: list[BenchmarkResult]):
    """Print a summary table of results."""
    if not results:
        return

    print("\n" + "=" * 110)
    print("SUMMARY TABLE")
    print("=" * 110)
    header = f"{'Backend':30s} {'SeqLen':>8s} {'Latency(ms)':>12s} {'Memory(MB)':>12s} {'L2 Error':>12s} {'Cosine Sim':>12s} {'Speedup':>8s}"
    print(header)
    print("-" * 110)

    # Find SDPA baseline latencies for speedup calculation
    sdpa_latencies: dict[int, float] = {}
    for r in results:
        if r.backend == "PyTorch SDPA":
            sdpa_latencies[r.seq_len] = r.latency_ms

    for r in results:
        baseline = sdpa_latencies.get(r.seq_len, r.latency_ms)
        speedup = baseline / r.latency_ms if r.latency_ms > 0 else 0
        l2_str = f"{r.l2_error:.6f}" if r.l2_error is not None else "N/A"
        cos_str = f"{r.cosine_sim:.6f}" if r.cosine_sim is not None else "N/A"
        print(
            f"{r.backend:30s} {r.seq_len:8d} {r.latency_ms:12.2f} {r.peak_memory_mb:12.1f} "
            f"{l2_str:>12s} {cos_str:>12s} {speedup:7.2f}x"
        )

    print("=" * 110)


def save_results(results: list[BenchmarkResult], output_path: str):
    """Save results to JSON."""
    data = {
        "device": get_device_info(),
        "results": [
            {
                "backend": r.backend,
                "seq_len": r.seq_len,
                "num_heads": r.num_heads,
                "head_dim": r.head_dim,
                "batch_size": r.batch_size,
                "latency_ms": r.latency_ms,
                "peak_memory_mb": r.peak_memory_mb,
                "l2_error": r.l2_error,
                "cosine_sim": r.cosine_sim,
            }
            for r in results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SGLang Attention Backends")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(BACKEND_REGISTRY.keys()),
        help="Backends to benchmark",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[4096, 16384, 32768],
        help="Sequence lengths to test",
    )
    parser.add_argument("--num-heads", type=int, default=24, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--output", type=str, default="attention_benchmark_results.json", help="Output JSON path")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Data type")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("SGLang Diffusion Attention Backend Benchmark")
    print(f"Backends: {args.backends}")
    print(f"Seq lengths: {args.seq_lens}")

    results = run_benchmark(
        backends=args.backends,
        seq_lens=args.seq_lens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        batch_size=args.batch_size,
        dtype=dtype,
    )

    print_summary_table(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
