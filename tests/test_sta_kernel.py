"""
Unit tests and benchmarks for Sliding Tile Attention (STA) kernel.

Tests cover:
  1. Tile / untile roundtrip correctness (no kernel dependency)
  2. STA kernel correctness vs PyTorch SDPA reference (requires st_attn + CUDA)
  3. Performance benchmarks: latency, memory, sparsity-accuracy trade-off

Target platforms: NVIDIA B200, NVIDIA H20
(AMD MI308 can run tile/untile tests; kernel tests auto-skip)

Usage:
    # Run all tests via pytest (kernel tests skipped if st_attn unavailable)
    python -m pytest tests/test_sta_kernel.py -v -s

    # Run only tile/untile tests (works on any GPU)
    python -m pytest tests/test_sta_kernel.py -v -k "TileUntile"

    # Run only correctness tests
    python -m pytest tests/test_sta_kernel.py -v -s -k "Correctness"

    # Run only benchmarks (use -s to see printed tables)
    python -m pytest tests/test_sta_kernel.py -v -s -k "Benchmark"

    # Run as script — full benchmark with JSON output
    CUDA_VISIBLE_DEVICES=0 python tests/test_sta_kernel.py [--warmup 5] [--repeat 20] [--output results.json]
"""

import argparse
import gc
import json
import platform
import time
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

# ---------------------------------------------------------------------------
# st_attn availability — the package imports OK even without CUDA .so,
# so we also probe whether the underlying C extension actually loaded.
# ---------------------------------------------------------------------------
ST_ATTN_AVAILABLE = False
try:
    from st_attn import sliding_tile_attention
    from st_attn_cuda import sta_fwd as _sta_fwd  # noqa: F401

    ST_ATTN_AVAILABLE = True
except Exception:
    pass

requires_st_attn = pytest.mark.skipif(
    not ST_ATTN_AVAILABLE,
    reason="st_attn CUDA kernel not available on this platform",
)

# ---------------------------------------------------------------------------
# Constants — mirroring SlidingTileAttentionImpl
# ---------------------------------------------------------------------------
BASE_TILE_SIZE = (6, 8, 8)  # (T, H, W) — 384 tokens per tile

# seq_shape_str → config
SHAPE_CONFIGS: dict[str, dict[str, Any]] = {
    "30x48x80": {
        "full_window": (5, 6, 10),
        "img_seq_len": 30 * 48 * 80,  # 115_200
        "has_text": True,
        "text_length": 256,  # HunyuanVideo
        "description": "HunyuanVideo 5s 720P",
    },
    "36x48x48": {
        "full_window": (6, 6, 6),
        "img_seq_len": 36 * 48 * 48,  # 82_944
        "has_text": False,
        "text_length": 0,
        "description": "StepVideo",
    },
    "18x48x80": {
        "full_window": (3, 6, 10),
        "img_seq_len": 18 * 48 * 80,  # 69_120
        "has_text": False,
        "text_length": 0,
        "description": "Wan",
    },
}

# Sparse window candidates from the paper / source code
SPARSE_WINDOW_CANDIDATES = [
    (3, 3, 3),
    (3, 1, 10),
    (1, 5, 7),
    (1, 6, 5),
    (1, 3, 10),
    (3, 6, 1),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "st_attn_available": ST_ATTN_AVAILABLE,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_mem / (1024**3), 2)
        info["sm_count"] = props.multi_processor_count
        cap = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{cap[0]}.{cap[1]}"
        hip = getattr(torch.version, "hip", None)
        if hip:
            info["rocm"] = hip
    return info


def flush_gpu():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()


def tile_tokens(x: torch.Tensor, full_window: tuple[int, int, int]) -> torch.Tensor:
    """Zigzag → tile-grouped order.  Layout: [B, S, H, D]."""
    n_t, n_h, n_w = full_window
    ts_t, ts_h, ts_w = BASE_TILE_SIZE
    return rearrange(
        x,
        "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
        n_t=n_t, n_h=n_h, n_w=n_w, ts_t=ts_t, ts_h=ts_h, ts_w=ts_w,
    )


def untile_tokens(x: torch.Tensor, full_window: tuple[int, int, int]) -> torch.Tensor:
    """Reverse tile_tokens."""
    n_t, n_h, n_w = full_window
    ts_t, ts_h, ts_w = BASE_TILE_SIZE
    return rearrange(
        x,
        "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
        n_t=n_t, n_h=n_h, n_w=n_w, ts_t=ts_t, ts_h=ts_h, ts_w=ts_w,
    )


def prepare_tiled_qkv(
    shape_str: str, batch: int, num_heads: int, head_dim: int,
    dtype: torch.dtype = torch.bfloat16,
):
    """Generate random QKV, tile image portion, transpose to [B,H,S,D].

    Returns (q, k, v) in [B, H, S, D] layout ready for sliding_tile_attention,
    plus (text_length, has_text, full_window).
    """
    cfg = SHAPE_CONFIGS[shape_str]
    full_window = cfg["full_window"]
    img_seq_len = cfg["img_seq_len"]
    has_text = cfg["has_text"]
    text_length = cfg["text_length"]
    total_seq_len = img_seq_len + text_length

    q = torch.randn(batch, total_seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(batch, total_seq_len, num_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(batch, total_seq_len, num_heads, head_dim, dtype=dtype, device="cuda")

    if text_length > 0:
        q_tiled = torch.cat([q[:, :text_length], tile_tokens(q[:, text_length:], full_window)], dim=1)
        k_tiled = torch.cat([k[:, :text_length], tile_tokens(k[:, text_length:], full_window)], dim=1)
        v_tiled = torch.cat([v[:, :text_length], tile_tokens(v[:, text_length:], full_window)], dim=1)
    else:
        q_tiled = tile_tokens(q, full_window)
        k_tiled = tile_tokens(k, full_window)
        v_tiled = tile_tokens(v, full_window)

    # [B, S, H, D] → [B, H, S, D]
    q_bhsd = q_tiled.transpose(1, 2).contiguous()
    k_bhsd = k_tiled.transpose(1, 2).contiguous()
    v_bhsd = v_tiled.transpose(1, 2).contiguous()
    return q_bhsd, k_bhsd, v_bhsd, text_length, has_text, full_window


def sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full attention via PyTorch SDPA.  Layout: [B, H, S, D]."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def compute_metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    out_f = output.float().flatten()
    ref_f = reference.float().flatten()
    l2_rel = (torch.norm(out_f - ref_f) / torch.norm(ref_f)).item()
    max_abs = (out_f - ref_f).abs().max().item()
    cosine = F.cosine_similarity(out_f.unsqueeze(0), ref_f.unsqueeze(0)).item()
    return {"l2_rel_error": l2_rel, "max_abs_error": max_abs, "cosine_sim": cosine}


def measure_latency(fn, warmup: int = 5, repeat: int = 20) -> float:
    """Return average latency in ms (CUDA-synchronised)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000


def clamp_window(window, full_window):
    """Clamp each dim of sparse window to not exceed full_window."""
    return tuple(min(s, f) for s, f in zip(window, full_window))


# ===========================================================================
# 1. Tile / Untile Roundtrip Tests (no kernel dependency)
# ===========================================================================
class TestTileUntile:

    @pytest.mark.parametrize("shape_str", list(SHAPE_CONFIGS.keys()))
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_roundtrip(self, shape_str: str, dtype: torch.dtype):
        cfg = SHAPE_CONFIGS[shape_str]
        seq_len = cfg["img_seq_len"]
        x = torch.randn(1, seq_len, 4, 128, dtype=dtype, device="cuda")
        recovered = untile_tokens(tile_tokens(x, cfg["full_window"]), cfg["full_window"])
        assert x.shape == recovered.shape
        assert torch.allclose(x, recovered, atol=0, rtol=0)

    @pytest.mark.parametrize("shape_str", list(SHAPE_CONFIGS.keys()))
    def test_tile_preserves_shape(self, shape_str: str):
        cfg = SHAPE_CONFIGS[shape_str]
        x = torch.randn(2, cfg["img_seq_len"], 8, 64, device="cuda")
        assert tile_tokens(x, cfg["full_window"]).shape == x.shape

    @pytest.mark.parametrize("shape_str", list(SHAPE_CONFIGS.keys()))
    def test_tile_reorders_tokens(self, shape_str: str):
        cfg = SHAPE_CONFIGS[shape_str]
        x = torch.arange(cfg["img_seq_len"], dtype=torch.float32, device="cuda").view(1, -1, 1, 1)
        assert not torch.equal(x, tile_tokens(x, cfg["full_window"]))


# ===========================================================================
# 2. STA Kernel Correctness Tests
# ===========================================================================
class TestSTACorrectness:

    @requires_st_attn
    @pytest.mark.parametrize("shape_str", list(SHAPE_CONFIGS.keys()))
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_full_window_vs_sdpa(self, shape_str: str, dtype: torch.dtype):
        """Full-window STA ≈ full SDPA (on the same tiled sequence)."""
        q, k, v, text_len, has_text, fw = prepare_tiled_qkv(shape_str, 1, 4, 128, dtype)

        with torch.no_grad():
            sta_out = sliding_tile_attention(q, k, v, [fw] * 4, text_len, has_text, shape_str)
            sdpa_out = sdpa_reference(q, k, v)

        m = compute_metrics(sta_out, sdpa_out)
        print(f"\n[{shape_str}][{dtype}] L2={m['l2_rel_error']:.6f} cos={m['cosine_sim']:.6f}")
        assert m["cosine_sim"] > 0.99
        assert m["l2_rel_error"] < 0.05

    @requires_st_attn
    @pytest.mark.parametrize("shape_str,sparse_window", [
        ("30x48x80", (3, 3, 3)),
        ("30x48x80", (1, 5, 7)),
        ("30x48x80", (3, 1, 10)),
        ("36x48x48", (3, 3, 3)),
        ("36x48x48", (1, 5, 7)),
        ("18x48x80", (1, 3, 10)),
        ("18x48x80", (3, 6, 1)),
    ])
    def test_sparse_window_no_nan(self, shape_str: str, sparse_window: tuple):
        """Sparse STA should produce finite output."""
        q, k, v, text_len, has_text, fw = prepare_tiled_qkv(shape_str, 1, 4, 128)
        clamped = clamp_window(sparse_window, fw)
        with torch.no_grad():
            out = sliding_tile_attention(q, k, v, [clamped] * 4, text_len, has_text, shape_str)
        assert out.shape == q.shape
        assert torch.isfinite(out).all()

    @requires_st_attn
    def test_sparsity_monotonic_error(self):
        """Sparser window → higher L2 error vs full attention."""
        shape_str = "30x48x80"
        q, k, v, text_len, has_text, fw = prepare_tiled_qkv(shape_str, 1, 4, 128)

        with torch.no_grad():
            full_out = sliding_tile_attention(q, k, v, [fw] * 4, text_len, has_text, shape_str)

        test_windows = [fw, (3, 3, 3), (1, 3, 10)]
        errors = []
        for w in test_windows:
            with torch.no_grad():
                out = sliding_tile_attention(q, k, v, [w] * 4, text_len, has_text, shape_str)
            errors.append(compute_metrics(out, full_out)["l2_rel_error"])
            print(f"  window={w} → L2_rel={errors[-1]:.6f}")

        assert errors[0] < 1e-5, "Full window self-error too large"
        for i in range(1, len(errors)):
            assert errors[i] > errors[0]

    @requires_st_attn
    @pytest.mark.parametrize("shape_str", list(SHAPE_CONFIGS.keys()))
    def test_per_head_mixed_window(self, shape_str: str):
        """Different heads can have different window sizes."""
        q, k, v, text_len, has_text, fw = prepare_tiled_qkv(shape_str, 1, 4, 128)
        sparse_w = clamp_window((3, 3, 3), fw)
        windows = [fw, sparse_w, fw, sparse_w]
        with torch.no_grad():
            out = sliding_tile_attention(q, k, v, windows, text_len, has_text, shape_str)
        assert out.shape == q.shape
        assert torch.isfinite(out).all()

    @requires_st_attn
    def test_deterministic(self):
        """Two calls with identical input produce identical output."""
        q, k, v, text_len, has_text, fw = prepare_tiled_qkv("18x48x80", 1, 2, 128)
        windows = [fw] * 2
        with torch.no_grad():
            o1 = sliding_tile_attention(q, k, v, windows, text_len, has_text, "18x48x80")
            o2 = sliding_tile_attention(q, k, v, windows, text_len, has_text, "18x48x80")
        assert torch.equal(o1, o2)


# ===========================================================================
# 3. Performance Benchmarks (pytest)
# ===========================================================================
class TestSTABenchmark:

    @requires_st_attn
    @pytest.mark.parametrize("shape_str", list(SHAPE_CONFIGS.keys()))
    def test_benchmark_latency(self, shape_str: str):
        """Benchmark STA vs SDPA latency."""
        cfg = SHAPE_CONFIGS[shape_str]
        fw = cfg["full_window"]
        num_heads, head_dim = 24, 128
        warmup, repeat = 5, 20

        q, k, v, text_len, has_text, _ = prepare_tiled_qkv(shape_str, 1, num_heads, head_dim)

        # SDPA baseline
        flush_gpu()
        sdpa_lat = measure_latency(lambda: sdpa_reference(q, k, v), warmup, repeat)
        with torch.no_grad():
            sdpa_out = sdpa_reference(q, k, v)

        # STA full
        flush_gpu()
        full_lat = measure_latency(
            lambda: sliding_tile_attention(q, k, v, [fw] * num_heads, text_len, has_text, shape_str),
            warmup, repeat,
        )

        # STA sparse (3,3,3)
        sw = clamp_window((3, 3, 3), fw)
        flush_gpu()
        sparse_lat = measure_latency(
            lambda: sliding_tile_attention(q, k, v, [sw] * num_heads, text_len, has_text, shape_str),
            warmup, repeat,
        )

        with torch.no_grad():
            full_out = sliding_tile_attention(q, k, v, [fw] * num_heads, text_len, has_text, shape_str)
            sparse_out = sliding_tile_attention(q, k, v, [sw] * num_heads, text_len, has_text, shape_str)

        fm = compute_metrics(full_out, sdpa_out)
        sm = compute_metrics(sparse_out, sdpa_out)
        full_tiles = fw[0] * fw[1] * fw[2]
        sparsity = 1.0 - (sw[0] * sw[1] * sw[2]) / full_tiles

        print(f"\n{'='*80}")
        print(f"  {shape_str} ({cfg['description']})  seq_len={cfg['img_seq_len'] + cfg['text_length']}")
        print(f"  {'Method':<28} {'Latency':>10} {'Speedup':>8} {'L2 Err':>10} {'CosSim':>10}")
        print(f"  {'─'*68}")
        print(f"  {'SDPA':<28} {sdpa_lat:>9.2f}ms {'1.00x':>8}")
        print(f"  {'STA full ' + str(fw):<28} {full_lat:>9.2f}ms {sdpa_lat/full_lat:>7.2f}x {fm['l2_rel_error']:>10.6f} {fm['cosine_sim']:>10.6f}")
        print(f"  {'STA sparse ' + str(sw):<28} {sparse_lat:>9.2f}ms {sdpa_lat/sparse_lat:>7.2f}x {sm['l2_rel_error']:>10.6f} {sm['cosine_sim']:>10.6f}")
        print(f"  Sparsity: {sparsity*100:.1f}%")
        print(f"{'='*80}")

    @requires_st_attn
    def test_benchmark_memory(self):
        """Peak GPU memory: STA vs SDPA."""
        shape_str = "18x48x80"
        cfg = SHAPE_CONFIGS[shape_str]
        fw = cfg["full_window"]
        num_heads = 24

        q, k, v, text_len, has_text, _ = prepare_tiled_qkv(shape_str, 1, num_heads, 128)

        flush_gpu(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = sdpa_reference(q, k, v)
        sdpa_mem = torch.cuda.max_memory_allocated() / (1024**2)

        flush_gpu(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = sliding_tile_attention(q, k, v, [fw] * num_heads, text_len, has_text, shape_str)
        sta_full_mem = torch.cuda.max_memory_allocated() / (1024**2)

        sw = clamp_window((3, 3, 3), fw)
        flush_gpu(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = sliding_tile_attention(q, k, v, [sw] * num_heads, text_len, has_text, shape_str)
        sta_sparse_mem = torch.cuda.max_memory_allocated() / (1024**2)

        print(f"\n  Memory ({shape_str}): SDPA={sdpa_mem:.0f}MB  STA_full={sta_full_mem:.0f}MB  STA_sparse={sta_sparse_mem:.0f}MB")


# ===========================================================================
# 4. CLI — full benchmark with JSON output for cross-platform comparison
# ===========================================================================
def run_full_benchmark(warmup: int = 5, repeat: int = 20) -> list[dict]:
    """Benchmark all shapes × window configs.  Returns structured results."""
    assert ST_ATTN_AVAILABLE, "st_attn CUDA kernel required for benchmarks"

    device_info = get_device_info()
    print(f"\nDevice: {device_info.get('gpu_name', 'Unknown')}")
    print(f"Torch: {device_info['torch']}, CC: {device_info.get('compute_capability', 'N/A')}")
    print(f"GPU Memory: {device_info.get('gpu_memory_gb', 'N/A')} GB")

    results: list[dict] = []
    num_heads, head_dim = 24, 128

    for shape_str, cfg in SHAPE_CONFIGS.items():
        fw = cfg["full_window"]
        total_seq = cfg["img_seq_len"] + cfg["text_length"]
        print(f"\n{'='*80}")
        print(f"Shape: {shape_str} ({cfg['description']}) — seq_len={total_seq}")
        print(f"{'='*80}")

        q, k, v, text_len, has_text, _ = prepare_tiled_qkv(shape_str, 1, num_heads, head_dim)

        base = {"shape": shape_str, "description": cfg["description"],
                "seq_len": total_seq, "num_heads": num_heads, "head_dim": head_dim}

        # ---- SDPA baseline ----
        flush_gpu()
        sdpa_lat = measure_latency(lambda: sdpa_reference(q, k, v), warmup, repeat)
        flush_gpu(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            sdpa_out = sdpa_reference(q, k, v)
        sdpa_mem = torch.cuda.max_memory_allocated() / (1024**2)

        results.append({**base, "backend": "SDPA", "window": "N/A",
                        "latency_ms": round(sdpa_lat, 3), "peak_memory_mb": round(sdpa_mem, 1),
                        "speedup_vs_sdpa": 1.0})
        print(f"  {'SDPA':<30} {sdpa_lat:>8.2f} ms  {sdpa_mem:>8.0f} MB")

        # ---- STA configs ----
        full_tiles = fw[0] * fw[1] * fw[2]
        window_configs = [
            ("full", fw),
            ("sparse_3x3x3", clamp_window((3, 3, 3), fw)),
            ("sparse_1x3x10", clamp_window((1, 3, 10), fw)),
            ("sparse_3x1x10", clamp_window((3, 1, 10), fw)),
            ("sparse_1x5x7", clamp_window((1, 5, 7), fw)),
            ("sparse_3x6x1", clamp_window((3, 6, 1), fw)),
        ]

        for name, window in window_configs:
            flush_gpu()
            try:
                lat = measure_latency(
                    lambda w=window: sliding_tile_attention(
                        q, k, v, [w] * num_heads, text_len, has_text, shape_str),
                    warmup, repeat)

                flush_gpu(); torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    out = sliding_tile_attention(q, k, v, [window] * num_heads, text_len, has_text, shape_str)
                mem = torch.cuda.max_memory_allocated() / (1024**2)

                m = compute_metrics(out, sdpa_out)
                sparse_tiles = window[0] * window[1] * window[2]
                sparsity = 1.0 - sparse_tiles / full_tiles
                speedup = sdpa_lat / lat

                results.append({
                    **base, "backend": f"STA_{name}", "window": str(window),
                    "latency_ms": round(lat, 3), "peak_memory_mb": round(mem, 1),
                    "speedup_vs_sdpa": round(speedup, 3),
                    "l2_rel_error": round(m["l2_rel_error"], 6),
                    "cosine_sim": round(m["cosine_sim"], 6),
                    "max_abs_error": round(m["max_abs_error"], 6),
                    "sparsity": round(sparsity, 4),
                })

                print(f"  {'STA ' + name:<30} {lat:>8.2f} ms  {mem:>8.0f} MB  "
                      f"{speedup:>5.2f}x  L2={m['l2_rel_error']:.4f}  "
                      f"cos={m['cosine_sim']:.4f}  sparsity={sparsity*100:.0f}%")
            except Exception as e:
                print(f"  {'STA ' + name:<30} ERROR: {e}")

        del q, k, v
        flush_gpu()

    return results


def print_summary(results: list[dict]):
    print(f"\n{'='*110}")
    print("SUMMARY")
    print(f"{'='*110}")
    print(f"{'Backend':<25} {'Shape':<12} {'SeqLen':>8} {'Latency':>10} {'Speedup':>8} "
          f"{'Memory':>10} {'L2 Err':>10} {'CosSim':>10} {'Sparsity':>9}")
    print(f"{'─'*110}")
    for r in results:
        l2 = f"{r['l2_rel_error']:.6f}" if 'l2_rel_error' in r else "N/A"
        cos = f"{r['cosine_sim']:.6f}" if 'cosine_sim' in r else "N/A"
        sp = f"{r['sparsity']*100:.0f}%" if 'sparsity' in r else "N/A"
        print(f"{r['backend']:<25} {r['shape']:<12} {r['seq_len']:>8} "
              f"{r['latency_ms']:>9.2f}ms {r.get('speedup_vs_sdpa', 1.0):>7.2f}x "
              f"{r['peak_memory_mb']:>9.0f}MB {l2:>10} {cos:>10} {sp:>9}")
    print(f"{'='*110}")


def main():
    parser = argparse.ArgumentParser(description="STA Kernel Benchmark")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output path (default: sta_benchmark_<gpu>.json)")
    args = parser.parse_args()

    device_info = get_device_info()
    results = run_full_benchmark(warmup=args.warmup, repeat=args.repeat)
    print_summary(results)

    output_path = args.output
    if output_path is None:
        gpu = device_info.get("gpu_name", "unknown").replace(" ", "_")
        output_path = f"sta_benchmark_{gpu}.json"

    report = {
        "device": device_info,
        "benchmark_config": {"warmup": args.warmup, "repeat": args.repeat},
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
