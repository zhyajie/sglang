# [Feature] Add profiling support for Diffusion multimodal models

## Motivation

The existing profiler for LLM models captures operations in the scheduler process (GPU kernels, CUDA operations). However, for **Diffusion multimodal models** (like Qwen-Image-Edit), we need visibility into both the HTTP server process and GPU worker processes to understand end-to-end latency.

This PR adds profiling support for Diffusion models, allowing us to:
- Profile the HTTP server process (request handling, image saving)
- Profile GPU workers (encode, denoise steps, decode)
- Correlate traces across processes using a unified `profile_id`

## What this PR does

**New API endpoints:**
- `POST /start_profile` - Start profiling across HTTP server and GPU workers
- `POST /stop_profile` - Stop profiling and save traces

**Trace file output:**
```
./sglang_qwen_profiling/
├── 1736694000-host.trace.json.gz     # HTTP server process
├── 1736694000-rank-0.trace.json.gz   # GPU worker rank 0
└── 1736694000-rank-1.trace.json.gz   # GPU worker rank 1
```

**Benchmark integration:**
- Added `--profile` flag to `bench_serving.py` for easy profiling during benchmarks

## Usage

### 1. Start the server

```bash
export SGLANG_TORCH_PROFILER_DIR=./sglang_qwen_profiling
export SGLANG_PROFILE_WITH_STACK=1
export SGLANG_PROFILE_RECORD_SHAPES=1

sglang serve \
    --model-path /path/to/Qwen-Image-Edit \
    --num-gpus 2 \
    --ulysses-degree 2 \
    --host 0.0.0.0 \
    --port 30000
```

### 2. Run benchmark with profiling

```bash
python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
    --backend sglang-image \
    --task ti2i \
    --port 30000 \
    --dataset vbench \
    --dataset-path /path/to/benchmark_data \
    --num-prompts 5 \
    --max-concurrency 1 \
    --profile
```

### 3. View traces

Open the `.trace.json.gz` files in [Perfetto UI](https://ui.perfetto.dev/) or Chrome's `chrome://tracing`.

## Trace visualization

**HTTP Server trace (`*-host.trace.json.gz`):**

<!-- TODO: Add screenshot here -->

**GPU Worker trace (`*-rank-0.trace.json.gz`):**

<!-- TODO: Add screenshot here -->

## Files changed

| File | Changes |
|------|---------|
| `bench_serving.py` | Added `--profile` flag and `/start_profile`, `/stop_profile` API calls |
| `http_server.py` | Implemented `/start_profile` and `/stop_profile` endpoints |
| `gpu_worker.py` | Added `start_profile()` and `stop_profile()` methods |
| `scheduler.py` | Added handlers for `StartProfileReq` and `StopProfileReq` |
| `profiler.py` | Simplified filename generation, reused `get_bool_env_var` |
| `utils.py` | Added `StartProfileReq` and `StopProfileReq` dataclasses |
