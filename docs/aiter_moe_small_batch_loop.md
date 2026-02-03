# AITER MoE Small Batch Loop Mode

## Background

AITER (AMD's MoE library) uses different kernels based on batch size:
- **Batch size = 1**: `moe_stage1/2_g1u1_small_batch1`
- **Batch size 2-16**: `moe_stage1/2_g1u1_small_batch`
- **Batch size > 16**: `fmoe_g1u1` (1-stage fused kernel)

This leads to different code paths for prefill (typically >16 tokens) vs decode (typically 1 token).

## Experiment

**Goal**: Unify prefill and decode to use the same small_batch operators by processing large batches in a for-loop (max 16 tokens per iteration).

**Implementation**:
- Added `SGLANG_AITER_MOE_USE_SMALL_BATCH_LOOP=1` environment variable
- Modified `fp8.py` and `compressed_tensors_moe.py` to add `_apply_small_batch_loop()` method
- When enabled, all MoE calls (including prefill) use small_batch operators via for-loop

## Test Results

Ran 5 identical requests with `temperature=0`:

| Run | Output Length | First Different Char Position |
|-----|--------------|------------------------------|
| 1   | 1855 chars   | -                            |
| 2   | 1821 chars   | char 8                       |
| 3   | 1787 chars   | char 8                       |
| 4   | 1681 chars   | char 8                       |
| 5   | 1868 chars   | char 8                       |

**Result**: Outputs are NOT consistent even when using the same small_batch operators for all batch sizes.

## Conclusion

**AITER MoE kernels are inherently non-deterministic at the assembly level.**

Using for-loop with small_batch operators does NOT achieve deterministic computation. The non-determinism comes from the kernel implementation itself, not from different code paths between prefill and decode.

## Usage

```bash
# Enable small_batch loop mode (for testing only, does not guarantee determinism)
export SGLANG_AITER_MOE_USE_SMALL_BATCH_LOOP=1
export AITER_MOE_SMALL_BATCH=1
python -m sglang.launch_server ... --disable-cuda-graph
```

## Alternative

For deterministic computation, use Triton MoE instead:
```bash
export SGLANG_USE_AITER_MOE=0
```
