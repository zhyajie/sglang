from __future__ import annotations

"""
end to end attention solution with aiter kernels
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, List

import torch
import triton

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import (
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

try:
    from aiter import (
        flash_attn_varlen_func,
        flash_attn_varlen_fp8_pertensor_func,
        mha_batch_prefill_func,
        paged_attention_ragged,
    )
    from aiter import dtypes, per_tensor_quant
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd
except ImportError:
    print(
        "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
    )

from sglang.srt.configs.model_config import AttentionArch

_use_aiter = get_bool_env_var("SGLANG_USE_AITER")
USING_PRESHUFFLE_LAYOUT = _use_aiter and get_bool_env_var("SGLANG_ROCM_USE_AITER_PA_ASM_PRESHUFFLE_LAYOUT")

import triton
import triton.language as tl


@triton.jit
def _quantize_qkv_fp8_kernel(
    q,
    k,
    v,
    q_out,
    k_out,
    v_out,
    scale: float,
    q_rows: int,
    k_rows: int,
    v_rows: int,
    q_cols: int,
    k_cols: int,
    v_cols: int,
    q_stride_r: int,
    k_stride_r: int,
    v_stride_r: int,
    q_out_stride_r: int,
    k_out_stride_r: int,
    v_out_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
):
    """
    Quantize q, k, v tensors to FP8 using per-tensor quantization with a single scale.
    Supports different number of rows for q, k, v (e.g., GQA/MQA).
    
    Args:
        q, k, v: Input tensors for q, k, v
        q_out, k_out, v_out: Output tensors for quantized q, k, v
        scale: Scale value (scalar float)
        q_rows, k_rows, v_rows: Number of rows for each tensor
        q_cols, k_cols, v_cols: Number of columns for each tensor
        q_stride_r, k_stride_r, v_stride_r: Row strides for input tensors
        q_out_stride_r, k_out_stride_r, v_out_stride_r: Row strides for output tensors
        NUM_COL_POW2: Power-of-2 rounded number of columns (for efficient masking)
    """
    pid = tl.program_id(axis=0)
    
    # Calculate scale reciprocal
    scale_recip = 1.0 / scale
    
    # Process Q tensor with mask
    q_offs = pid * q_stride_r + tl.arange(0, NUM_COL_POW2)
    q_row_mask = pid < q_rows
    q_col_mask = tl.arange(0, NUM_COL_POW2) < q_cols
    q_mask = q_row_mask & q_col_mask
    q_val = tl.load(q + q_offs, mask=q_mask, other=0.0, cache_modifier=".cg")
    q_out_val = (q_val * scale_recip).to(tl.float8e4b8)
    q_out_offs = pid * q_out_stride_r + tl.arange(0, NUM_COL_POW2)
    tl.store(q_out + q_out_offs, q_out_val, mask=q_mask)
    
    # Process K tensor with mask
    k_offs = pid * k_stride_r + tl.arange(0, NUM_COL_POW2)
    k_row_mask = pid < k_rows
    k_col_mask = tl.arange(0, NUM_COL_POW2) < k_cols
    k_mask = k_row_mask & k_col_mask
    k_val = tl.load(k + k_offs, mask=k_mask, other=0.0, cache_modifier=".cg")
    k_out_val = (k_val * scale_recip).to(tl.float8e4b8)
    k_out_offs = pid * k_out_stride_r + tl.arange(0, NUM_COL_POW2)
    tl.store(k_out + k_out_offs, k_out_val, mask=k_mask)
    
    # Process V tensor with mask
    v_offs = pid * v_stride_r + tl.arange(0, NUM_COL_POW2)
    v_row_mask = pid < v_rows
    v_col_mask = tl.arange(0, NUM_COL_POW2) < v_cols
    v_mask = v_row_mask & v_col_mask
    v_val = tl.load(v + v_offs, mask=v_mask, other=0.0, cache_modifier=".cg")
    v_out_val = (v_val * scale_recip).to(tl.float8e4b8)
    v_out_offs = pid * v_out_stride_r + tl.arange(0, NUM_COL_POW2)
    tl.store(v_out + v_out_offs, v_out_val, mask=v_mask)


def quantize_qkv_fp8_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize q, k, v tensors to FP8 using Triton kernel in a single fused operation.
    Supports different number of rows for q, k, v (e.g., GQA/MQA).
    
    Args:
        q: Query tensor of shape [M_q, N_q]
        k: Key tensor of shape [M_k, N_k]
        v: Value tensor of shape [M_v, N_v]
        scale: Scale value (scalar float)
        quant_dtype: Target quantization dtype (e.g., torch.float8_e4m3fnuz)
    
    Returns:
        q_out, k_out, v_out: Quantized tensors
    """
    q_rows = q.shape[0]
    k_rows = k.shape[0]
    v_rows = v.shape[0]
    q_cols = q.shape[1]
    k_cols = k.shape[1]
    v_cols = v.shape[1]
    
    # Create output tensors
    q_out = torch.empty_like(q, dtype=quant_dtype)
    k_out = torch.empty_like(k, dtype=quant_dtype)
    v_out = torch.empty_like(v, dtype=quant_dtype)
    
    # Calculate power-of-2 rounded columns for efficient masking
    max_cols = max(q_cols, k_cols, v_cols)
    NUM_COL_POW2 = triton.next_power_of_2(max_cols)
    
    # Launch kernel with grid size covering all three tensors
    max_rows = max(q_rows, k_rows, v_rows)
    grid = lambda meta: (max_rows,)
    _quantize_qkv_fp8_kernel[grid](
        q,
        k,
        v,
        q_out,
        k_out,
        v_out,
        scale,
        q_rows,
        k_rows,
        v_rows,
        q_cols,
        k_cols,
        v_cols,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        v_out.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
    )
    
    return q_out, k_out, v_out


@triton.jit
def _quantize_q_fp8_kernel(
    q,
    q_out,
    scale: float,
    q_cols: int,
    q_stride_r: int,
    q_out_stride_r: int,
    NUM_COL_POW2: tl.constexpr,
):
    """
    Quantize q tensor to FP8 using per-tensor quantization.
    
    Args:
        q: Input tensor for q
        q_out: Output tensor for quantized q
        scale: Scale value (scalar float)
        q_cols: Number of columns
        q_stride_r: Row stride for input tensor
        q_out_stride_r: Row stride for output tensor
        NUM_COL_POW2: Power-of-2 rounded number of columns
    """
    pid = tl.program_id(axis=0)
    
    # Calculate scale reciprocal
    scale_recip = 1.0 / scale
    
    # Process Q tensor
    q_offs = pid * q_stride_r + tl.arange(0, NUM_COL_POW2)
    q_mask = tl.arange(0, NUM_COL_POW2) < q_cols
    q_val = tl.load(q + q_offs, mask=q_mask, cache_modifier=".cg")
    q_out_val = (q_val * scale_recip).to(tl.float8e4b8)
    q_out_offs = pid * q_out_stride_r + tl.arange(0, NUM_COL_POW2)
    tl.store(q_out + q_out_offs, q_out_val, mask=q_mask)


def quantize_q_fp8_triton(
    q: torch.Tensor,
    scale: float,
    quant_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Quantize q tensor to FP8 using Triton kernel.
    
    Args:
        q: Query tensor of shape [M, N_q]
        scale: Scale value (scalar float)
        quant_dtype: Target quantization dtype (e.g., torch.float8_e4m3fnuz)
    
    Returns:
        q_out: Quantized tensor
    """
    M = q.shape[0]
    q_cols = q.shape[1]
    
    # Create output tensor
    q_out = torch.empty_like(q, dtype=quant_dtype)
    
    # Calculate power-of-2 rounded columns for efficient masking
    NUM_COL_POW2 = triton.next_power_of_2(q_cols)
    
    # Launch kernel
    grid = lambda meta: (M,)
    _quantize_q_fp8_kernel[grid](
        q,
        q_out,
        scale,
        q_cols,
        q.stride(0),
        q_out.stride(0),
        NUM_COL_POW2=NUM_COL_POW2,
    )
    
    return q_out

class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class ForwardMetadata:
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    kv_last_page_len: torch.Tensor
    max_q_len: int
    max_kv_len: Optional[int]
    page_table: Optional[torch.Tensor]
    kv_lens: Optional[torch.Tensor]


global_workspace_buffer = None

_AITER_PARTITION_SIZE_ROCM = 256


class AiterAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            extend_attention_fwd,
        )

        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)

        self.device = model_runner.device
        self.is_multimodal = model_runner.model_config.is_multimodal
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.page_size = model_runner.page_size
        self.head_dim = model_runner.model_config.head_dim
        mapping = getattr(
            model_runner.token_to_kv_pool, "full_attention_layer_id_mapping", None
        )

        if isinstance(mapping, dict) and mapping:
            first_full_attn_id = next(iter(mapping.keys()))
        else:
            first_full_attn_id = 0
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(
            first_full_attn_id
        ).shape[-1]
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.kv_cache_dtype = model_runner.kv_cache_dtype

        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA

        # Parse constants
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill

        max_bs = model_runner.req_to_token_pool.size

        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        self.seq_lens = torch.zeros(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.page_table = torch.zeros(
            (max_bs, self.max_context_len // self.page_size), dtype=torch.int32, device=model_runner.device
        )

        # Create prefill indices updater
        if not skip_prefill:
            self.indices_updater_prefill = AiterIndicesUpdaterPrefill(
                model_runner, self
            )
            if self.use_mla:
                self.mla_indices_updater_prefill = AiterMlaIndicesUpdaterPrefill(
                    model_runner, self
                )

        # aiter kernel related initialization
        self.max_num_partitions = (
            self.max_context_len + _AITER_PARTITION_SIZE_ROCM - 1
        ) // _AITER_PARTITION_SIZE_ROCM

        nbyes_per_qo_elem = torch.finfo(torch.float32).bits // 8

        if not self.use_mla:
            self.workspace_buffer = torch.empty(
                (max_bs * self.num_head * self.max_num_partitions * self.head_dim)
                * nbyes_per_qo_elem
                + 2 * (max_bs * self.num_head * self.max_num_partitions) * 4,
                dtype=torch.uint8,
                device=self.device,
            )

        self.scale = float(1.0 / (self.head_dim**0.5))
        self.k_scale = self.v_scale = torch.tensor([1.0], dtype=torch.float32).to(
            self.device
        )
        self.fp8_scale = 1.0  # Store as scalar to avoid sync in kernel calls

        self.logits_soft_cap = 0.0

        self.forward_metadata: ForwardMetadata = None

        if self.use_mla:
            self.qo_indptr_ = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )

            self.enable_dp_attention = is_dp_attention_enabled()

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        spec_info = forward_batch.spec_info
        qo_indptr = None
        kv_last_page_len = None
        max_q_len = None
        page_table = None

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    forward_batch.seq_lens_sum, dtype=torch.int32, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(self.kv_last_page_len[:bs], dim=0)
                kv_last_page_len = self.kv_last_page_len[:bs]
                max_q_len = 1
            page_table = forward_batch.req_to_token_pool.req_to_token[forward_batch.req_pool_indices, :]


            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                None,
                page_table,
                forward_batch.seq_lens,
            )

        elif forward_batch.forward_mode.is_draft_extend():
            if self.use_mla:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        forward_batch.seq_lens_sum,
                        self.req_to_token,
                    )
                )
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    # self.mla_indices_updater_prefill.kv_last_page_len,
                    self.kv_last_page_len[:bs],
                    max(forward_batch.extend_seq_lens_cpu),
                    forward_batch.seq_lens_cpu.max().item(),
                    None,
                    None,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                    None,
                    None,
                )
        elif forward_batch.forward_mode.is_target_verify():
            if self.use_mla:
                draft_num = spec_info.draft_token_num
                kv_lens = forward_batch.seq_lens + draft_num
                kv_lens_sum = forward_batch.seq_lens_sum + draft_num * bs
                device = forward_batch.seq_lens.device

                qo_indptr = torch.arange(
                    0,
                    (1 + bs) * draft_num,
                    step=draft_num,
                    dtype=torch.int32,
                    device=device,
                )
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    kv_lens_sum,
                    dtype=torch.int32,
                    device=device,
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    kv_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    # self.mla_indices_updater_prefill.kv_last_page_len,
                    self.kv_last_page_len[:bs],
                    draft_num,
                    None,
                    None,
                    None,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens=None,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=forward_batch.spec_info,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                    None,
                    None,
                )
        else:
            prefix_lens = forward_batch.extend_prefix_lens
            prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

            if self.is_multimodal:
                extend_no_prefix = False
            else:
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
            if self.use_mla:
                self.mla_indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    forward_batch.extend_seq_lens,
                    forward_batch.extend_seq_lens.max().item(),
                    forward_batch.seq_lens.max().item(),
                    spec_info=None,
                )

                kv_indices = self.mla_indices_updater_prefill.kv_indices

                self.forward_metadata = ForwardMetadata(
                    self.mla_indices_updater_prefill.kv_indptr,
                    kv_indices,
                    self.mla_indices_updater_prefill.qo_indptr,
                    self.kv_last_page_len[:bs],
                    self.mla_indices_updater_prefill.max_q_len,
                    self.mla_indices_updater_prefill.max_kv_len,
                    None,
                    None,
                )
            else:
                self.indices_updater_prefill.update(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    forward_batch.seq_lens_sum,
                    prefix_lens,
                    prefix_lens_cpu=prefix_lens_cpu,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                    encoder_lens=forward_batch.encoder_lens,
                    spec_info=None,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                    None,
                    forward_batch.seq_lens,
                )

        if self.page_size > 1 and self.forward_metadata.page_table is not None:
            strided_indices = torch.arange(
                0, self.forward_metadata.page_table.shape[1], self.page_size, device=self.device
            )
            self.forward_metadata.page_table = (
                self.forward_metadata.page_table[:, strided_indices] // self.page_size
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_kv_last_page_len = torch.ones(max_bs, dtype=torch.int)
        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_bs * self.max_context_len),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )
        # Always use preshuffle layout for pa_fwd_asm
        self.page_table = torch.zeros(
            (max_bs, self.max_context_len // self.page_size), dtype=torch.int32, device=self.device
        )
        self.seq_lens = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self.strided_indices = torch.arange(
            0, self.max_context_len, self.page_size, device=self.device
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        if forward_mode.is_decode_or_idle():
            qo_indptr = None
            kv_last_page_len = None
            max_q_len = None

            if spec_info is None:
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            if self.use_mla:
                qo_indptr = self.qo_indptr_[: bs + 1]
                qo_indptr[1 : bs + 1] = torch.cumsum(
                    self.cuda_graph_kv_last_page_len[:bs], dim=0
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = 1
            # Always use preshuffle layout for pa_fwd_asm
            page_table = self.page_table[:bs, :]
            self.seq_lens[:bs].copy_(seq_lens, non_blocking=True)
            seq_lens = self.seq_lens[:bs]
            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                None,
                page_table,
                seq_lens,
            )

        elif forward_mode.is_target_verify():
            if self.use_mla:
                qo_indptr = self.qo_indptr[: bs + 1]
                qo_indptr[: bs + 1] = torch.arange(
                    0,
                    (1 + bs) * self.num_draft_tokens,
                    step=self.num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                )
                kv_indptr = self.kv_indptr[: bs + 1]
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
                max_q_len = self.num_draft_tokens

                self.forward_metadata = ForwardMetadata(
                    kv_indptr,
                    kv_indices,
                    qo_indptr,
                    kv_last_page_len,
                    max_q_len,
                    None,
                )
            else:
                seq_lens_sum = seq_lens.sum().item()
                self.indices_updater_prefill.update(
                    req_pool_indices,
                    seq_lens,
                    seq_lens_sum,
                    prefix_lens=None,
                    encoder_lens=encoder_lens,
                    spec_info=spec_info,
                )
                self.forward_metadata = ForwardMetadata(
                    self.indices_updater_prefill.kv_indptr,
                    self.indices_updater_prefill.kv_indices,
                    None,
                    None,
                    self.indices_updater_prefill.max_q_len,
                    self.indices_updater_prefill.max_kv_len,
                )
        elif forward_mode.is_draft_extend():
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            kv_last_page_len = self.cuda_graph_kv_last_page_len[:bs]
            max_q_len = num_tokens_per_bs
            self.forward_metadata = ForwardMetadata(
                kv_indptr,
                kv_indices,
                qo_indptr,
                kv_last_page_len,
                max_q_len,
                None,
            )
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        if forward_mode.is_decode_or_idle():
            kv_indptr = self.kv_indptr
            kv_indices = self.cuda_graph_kv_indices
            # Always use preshuffle layout for pa_fwd_asm
            page_table_persistent = self.page_table
            seq_lens_persistent = self.seq_lens
            seq_lens_persistent.fill_(0)
            page_table_persistent.fill_(0)
            seq_lens_persistent[:bs].copy_(seq_lens, non_blocking=True)
            max_seq_pages = (seq_lens_cpu.max().item() + self.page_size - 1) // self.page_size + 1
            page_table = self.req_to_token[req_pool_indices[:, None], self.strided_indices[:max_seq_pages][None, :],]
            page_table_persistent[:bs, :max_seq_pages].copy_(page_table // self.page_size, non_blocking=True)
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
            else:
                kv_indptr[: spec_info.kv_indptr.shape[0]] = spec_info.kv_indptr
                kv_indices[: spec_info.kv_indices.shape[0]] = spec_info.kv_indices

        elif forward_mode.is_target_verify():
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_lens = seq_lens + self.num_draft_tokens
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
        elif forward_mode.is_draft_extend():
            seq_lens = seq_lens[:bs]
            accept_lens = spec_info.accept_length[:bs]
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[1 : bs + 1] = torch.cumsum(accept_lens, dim=0)
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
        else:
            raise ValueError("Invalid forward mode")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        self.logits_soft_cap = layer.logit_cap

        if save_kv_cache:
            assert k is not None
            assert v is not None
            if self.use_mla:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)
            else:
                # Shuffle operation is already fused in rotary_emb, so just save directly
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        if self.use_mla:
            max_q_len = self.forward_metadata.max_q_len
            max_kv_len = self.forward_metadata.max_kv_len
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            qo_indptr = self.forward_metadata.qo_indptr
            K_Buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            V_Buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
            kv_lora_rank = V_Buffer.shape[-1]
            qk_rope_head_dim = K_Buffer.shape[-1] - kv_lora_rank
            qk_nope_head_dim = k.shape[-1] - qk_rope_head_dim
            assert len(q.shape) == 3
            assert len(k.shape) == 3
            assert len(v.shape) == 3

            if (
                forward_batch.forward_mode.is_extend()
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                if kv_indices.shape[0] == 0:
                    o = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        qo_indptr,
                        qo_indptr,
                        max_q_len,
                        max_q_len,
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    return o
                elif layer.qk_head_dim != (kv_lora_rank + qk_rope_head_dim):
                    K_Buffer = torch.index_select(K_Buffer, 0, kv_indices)
                    kvc, k_pe = torch.split(
                        K_Buffer, [kv_lora_rank, qk_rope_head_dim], dim=-1
                    )
                    kvprefix = layer.kv_b_proj(kvc.contiguous())[0]

                    kvprefix = kvprefix.view(
                        -1, layer.tp_k_head_num, qk_nope_head_dim + layer.v_head_dim
                    )
                    k_prefix, v_prefix = torch.split(
                        kvprefix, [qk_nope_head_dim, layer.v_head_dim], dim=-1
                    )
                    k_prefix = torch.cat(
                        [
                            k_prefix,
                            torch.broadcast_to(
                                k_pe,
                                (k_pe.shape[0], layer.tp_k_head_num, k_pe.shape[2]),
                            ),
                        ],
                        dim=-1,
                    )
                    assert (
                        forward_batch.extend_prefix_lens.shape
                        == forward_batch.extend_seq_lens.shape
                    )

                    k = k_prefix
                    v = v_prefix

                    o = flash_attn_varlen_func(
                        q,
                        k,
                        v,
                        qo_indptr,
                        kv_indptr,
                        max_q_len,
                        max_kv_len,
                        softmax_scale=layer.scaling,
                        causal=True,
                    )
                    return o

                else:
                    if layer.qk_head_dim != layer.v_head_dim:
                        o = q.new_empty(
                            (q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
                        )
                    else:
                        o = torch.empty_like(q)

                    mla_prefill_fwd(
                        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                        K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                        o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        self.forward_metadata.kv_last_page_len,
                        self.forward_metadata.max_q_len,
                        layer.scaling,
                        layer.logit_cap,
                    )
                    K_Buffer = K_Buffer.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                    return o
            elif forward_batch.forward_mode.is_target_verify():
                o = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
                mla_decode_fwd(
                    q,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    o,
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.kv_last_page_len,
                    self.forward_metadata.max_q_len,
                    layer.scaling,
                    layer.logit_cap,
                )
                K_Buffer = K_Buffer.view(-1, 1, layer.qk_head_dim)
                return o
            elif forward_batch.forward_mode.is_draft_extend():
                o = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))
                kv_indptr = self.forward_metadata.kv_indptr
                kv_indices = self.forward_metadata.kv_indices
                mla_prefill_fwd(
                    q,
                    K_Buffer.view(-1, 1, 1, layer.qk_head_dim),
                    o,
                    self.forward_metadata.qo_indptr,
                    self.forward_metadata.kv_indptr,
                    self.forward_metadata.kv_indices,
                    self.forward_metadata.kv_last_page_len,
                    self.forward_metadata.max_q_len,
                    layer.scaling,
                    layer.logit_cap,
                )
                K_Buffer = K_Buffer.view(-1, 1, layer.qk_head_dim)
                return o
            else:
                raise ValueError(
                    f"Invalid forward mode for MLA prefill: {forward_batch.forward_mode=}"
                )
        else:
            bs0 = forward_batch.batch_size + 1

            # FP8 dtype set for checking
            FP8_DTYPES = (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
                torch.float8_e5m2,
                torch.float8_e5m2fnuz,
            )
            
            # Convert q, k, v to FP8 format
            # Reshape q to [num_tokens, num_heads, head_dim] format expected by flash_attn_varlen_fp8_pertensor_func
            q_reshaped = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            
            # Check if k and v are bf16 - if so, quantize all three; otherwise only quantize q
            k_is_bf16 = k.dtype == torch.bfloat16
            v_is_bf16 = v.dtype == torch.bfloat16
            
            if k_is_bf16 and v_is_bf16:
                # Ensure k and v are in [num_tokens, num_kv_heads, head_dim] format
                if k.dim() == 2:
                    # k is [num_tokens, num_kv_heads * qk_head_dim], reshape to [num_tokens, num_kv_heads, qk_head_dim]
                    k_reshaped = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
                else:
                    k_reshaped = k
                    
                if v.dim() == 2:
                    # v is [num_tokens, num_kv_heads * v_head_dim], reshape to [num_tokens, num_kv_heads, v_head_dim]
                    v_reshaped = v.view(-1, layer.tp_k_head_num, layer.v_head_dim)
                else:
                    v_reshaped = v
                
                # Flatten to 2D for quantization: [num_tokens * num_heads, head_dim]
                q_flat = q_reshaped.reshape(-1, layer.head_dim)
                k_flat = k_reshaped.reshape(-1, k_reshaped.shape[-1])
                v_flat = v_reshaped.reshape(-1, v_reshaped.shape[-1])
                
                # Quantize q, k, v together in a single fused kernel
                # The kernel supports different number of rows (GQA/MQA case)
                q_fp8_flat, k_fp8_flat, v_fp8_flat = quantize_qkv_fp8_triton(
                    q_flat,
                    k_flat,
                    v_flat,
                    self.fp8_scale,
                    dtypes.fp8,
                )
                
                # Reshape back to [num_tokens, num_heads, head_dim] format
                q_fp8 = q_fp8_flat.view(q_reshaped.shape)
                k_fp8 = k_fp8_flat.view(k_reshaped.shape)
                v_fp8 = v_fp8_flat.view(v_reshaped.shape)
            else:
                # Only quantize q, k and v are already FP8
                q_flat = q_reshaped.reshape(-1, layer.head_dim)
                q_fp8_flat = quantize_q_fp8_triton(
                    q_flat,
                    self.fp8_scale,
                    dtypes.fp8,
                )
                q_fp8 = q_fp8_flat.view(q_reshaped.shape)
                k_fp8 = k  # Use k as-is (already FP8)
                v_fp8 = v  # Use v as-is (already FP8)

            o = flash_attn_varlen_fp8_pertensor_func(
                q=q_fp8,
                k=k_fp8,
                v=v_fp8,
                cu_seqlens_q=self.qo_indptr[:bs0],
                cu_seqlens_k=self.forward_metadata.kv_indptr[:bs0],
                max_seqlen_q=self.forward_metadata.max_q_len,
                max_seqlen_k=self.forward_metadata.max_kv_len,
                softmax_scale=layer.scaling,
                causal=True,
            )

            return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # Create o as 3D tensor [batch_size, num_heads, head_dim] for both MLA and pa_fwd_asm
        # In decode mode, q.shape[0] equals batch_size (each sequence has 1 token)
        # Use q.shape[0] instead of forward_batch.batch_size to be safe
        batch_size = q.shape[0]
        head_dim_out = layer.v_head_dim if layer.qk_head_dim != layer.v_head_dim else layer.head_dim
        o = q.new_empty((batch_size, layer.tp_q_head_num, head_dim_out))

        if save_kv_cache:
            if self.use_mla:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )
            else:
                # Shuffle operation is already fused in rotary_emb, so just save directly
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        if self.use_mla:
            k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            mla_decode_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer.view(-1, 1, 1, layer.qk_head_dim),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                self.forward_metadata.qo_indptr,
                self.forward_metadata.kv_indptr,
                self.forward_metadata.kv_indices,
                self.forward_metadata.kv_last_page_len,
                self.forward_metadata.max_q_len,
                layer.scaling,
                layer.logit_cap,
            )
            k_buffer = k_buffer.view(-1, 1, layer.qk_head_dim)
        else:
            # Use pa_fwd_asm for decode with shuffle layout (shuffle is fused in rotary_emb)
            import aiter
            k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

            # Use page_size as block_size (pa_fwd_asm expects block_size to match page_size)
            block_size = self.page_size
            block_size = 16
            
            num_slots, num_kv_heads, head_size = k_buffer.shape
            num_blocks = num_slots // block_size
            k_buffer = k_buffer[:num_blocks * block_size].view(num_blocks, block_size, num_kv_heads, head_size)
            v_buffer = v_buffer[:num_blocks * block_size].view(num_blocks, block_size, num_kv_heads, head_size)

            # x is the number of elements per 16-byte aligned chunk
            x = 16 // k_buffer.element_size()
            
            # Convert to shuffle layout for pa_fwd_asm
            # K: [num_blocks, block_size, num_kv_heads, head_size] -> [num_blocks, num_kv_heads, head_size // x, block_size, x]
            k_cache_template = torch.empty(
                [num_blocks, num_kv_heads, head_size // x, block_size, x],
                dtype=k_buffer.dtype,
                device="meta",
            )
            # V: [num_blocks, block_size, num_kv_heads, head_size] -> [num_blocks, num_kv_heads, block_size // x, head_size, x]
            v_cache_template = torch.empty(
                [num_blocks, num_kv_heads, block_size // x, head_size, x],
                dtype=v_buffer.dtype,
                device="meta",
            )
            
            new_key_cache = k_buffer.view_as(k_cache_template)
            new_value_cache = v_buffer.view_as(v_cache_template)
            
            # Reshape q to [batch_size, num_heads, head_dim] for pa_fwd_asm
            q = q.contiguous().view(batch_size, layer.tp_q_head_num, layer.head_dim)
            
            # o is already created as 3D tensor [batch_size, num_heads, head_dim]
            aiter.pa_fwd_asm(
                Q=q,
                K=new_key_cache,
                V=new_value_cache,
                block_tables=self.forward_metadata.page_table,
                context_lens=self.forward_metadata.kv_lens,
                block_tables_stride0=self.forward_metadata.page_table.stride(0),
                K_QScale=self.k_scale,
                V_QScale=self.v_scale,
                out_=o,
            )

        # Return o as 2D tensor [batch_size, num_heads * head_dim] to match other backends
        return o.view(-1, layer.tp_q_head_num * head_dim_out)


class AiterIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.update = self.update_single_wrapper

        self.kv_indices = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: Optional[List[int]],
        seq_lens_cpu: Optional[torch.Tensor],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: Optional[List[int]],
        seq_lens_cpu: Optional[torch.Tensor],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
    ):

        kv_start_idx = None
        kv_indptr = self.kv_indptr
        qo_indptr = self.qo_indptr
        paged_kernel_lens = seq_lens
        paged_kernel_lens_sum = seq_lens_sum

        bs = len(req_pool_indices)
        if spec_info is None:
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            # (TODO: Kk) WA - CI test_moe_eval_accuracy_large.py
            # mha_batch_prefill reads 128 data to do computatoin
            # if real data is not long enough then original padding value 0 is used
            # but the 0 location will be made nan (noqa) in cuda graph capture mode
            # this will cause the output tensor value becomes nan
            # WA is to assure that last index of pool not changed
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            if seq_lens_cpu is None:
                token_num = kv_indptr[-1]
                kv_indices[token_num:] = kv_indices[0]
            else:
                token_num = torch.cumsum(seq_lens_cpu, dim=0)[-1]
                kv_indices[token_num:] = kv_indices[0]

            if seq_lens_cpu is None:
                self.max_kv_len = torch.max(paged_kernel_lens).item()
            else:
                self.max_kv_len = torch.max(seq_lens_cpu).item()

            extend_lens = seq_lens - prefix_lens
            if seq_lens_cpu is not None and prefix_lens_cpu is not None:
                prefix_lens_cpu_torch = torch.tensor(prefix_lens_cpu)
                extend_lens_cpu = seq_lens_cpu - prefix_lens_cpu_torch
                self.max_q_len = torch.max(extend_lens_cpu).item()
            else:
                self.max_q_len = torch.max(extend_lens).item()

            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    self.req_to_token,
                )
            )

        self.kv_indices = kv_indices


class AiterMlaIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: AttentionBackend):
        # Parse Constants
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.update = self.update_single_wrapper

        self.kv_indptr = None
        self.kv_indices = None
        self.qo_indptr = None
        self.kv_last_page_len = None
        self.max_q_len = 0
        self.max_kv_len = 0

    def update(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_lens_sum: int,
        extend_lens: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        spec_info: Optional[SpecInput],
    ):
        bs = len(req_pool_indices)

        kv_indptr = self.attn_backend.kv_indptr

        if spec_info is None:
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_lens_sum,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                kv_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            qo_indptr = self.attn_backend.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(extend_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
        else:
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    kv_lens,
                    kv_lens_sum,
                    self.req_to_token,
                )
            )

        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.qo_indptr = qo_indptr
        self.max_q_len = max_q_len
        self.max_kv_len = max_kv_len


class AiterMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices

        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                AiterAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size
        assert self.page_size == 1, "Page size must be 1"

    def common_template(
        self, forward_batch: ForwardBatch, kv_indices_buffer: torch.Tensor, call_fn: int
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            triton.next_power_of_2(num_seqs),
            triton.next_power_of_2(self.speculative_num_steps),
            triton.next_power_of_2(bs),
            self.page_size,
        )

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_num_tokens * self.max_context_len),
            dtype=torch.int32,
            device=self.device,
        )
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                seq_lens_sum=-1,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=None,
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)
