# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import os
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sequence_parallel_world_size,
    get_sp_parallel_rank,
    get_sp_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
    ring_attn,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    ForwardContext,
    get_forward_context,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.utils import get_compute_dtype

# ---- DEBUG: force standard torch SDPA for precision debugging ----
# Set SGLANG_FORCE_TORCH_SDPA=1 to bypass aiter/flash_attn and use
# torch.nn.functional.scaled_dot_product_attention everywhere.
_FORCE_TORCH_SDPA = os.environ.get("SGLANG_FORCE_TORCH_SDPA", "0") == "1"


def _torch_sdpa_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Pure PyTorch manual attention for precision debugging.

    Uses individual torch ops (matmul, softmax, etc.) instead of
    F.scaled_dot_product_attention, which may internally dispatch to
    flash-attention or memory-efficient backends.

    Args:
        query/key/value: [B, S, H, D]
        softmax_scale: scaling factor for attention
        causal: whether to use causal masking
        dropout_p: dropout probability

    Returns:
        output: [B, S, H, D]
    """
    orig_dtype = query.dtype
    # [B, S, H, D] -> [B, H, S, D]
    q = query.transpose(1, 2).float()
    k = key.transpose(1, 2).float()
    v = value.transpose(1, 2).float()

    # Handle GQA: repeat KV heads to match Q heads
    if q.shape[1] != k.shape[1]:
        num_q_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        repeat_factor = num_q_heads // num_kv_heads
        # [B, H_kv, S, D] -> [B, H_q, S, D]
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    # attn_weights: [B, H, S_q, S_k]
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    # causal mask
    if causal:
        S_q, S_k = attn_weights.shape[-2], attn_weights.shape[-1]
        causal_mask = torch.triu(
            torch.ones(S_q, S_k, device=attn_weights.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = torch.softmax(attn_weights, dim=-1)

    if dropout_p > 0.0 and query.requires_grad:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # [B, H, S_q, D]
    output = torch.matmul(attn_weights, v)
    # [B, H, S, D] -> [B, S, H, D]
    return output.transpose(1, 2).to(orig_dtype)


class UlyssesAttention(nn.Module):
    """Ulysses-style SequenceParallelism attention layer."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale

        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(
            head_size, dtype, supported_attention_backends=supported_attention_backends
        )
        impl_cls = attn_backend.get_impl_cls()

        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.impl",
            **extra_impl_args,
        )
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = attn_backend.get_enum()
        self.dtype = dtype

    @torch.compiler.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for distributed attention.

        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"
        batch_size, seq_len, num_heads, head_dim = q.shape
        local_rank = get_sp_parallel_rank()
        world_size = get_sp_world_size()

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        # Stack QKV
        qkv = torch.cat([q, k, v], dim=0)  # [3, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
        # Apply backend-specific preprocess_qkv
        qkv = self.attn_impl.preprocess_qkv(qkv, ctx_attn_metadata)

        # Concatenate with replicated QKV if provided
        if replicated_q is not None:
            assert replicated_k is not None and replicated_v is not None
            replicated_qkv = torch.cat(
                [replicated_q, replicated_k, replicated_v], dim=0
            )  # [3, seq_len, num_heads, head_dim]
            heads_per_rank = num_heads // world_size
            replicated_qkv = replicated_qkv[
                :, :, local_rank * heads_per_rank : (local_rank + 1) * heads_per_rank
            ]
            qkv = torch.cat([qkv, replicated_qkv], dim=1)

        q, k, v = qkv.chunk(3, dim=0)

        if _FORCE_TORCH_SDPA:
            output = _torch_sdpa_forward(q, k, v, self.softmax_scale)
        else:
            output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)

        # Redistribute back if using sequence parallelism
        replicated_output = None
        if replicated_q is not None:
            replicated_output = output[:, seq_len * world_size :]
            output = output[:, : seq_len * world_size]
            # TODO: make this asynchronous
            replicated_output = sequence_model_parallel_all_gather(
                replicated_output.contiguous(), dim=2
            )
        # Apply backend-specific postprocess_output
        output = self.attn_impl.postprocess_output(output, ctx_attn_metadata)

        output = sequence_model_parallel_all_to_all_4D(
            output, scatter_dim=1, gather_dim=2
        )
        return output, replicated_output


class UlyssesAttention_VSA(UlyssesAttention):
    """Distributed attention layer with VSA support."""

    @torch.compiler.disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
        gate_compress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for distributed attention.

        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            gate_compress (torch.Tensor): Gate compress tensor [batch_size, seq_len, num_heads, head_dim]
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        # Check text tokens are not supported for VSA now
        assert (
            replicated_q is None and replicated_k is None and replicated_v is None
        ), "Replicated QKV is not supported for VSA now"
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        # Stack QKV
        qkvg = torch.cat(
            [q, k, v, gate_compress], dim=0
        )  # [3, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        qkvg = sequence_model_parallel_all_to_all_4D(qkvg, scatter_dim=2, gather_dim=1)

        qkvg = self.attn_impl.preprocess_qkv(qkvg, ctx_attn_metadata)

        q, k, v, gate_compress = qkvg.chunk(4, dim=0)
        if _FORCE_TORCH_SDPA:
            output = _torch_sdpa_forward(q, k, v, self.softmax_scale)
        else:
            output = self.attn_impl.forward(
                q, k, v, gate_compress=gate_compress, attn_metadata=ctx_attn_metadata
            )  # type: ignore[call-arg]

        # Apply backend-specific postprocess_output
        output = self.attn_impl.postprocess_output(output, ctx_attn_metadata)

        output = sequence_model_parallel_all_to_all_4D(
            output, scatter_dim=1, gather_dim=2
        )

        return output


class LocalAttention(nn.Module):
    """Attention layer."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale
        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(
            head_size, dtype, supported_attention_backends=supported_attention_backends
        )
        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=self.softmax_scale,
            num_kv_heads=num_kv_heads,
            causal=causal,
            **extra_impl_args,
        )
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = attn_backend.get_enum()
        self.dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply local attention between query, key and value tensors.

        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim]

        Returns:
            torch.Tensor: Output tensor after local attention
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        if _FORCE_TORCH_SDPA:
            output = _torch_sdpa_forward(q, k, v, self.softmax_scale)
        else:
            output = self.attn_impl.forward(q, k, v, attn_metadata=ctx_attn_metadata)
        return output


class USPAttention(nn.Module):
    """
    Ulysses Sequence Parallelism with Ring Attention.

    This class implements the USP algorithm, which is a combination of
    Ulysses-style all-to-all communication for sequence-head dimension sharding
    and Ring Attention for fine-grained sequence parallelism within subgroups.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        dropout_rate: float = 0.0,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale

        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(
            head_size, dtype, supported_attention_backends=supported_attention_backends
        )
        impl_cls: Type["AttentionImpl"] = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.impl",
            **extra_impl_args,
        )
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = attn_backend.get_enum()
        self.dtype = dtype
        self.causal = causal
        self.dropout_p = dropout_rate

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for USPAttention.

            q, k, v: [B, S_local, H, D]

        Note: Replicated tensors are not supported in this implementation.
        """
        assert (
            replicated_q is None and replicated_k is None and replicated_v is None
        ), "USPAttention does not support replicated_qkv."
        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata
        if get_sequence_parallel_world_size() == 1:
            # No sequence parallelism, just run local attention.
            if _FORCE_TORCH_SDPA:
                out = _torch_sdpa_forward(
                    q, k, v, self.softmax_scale, self.causal, self.dropout_p
                )
            else:
                out = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
            return out

        # Ulysses-style All-to-All for sequence/head sharding
        if get_ulysses_parallel_world_size() > 1:
            # -> [B, S, H_local, D]
            q = _usp_input_all_to_all(q, head_dim=2)
            k = _usp_input_all_to_all(k, head_dim=2)
            v = _usp_input_all_to_all(v, head_dim=2)

        # Ring Attention within subgroups or local attention
        if get_ring_parallel_world_size() > 1:
            if _FORCE_TORCH_SDPA:
                out = _torch_sdpa_forward(
                    q, k, v, self.softmax_scale, self.causal, self.dropout_p
                )
            else:
                out = ring_attn(
                    q,
                    k,
                    v,
                    attn_impl=self.attn_impl,
                    is_causal=self.causal,
                    dropout_p=self.dropout_p,
                )
        else:
            # -> [B, S, H_local, D]
            if _FORCE_TORCH_SDPA:
                out = _torch_sdpa_forward(
                    q, k, v, self.softmax_scale, self.causal, self.dropout_p
                )
            else:
                out = self.attn_impl.forward(q, k, v, ctx_attn_metadata)

        # Ulysses-style All-to-All to restore original sharding
        if get_ulysses_parallel_world_size() > 1:
            # -> [B, S_local, H, D]
            out = _usp_output_all_to_all(out, head_dim=2)

        return out
