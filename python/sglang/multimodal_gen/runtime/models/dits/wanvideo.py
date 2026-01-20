# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits import WanVideoConfig
from sglang.multimodal_gen.configs.sample.wan import WanTeaCacheParams
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_world_size,
    get_sp_parallel_rank,
    get_sp_group,
)
from sglang.multimodal_gen.runtime.layers.attention import (
    MinimalA2AAttnOp,
    SparseLinearAttention,
    UlyssesAttention_VSA,
    USPAttention,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    RMSNorm,
    ScaleResidual,
    ScaleResidualLayerNormScaleShift,
)
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    NDRotaryEmbedding,
    _apply_rotary_emb,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    ModulateProjection,
    PatchEmbed,
    TimestepEmbedder,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ============================================================================
# xDiT/Diffusers-style RoPE for WAN models
# ============================================================================

def apply_rotary_emb_wan(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings using xDiT/Diffusers style.
    
    Args:
        hidden_states: [batch, seq_len, num_heads, head_dim]
        freqs_cos: [1, seq_len, 1, head_dim] (with repeat_interleave)
        freqs_sin: [1, seq_len, 1, head_dim] (with repeat_interleave)
    
    Returns:
        Tensor with rotary embeddings applied
    """
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class WanRotaryPosEmbed(nn.Module):
    """
    xDiT/Diffusers-style Rotary position embeddings for WAN 3D video data.
    Lazy initialization to avoid meta device issues with FSDP.
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Lazy initialization - tensors will be created on first forward call
        self._freqs_cos = None
        self._freqs_sin = None
        self._initialized_device = None

    def _initialize_freqs(self, device: torch.device):
        """Lazily initialize frequency tensors on the target device."""
        if self._initialized_device == device:
            return
            
        # Split dimensions for temporal, height, width (same as Diffusers)
        h_dim = w_dim = 2 * (self.attention_head_dim // 6)
        t_dim = self.attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos_list = []
        freqs_sin_list = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = self._get_1d_rotary_pos_embed(dim, self.max_seq_len, self.theta, freqs_dtype, device)
            freqs_cos_list.append(freq_cos)
            freqs_sin_list.append(freq_sin)

        self._freqs_cos = torch.cat(freqs_cos_list, dim=1)
        self._freqs_sin = torch.cat(freqs_sin_list, dim=1)
        self._initialized_device = device

    @staticmethod
    def _get_1d_rotary_pos_embed(
        dim: int,
        max_seq_len: int,
        theta: float,
        freqs_dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate 1D rotary position embeddings with repeat_interleave."""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
        t = torch.arange(max_seq_len, dtype=freqs_dtype, device=device)
        freqs = torch.outer(t, freqs)
        # Repeat interleave for real representation (key difference from SGLang's NDRotaryEmbedding)
        freqs_cos = freqs.cos().repeat_interleave(2, dim=-1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=-1)
        return freqs_cos.float(), freqs_sin.float()

    def forward(
        self,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate 4D RoPE for the given video dimensions.
        
        Returns:
            freqs_cos: [1, seq_len, 1, head_dim]
            freqs_sin: [1, seq_len, 1, head_dim]
        """
        # Lazy initialization on first call
        self._initialize_freqs(device)

        p_t, p_h, p_w = self.patch_size
        ppf = num_frames // p_t
        pph = height // p_h
        ppw = width // p_w

        # Split sizes must match the dimension split in _initialize_freqs
        # h_dim = w_dim = 2 * (attention_head_dim // 6)
        # t_dim = attention_head_dim - h_dim - w_dim
        h_dim = w_dim = 2 * (self.attention_head_dim // 6)
        t_dim = self.attention_head_dim - h_dim - w_dim
        split_sizes = [t_dim, h_dim, w_dim]

        freqs_cos = self._freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self._freqs_sin.split(split_sizes, dim=1)

        # Expand each dimension
        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        # Concatenate and reshape to [1, seq_len, 1, head_dim]
        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin


class WanImageEmbedding(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = MLP(in_features, in_features, out_features, act_type="gelu")
        self.norm2 = FP32LayerNorm(out_features)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        dtype = encoder_hidden_states_image.dtype
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states).to(dtype)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
    ):
        super().__init__()

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu"
        )
        self.time_modulation = ModulateProjection(dim, factor=6, act_layer="silu")
        self.text_embedder = MLP(
            text_embed_dim, dim, dim, bias=True, act_type="gelu_pytorch_tanh"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        temb = self.time_embedder(timestep, timestep_seq_len)
        timestep_proj = self.time_modulation(temb)

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            assert self.image_embedder is not None
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanSelfAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        parallel_attention=False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention

        # layers
        self.to_q = ReplicatedLinear(dim, dim)
        self.to_k = ReplicatedLinear(dim, dim)
        self.to_v = ReplicatedLinear(dim, dim)
        self.to_out = ReplicatedLinear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Scaled dot product attention
        self.attn = USPAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_lens: int):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        pass


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
                v = self.to_v(context)[0].view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
            v = self.to_v(context)[0].view(b, -1, n, d)

        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        x, _ = self.to_out(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ) -> None:
        # VSA should not be in supported_attention_backends
        super().__init__(
            dim,
            num_heads,
            window_size,
            qk_norm,
            eps,
            supported_attention_backends=supported_attention_backends,
        )

        self.add_k_proj = ReplicatedLinear(dim, dim)
        self.add_v_proj = ReplicatedLinear(dim, dim)
        self.norm_added_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_added_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)
        k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
        v = self.to_v(context)[0].view(b, -1, n, d)
        k_img = self.norm_added_k(self.add_k_proj(context_img)[0]).view(b, -1, n, d)
        v_img = self.add_v_proj(context_img)[0].view(b, -1, n, d)
        img_x = self.attn(q, k_img, v_img)
        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x, _ = self.to_out(x)
        return x


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        attention_type: str = "original",
        sla_topk: float = 0.1,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        if attention_type == "sla":
            self.attn1 = MinimalA2AAttnOp(
                SparseLinearAttention(
                    dim // num_heads, topk=sla_topk, BLKQ=128, BLKK=64
                )
            )
        else:
            self.attn1 = USPAttention(
                num_heads=num_heads,
                head_size=dim // num_heads,
                causal=False,
                supported_attention_backends=supported_attention_backends,
                prefix=f"{prefix}.attn1",
            )

        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            logger.error("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )

        # 2. Cross-attention
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                supported_attention_backends=supported_attention_backends,
            )
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                supported_attention_backends=supported_attention_backends,
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        if temb.dim() == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            e = self.scale_shift_table + temb.float()
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                e.chunk(6, dim=1)
            )

        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm1 = self.norm1(hidden_states.float())
        norm_hidden_states = (norm1 * (1 + scale_msa) + shift_msa).to(orig_dtype)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        # Apply rotary embeddings (xDiT/Diffusers style)
        freqs_cos, freqs_sin = freqs_cis
        query = apply_rotary_emb_wan(query, freqs_cos, freqs_sin)
        key = apply_rotary_emb_wan(key, freqs_cos, freqs_sin)
        attn_output = self.attn1(query, key, value)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.zeros(
            (1,), device=hidden_states.device, dtype=hidden_states.dtype
        )
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(
            norm_hidden_states, context=encoder_hidden_states, context_lens=None
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class WanTransformerBlock_VSA(nn.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        self.to_gate_compress = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = UlyssesAttention_VSA(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1",
        )
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            logger.error("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )

        if AttentionBackendEnum.VIDEO_SPARSE_ATTN in supported_attention_backends:
            supported_attention_backends.remove(AttentionBackendEnum.VIDEO_SPARSE_ATTN)
        # 2. Cross-attention
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                supported_attention_backends=supported_attention_backends,
            )
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(
                dim,
                num_heads,
                qk_norm=qk_norm,
                eps=eps,
                supported_attention_backends=supported_attention_backends,
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        # assert orig_dtype != torch.float32
        e = self.scale_shift_table + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
            6, dim=1
        )
        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).to(orig_dtype)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        gate_compress, _ = self.to_gate_compress(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        gate_compress = gate_compress.squeeze(1).unflatten(
            2, (self.num_attention_heads, -1)
        )

        # Apply rotary embeddings (xDiT/Diffusers style)
        freqs_cos, freqs_sin = freqs_cis
        query = apply_rotary_emb_wan(query, freqs_cos, freqs_sin)
        key = apply_rotary_emb_wan(key, freqs_cos, freqs_sin)

        attn_output = self.attn1(query, key, value, gate_compress=gate_compress)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.zeros((1,), device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 2. Cross-attention
        attn_output = self.attn2(
            norm_hidden_states, context=encoder_hidden_states, context_lens=None
        )
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa
        )
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype
        ), hidden_states.to(orig_dtype)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class WanTransformer3DModel(CachableDiT, OffloadableDiTMixin):
    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig()._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Transformer blocks
        attn_backend = get_global_server_args().attention_backend
        transformer_block = (
            WanTransformerBlock_VSA
            if (attn_backend and attn_backend.lower() == "video_sparse_attn")
            else WanTransformerBlock
        )
        self.blocks = nn.ModuleList(
            [
                transformer_block(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                    self._supported_attention_backends
                    | {AttentionBackendEnum.VIDEO_SPARSE_ATTN},
                    prefix=f"{config.prefix}.blocks.{i}",
                    attention_type=config.attention_type,
                    sla_topk=config.sla_topk,
                )
                for i in range(config.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            norm_type="layer",
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size)
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        # For type checking
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.is_even = True
        self.should_calc_even = True
        self.should_calc_odd = True
        self.accumulated_rel_l1_distance_even = 0
        self.accumulated_rel_l1_distance_odd = 0
        self.cnt = 0
        self.__post_init__()

        # misc
        self.sp_size = get_sp_world_size()

        # Get rotary embeddings (xDiT/Diffusers style)
        d = self.hidden_size // self.num_attention_heads
        self.rotary_emb = WanRotaryPosEmbed(
            attention_head_dim=d,
            patch_size=self.patch_size,
            max_seq_len=1024,  # Same as Diffusers default
            theta=10000.0,
        )

        self.layer_names = ["blocks"]

    @torch.compiler.disable
    def _generate_rope_embeddings(
        self,
        num_frames: int,
        height: int,
        width: int,
        seq_pad_amount: int,
        sp_world_size: int,
        sp_rank: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate RoPE embeddings using xDiT/Diffusers style with padding and chunking
        for sequence parallelism.
        
        Disabled from torch.compile due to dynamic tensor creation in lazy initialization.
        
        Returns:
            freqs_cos, freqs_sin: [1, chunk_seq_len, 1, head_dim]
        """
        # Generate full RoPE using xDiT/Diffusers style
        # Output shape: [1, seq_len, 1, head_dim]
        freqs_cos, freqs_sin = self.rotary_emb(num_frames, height, width, device=device)

        # Pad RoPE with zeros (xDiT style - padding tokens get zero RoPE)
        # Shape after pad: [1, seq_len + pad, 1, head_dim]
        if seq_pad_amount > 0:
            freqs_cos = torch.cat([
                freqs_cos,
                torch.zeros(1, seq_pad_amount, 1, freqs_cos.shape[3], device=device, dtype=freqs_cos.dtype)
            ], dim=1)
            freqs_sin = torch.cat([
                freqs_sin,
                torch.zeros(1, seq_pad_amount, 1, freqs_sin.shape[3], device=device, dtype=freqs_sin.dtype)
            ], dim=1)

        # Chunk RoPE for SP along seq_len dimension
        # Shape after chunk: [1, chunk_seq_len, 1, head_dim]
        if sp_world_size > 1:
            freqs_cos = torch.chunk(freqs_cos, sp_world_size, dim=1)[sp_rank]
            freqs_sin = torch.chunk(freqs_sin, sp_world_size, dim=1)[sp_rank]

        return freqs_cos, freqs_sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        guidance=None,
        **kwargs,
    ) -> torch.Tensor:
        forward_batch = get_forward_context().forward_batch
        enable_teacache = forward_batch is not None and forward_batch.enable_teacache

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if (
            isinstance(encoder_hidden_states_image, list)
            and len(encoder_hidden_states_image) > 0
        ):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # === xDiT-style sequence parallel handling ===
        # Padding is done at seq_len level, NOT frame level
        sp_world_size = self.sp_size
        sp_rank = get_sp_parallel_rank() if sp_world_size > 1 else 0

        # Calculate original sequence length
        original_seq_len = post_patch_num_frames * post_patch_height * post_patch_width

        # Calculate padded sequence length (divisible by sp_world_size)
        if sp_world_size > 1 and original_seq_len % sp_world_size != 0:
            padded_seq_len = int(math.ceil(original_seq_len / sp_world_size)) * sp_world_size
            seq_pad_amount = padded_seq_len - original_seq_len
        else:
            padded_seq_len = original_seq_len
            seq_pad_amount = 0

        # Generate RoPE using xDiT/Diffusers style
        # Pass original dimensions (before patching), the method handles patch_size internally
        freqs_cis = self._generate_rope_embeddings(
            num_frames, height, width,
            seq_pad_amount, sp_world_size, sp_rank, hidden_states.device
        )

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Pad hidden_states at seq_len level (xDiT style)
        if seq_pad_amount > 0:
            hidden_states = torch.cat([
                hidden_states,
                torch.zeros(
                    batch_size, seq_pad_amount, hidden_states.shape[2],
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
            ], dim=1)

        # Chunk hidden_states for SP
        if sp_world_size > 1:
            hidden_states = torch.chunk(hidden_states, sp_world_size, dim=1)[sp_rank]

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.dim() == 2:
            # ti2v
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
            )
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))

            # For ti2v: pad and chunk temb and timestep_proj
            if sp_world_size > 1:
                # Pad temb: [batch_size, seq_len, inner_dim]
                if seq_pad_amount > 0:
                    temb = torch.cat([
                        temb,
                        torch.zeros(
                            batch_size, seq_pad_amount, temb.shape[2],
                            device=temb.device, dtype=temb.dtype
                        )
                    ], dim=1)
                    # Pad timestep_proj: [batch_size, seq_len, 6, inner_dim]
                    timestep_proj = torch.cat([
                        timestep_proj,
                        torch.zeros(
                            batch_size, seq_pad_amount, timestep_proj.shape[2], timestep_proj.shape[3],
                            device=timestep_proj.device, dtype=timestep_proj.dtype
                        )
                    ], dim=1)
                # Chunk temb and timestep_proj
                temb = torch.chunk(temb, sp_world_size, dim=1)[sp_rank]
                timestep_proj = torch.chunk(timestep_proj, sp_world_size, dim=1)[sp_rank]
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            # Wan2.1: cross attention with image embeddings - don't chunk
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )
        else:
            # Wan2.1 T2V or other modes: chunk encoder_hidden_states for SP
            if sp_world_size > 1:
                encoder_hidden_states = torch.chunk(
                    encoder_hidden_states, sp_world_size, dim=-2
                )[sp_rank]

        encoder_hidden_states = (
            encoder_hidden_states.to(orig_dtype)
            if current_platform.is_mps()
            else encoder_hidden_states
        )  # cast to orig_dtype for MPS

        assert encoder_hidden_states.dtype == orig_dtype

        # 4. Transformer blocks
        # if caching is enabled, we might be able to skip the forward pass
        should_skip_forward = self.should_skip_forward_for_cached_states(
            timestep_proj=timestep_proj, temb=temb
        )

        if should_skip_forward:
            hidden_states = self.retrieve_cached_states(hidden_states)
        else:
            # if teacache is enabled, we need to cache the original hidden states
            if enable_teacache:
                original_hidden_states = hidden_states.clone()

            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, freqs_cis
                )
            # if teacache is enabled, we need to cache the original hidden states
            if enable_teacache:
                self.maybe_cache_states(hidden_states, original_hidden_states)
        # 5. Output norm, projection & unpatchify
        if temb.dim() == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (
                self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        # === xDiT-style all_gather and unpad ===
        # Gather hidden_states from all SP ranks
        if sp_world_size > 1:
            sp_group = get_sp_group()
            hidden_states = sp_group.all_gather(hidden_states, dim=-2)

        # Remove padding to get back to original sequence length (xDiT style)
        # Use the product of post_patch dimensions as the target length
        original_seq_len = post_patch_num_frames * post_patch_height * post_patch_width
        hidden_states = hidden_states[:, :original_seq_len, :]

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        if self.is_even:
            self.previous_residual_even = (
                hidden_states.squeeze(0) - original_hidden_states
            )
        else:
            self.previous_residual_odd = (
                hidden_states.squeeze(0) - original_hidden_states
            )

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        if forward_batch is None or not forward_batch.enable_teacache:
            return False
        teacache_params = forward_batch.teacache_params
        assert teacache_params is not None, "teacache_params is not initialized"
        assert isinstance(
            teacache_params, WanTeaCacheParams
        ), "teacache_params is not a WanTeaCacheParams"
        current_timestep = forward_context.current_timestep
        num_inference_steps = forward_batch.num_inference_steps

        # initialize the coefficients, cutoff_steps, and ret_steps
        coefficients = teacache_params.coefficients
        use_ret_steps = teacache_params.use_ret_steps
        cutoff_steps = teacache_params.get_cutoff_steps(num_inference_steps)
        ret_steps = teacache_params.ret_steps
        teacache_thresh = teacache_params.teacache_thresh

        if current_timestep == 0:
            self.cnt = 0

        timestep_proj = kwargs["timestep_proj"]
        temb = kwargs["temb"]
        modulated_inp = timestep_proj if use_ret_steps else temb

        if self.cnt % 2 == 0:  # even -> condition
            self.is_even = True
            if self.cnt < ret_steps or self.cnt >= cutoff_steps:
                self.should_calc_even = True
                self.accumulated_rel_l1_distance_even = 0
            else:
                assert (
                    self.previous_e0_even is not None
                ), "previous_e0_even is not initialized"
                assert (
                    self.accumulated_rel_l1_distance_even is not None
                ), "accumulated_rel_l1_distance_even is not initialized"
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(
                    (
                        (modulated_inp - self.previous_e0_even).abs().mean()
                        / self.previous_e0_even.abs().mean()
                    )
                    .cpu()
                    .item()
                )
                if self.accumulated_rel_l1_distance_even < teacache_thresh:
                    self.should_calc_even = False
                else:
                    self.should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_e0_even = modulated_inp.clone()

        else:  # odd -> unconditon
            self.is_even = False
            if self.cnt < ret_steps or self.cnt >= cutoff_steps:
                self.should_calc_odd = True
                self.accumulated_rel_l1_distance_odd = 0
            else:
                assert (
                    self.previous_e0_odd is not None
                ), "previous_e0_odd is not initialized"
                assert (
                    self.accumulated_rel_l1_distance_odd is not None
                ), "accumulated_rel_l1_distance_odd is not initialized"
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(
                    (
                        (modulated_inp - self.previous_e0_odd).abs().mean()
                        / self.previous_e0_odd.abs().mean()
                    )
                    .cpu()
                    .item()
                )
                if self.accumulated_rel_l1_distance_odd < teacache_thresh:
                    self.should_calc_odd = False
                else:
                    self.should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_e0_odd = modulated_inp.clone()
        self.cnt += 1
        should_skip_forward = False
        if self.is_even:
            if not self.should_calc_even:
                should_skip_forward = True
        else:
            if not self.should_calc_odd:
                should_skip_forward = True

        return should_skip_forward

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.is_even:
            return hidden_states + self.previous_residual_even
        else:
            return hidden_states + self.previous_residual_odd


EntryClass = WanTransformer3DModel
