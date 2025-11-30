import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, network_alpha: Optional[int] = None):
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features
        self.scale = network_alpha / rank if network_alpha is not None else 1.0 / rank
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_up(self.lora_down(x)) * self.scale


class IPAttnProcessor_mask2_0(nn.Module):
    r"""
    IP-Adapter attention processor with optional mask gating.

    Stage1: train IP projections (to_k_ip/to_v_ip), mask off by default.
    Stage2: freeze IP projections, enable mask branch only.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        rank: int = 8,
        network_alpha: Optional[int] = None,
        lora_scale: float = 1.0,
        scale_img: float = 1.0,
        scale_text: float = 1.0,
        num_tokens: int = 4,
        use_mask: bool = False,
        train_ip: bool = True,
        mask_clamp: Tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()

        self.rank = rank
        self.lora_scale = lora_scale
        self.num_tokens = num_tokens
        self.use_mask = use_mask
        self.mask_clamp = mask_clamp

        # mask branch (Stage2)
        self.to_k_ip_mask = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip_mask = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_out_ip_mask = nn.Linear(hidden_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale_img = scale_img
        self.scale_text = scale_text

        # IP branch (Stage1 trainable, Stage2 freeze)
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if not train_ip:
            self.freeze_ip()

        # LoRA hooks
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def freeze_ip(self) -> None:
        for module in (self.to_k_ip, self.to_v_ip):
            for param in module.parameters():
                param.requires_grad_(False)

    def freeze_mask(self) -> None:
        for module in (self.to_k_ip_mask, self.to_v_ip_mask, self.to_out_ip_mask):
            for param in module.parameters():
                param.requires_grad_(False)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, channel, height, width = hidden_states.shape[0], None, None, None

        use_ip = encoder_hidden_states is not None

        if not use_ip:
            encoder_hidden_states = hidden_states

        batch_size_enc, sequence_length, _ = encoder_hidden_states.shape

        if use_ip:
            if encoder_hidden_states.shape[1] < self.num_tokens:
                raise ValueError(
                    f"encoder_hidden_states length {encoder_hidden_states.shape[1]} < num_tokens={self.num_tokens}"
                )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length - self.num_tokens, batch_size_enc)

            # split text/cross and ip tokens
            end_pos = sequence_length - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            ip_tokens = ip_hidden_states
        else:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size_enc)
            ip_hidden_states = None

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + self.lora_scale * self.to_q_lora(hidden_states)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + self.lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + self.lora_scale * self.to_v_lora(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # ip branch
        if use_ip and ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)
        else:
            ip_hidden_states = torch.zeros_like(hidden_states)

        # mask branch
        if use_ip and self.use_mask:
            ip_key_mask = self.to_k_ip_mask(ip_tokens)
            ip_value_mask = self.to_v_ip_mask(ip_tokens)

            ip_key_mask = ip_key_mask.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value_mask = ip_value_mask.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            ip_hidden_states_mask = F.scaled_dot_product_attention(
                query, ip_key_mask, ip_value_mask, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            ip_hidden_states_mask = ip_hidden_states_mask.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

            ip_hidden_states_mask = self.to_out_ip_mask(ip_hidden_states_mask)
            ip_hidden_states_mask = self.sigmoid(ip_hidden_states_mask)
            ip_hidden_states_mask_out = ip_hidden_states_mask.repeat(1, 1, ip_hidden_states.shape[-1])
        else:
            ip_hidden_states_mask_out = 1

        hidden_states = self.scale_text * hidden_states + (self.scale_img * ip_hidden_states) * ip_hidden_states_mask_out

        hidden_states = attn.to_out[0](hidden_states) + self.lora_scale * self.to_out_lora(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
