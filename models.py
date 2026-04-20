# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.nn.attention.varlen import varlen_attn
from functools import partial


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        # prevent explosion by projecting onto hypersphere
        # TODO: temperature?
        self.q_norm = nn.RMSNorm(head_dim, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(head_dim, elementwise_affine=False)

        assert hidden_size % num_heads == 0
        return

    def forward(self, x, cond_tokens, attn_mask):
        N, T, D = x.shape
        B, T_, D_ = cond_tokens.shape # B is total number of attended frames (packed among all images in batch, not necessarily even)

        assert B >= N and B == torch.sum(attn_mask).item()
        assert T == T_
        assert D == D_

        L = attn_mask.shape[-1]

        assert attn_mask.shape == (N, L)
        assert D % self.num_heads == 0

        q = self.q(x)
        k = self.k(cond_tokens)
        v = self.v(cond_tokens)

        assert q.shape == (N, T, D)
        assert k.shape == v.shape == (B, T, D)

        query = q.reshape(N * T, self.num_heads, D // self.num_heads)
        assert (query[:T].view(T, D) == q[0]).all()
        cu_seq_q = torch.arange(N + 1, device=x.device, dtype=torch.int32) * T
        max_q = T

        key = k.reshape(B * T, self.num_heads, D // self.num_heads)
        value = v.reshape(B * T, self.num_heads, D // self.num_heads)
        # TODO: debug statements if this goes wrong

        kv_seq_len = attn_mask.sum(dim=-1) * T # T tokens per frame (and each query frame attends to multiple frames)
        assert (kv_seq_len >= T).all() and kv_seq_len.shape == (N,)
        cu_seq_kv = torch.zeros(N + 1, device=x.device, dtype=torch.int32)
        cu_seq_kv[1:] = kv_seq_len.cumsum(dim=0)
        assert cu_seq_kv[-1] == B * T
        max_kv = kv_seq_len.max().item()
        assert max_kv >= max_q

        # TODO: investigate hardcoded scale if this is still unstable
        out = varlen_attn(
            query=self.q_norm(query),
            key=self.k_norm(key),
            value=value,
            cu_seq_q=cu_seq_q,
            cu_seq_k=cu_seq_kv,
            max_q=max_q,
            max_k=max_kv,
            # scale=self.scale,
        )
        assert out.shape == (N * T, self.num_heads, D // self.num_heads)
        out = out.reshape(N, T, D)

        return self.proj(out)

class DiTBlockWithCrossAttention(DiTBlock):
    """
    An (inherited) DiT block with adaptive layer norm zero (adaLN-Zero) conditioning and cross attention.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True) # TODO: **kwargs? dropout?
        self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 12 * hidden_size, bias=True)
        )
        return

    def forward(self, x, c, cond_tokens, cond_attn_mask):
        shift_msa, scale_msa, gate_msa, shift_xa, scale_xa, gate_xa, shift_cond, scale_cond, gate_cond, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(12, dim=1)

        assert isinstance(self.attn.q_norm, nn.RMSNorm)
        assert isinstance(self.attn.k_norm, nn.RMSNorm)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        N, T, D = x.shape
        B, T_, D_ = cond_tokens.shape
        L = cond_attn_mask.shape[-1]

        assert T == T_ and D == D_ 
        assert B >= N and B == torch.sum(cond_attn_mask).item()
        assert cond_attn_mask.dtype == torch.bool
        assert shift_cond.shape == scale_cond.shape == (N, D)
        assert cond_attn_mask.shape == (N, L)

        shift_cond = shift_cond.unsqueeze(1).expand(-1, L, -1)[cond_attn_mask]
        scale_cond = scale_cond.unsqueeze(1).expand(-1, L, -1)[cond_attn_mask]
        
        x = x + gate_xa.unsqueeze(1) * self.cross_attn(
            modulate(self.norm_cross(x), shift_xa, scale_xa), 
            modulate(self.norm_cond(cond_tokens), shift_cond, scale_cond), 
            attn_mask=cond_attn_mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiTWithCrossAttention(DiT):
    """
    A DiT model (inherited) with cross attention.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        sde=None,
    ):
        super().__init__(
            input_size=input_size, patch_size=patch_size, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, class_dropout_prob=class_dropout_prob, num_classes=num_classes,
            learn_sigma=False,
        )
        assert not self.learn_sigma, "DiTWithCrossAttention must be initialized with learn_sigma=False"

        # have separate patch and timestep embedders for the conditioning images
        self.cond_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        cond_num_patches = self.cond_embedder.num_patches
        # separate cond_pos_embed not necessary, as positional embeddings are fixed
        self.cond_t_embedder = TimestepEmbedder(hidden_size)

        # destination timestep embedder (tells us when the next waypoint is)
        self.next_t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlockWithCrossAttention(
                hidden_size=hidden_size, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                qk_norm=True,
                norm_layer=partial(nn.RMSNorm, elementwise_affine=False)
            ) for i in range(depth)
        ])

        self.initialize_weights() # call initialization again to initialize the cross attention blocks

        self.time_scale = 1000 # we have timesteps between 0 and 1, but we want to embed them in the desired range (i.e., 0 to 1000)

        self.sde = sde

        return
    
    def initialize_weights(self):
        super().initialize_weights() # this takes care of the blocks
        if not hasattr(self, 'cond_embedder'):
            return

        # Initialize next timestep embedding MLP:
        nn.init.normal_(self.next_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.next_t_embedder.mlp[2].weight, std=0.02)

        # Initialize conditioning's timestep embedding MLP:
        nn.init.normal_(self.cond_t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.cond_t_embedder.mlp[2].weight, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.cond_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.cond_embedder.proj.bias, 0)

        print("Initialized conditioning embedders")

        return
    
    def get_brownian_bridge_drift(self, x_t, t, cond_times, cond_masks, cond_images):
        """
        Get the Brownian bridge drift term for the next pinning time. return zero if there are no valid pinning times.
        """
        B, C, H, W = x_t.shape
        L = cond_images.shape[1]
        assert cond_images.shape == (B, L, C, H, W)
        assert t.shape == (B,)
        assert cond_times.shape == cond_masks.shape == (B, L)
        assert C in (3, 4)
        assert H == W
        
        is_valid_pinning_time = (cond_times > t.unsqueeze(1)) & cond_masks
        has_valid_pinning_times = torch.any(is_valid_pinning_time, dim=1)
        assert is_valid_pinning_time.shape == (B, L)
        assert has_valid_pinning_times.shape == (B,)

        pinning_times = cond_times.clone()
        pinning_times[~is_valid_pinning_time] = float('inf') # cannot select these
        
        next_pinning_time = pinning_times.min(dim=1).values
        next_pinning_idx = pinning_times.argmin(dim=1)
        assert next_pinning_time.shape == (B,)
        assert next_pinning_idx.shape == (B,)

        next_pinning_time[~has_valid_pinning_times] = 1 + 1e-4 # avoid numerical instability, in the case that there are no valid pinning times

        next_pinning_idx = next_pinning_idx.reshape(B, 1, 1, 1, 1).expand(-1, 1, C, H, W)
        next_pinning_state = cond_images.gather(dim=1, index=next_pinning_idx).squeeze(1)
        assert next_pinning_state.shape == (B, C, H, W) == x_t.shape
        assert self.sde.A == 0, "A must be 0 for Brownian bridge drift"
        numerator = next_pinning_state - x_t
        denominator = self.sde.C(start=t, t_a=next_pinning_time, t_b=next_pinning_time) + 1e-7
        denominator = denominator.reshape(B, 1, 1, 1)
        drift = numerator / denominator
        # NOTE: this code assumes that phi = 1

        drift[~has_valid_pinning_times] = 0.0

        assert drift.shape == (B, C, H, W)
        assert drift.isnan().sum() == 0
        assert -math.ceil(0.1 * B) <= (
            torch.count_nonzero(drift.abs().sum(dim=(1, 2, 3))) \
            - torch.sum(has_valid_pinning_times)
        ).item() <= 0

        return drift

    def forward(self, x, t, t_next, y, cond_images, cond_times, cond_masks):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps: must be between 0 and 1
        t_next: (N,) tensor of diffusion timesteps of the next waypoint in the sequence: must be between 0 and 1
        y: (N,) tensor of class labels
        cond_images: (N, max_cond_images, C, H, W) tensors of conditional images
        cond_times: (N, max_cond_images) tensors of diffusion timesteps corresponding to cond_images: must be between 0 and 1
        cond_masks: (N, max_cond_images) tensors of masks corresponding to cond_images. cond_masks[i, j] = 1 means that x[i] should condition on cond_images[i, ..., j]
        """
        orig_input_x_t = x.clone()
        orig_input_t = t.clone()

        N, L, C, H, W = cond_images.shape
        assert x.shape == (N, C, H, W)
        assert C in (3, 4)
        assert H == W
        assert cond_times.shape == (N, L)
        assert cond_masks.shape == (N, L)

        assert torch.all(t < 1.0) and torch.all(t_next <= 1.0) and torch.all(cond_times <= 1.0), "Timesteps must be between 0 and 1"
        assert torch.all(t >= 0.0) and torch.all(t_next > 0.0) and torch.all(cond_times >= 0.0), "Timesteps must be non-negative"
        # assert torch.all(t_next > t), "t_next must be greater than t"
        assert t.dtype == t_next.dtype == cond_times.dtype == torch.float32
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        # TODO: need to adjust this due to numerical precision
        t = self.t_embedder(t * self.time_scale)                   # (N, D)
        t_next = self.next_t_embedder(t_next * self.time_scale)    # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = (t + t_next) / 2 + y                 # (N, D)

        assert torch.all(torch.sum(cond_masks, dim=-1) > 0), "At least one mask should be 1 for each image"
        assert cond_masks[:, 0].all(), "The first frame should always be visible"

        assert cond_masks.dtype == torch.bool
        valid = cond_masks.bool()
        cond_images_packed = cond_images[valid]      # (B, C, H, W)
        cond_times_packed = cond_times[valid]        # (B,)
        B = cond_images_packed.shape[0]

        assert B == torch.sum(valid).item() and B >= N
        assert cond_images_packed.shape == (B, C, H, W)
        assert cond_times_packed.shape == (B,)

        cond_patch_embeddings = self.cond_embedder(cond_images_packed)  # (B, T, D)
        cond_times_embeddings = self.cond_t_embedder(cond_times_packed * self.time_scale)
        cond_tokens = cond_patch_embeddings + self.pos_embed + cond_times_embeddings.unsqueeze(1) # needs space and time info

        B_, T, D = cond_patch_embeddings.shape
        assert B_ == B
        assert cond_times_embeddings.shape == (B, D)
        assert cond_tokens.shape == (B, T, D)
        assert x.shape == (N, T, D)

        # TODO: add back sanity check

        for block_idx, block in enumerate(self.blocks):
            x = block(x, c, cond_tokens, cond_masks)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.sde is not None:
            brownian_bridge_drift = self.get_brownian_bridge_drift(
                x_t=orig_input_x_t, # use original input
                t=orig_input_t, # use un-embedded timestep
                cond_times=cond_times, 
                cond_masks=cond_masks, 
                cond_images=cond_images,
            )
            if (torch.rand(1).item() < 0.005) \
            or ((not self.training) and int(orig_input_t[0].item() * 1000) % 25 == 0): # print in inference mode
                print(f"t: {orig_input_t[0].item() * 1000}")
                print(f"\tx magnitude^2: {(x**2).mean()}, brownian_bridge_drift magnitude^2: {(brownian_bridge_drift**2).mean()}")
            x = x + brownian_bridge_drift
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        raise NotImplementedError("Not implemented yet")


class LogvarNet(nn.Module):
    def __init__(self, seq_len, hidden_size):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.curr_t_embedder = TimestepEmbedder(hidden_size=hidden_size)
        self.next_t_embedder = TimestepEmbedder(hidden_size=hidden_size)

        self.time_scale = 1000 # we have timesteps between 0 and 1, but we want to embed them in the desired range (i.e., 0 to 1000)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + seq_len, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, t, t_next, cond_masks):
        B = cond_masks.shape[0]
        assert t.shape == (B,)
        assert t_next.shape == (B,)
        assert cond_masks.shape == (B, self.seq_len)
        assert cond_masks.dtype == torch.bool

        cond_masks = cond_masks.to(t.dtype)
        
        t = self.curr_t_embedder(t * self.time_scale)
        t_next = self.next_t_embedder(t_next * self.time_scale)

        assert t.shape == t_next.shape == (B, self.hidden_size)

        x = torch.cat([t, t_next, cond_masks], dim=-1)
        x = self.mlp(x)
        return torch.clamp(x, min=-7.5, max=7.5)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

# DiT Cross Attention Models
def DiTXA_XL_2(**kwargs):
    return DiTWithCrossAttention(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiTXA_B_2(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiTXA_B_4(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiTXA_B_8(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiTXA_S_2(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiTXA_S_4(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiTXA_S_8(**kwargs):
    return DiTWithCrossAttention(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiTXA-B/2': DiTXA_B_2, 'DiTXA-B/4': DiTXA_B_4, 'DiTXA-B/8': DiTXA_B_8,
    'DiTXA-S/2': DiTXA_S_2, 'DiTXA-S/4': DiTXA_S_4, 'DiTXA-S/8': DiTXA_S_8,
    'DiTXA-XL/2': DiTXA_XL_2,
}