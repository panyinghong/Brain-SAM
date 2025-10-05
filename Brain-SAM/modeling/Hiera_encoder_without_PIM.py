# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Optional, List, Tuple, Union,Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything.modeling.common import LayerNorm2d, MLPBlock
from segment_anything.modeling.image_encoder import Attention, PatchEmbed, window_partition, window_unpartition

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W,D, C) -> (B, C, H, W,D)
    x = x.permute(0, 4, 1, 2,3)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3,4, 1)
    if norm:
        x = norm(x)

    return x




class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 96,
        num_heads: int = 1, 
        drop_path_rate: float = 0.1,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (1,2,7,2),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            5,
            7,
            9,
        ),
        return_interm_layers=True,  # return feats from every stage
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        # self.img_size = img_size
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers
        self.patch_embed = PatchEmbed3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed: Optional[nn.Parameter] = None


        self.pos_embed = nn.Parameter(
            torch.zeros(1, img_size // patch_size, img_size // patch_size, img_size // patch_size, embed_dim)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0],self.window_spec[0])
        )

        self.global_att_blocks = global_att_blocks

        cur_stage = 1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks=nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim

            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            self.blocks.append(block)
            


            embed_dim = dim_out

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )
    def _get_pos_embed(self, dhw: Tuple[int, int, int]) -> torch.Tensor:
        d, h, w = dhw 

        pos_embed = F.interpolate(self.pos_embed.permute(0,4,1,2,3), size=(d, h, w), mode="trilinear", align_corners=False)  

        window_embed = self.pos_embed_window 

        tile_shape = [x // y for x, y in zip(pos_embed.shape[2:], window_embed.shape[2:])]
        window_embed = window_embed.repeat(1, 1, *tile_shape)

        pos_embed = pos_embed + window_embed  
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1)  # [1, D, H, W, C]

        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self._get_pos_embed(x.shape[1:4])


        outputs = []
        for i, blk in enumerate(self.blocks):

            x = blk(x)

            if (i == self.stage_ends[-1] or i in self.stage_ends) and self.return_interm_layers:
                feats = x.permute(0, 4, 1, 2, 3)
                outputs.append(feats)    
        return outputs
    

class MultiScaleBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            dim_out:int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            drop_path: float = 0.0,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            q_stride: Tuple[int, int] = None,

            window_size: int = 0,

    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool3d(
                kernel_size=(q_stride[0], q_stride[0], q_stride[0]), stride=(q_stride[0], q_stride[0],q_stride[0]), ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,

        )


        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out,num_layers=2,activation=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        window_size=self.window_size

        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        if self.window_size > 0:
            H, W, D = x.shape[1], x.shape[2], x.shape[3]

            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        if self.q_stride:

            window_size = self.window_size // self.q_stride[0]
            H, W,D = shortcut.shape[1:4]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_d = (window_size - D % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w,D+pad_d)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W, D))

        B,H,W,D,C=x.shape[0],x.shape[1],x.shape[2],x.shape[3],x.shape[4]

        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MultiScaleAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
            self,
            dim: int,
            dim_out:int,
            num_heads: int = 8,
            q_pool: nn.Module = None,

    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads


        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)#self.qkv: 用于将输入特征映射到查询（Q）、键（K）和值（V）。它的输出维度是 dim * 3，因为查询、键和值是分开的。


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, _ = x.shape

        qkv = self.qkv(x).reshape(B, H * W * D, 3, self.num_heads, -1)

        q, k, v = torch.unbind(qkv, 2)


        if self.q_pool:
            q = do_pool(q.reshape(B, H, W,D ,-1), self.q_pool)
            H, W,D = q.shape[1:4]  # downsampled shape
            q = q.reshape(B, H * W*D, self.num_heads, -1)

        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        x = x.transpose(1, 2).reshape(B, H, W, D, -1)
        x = self.proj(x)




        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:

    B, H, W, D, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    pad_d = (window_size - D % window_size) % window_size
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, Dp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Hp, Wp, Dp)


def window_unpartition(
        windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int, int], hw: Tuple[int, int, int]
) -> torch.Tensor:

    Hp, Wp, Dp = pad_hw
    H, W, D = hw
    B = windows.shape[0] // (Hp * Wp * Dp // window_size // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, Dp // window_size, window_size, window_size, window_size,
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Hp, Wp, Dp, -1)

    if Hp > H or Wp > W or Dp > D:
        x = x[:, :H, :W, :D, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).
    Returns:
        Extracted positional embeddings according to relative positions.
    
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
        attn: torch.Tensor,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        rel_pos_d: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
        lr,
) -> torch.Tensor:

    q_h, q_w, q_d = q_size
    k_h, k_w, k_d = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    Rd = get_rel_pos(q_d, k_d, rel_pos_d)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, q_d, dim)
    rel_h = torch.einsum("bhwdc,hkc->bhwdk", r_q, Rh)
    rel_w = torch.einsum("bhwdc,wkc->bhwdk", r_q, Rw)
    rel_d = torch.einsum("bhwdc,dkc->bhwdk", r_q, Rd)

    attn = (
            attn.view(B, q_h, q_w, q_d, k_h, k_w, k_d) +
            lr * rel_h[:, :, :, :, :, None, None] +
            lr * rel_w[:, :, :, :, None, :, None] +
            lr * rel_d[:, :, :, :, None, None, :]
    ).view(B, q_h * q_w * q_d, k_h * k_w * k_d)

    return attn




class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class DropPath(nn.Module):
    # adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor  

class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16, 16),
            stride: Tuple[int, int] = (16, 16, 16),
            padding: Tuple[int, int] = (0, 0, 0),
            in_chans: int = 1,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv3d(in_chans,
                              embed_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C X Y Z -> B X Y Z C
        x = x.permute(0, 2, 3, 4, 1)
        return x


