# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets import ViT
# from modeling.unet import Unet_decoder, Conv, TwoConv
from .unet import Unet_decoder, Conv, TwoConv

class UpBlockConvTranspose3d(nn.Module):
    def __init__(self, in_channels=48, out_channels=6):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,     # 放大2倍
            stride=2,
            padding=1
        )
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)       # (B, 6, 128, 128, 128)
        x = self.norm(x)
        x = self.relu(x)
        return x


    


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x

class new_UNETR_Decoder_combine_auto_prompt(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        multiple_outputs: bool = False,
        num_multiple_outputs: int = 0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.hidden_size = hidden_size
        self.classification = False
        self.multiple_outputs = multiple_outputs
        self.num_multiple_outputs = num_multiple_outputs
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(768, 768, 32, 3) for i in range(num_multiple_outputs + 1)])
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=48,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=768,
            out_channels=384,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=384,
            out_channels=192,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            # dropout=0
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=192,
            out_channels=96,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=96,
            out_channels=48,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UpBlockConvTranspose3d(48, 6)
        self.out1= UnetOutBlock(spatial_dims=3, in_channels=6, out_channels=32)  # type: ignore
        self.out = UnetOutBlock(spatial_dims=3, in_channels=6, out_channels=2) 



    def forward(self, x_in,prompt_embeddings, prompt_logits=None,prev_masks=None):
        if prompt_logits is None:
            enc1 = self.encoder1(x_in[4])
            dec3 = self.decoder5(x_in[3], x_in[2])
            dec2 = self.decoder4(dec3, x_in[1])
            dec1 = self.decoder3(dec2, x_in[0])
            out = self.decoder2(dec1, enc1)
            dec_out = self.decoder1(out) 
            masks_auto=self.out(dec_out)
            prompt_logits = self.out1(dec_out)
            masks_prompt, iou_pred = self._predict_mask(prompt_logits, prompt_embeddings)
            return masks_prompt,masks_auto,prompt_logits
            # return masks_auto,logits
        else:
            if prev_masks is not None:
                # prev_masks= self.encoder2(prev_masks)
                masks_prompt, iou_pred = self._predict_mask(prev_masks, prompt_embeddings)
            else:
                masks_prompt, iou_pred = self._predict_mask(prompt_logits, prompt_embeddings)
            return masks_prompt,prompt_logits

    
    
    def _predict_mask(self, upscaled_embedding, prompt_embeddings):
        b, c, x, y, z = upscaled_embedding.shape
        mask_token_out = prompt_embeddings[:, 1, :]          # (B, C)

        hyper_in = self.output_hypernetworks_mlps[0](mask_token_out)  # (B, C_out)
        hyper_in = hyper_in.unsqueeze(1)  


        masks = (hyper_in @ upscaled_embedding.view(b, c, x * y * z)).view(b, 1, x, y, z)

        iou_pred=None

        return masks,iou_pred
