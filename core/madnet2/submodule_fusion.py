"""
From https://github.com/JiaRenChang/PSMNet
Licensed under MIT
"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch import Tensor
import torch.utils.data
from typing import Optional

from core.madnet2.attention import MultiheadAttentionRelative

from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def conv2d(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation if dilation > 1 else pad,
            dilation=dilation,
            bias=True,
        )
    )

class guidance_encoder(nn.Module):
    def __init__(self, img_w=1280):
        super(guidance_encoder, self).__init__()


        self.block1 = nn.Sequential(
            conv2d(1, 64, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(64, 64, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.block2 = nn.Sequential(
            conv2d(64, 128, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(128, 128, 3, 1, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.pool2x = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(128, 5, kernel_size=1)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 5, kernel_size=1)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(128, 5, kernel_size=1)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(128, 5, kernel_size=1)
        )

        self.conv_6 = nn.Sequential(
            nn.Conv2d(128, 5, kernel_size=1)
        )

    def forward(self, x, mad=False):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out2_ = self.conv_2(out2)

        out3 = self.pool2x(out2)
        out3_ = self.conv_3(out3)

        out4 = self.pool2x(out3)
        out4_ = self.conv_4(out4)

        out5 = self.pool2x(out4)
        out5_ = self.conv_5(out5)

        out6 = self.pool2x(out5)
        out6_ = self.conv_6(out6)

        return x, out1, out2_, out3_, out4_, out5_, out6_

class guidance_encoder_small(nn.Module):
    def __init__(self):
        super(guidance_encoder_small, self).__init__()

        self.block1 = nn.Sequential(
            conv2d(1, 32, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(32, 64, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block2 = nn.Sequential(
            conv2d(64, 96, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(96, 96, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.block3 = nn.Sequential(
            conv2d(96, 128, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(128, 128, 3, 2, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv2d(128, 20, 1, 1, 0, 1),

        )


        # self.block1 = nn.Sequential(
        #     conv2d(1, 64, 3, 4, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     conv2d(64, 64, 3, 1, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.MaxPool2d(3,2,1),
        # )
        #
        # self.block2 = nn.Sequential(
        #     conv2d(64, 192, 3, 4, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     conv2d(192, 192, 3, 1, 1, 1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.MaxPool2d(3, 2,1),
        #
        # )

    def forward(self, x, mad=False):
        out1 = self.block1(x)
        out2 = self.block2(out1 if not mad else out1.detach())
        out = self.block3(out2 if not mad else out1.detach())



        return out
class fusion_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(fusion_block, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            # conv2d(in_channels, out_channels, 1, 1, 1, 1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # conv2d(16, 192*2, 1, 1, 1, 1),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )


    def forward(self, x):
        out = self.block1(x)

        return out

class TransformerCrossAttnLayer(nn.Module):
    """
    Cross attention layer
    """

    def __init__(self, hidden_dim: int, nhead: int):
        super().__init__()
        self.cross_attn = MultiheadAttentionRelative(hidden_dim, nhead)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, feat_left: Tensor, feat_right: Tensor,
                pos: Optional[Tensor] = None,
                pos_indexes: Optional[Tensor] = None,
                last_layer: Optional[bool] = False):
        """
        :param feat_left: left image feature, [W,HN,C]
        :param feat_right: right image feature, [W,HN,C]
        :param pos: pos encoding, [2W-1,HN,C]
        :param pos_indexes: indexes to slicer pos encoding [W,W]
        :param last_layer: Boolean indicating if the current layer is the last layer
        :return: update image feature and attention weight
        """
        feat_left_2 = self.norm1(feat_left)
        feat_right_2 = self.norm1(feat_right)

        # torch.save(torch.cat([feat_left_2, feat_right_2], dim=1), 'feat_cross_attn_input_' + str(layer_idx) + '.dat')

        # # update right features
        # if pos is not None:
        #     pos_flipped = torch.flip(pos, [0])
        # else:
        #     pos_flipped = pos
        # feat_right_2 = self.cross_attn(query=feat_right_2, key=feat_left_2, value=feat_left_2, pos_enc=pos_flipped,
        #                                pos_indexes=pos_indexes)[0]
        #
        # feat_right = feat_right + feat_right_2

        # update left features
        # use attn mask for last layer
        if last_layer:
            w = feat_left_2.size(0)
            attn_mask = self._generate_square_subsequent_mask(w).to(feat_left.device)  # generate attn mask
        else:
            attn_mask = None

        # # normalize again the updated right features
        # feat_right_2 = self.norm2(feat_right)
        feat_left_2, attn_weight, raw_attn = self.cross_attn(query=feat_left_2, key=feat_right_2, value=feat_right_2,
                                                             attn_mask=attn_mask, pos_enc=pos,
                                                             pos_indexes=pos_indexes)

        # torch.save(attn_weight, 'cross_attn_' + str(layer_idx) + '.dat')

        feat_left = feat_left + feat_left_2

        # # concat features
        # feat = torch.cat([feat_left, feat_right], dim=1)  # Wx2HNxC

        return feat_left, raw_attn