import torch
import torch.nn as nn
import torch.nn.functional as F
from .corr import CorrBlock1D
from .submodule import *
from .submodule_fusion import *
from ..losses import *

from .madnet2 import MADNet2

class MADNet2Fusion(MADNet2):
    def __init__(self, args,  hidden_dim: int = 128, nhead: int = 1):
        super().__init__(args)

        # self.guidance_encoder = guidance_encoder()
        # self.fusion_block6 = fusion_block(5+192+192, 5+192)
        # self.fusion_block5 = fusion_block(5+128+1+128, 5+128+1)
        # self.fusion_block4 = fusion_block(5+96+1+96,5+96+1)
        # self.fusion_block3 = fusion_block(5+64+1+64,5+64+1)
        # self.fusion_block2 = fusion_block(5+32+1+32,5+32+1)
        w = args.image_size[1]
        pad_wd = (((w // 128) + 1) * 128 - w) % 128

        self.guidance_encoder = guidance_encoder()
        # self.guidance_encoder_small = guidance_encoder_small()
        # self.fusion_block = fusion_block(5+192+32, 5+192)


        self.cross_attn_layer_2 = TransformerCrossAttnLayer(hidden_dim=5, nhead=nhead)
        self.cross_attn_layer_3 = TransformerCrossAttnLayer(hidden_dim=5, nhead=nhead)
        self.cross_attn_layer_4 = TransformerCrossAttnLayer(hidden_dim=5, nhead=nhead)
        self.cross_attn_layer_5 = TransformerCrossAttnLayer(hidden_dim=5, nhead=nhead)
        self.cross_attn_layer_6 = TransformerCrossAttnLayer(hidden_dim=5, nhead=nhead)



    def forward(self, image2, image3, guide):
        """ Estimate optical flow between pair of frames """

        im2_fea = self.feature_extraction(image2)
        im3_fea = self.feature_extraction(image3)

        guide_fea = self.guidance_encoder(guide)
        guide_fea = [guide_fea[i].permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0) for i in range(len(guide_fea))]
        # guide_fea = self.guidance_encoder_small(guide)
        # guide_fea = guide_fea.permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)


        corr_block = CorrBlock1D

        corr_fn6 = corr_block(im2_fea[6], im3_fea[6], radius=2, num_levels=1)
        corr_fn5 = corr_block(im2_fea[5], im3_fea[5], radius=2, num_levels=1)
        corr_fn4 = corr_block(im2_fea[4], im3_fea[4], radius=2, num_levels=1)
        corr_fn3 = corr_block(im2_fea[3], im3_fea[3], radius=2, num_levels=1)
        corr_fn2 = corr_block(im2_fea[2], im3_fea[2], radius=2, num_levels=1)

        coords0, coords1_6 = self.initialize_flow(im2_fea[6])
        coords0, coords1_5 = self.initialize_flow(im2_fea[5])
        coords0, coords1_4 = self.initialize_flow(im2_fea[4])
        coords0, coords1_3 = self.initialize_flow(im2_fea[3])
        coords0, coords1_2 = self.initialize_flow(im2_fea[2])

        ## cross-attention with img fea and sgm fea
        # feat_left = im2_fea[5].clone().permute(1, 3, 2, 0).flatten(2).permute(1, 2, 0)  # CxWxHxN -> CxWxHN -> WxHNxC
        # fusion_fea, fusion_attn = self.cross_attn_layer(feat_left, guide_fea)
        # fusion_fea = fusion_fea.unsqueeze(0).permute(0,3,2,1)


        ## original correlation
        # corr6 = corr_fn6(coords1_6)
        ## correlation cross-attention fusion
        corr6 = corr_fn6(coords1_6, guide=guide_fea[6], cross_attn_layer=self.cross_attn_layer_6)

        # fusion6 = self.fusion_block6(torch.cat((im2_fea[6], corr6, guide_fea[6]),1))
        # fusion6 = self.fusion_block(torch.cat((im2_fea[6], corr6, guide_fea),1))
        # fusion6 = torch.cat((im2_fea[6]+guide_fea,corr6),1)
        # fusion6 = torch.cat((im2_fea[6]+guide_fea[6],corr6),1)
        # fusion6 = torch.cat((fusion_fea,corr6),1)
        # disp6 = self.decoder6(fusion6)
        disp6 = self.decoder6(torch.cat((im2_fea[6], corr6), 1))

        disp6_u = F.interpolate(disp6, scale_factor=2) * 20. / 32

        coords1_5 = coords1_5 + disp6_u
        # original correlation
        # corr5 = corr_fn5(coords1_5)
        # ##correlation cross-attention fusion
        corr5 = corr_fn5(coords1_5, guide=guide_fea[5], cross_attn_layer=self.cross_attn_layer_5)

        # fusion5 = self.fusion_block5(torch.cat((im2_fea[5], corr5, disp6_u, guide_fea[5]),1))
        # fusion5 = torch.cat((im2_fea[5]+guide_fea[5],corr5,disp6_u),1)
        # disp5 = self.decoder5(fusion5)
        disp5 = self.decoder5(torch.cat((im2_fea[5], corr5, disp6_u), 1))
        # disp5 = self.decoder5(torch.cat((fusion_fea, corr5,disp6_u), 1)


        disp5_u = F.interpolate(disp5, scale_factor=2) * 20. / 16

        coords1_4 = coords1_4 + disp5_u
        # original correlation
        # corr4 = corr_fn4(coords1_4)
        ## correlation cross-attention fusion
        corr4 = corr_fn4(coords1_4, guide=guide_fea[4], cross_attn_layer=self.cross_attn_layer_4)

        # fusion4 = self.fusion_block4(torch.cat((im2_fea[4], corr4, disp5_u, guide_fea[4]),1))
        # fusion4 = torch.cat((im2_fea[4]+guide_fea[4],corr4,disp5_u),1)
        # disp4 = self.decoder4(fusion4)
        disp4 = self.decoder4(torch.cat((im2_fea[4], corr4, disp5_u), 1))
        disp4_u = F.interpolate(disp4, scale_factor=2) * 20. / 8

        coords1_3 = coords1_3 + disp4_u
        # original correlation
        # corr3 = corr_fn3(coords1_3)
        ## correlation cross-attention fusion
        corr3 = corr_fn3(coords1_3, guide=guide_fea[3], cross_attn_layer=self.cross_attn_layer_3)

        # fusion3 = self.fusion_block3(torch.cat((im2_fea[3], corr3, disp4_u, guide_fea[3]),1))
        # fusion3 = torch.cat((im2_fea[3]+guide_fea[3],corr3,disp4_u),1)
        # disp3 = self.decoder3(fusion3)
        disp3 = self.decoder3(torch.cat((im2_fea[3], corr3, disp4_u), 1))
        disp3_u = F.interpolate(disp3, scale_factor=2) * 20. / 4

        coords1_2 = coords1_2 + disp3_u
        # original correlation
        # corr2 = corr_fn2(coords1_2)
        ## correlation cross-attention fusion
        corr2 = corr_fn2(coords1_2, guide=guide_fea[2], cross_attn_layer=self.cross_attn_layer_2)

        # fusion2 = self.fusion_block2(torch.cat((im2_fea[2], corr2, disp3_u, guide_fea[2]),1))
        # fusion2 = torch.cat((im2_fea[2]+guide_fea[2],corr2,disp3_u),1)
        # disp2 = self.decoder2(fusion2)
        disp2 = self.decoder2(torch.cat((im2_fea[2], corr2, disp3_u), 1))

        return disp2, disp3, disp4, disp5, disp6