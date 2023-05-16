import cv2
import torch
import torch.nn as nn
import numpy as np
from math import log2
import torch.nn.functional as nnf
from NetWorks.transformer import TransformerEncoderLayer
from NetWorks.dwt import multi_level_dwt
if __name__ == "__main__":
    from PixelShuffleUpsample import PixelShuffleUpsample, Blur
else:
    from NetWorks.PixelShuffleUpsample import PixelShuffleUpsample, Blur


class NeuralRenderer(nn.Module):

    def __init__(
            self, bg_type = "white", feat_nc=256, out_dim=3, final_actvn=True, min_feat=32, featmap_size=32, img_size=256, 
            **kwargs):
        super().__init__()
        # assert n_feat == input_dim
        
        self.bg_type = bg_type
        self.featmap_size = featmap_size
        self.final_actvn = final_actvn
        # self.input_dim = input_dim
        self.n_feat = feat_nc
        self.out_dim = out_dim
        self.n_blocks = int(log2(img_size) - log2(featmap_size))
        self.min_feat = min_feat
        self._make_layer()
        self._build_bg_featmap()


    def _build_bg_featmap(self): 
        
        if self.bg_type == "white":
            bg_featmap = torch.ones((1, self.n_feat, self.featmap_size, self.featmap_size), dtype=torch.float32)
        elif self.bg_type == "black":
            bg_featmap = torch.zeros((1, self.n_feat, self.featmap_size, self.featmap_size), dtype=torch.float32)
        else:
            bg_featmap = None
            print("Error bg_type")
            exit(0)
        
        self.register_parameter("bg_featmap", torch.nn.Parameter(bg_featmap))


    def get_bg_featmap(self, bg=None):
        if bg is not None:
            bg = nnf.interpolate(bg, size=(32, 32), mode='bicubic', align_corners=False)
            bg = self.bg_layer(bg) # [5,3,32,32] --> [5,256,32,32]
            return bg
        else:
            return self.bg_featmap
    

    def _make_layer(self):


        self.feat_upsample_list = nn.ModuleList(
            [PixelShuffleUpsample(max(self.n_feat // (2 ** (i)), self.min_feat)) for i in range(self.n_blocks)]
        )
        
        self.rgb_upsample = nn.Sequential(nn.Upsample(
            scale_factor=2, mode='bicubic', align_corners=False), Blur())

        self.feat_2_rgb_list = nn.ModuleList(
                [nn.Conv2d(self.n_feat , self.out_dim, 1, 1, padding=0)] +
                [nn.Conv2d(max(self.n_feat // (2 ** (i + 1)), self.min_feat),
                           self.out_dim, 1, 1, padding=0) for i in range(0, self.n_blocks)]
            )

        self.feat_layers = nn.ModuleList(
            [nn.Conv2d(max(self.n_feat // (2 ** (0)), self.min_feat),
                       max(self.n_feat // (2 ** (1)), self.min_feat), 1, 1,  padding=0)] +
            [nn.Conv2d(max(self.n_feat // (2 ** (i)), self.min_feat),
                       max(self.n_feat // (2 ** (i + 1)), self.min_feat), 1, 1,  padding=0)
                for i in range(1, self.n_blocks)]
        )
        
        self.actvn = nn.LeakyReLU(0.2, inplace=True)
        
        self.cross_attn1 = TransformerEncoderLayer(d_model=256, nhead=8, activation='gelu')
        self.cross_attn2 = TransformerEncoderLayer(d_model=256, nhead=8, activation='gelu')
        self.cross_attn3 = TransformerEncoderLayer(d_model=256, nhead=8, activation='gelu')

        self.bg_layer = nn.Conv2d(3, 256, kernel_size=1, stride=1, padding=0)

        
    def forward(self, x, aud=None):
        B, D, H, W = x.shape

        # cross-attention module
        if aud is not None:
            if aud.dim() == 2: # training
                aud = aud.unsqueeze(1).permute(1,0,2) # [5,1,dim]
            else: # rendering
                aud = aud.unsqueeze(0).unsqueeze(1).permute(1,0,2) # [1,1,dim]
            x = x.view(B, D, -1).permute(2,0,1) # [5,1024,dim]
            mm = torch.cat([x, aud], dim=0) # [5,1025,dim]
            x = self.cross_attn1(x, mm, mm) # [5,1024,dim]

            mm = torch.cat([x, aud], dim=0) # [5,1025,dim]
            x = self.cross_attn2(x, mm, mm) # [5,1024,dim]
            
            mm = torch.cat([x, aud], dim=0) # [5,1025,dim]
            x = self.cross_attn3(x, mm, mm) # [5,1024,dim]

            x = x.view(H,W,B,D).permute(2,3,0,1)


        rgb = self.rgb_upsample(self.feat_2_rgb_list[0](x)) # [5,3,64,64]
        sync_mid = rgb.clone()

        net = x
        for idx in range(self.n_blocks):
            hid = self.feat_layers[idx](self.feat_upsample_list[idx](net))
            net = self.actvn(hid)
            
            rgb = rgb + self.feat_2_rgb_list[idx + 1](net)
            
            if idx < self.n_blocks - 1:
                rgb = self.rgb_upsample(rgb)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)


        dwt = multi_level_dwt(rgb, levels=3) # [5,12,256]

        return rgb, dwt, sync_mid