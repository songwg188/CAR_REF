#
# reference based jpeg image artifacts removal

import torch
import torch.nn as nn
import math

from torch.nn.modules.pooling import AvgPool2d

from model import base


def make_model(args):
    return Net(
        in_channel=args.in_channel,
        out_channel=args.n_colors,
        n_feats=args.n_feats,
        n_resblocks=args.n_resblocks,
        res_scale=args.res_scale)


class Net(nn.Module):
    def __init__(self, in_channel, out_channel, n_feats=64, n_resblocks=8, res_scale=0.1, down_scale=2):
        super(Net, self).__init__()

        self.n_resblocks = n_resblocks
        self.conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False)
        self.head_1 = nn.Conv2d(
            in_channel, n_feats, kernel_size=3, stride=1, padding=1, bias=False)
        
        residual = [
            base.Residual_Block(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks)
        ]
        self.down1 = nn.Sequential(*residual) # 1

        residual = [
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            ]
        residual.extend( 
            [
                base.Residual_Block(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks)
            ]
        )
        self.down2 = nn.Sequential(*residual) # 1/2 

        residual = [
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            ]
        residual.extend( 
            [
                base.Residual_Block(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks)
            ]
        )
        self.down3 = nn.Sequential(*residual) # 1/4
        
        # nonlocal  
        # TODO multi scale 
        self.cc1 = base.CCmodule_Ref_att(n_feats) # 1
        self.cc2 = base.CCmodule_Ref_att(n_feats) # 1/2
        self.cc3 = base.CCmodule_Ref_att(n_feats) # 1/4

        residual = [
            nn.Conv2d(2 * n_feats, n_feats, 1, padding=0, stride=1),
        ]
        residual.extend(
            [base.Residual_Block(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks)]
        )
        residual.extend(
            [
                nn.Conv2d(n_feats, n_feats * (down_scale**2), kernel_size=3, padding=1, bias=False),
                nn.PixelShuffle(down_scale),
            ]
        )
        
        self.up1 = nn.Sequential(*residual) # 1/2


        residual = [
            nn.Conv2d(3 * n_feats, n_feats, 1, padding=0, stride=1),
        ]
        residual.extend(
            [base.Residual_Block(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks)]
        )
        residual.extend(
            [
                nn.Conv2d(n_feats, n_feats * (down_scale**2), kernel_size=3, padding=1, bias=False),
                nn.PixelShuffle(down_scale),
            ]
        )
        self.up2 = nn.Sequential(*residual) # 1

        residual = [
            nn.Conv2d(3 * n_feats, n_feats, 1, padding=0, stride=1),
        ]
        residual.extend(
            [base.Residual_Block(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks)]
        )
        self.up3 = nn.Sequential(*residual)

        self.conv_merge_1 = nn.Conv2d(2 * n_feats, n_feats, 1, padding=0, stride=1)
        self.conv_merge_2 = nn.Conv2d(2 * n_feats, n_feats, 1, padding=0, stride=1)
        self.conv_merge_3 = nn.Conv2d(2 * n_feats, n_feats, 1, padding=0, stride=1)

        self.tail_2 = nn.Conv2d(n_feats, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(True)


    def forward(self, x):
        low, ref = x[0], x[1]
        low = (low - 0.5) / 0.5
        ref = (ref - 0.5) / 0.5
        
        # head1
        f_low = self.head_1(low)
        f_ref = self.head_1(ref)
        f_skip = f_low

        # body1
        f_low_down1 = self.down1(f_low) #1
        f_ref_down1 = self.down1(f_ref)
        # nonlocal_1
        f_cc1 = self.cc1([f_low_down1, f_ref_down1])

        f_low_down2 = self.down2(f_low_down1) #1/2
        f_ref_down2 = self.down2(f_ref_down1)
        f_cc2 = self.cc2([f_low_down2, f_ref_down2])

        f_low_down3 = self.down3(f_low_down2) #1/4
        f_ref_down3 = self.down3(f_ref_down2)
        f_cc3 = self.cc3([f_low_down3, f_ref_down3])

        f_up1 = self.up1(torch.cat((f_low_down3, f_cc3),1)) # 1/2
        #import pdb;pdb.set_trace()
        f_up2 = self.up2(torch.cat((f_low_down2, f_up1, f_cc2),1)) # 1

        f_up3 = self.up3(torch.cat((f_low_down1, f_up2, f_cc1),1))
        

        output = self.tail_2(f_up3)
        output = output * 0.5 + 0.5
        return output


    