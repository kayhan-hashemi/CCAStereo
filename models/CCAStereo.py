import math
import torch
import torch.utils.data
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
from .submodule import *



class SubModule(nn.Module):

    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class Feature(nn.Module):

    def __init__(self):
        super().__init__()
   
        self.model =torch.hub.load('mit-han-lab/ProxylessNAS', 'proxyless_gpu', pretrained=True, trust_repo=True)

        self.block0 = nn.Sequential(self.model.first_conv,
                                    self.model.blocks[0])
        self.block1 = nn.Sequential(*self.model.blocks[1:5])
        self.block2 = nn.Sequential(*self.model.blocks[5:9])
        self.block3 = nn.Sequential(*self.model.blocks[9:17])
        self.block4 = nn.Sequential(*self.model.blocks[17:])

        channels = [24, 32, 56, 128, 432]

    def forward(self, x):

        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        return [x4, x8, x16, x32]



class FeatUp(SubModule):

    def __init__(self):
        super().__init__()

        channels = [24, 32, 56, 128, 432]
        self.deconv32_16 = Conv2x(channels[4], channels[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(channels[3]*2, channels[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(channels[2]*2, channels[1], deconv=True, concat=True)
        self.conv4 = BasicConv(channels[1]*2, channels[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):

        x4, x8, x16, x32 = featL
        y4, y8, y16, y32 = featR

        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)        
        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)
        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]
    


class Cross_Attention(SubModule):

    def __init__(self,cost_chan,fmap_chan,kernel,heads):
        super().__init__()

        self.scale = cost_chan**-0.5
        self.heads = heads
        self.kernel = kernel
        self.norm = nn.LayerNorm(cost_chan)

        self.to_qk = nn.Sequential(BasicConv(cost_chan, cost_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,5,5),padding=(0,2,2), stride=1, dilation=1),
                                  nn.Conv3d(cost_chan,cost_chan,1))

        self.to_v = nn.Sequential(BasicConv(fmap_chan, fmap_chan, is_3d=False, bn=True, relu=True, kernel_size=3,padding=1, stride=1, dilation=1),
                                  BasicConv(fmap_chan, fmap_chan//2, is_3d=False, bn=True, relu=True, kernel_size=1,padding=0, stride=1, dilation=1),
                                  nn.Conv2d(fmap_chan//2,cost_chan,1))

        self.att = nn.Sequential(BasicConv(cost_chan, cost_chan, is_3d=True, bn=True, relu=True, kernel_size=1,padding=0, stride=1, dilation=1),
                                  nn.Conv3d(cost_chan,cost_chan,1))

        self.agg = nn.Sequential(BasicConv(cost_chan, cost_chan, is_3d=True, bn=True, relu=True, kernel_size=1,padding=0, stride=1, dilation=1),
                                  nn.Conv3d(cost_chan,cost_chan,1))
        
        if cost_chan%heads!=0 :
            raise Exception(' Input dimension is not divisble by heads ')
        
        self.weight_init()

    def forward(self,Cost,Fmap):

        qk = self.to_qk(Cost)
        q = rearrange(qk,'b (m c) d h w -> b m d h w 1 c', m=self.heads)

        k = unfold_3d(qk,self.kernel)
        k = rearrange(k,'b (m c) k d h w -> b m d h w c k', m=self.heads)

        att = einsum('b m d h w o c, b m d h w c k -> b m d h w o k', q, k)
        att = att.softmax(dim=-1)

        v = self.to_v(Fmap)
        v = unfold_2d(v,self.kernel)
        v = rearrange(v,'b (m c) k h w -> b m 1 h w k c', m=self.heads)

        excite = einsum('b m d h w o k, b m o h w k c -> b m d h w o c', att, v)
        excite = self.norm(rearrange(excite,'b m d h w 1 c -> b d h w (m c)'))
        excite = rearrange(excite,'b d h w c -> b c d h w')

        excite = torch.sigmoid(self.att(excite))
        Cost = excite * Cost
        Cost = self.agg(Cost)

        return Cost
    


class hourglass_fusion(nn.Module):
    
    def __init__(self, in_channels):
        super(hourglass_fusion, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.out = BasicConv(in_channels, 1, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.GCC_8 = Cross_Attention(in_channels*2, 112, 3, 2)
        self.GCC_16 = Cross_Attention(in_channels*4, 256, 5, 8)
        self.GCC_32 = Cross_Attention(in_channels*6, 432, 7, 8)
        self.GCC_16_up = Cross_Attention(in_channels*4, 256, 5, 8)
        self.GCC_8_up = Cross_Attention(in_channels*2, 112, 3, 2)


    def forward(self, x, imgs):

        conv1 = self.conv1(x)
        conv1 = self.GCC_8(conv1, imgs[1])

        conv2 = self.conv2(conv1)
        conv2 = self.GCC_16(conv2, imgs[2])

        conv3 = self.conv3(conv2)
        conv3 = self.GCC_32(conv3, imgs[3])

        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.GCC_16_up(conv2, imgs[2])

        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.GCC_8_up(conv1, imgs[1])

        conv = self.conv1_up(conv1)
        out = self.out(conv)

        return out , conv



class Refinement(SubModule):

    def __init__(self, geo_chan, con_chan):
        super().__init__()

        chan = geo_chan+con_chan+9
        
        self.chan_att = nn.Sequential(
            nn.Conv2d(chan,chan,1,1),
            nn.ReLU(),
            nn.Conv2d(chan,chan//7,1,1),
            nn.ReLU(),
            nn.Conv2d(chan//7,chan,1,1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(chan,chan*2,3,1,1),
            nn.ReLU(),
            nn.Conv2d(chan*2,chan*2,5,1,2),
            nn.BatchNorm2d(chan*2),
            nn.ReLU()
        )
        self.conv_d1 = nn.Sequential(nn.Conv2d(chan*2,chan,3,1,1),
                                     nn.ReLU(),
                                     nn.Conv2d(chan,9,3,1,1),
        )
        self.conv_d3 = nn.Sequential(nn.Conv2d(chan*2,chan,5,1,6,3),
                                     nn.ReLU(),
                                     nn.Conv2d(chan,25,5,1,6,3),

        )

    def forward(self, disparity, geometry, context, variance):

        variance = unfold_2d(variance,3).squeeze(1)

        weights = torch.concat((geometry,context,variance),dim=1) 
        att = self.chan_att(weights)
        weights = weights * att

        weights = self.conv(weights)
        weights_d1 = self.conv_d1(weights)
        weights_d3 = self.conv_d3(weights)

        weights = torch.concat((weights_d1,weights_d3),1)
        weights = F.softmax(weights,1)

        disparity_s1 = unfold_2d(disparity,3,1).squeeze(1)
        disparity_s3 = unfold_2d(disparity,5,3).squeeze(1)

        disparity_refined = torch.concat((disparity_s1,disparity_s3),1)
        disparity_refined = torch.sum(disparity * weights, 1, True)

        return disparity_refined



class CCAStereo(nn.Module):

    def __init__(self, maxdisp):
        super().__init__()

        self.maxdisp = maxdisp 
        self.feature = Feature()
        self.feature_up = FeatUp()
        chans = [16, 24, 32, 96, 160]

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(112, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )

        self.conv = BasicConv(112, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.semantic = nn.Sequential(
            BasicConv(112, 64, kernel_size=3, stride=1, padding=1),
            BasicConv(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0, stride=1, bias=False)
            )
        
        self.agg_left = BasicConv(8, 8, is_3d=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1)
        self.agg_right = BasicConv(8, 8, is_3d=True, kernel_size=(5,5,1), padding=(2,2,0), stride=1)
        self.hourglass_fusion = hourglass_fusion(16)
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(2*torch.ones(1))

        self.refinement = Refinement(16, 32)

    def build_slided_volume(self, targetimg_fea, maxdisp):
        B, C, H, W = targetimg_fea.shape
        volume = targetimg_fea.new_zeros([B, C, maxdisp, H, W])
        for i in range(maxdisp):
            if i > 0:
                volume[:, :, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                volume[:, :, i, :, :] = targetimg_fea
        volume = volume.contiguous()
        return volume
    
    def forward(self, left, right):

        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)
        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        corr_volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp//4)
        corr_volume = self.corr_stem(corr_volume)

        left_feat_volume = self.semantic(features_left[0]).unsqueeze(2)
        right_feat_volume = self.semantic(features_right[0])
        right_feat_volume = self.build_slided_volume(right_feat_volume, self.maxdisp//4)
        
        volume = torch.concat((self.agg_right(corr_volume * right_feat_volume) , self.agg_left(left_feat_volume * corr_volume)),dim=1)
        cost, cost_feat = self.hourglass_fusion(volume, features_left)

        disp_samples = torch.arange(0, self.maxdisp//4, dtype=cost.dtype, device=cost.device)
        disp_samples = disp_samples.view(1, self.maxdisp//4, 1, 1).repeat(cost.shape[0],1,cost.shape[3],cost.shape[4])
        pred = regression_topk(cost.squeeze(1), disp_samples, 2)

        ind = torch.round(pred).to(torch.int64)
        ind = ind.expand(-1, cost_feat.shape[1],-1,-1).unsqueeze(2)
        cost_feat = cost_feat.permute(0, 1, 3, 4, 2)
        geo_features = torch.gather(cost_feat, 4, ind).squeeze(2) 

        variance = disparity_variance(F.softmax(cost.squeeze(1),1),self.maxdisp//4,pred)
        variance = self.beta + self.gamma * variance
        variance = torch.sigmoid(variance)

        xspx_4 = self.spx_4(features_left[0])
        xspx_2 = self.spx_2(xspx_4, stem_2x)
        spx_pred = self.spx(xspx_2)
        spx_pred = F.softmax(spx_pred, 1)

        pred = self.refinement(pred, geo_features, xspx_4, variance)
        pred_up = context_upsample(pred, spx_pred)
        

        if self.training:
            return [pred_up*4, pred.squeeze(1)*4]

        else:
            return [pred_up*4]
