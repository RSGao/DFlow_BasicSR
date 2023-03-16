import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import flow_warp, make_layer



@ARCH_REGISTRY.register()
class DFlow(nn.Module):
    """A recurrent network for efficient video SR based in difference flow.
       Code structure is based on BasicVSR. 
       Only x4 is supported.

    Args:
        num_feat (int): Number of channels.
        num_block (int): Number of RFDB blocks per stage (Default: 3 stage).
        return_wimage (bool): True for returning warped image.
    """

    def __init__(self, num_feat=64, num_block=2, return_wimage=False):
        super().__init__()
        self.return_wimage = return_wimage
        self.num_feat = num_feat

        # alignment
        self.createflowf = IME(in_planes=3, num_lka=2, num_feat=64)
        self.createflowb = IME(in_planes=3, num_lka=2, num_feat=64)
        
        # flow modification
        self.flowconv1 = nn.Conv2d(64, 2, 1, 1, 0, bias=True)
        self.flowconv2 = nn.Conv2d(64, 2, 1, 1, 0, bias=True)
        self.flowconv31 = nn.Conv2d(3+2, 64, 3, 1, 1, bias=True)
        self.flowconv32 = nn.Conv2d(3+2, 64, 3, 1, 1, bias=True)
        
        # propagation
        self.backward_trunk = DFD(num_feat + 3, num_feat, num_block)
        self.forward_trunk = DFD(num_feat*2 + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def get_initial_flow(self, x):
        b, n, c, h, w = x.size()

        x_0 = x[:, 0, :, :, :]
        x_1 = x[:, 1, :, :, :]
        x_last = x[:, -1, :, :, :]
        x_lastpre = x[:, -2, :, :, :]

        flow_backward = self.createflowb(x_lastpre, x_last).view(b, 1, 2, h, w)
        flow_forward = self.createflowf(x_1, x_0).view(b, 1, 2, h, w)

        return flow_forward.squeeze(1), flow_backward.squeeze(1)


    def forward(self, x):
        """Forward function of DFlow.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flow_forward, flow_backward = self.get_initial_flow(x)
        b, n, _, h, w = x.size()
        
        img_fwarp = []
        img_bwarp = []

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            
            x_i = x[:, i, :, :, :]
            if i < n - 2:
                #modify hidden_flowb#
                x_next = x[:, i+1, :, :, :]
                residual = x_i - x_next
                t = self.flowconv31(torch.cat((residual, hidden_flowb),1))
                hidden_flowb = self.flowconv1(t)
                
                #####################
                
                #img_align
                img_bwarp.append(flow_warp(x_i, hidden_flowb.permute(0, 2, 3, 1)))
                
                # feat_align
                feat_prop = flow_warp(feat_prop, hidden_flowb.permute(0, 2, 3, 1))
                feat_prop = torch.cat([x_i, feat_prop], dim=1)
                feat_prop = self.backward_trunk(feat_prop)
                
            elif i == n - 2:
                
                hidden_flowb = flow_backward
                
                img_bwarp.append(flow_warp(x_i, hidden_flowb.permute(0, 2, 3, 1)))   # flow 5->4 makes 4 looks like 5
                
                feat_prop = flow_warp(feat_prop, hidden_flowb.permute(0, 2, 3, 1))
                feat_prop = torch.cat([x_i, feat_prop], dim=1)
                feat_prop = self.backward_trunk(feat_prop)
            
            else:
                
                feat_prop = torch.cat([x_i, feat_prop], dim=1)
                feat_prop = self.backward_trunk(feat_prop)

            out_l.insert(0, feat_prop)
            
        img_bwarp.append(x[:, 0, :, :, :])
        img_bwarp = img_bwarp[::-1]      # reverse the img_bwarp list
        
        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            
            x_i = x[:, i, :, :, :]
            if i > 1:
                #modify hidden_flowf#
                x_pre = x[:, i-1, :, :, :]
                residual = x_i - x_pre
                t = self.flowconv32(torch.cat((residual, hidden_flowf),1))
                hidden_flowf = self.flowconv2(t)
                
                #####################
                
                # img_align
                img_fwarp.append(flow_warp(x_i, hidden_flowf.permute(0, 2, 3, 1)))
                
                # fea_align
                feat_prop = flow_warp(feat_prop, hidden_flowf.permute(0, 2, 3, 1))
                feat_prop = torch.cat([x_i, feat_prop, out_l[i]], dim=1)
                feat_prop = self.forward_trunk(feat_prop)
            
            elif i==1:
                hidden_flowf = flow_forward
                img_fwarp.append(flow_warp(x_i, hidden_flowf.permute(0, 2, 3, 1)))
                
                feat_prop = flow_warp(feat_prop, hidden_flowf.permute(0, 2, 3, 1))
                feat_prop = torch.cat([x_i, feat_prop, out_l[i]], dim=1)
                feat_prop = self.forward_trunk(feat_prop)
                
            else:
                
                feat_prop = torch.cat([x_i, feat_prop, out_l[i]], dim=1)
                feat_prop = self.forward_trunk(feat_prop)

            
            # upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)

            
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            out_l[i] = out
        
        img_fwarp.append(x[:, -1, :, :, :])
        
        if self.return_wimage:
            return torch.stack(out_l, dim=1), torch.stack(img_fwarp,dim=1), torch.stack(img_bwarp,dim=1)
        else:
            return torch.stack(out_l, dim=1)





############# IME ###################
class IME(nn.Module):             
    """
    IME (Initial Motion Estimation) module.

    Args:
        in_planes (int): Input channels.
        num_lka (int): Number of LKA module.
        num_feat (int): Number of features.
    """
    def __init__(self, in_planes=3, num_lka=2, num_feat=64):
        super().__init__()
        self.lka1 = make_layer(LKA, num_lka, d_model=num_feat)
        self.lka2 = make_layer(LKA, num_lka, d_model=num_feat)
        self.lka3 = make_layer(LKA, num_lka, d_model=num_feat)


        self.conv1 = nn.Conv2d(in_planes,num_feat,3,1,1)
        self.conv2 = nn.Conv2d(in_planes,num_feat,3,1,1)
        self.conv3 = nn.Conv2d(in_planes*2,num_feat,3,1,1)
        
        self.convout = nn.Conv2d(num_feat*3,2,1,1,0)
    
    def forward(self, x1, x2):
        image_res = self.conv1(x1 - x2)
        feat_res = self.conv2(x1) - self.conv2(x2)
        image_feat = self.conv3(torch.cat((x1,x2),dim=1))
        
        o1 = self.lka1(image_res)
        o2 = self.lka2(image_feat)
        o3 = self.lka3(feat_res)
        
        o = torch.cat((o1,o2,o3),dim=1)
        return self.convout(o)
############# IME ENDS ##############


################ LKA ############################

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn

class LKA(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
#    Adapted from 'Visual attention network'    #
################ LKA ENDS #######################





######################  DFD ##################################

#    Adapted from the original code  #
class RFDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.5):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)
        
        
    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.esa(self.c5(out)) + input

        return out

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 2
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m



class DFD(nn.Module):
    """DFD (Dense Feature Distillation) module.

    Args:
        num_in_ch (int): Number of input channels. 
        num_out_ch (int): Number of output channels. 
        num_rfdb (int): Number of RFDB blocks per stage (Default: 3 stage). 
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_rfdb=2):
        super().__init__()
        
        self.inconv = nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.r1 = make_layer(RFDB, num_rfdb, in_channels=num_out_ch)

        self.conv1 = nn.Conv2d(num_out_ch, num_out_ch*3//4, 1, 1, 0, bias=True)
        self.r2 = make_layer(RFDB, num_rfdb, in_channels=num_out_ch)

        self.conv2 = nn.Conv2d(num_out_ch, num_out_ch//2, 1, 1, 0, bias=True)
        self.r3 = make_layer(RFDB, num_rfdb, in_channels=num_out_ch)

        self.conv3 = nn.Conv2d(num_out_ch, num_out_ch//4, 1, 1, 0, bias=True)

        self.fusion = nn.Conv2d(num_out_ch*5//2, num_out_ch, 1, 1, 0, bias=True)
            

    def forward(self, fea):
        
        fea = self.lrelu(self.inconv(fea))
        i = fea.clone()             
        
        dis1 = self.conv1(fea)    
        
        l1 = self.r1(fea)
        dis2 = self.conv2(l1)
        
        l2 = self.r2(l1)
        dis3 = self.conv3(l2)
 
        l3 = self.r3(l2)
        
        o = self.fusion(torch.cat((l3,dis1,dis2,dis3),dim=1))

        return o + i
###################### DFD ENDS ##############################
















if __name__ == "__main__":

  from thop import profile
  i = torch.randn(1,7,3,64,64)
  n = DFlow(64,2,False)

  macs, params = profile(n, inputs=(i, ))
  o = n(i)
  print(o.shape)
  
  print("FLOPs[G]",(macs*2)/10**9)
  print("n-num",params)


