import torch
import torch.nn as nn

class Residual_Block(nn.Module):
    def __init__(self, n_feats=64, res_scale=1, n_padding=1, n_kernel=3):
        super(Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=n_kernel, stride=1, padding=n_padding, bias=False)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=n_kernel, stride=1, padding=n_padding, bias=False)
        self.res_scale = res_scale

    def forward(self, x):
        output = self.conv2(self.relu(self.conv1(x)))
        output = output.mul(self.res_scale)
        output += x
        return output

class Residual_Block_Dilate(nn.Module):
    def __init__(self, n_feats=64, res_scale=1, n_dilation=2, n_padding=2):
        super(Residual_Block_Dilate, self).__init__()

        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=n_padding, dilation=n_dilation, bias=False)
        self.res_scale = res_scale

    def forward(self, x):
        output = self.conv2(self.relu(self.conv1(x)))
        output = output.mul(self.res_scale)
        output += x
        return output



def default_conv(in_channels, out_channels, kernel_size, dilation=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), dilation=dilation, bias=bias)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, conv=default_conv):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                conv(channel, channel // reduction, 1),
                nn.ReLU(inplace=True),
                conv(channel // reduction, channel, 1),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class NonLocalBlockRef(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlockRef, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, f):
        # [N, C, H1 , W1]
        x1, x2 = f[0],f[1]
        b, c, h1, w1 = x1.size()
        b, c, h2, w2 = x2.size()
        # [N, H1 * W1, C/2]
        x_theta = self.conv_theta(x1).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, C/2, H2 * W2]
        x_phi = self.conv_phi(x2).view(b, c, -1)
        # [N, H2 * W2, C/2]
        x_g = self.conv_g(x2).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H1 * W1, H2 * W2]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H1 * W1, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H1, W1]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h1, w1)
        # [N, C, H1 , W1]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x1
        return out


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class CCmodule(nn.Module):
    def __init__(self, n_feats):
        super(CCmodule, self).__init__()

        R = 2
        att_blocks = [CrissCrossAttention(n_feats) for _ in range(R)]
        self.att = nn.Sequential(*att_blocks)
        self.conv = nn.Conv2d(2 * n_feats, n_feats, 1, padding=0, stride=1)

    def forward(self, x):
        H = self.att(x)
        out = self.conv(torch.cat([x, H], 1))
        return out

class CrissCrossAttention_Ref(nn.Module):
    """ Reference Based Criss-Cross Attention Module"""
    # fixed 20210710
    def __init__(self, in_dim):
        super(CrissCrossAttention_Ref,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, f):
        f_low, f_ref = f[0], f[1]
        #(1,16,8,8)
        m_batchsize, _, height, width = f_low.size()
        #(1,16,4,4)
        _, _, height_ref, width_ref = f_ref.size()
        proj_query = self.query_conv(f_low)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(f_ref)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width_ref,-1,height_ref)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height_ref,-1,width_ref)
        proj_value = self.value_conv(f_ref)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width_ref,-1,height_ref)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height_ref,-1,width_ref)
        #print('proj_query_H', proj_query_H.size())
        #print('proj_key_H', proj_key_H.size())
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        return self.gamma*(out_H + out_W) + f_low


class CCmodule_Ref_att(nn.Module):
    def __init__(self, n_feats):
        super(CCmodule_Ref_att, self).__init__()
        R = 1
        att_blocks_ref = [
            CrissCrossAttention_Ref(n_feats) for _ in range(R)
        ]

        self.att_ref = nn.Sequential(*att_blocks_ref)
        # self.att_ref2 = nn.Sequential(*att_blocks_ref)

    def forward(self, f):
        f_low, f_ref = f[0], f[1]
        # CrissCrossAttention needs two operation to acquare global info
        H = self.att_ref([f_low, f_ref])
        # diff with old and new
        H = self.att_ref([H, f_ref])
        # print(H.size())
        # print(out.size())
        return H


# input ref_jpeg ref
if __name__=='__main__':
    model = CCmodule_Ref_att(n_feats=64)
    print(model)

    input = torch.randn(2, 64, 6, 6)
    ref = torch.randn(2, 64, 6, 6)
    out = model([input,ref])
    print(out.shape)