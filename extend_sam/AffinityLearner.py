import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = doubleConv(in_channels,out_channels,in_channels//2) 
        else:
            self.up = nn.ConvTranspose2d(in_channels,out_channels//2,kernel_size=2,stride=2)
            self.conv = doubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x1,x2],dim=1)
        x = self.conv(x)
        return x

def doubleConv(in_channels,out_channels,mid_channels=None):
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    layer.append(nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm2d(mid_channels))
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm2d(out_channels))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


def down(in_channels,out_channels):
    layer = []
    layer.append(nn.MaxPool2d(2,stride=2))
    layer.append(doubleConv(in_channels,out_channels))
    return nn.Sequential(*layer)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
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



class SelfAttention(nn.Module):
    """ Self-Attention Module"""
    def __init__(self, in_dim):
        super(SelfAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query = proj_query.view(m_batchsize, -1, width*height).permute(0, 2, 1) # (B, N, C)

        proj_key = self.key_conv(x)
        proj_key = proj_key.view(m_batchsize, -1, width*height) # (B, C, N)

        energy = torch.bmm(proj_query, proj_key) # (B, N, N)
        attention = self.softmax(energy) # (B, N, N)

        proj_value = self.value_conv(x)
        proj_value = proj_value.view(m_batchsize, -1, width*height) # (B, C, N)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # (B, C, N)
        out = out.view(m_batchsize, -1, height, width) # (B, C, H, W)

        return self.gamma*out + x # (B, C, H, W)


class AffinityLearner(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True,base_channel=64):

        super(AffinityLearner, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.in_conv = doubleConv(self.in_channels,base_channel)
        self.CCAtn1 = SelfAttention(base_channel)
        self.CCAtn2 = SelfAttention(base_channel)
        self.CCAtn3 = SelfAttention(base_channel)
        self.CCAtn4 = SelfAttention(base_channel)
        self.out = nn.Conv2d(in_channels=base_channel,out_channels=self.out_channels,kernel_size=1)

    def forward(self,x):
        x = self.in_conv(x)
        x= self.CCAtn1(x)
        x= self.CCAtn2(x)
        x= self.CCAtn3(x)
        x= self.CCAtn4(x)
        out = self.out(x)

        return out
    

