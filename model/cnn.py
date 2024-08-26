import numpy
from torch import nn
import torch

import torch.nn.functional as F

from .cnn_liabrary import *


class LayerNorm(nn.Module):
    def __init__(self, shape=(1, 7, 1, 1), dim_index=1):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))
        self.dim_index = dim_index
        self.eps = 1e-6

    def forward(self, x):
        """

        :param x: bsz x dim x max_len x max_len
        :param mask: bsz x dim x max_len x max_len, 为1的地方为pad
        :return:
        """
        u = x.mean(dim=self.dim_index, keepdim=True)
        s = (x - u).pow(2).mean(dim=self.dim_index, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight * x + self.bias
        return x


class MaskConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=1, flag=1,dilation=1,stride=1,theta=1):
        super(MaskConv2d, self).__init__()
        self.in_ch = in_ch
        self.flag = flag
        if flag == 1 or flag == 2:
            self.conv2d_3 = Conv2d_selfAdapt(in_ch, out_ch, kernel_size=kernel_size, padding=padding,
                                    bias=False, groups=groups,dilation=dilation,stride=stride,theta=theta)
        elif flag == 3 or flag == 4:
            self.conv2d_3 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding,
                                  bias=False, groups=groups,dilation=dilation,stride=stride)
    
    def forward(self, x, mask,init_flag):
        """

        :param x:
        :param mask:
        :return:
        """
        if self.flag !=4:
            x_1 = x.masked_fill(mask, 0)
        else:
            x_1 =x
        if self.flag == 1 or self.flag == 2:
            return self.conv2d_3(x_1,init_flag)
        return self.conv2d_3(x_1)


class MaskCNN_1(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3,theta=1):
        super(MaskCNN_1, self).__init__()
        
        self.theta = theta
        self.inchannels = input_channels
        layers1 = []
        layers2 = []
        layers3 = []
        layers4 = []

        self.c1 = MaskConv2d(input_channels, input_channels, kernel_size=kernel_size, padding='same', flag=1,theta=self.theta)
        self.f1 = MaskConv2d(input_channels, input_channels, kernel_size=1, padding='same', flag=3)
        layers1.extend([
            self.c1,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(), 
            self.c1,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(), 
            self.f1,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
        ])
        self.c2 = MaskConv2d(input_channels, input_channels, kernel_size=kernel_size, padding='same', flag=1,theta=self.theta)
        self.f2 = MaskConv2d(input_channels, input_channels, kernel_size=1, padding='same', flag=3)
        layers2.extend([
            self.c2,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(), 
            self.c2,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
            self.f2,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
        ])
        
        layers4.extend([
            MaskConv2d(input_channels, input_channels, kernel_size=3, padding=1, flag=4,stride=2),
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
            MaskConv2d(input_channels, input_channels, kernel_size=3, padding=1, flag=4,stride=2),
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
        ])
        self.cnns1 = nn.ModuleList(layers1)
        self.cnns2 = nn.ModuleList(layers2)
        self.cnns3 = nn.ModuleList(layers3)
        self.cnns4 = nn.ModuleList(layers4)
        self.lastLinear_dc = nn.Linear(in_features=input_channels*2, out_features=input_channels)
        self.t4_trans = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding='same',bias=False)

        torch.nn.init.xavier_normal_(self.lastLinear_dc.weight.data)

    def forward(self, x, mask,train):
        _x1 = x  # 用作residual
        _x2 = x
        x_1 =x.clone()
        x_2 =x.clone()
        x_3 =x.clone()
        x_4 =x.clone()
        _x1 = x_1  # 用作residual
        _x2 = x_2
        _x3 = x_3
        
        t_hor = []
        t_ver = []
        t_real = []
        t_se = [x_4]

        cnn_type =[0,3]
        
        if 0 in cnn_type:
            direction_flag = True
            direction = 1
            for layer1 in self.cnns1:
                if isinstance(layer1, LayerNorm):
                    x_1 = layer1(x_1)
                elif isinstance(layer1, MaskConv2d):
                    _x1 = x_1
                    x_1 = layer1(x_1, mask, True)
                    if  isinstance(layer1.conv2d_3,nn.Conv2d):
                        direction_flag = False
                        x_1 = x_1 + _x1
                    else:
                        layer1.conv2d_3.diff_conv2d_weight = layer1.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_1 = layer1(x_1)
                    if direction_flag:
                        x_1 = x_1 + _x1
                    t_ver.append(x_1)
                    
        if 1 in cnn_type:
            direction_flag = True
            direction = 1
            for layer in self.cnns2:
                if isinstance(layer, LayerNorm):
                    x_2 = layer(x_2)
                elif isinstance(layer, MaskConv2d):
                    _x2 = x_2
                    x_2 = layer(x_2, mask,True)
                    if  isinstance(layer.conv2d_3,nn.Conv2d):
                        direction_flag = False
                        x_2 = x_2 + _x2
                    else:
                        layer.conv2d_3.diff_conv2d_weight = layer.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_2 = layer(x_2)
                    if direction_flag:
                        x_2 = x_2 + _x2
                    t_hor.append(x_2)
        if 2 in cnn_type:
            direction_flag = True
            direction = 1
            for layer in self.cnns3:
                if isinstance(layer, LayerNorm):
                    x_3 = layer(x_3)
                elif isinstance(layer, MaskConv2d):
                    _x3 = x_3
                    x_3 = layer(x_3, mask,True)
                    if  isinstance(layer.conv2d_3,nn.Conv2d):
                        direction_flag = False
                        x_3 = x_3 + _x3
                    else:
                        layer.conv2d_3.diff_conv2d_weight = layer.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_3 = layer(x_3)
                    if direction_flag:
                        x_3 = x_3 + _x3
                    t_real.append(x_3)

        if 3 in cnn_type:
            for layer in self.cnns4:
                if isinstance(layer, LayerNorm):
                    x_4 = layer(x_4)
                elif isinstance(layer, MaskConv2d):
                    x_4 = layer(x_4, mask,True)
                else:
                    x_4 = layer(x_4)
                    t_se.append(x_4)

        for i in [1]:
            i_up = F.upsample(t_se[i],size=(t_se[i-1].shape[2],t_se[i-1].shape[3]),mode='nearest')
            if i !=1:
                t_se[i-1] = t_se[i-1] + i_up
            else:
                t_se[i-1] = self.t4_trans(i_up)
            
        t_list = [t_ver,t_hor,t_real,t_se]
        # atts_dc1 = torch.cat([t_list[cnn_type[0]][2],t_list[cnn_type[1]][2]],dim=1)
        # atts_dc1 = self.lastLinear_dc(atts_dc1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # linear_atts_dc1 = torch.cat([atts_dc1,t_se[0]],dim=1)
        linear_atts_dc1 = torch.cat([t_list[cnn_type[0]][2],t_se[0]],dim=1).masked_fill(mask, 0)
        return linear_atts_dc1

class MaskCNN_2(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, depth=3,theta=1):
        super(MaskCNN_2, self).__init__()
        
        self.theta = theta
        self.inchannels = input_channels
        layers1 = []
        layers2 = []
        layers3 = []
        layers4 = []

        self.c1 = MaskConv2d(input_channels, input_channels, kernel_size=kernel_size, padding='same', flag=1,theta=self.theta)
        self.f1 = MaskConv2d(input_channels, input_channels, kernel_size=1, padding='same', flag=3)
        layers1.extend([
            self.c1,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(), 
            self.c1,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(), 
            self.f1,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
        ])
        self.c2 = MaskConv2d(input_channels, input_channels, kernel_size=kernel_size, padding='same', flag=1,theta=self.theta)
        self.f2 = MaskConv2d(input_channels, input_channels, kernel_size=1, padding='same', flag=3)
        layers2.extend([
            self.c2,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(), 
            self.c2,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
            self.f2,
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
        ])
        
        layers4.extend([
            MaskConv2d(input_channels, input_channels, kernel_size=3, padding=1, flag=4,stride=2),
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
            MaskConv2d(input_channels, input_channels, kernel_size=3, padding=1, flag=4,stride=2),
            LayerNorm((1, input_channels, 1, 1), dim_index=1),
            nn.GELU(),
        ])
        self.cnns1 = nn.ModuleList(layers1)
        self.cnns2 = nn.ModuleList(layers2)
        self.cnns3 = nn.ModuleList(layers3)
        self.cnns4 = nn.ModuleList(layers4)
        self.lastLinear_dc = nn.Linear(in_features=input_channels*2, out_features=input_channels)
        self.t4_trans = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding='same',bias=False)

        torch.nn.init.xavier_normal_(self.lastLinear_dc.weight.data)

    def forward(self, x, mask,train):
        _x1 = x  # 用作residual
        _x2 = x
        # x_copy = self.t4_head(x)
        x_1 =x.clone()
        x_2 =x.clone()
        x_3 =x.clone()
        x_4 =x.clone()
        _x1 = x_1  # 用作residual
        _x2 = x_2
        _x3 = x_3
        
        t_hor = []
        t_ver = []
        t_real = []
        t_se = [x_4]

        cnn_type =[0,1,3]
        
        if 0 in cnn_type:
            direction_flag = True
            direction = 1
            for layer1 in self.cnns1:
                if isinstance(layer1, LayerNorm):
                    x_1 = layer1(x_1)
                elif isinstance(layer1, MaskConv2d):
                    _x1 = x_1
                    x_1 = layer1(x_1, mask, True)
                    if  isinstance(layer1.conv2d_3,nn.Conv2d):
                        direction_flag = False
                        x_1 = x_1 + _x1
                    else:
                        layer1.conv2d_3.diff_conv2d_weight = layer1.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_1 = layer1(x_1)
                    if direction_flag:
                        x_1 = x_1 + _x1
                    t_ver.append(x_1)
                    
        if 1 in cnn_type:
            direction_flag = True
            direction = 1
            for layer in self.cnns2:
                if isinstance(layer, LayerNorm):
                    x_2 = layer(x_2)
                elif isinstance(layer, MaskConv2d):
                    _x2 = x_2
                    x_2 = layer(x_2, mask,True)
                    if  isinstance(layer.conv2d_3,nn.Conv2d):
                        direction_flag = False
                        x_2 = x_2 + _x2
                    else:
                        layer.conv2d_3.diff_conv2d_weight = layer.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_2 = layer(x_2)
                    if direction_flag:
                        x_2 = x_2 + _x2
                    t_hor.append(x_2)
        if 2 in cnn_type:
            direction_flag = True
            direction = 1
            for layer in self.cnns3:
                if isinstance(layer, LayerNorm):
                    x_3 = layer(x_3)
                elif isinstance(layer, MaskConv2d):
                    _x3 = x_3
                    x_3 = layer(x_3, mask,True)
                    if  isinstance(layer.conv2d_3,nn.Conv2d):
                        direction_flag = False
                        x_3 = x_3 + _x3
                    else:
                        layer.conv2d_3.diff_conv2d_weight = layer.conv2d_3.diff_conv2d_weight * direction
                        direction = direction * -1
                        direction_flag = True
                else:
                    x_3 = layer(x_3)
                    if direction_flag:
                        x_3 = x_3 + _x3
                    t_real.append(x_3)

        if 3 in cnn_type:
            for layer in self.cnns4:
                if isinstance(layer, LayerNorm):
                    x_4 = layer(x_4)
                elif isinstance(layer, MaskConv2d):
                    x_4 = layer(x_4, mask,True)
                else:
                    x_4 = layer(x_4)
                    t_se.append(x_4)

        for i in [1]:
            i_up = F.upsample(t_se[i],size=(t_se[i-1].shape[2],t_se[i-1].shape[3]),mode='nearest')
            if i !=1:
                t_se[i-1] = t_se[i-1] + i_up
            else:
                t_se[i-1] = self.t4_trans(i_up)
            
        t_list = [t_ver,t_hor,t_real,t_se]
        atts_dc1 = torch.cat([t_list[cnn_type[0]][2],t_list[cnn_type[1]][2]],dim=1)
        atts_dc1 = self.lastLinear_dc(atts_dc1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # linear_atts_dc1 = torch.cat([atts_dc1,t_se[0]],dim=1)
        linear_atts_dc1 = torch.cat([atts_dc1,t_se[0]],dim=1).masked_fill(mask, 0)
        return linear_atts_dc1
