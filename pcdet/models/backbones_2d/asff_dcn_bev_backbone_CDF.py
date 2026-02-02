import numpy as np
import torch
import torch.nn as nn
from pcdet.models.backbones_2d.ASFF import ASFF
import torch.nn.functional as F 
from pcdet.models.backbones_2d.utils import get_activation

try:
    from pcdet.ops.DeformableConvolutionV2PyTorch.modules.mdeformable_conv_block import MdeformConvBlock 
except:
    print("Deformable Convolution not built!")
class asff_DCNBEVBackbone_CDF(nn.Module):
    def __init__(self, model_cfg ,input_channels):
        super().__init__()
        layer_nums = [5,5,5] # [5, 5, 5]
        layer_strides = [1,2,2] # [1, 2, 2]
        num_filters = [128,256,256] # [128, 256, 256]
        num_upsample_filters = [128, 256, 256]  #[128, 256, 256]
        upsample_strides = [1, 2, 4]
        use_dcn = True
        self.fusion1 = CSPRepLayer(256, 128, num_blocks=2,act="silu")  # 256, 128
        self.fusion2 = CSPRepLayer(512, 256, num_blocks=2,act="silu") # 3  # 512, 256
        self.fusion3 = CSPRepLayer(512, 256, num_blocks=2,act="silu")  
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.downscale = nn.Sequential(
            nn.Conv2d(640, 256 , kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()            
        )
        for idx in range(num_levels): # 3
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]): # 5
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    deblocks_list = []
                    if use_dcn:
                        deblocks_list.extend([
                            MdeformConvBlock(num_filters[idx], num_filters[idx],
                                deformable_groups=1),
                            nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU()
                        ])
                    deblocks_list.extend([
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx], 
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ])
                    self.deblocks.append(nn.Sequential(*deblocks_list))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
       
        self.num_bev_features_post = 256

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features_mm = data_dict['spatial_features_mm']
        spatial_features00 = data_dict['spatial_features00']  
        ups = []
        ret_dict = {}
        x = spatial_features00  
        x1 = spatial_features_mm

        for i in range(len(self.blocks)):
            x = self.blocks[i](x) # 128 256 256            
            stride = int(spatial_features00.shape[2] / x.shape[2])  # 1 2 4
            ret_dict['spatial_features00_%dx' % stride] = x  # [1X:[B,128,200,176] 2X:[B,256,100,88] 4X:[B,256,50,44]]
            x1 = self.blocks[i](x1) # 128 256 256           
            stride1 = int(spatial_features_mm.shape[2] / x1.shape[2])  # 1 2 4
            ret_dict['spatial_features_mm_%dx' % stride1] = x1  # [1X:[B,128,200,176] 2X:[B,256,100,88] 4X:[B,256,50,44]]
        stride_f_list = [1,2,4]
        ret_dict['spatial_features_1x'] = self.fusion1(torch.cat((ret_dict['spatial_features00_1x'],ret_dict['spatial_features_mm_1x']),dim=1))
        ret_dict['spatial_features_2x'] = self.fusion2(torch.cat((ret_dict['spatial_features00_2x'],ret_dict['spatial_features_mm_2x']),dim=1))
        ret_dict['spatial_features_4x'] = self.fusion3(torch.cat((ret_dict['spatial_features00_4x'],ret_dict['spatial_features_mm_4x']),dim=1))
        for k in range (len(self.deblocks)):
            stridef = stride_f_list[k]
            ups.append(self.deblocks[k](ret_dict['spatial_features_%dx' % stridef]))  #[0:[B, 128, 200, 176], 1:[B,256,100,88]],3:[B,256,50,44]
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        data_dict['st_features_2d'] = self.downscale(x)

        return data_dict


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="relu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)
   



