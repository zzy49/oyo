# Lite-Mono 风格深度解码器 —— 适配 ResNet18 encoder
#
# 与标准 Monodepth2 DepthDecoder 的关键区别:
#   1. 3 层上采样 (非 5 层), 更轻量, 更少过拟合
#   2. bilinear 上采样 (非 nearest), 保留更多细节
#   3. Truncated Normal 初始化, 更好的梯度流
#   4. ReflectionPad2d 卷积 (非 ZeroPad), 减少边缘伪影
#
# 用法:
#   from networks.litemono_decoder import LiteMonoDepthDecoder
#   decoder = LiteMonoDepthDecoder(encoder.num_ch_enc, scales=range(3))

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ── 内嵌 Lite-Mono 基础模块 (避免与 monodepth2 layers.py 冲突) ──

class _Conv3x3(nn.Module):
    """ReflectionPad2d + Conv3x3 (Lite-Mono 风格)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ConvBlock(nn.Module):
    """Conv3x3 + ELU + BatchNorm (防激活值发散)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = _Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.nonlin(self.conv(x)))


def _upsample(x, scale_factor=2, mode='bilinear'):
    """Lite-Mono 风格上采样: bilinear + align_corners=False."""
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)


# ── 解码器 ──

class LiteMonoDepthDecoder(nn.Module):
    """Lite-Mono 风格深度解码器, 适配 ResNet18 encoder 特征.

    ResNet18 encoder 输出 (5 层):
      enc[0]: 64ch, H/2, W/2
      enc[1]: 64ch, H/4, W/4
      enc[2]: 128ch, H/8, W/8
      enc[3]: 256ch, H/16, W/16
      enc[4]: 512ch, H/32, W/32

    本解码器使用后 3 层 (enc[2], enc[3], enc[4]):
      Level 2 (最粗): enc[4] 512ch → 256ch
      Level 1:        dec[2] 256ch + enc[3] 256ch skip → 128ch
      Level 0 (最细): dec[1] 128ch + enc[2] 128ch skip → 64ch

    Args:
        num_ch_enc: ResNet18 encoder 通道数 [64, 64, 128, 256, 512]
        scales: 输出尺度列表, 默认 range(3)
        num_output_channels: 输出通道数, 默认 1 (disparity)
        use_skips: 是否使用 skip connections
    """
    def __init__(self, num_ch_enc, scales=range(3), num_output_channels=1, use_skips=True, num_bins=0):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales
        self.num_bins = num_bins

        # 使用后 3 层 encoder 特征 (enc[2], enc[3], enc[4])
        self.num_ch_enc = num_ch_enc[-3:]  # [128, 256, 512]
        # Lite-Mono 公式: decoder 通道 = encoder 通道 / 2
        self.num_ch_dec = np.array([64, 128, 256])  # 3 层 decoder

        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):  # 2, 1, 0
            # upconv_0: 从更粗级别上采样
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = _ConvBlock(num_ch_in, num_ch_out)

            # upconv_1: 融合 skip connection
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]  # 加前方 encoder 特征
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = _ConvBlock(num_ch_in, num_ch_out)

        # 每个尺度的输出头
        for s in self.scales:
            if self.num_bins > 0:
                # 分类+残差混合头 (与 monodepth2 DepthDecoder 一致)
                self.convs[("binconv", s)] = _Conv3x3(self.num_ch_dec[s], self.num_bins)
                self.convs[("resconv", s)] = _Conv3x3(self.num_ch_dec[s], 1)
            else:
                # disparity 输出头 (标记 is_dispconv=True 用于负偏置初始化)
                disp_conv = _Conv3x3(self.num_ch_dec[s], self.num_output_channels)
                disp_conv.conv.is_dispconv = True
                self.convs[("dispconv", s)] = disp_conv

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Truncated Normal 初始化 (Lite-Mono 风格)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.005)
            if m.bias is not None:
                # dispconv 偏置初始化为负值, 让 sigmoid 初始在 ~0.05
                # (sigmoid(-3)≈0.047), 避免初始化即饱和
                if hasattr(m, 'is_dispconv'):
                    nn.init.constant_(m.bias, -3.0)
                else:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        """前向传播.

        Args:
            input_features: list of 5 tensors [f0, f1, f2, f3, f4]
                            from ResNet18 encoder.

        Returns:
            dict: {("disp", s): tensor} for s in self.scales
        """
        self.outputs = {}

        # 取后 3 层特征: f2, f3, f4 → indices 0, 1, 2
        dec_features = input_features[-3:]  # [128ch, 256ch, 512ch]

        x = dec_features[-1]  # 512ch, 最粗层
        for i in range(2, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [_upsample(x)]

            if self.use_skips and i > 0:
                x += [dec_features[i - 1]]

            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                if self.num_bins > 0:
                    # 分类头: raw logits (不做 softmax)
                    f_bin = _upsample(self.convs[("binconv", i)](x), mode='bilinear')
                    f_bin = _upsample(f_bin, mode='bilinear')
                    self.outputs[("bins", i)] = f_bin
                    # 残差头: tanh → [-1, 1]
                    f_res = _upsample(self.convs[("resconv", i)](x), mode='bilinear')
                    f_res = _upsample(f_res, mode='bilinear')
                    self.outputs[("residual", i)] = self.tanh(f_res)
                else:
                    f = _upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                    f = _upsample(f, mode='bilinear')
                    self.outputs[("disp", i)] = self.sigmoid(f)

        return self.outputs
