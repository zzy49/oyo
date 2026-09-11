# Motion Encoder: 轻量帧间运动编码器
#
# 输入帧对 (6通道) → 提取4尺度运动特征 → 注入深度decoder的skip连接
# 使深度图感知相机运动, 弥补单帧深度模型的Z轴盲区。

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    """轻量运动编码器: 提取帧间运动特征用于注入深度decoder。

    输入: [B, 6, H, W] (两帧RGB拼接, 当前帧+下一帧)
    输出: 4个尺度的运动特征 [f0, f1, f2, f3]
          通道数匹配ResNet18 encoder skip连接: 64, 64, 128, 256
    """
    def __init__(self):
        super(MotionEncoder, self).__init__()

        # ── Encoder: 5 stride-2 conv, 输出4个下采样层级 ──
        # Level 0 (1/4): 64 ch → 匹配 ResNet layer0 skip
        # Level 1 (1/8): 64 ch → 匹配 ResNet layer1 skip
        # Level 2 (1/16): 128 ch → 匹配 ResNet layer2 skip
        # Level 3 (1/32): 256 ch → 匹配 ResNet layer3 skip

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        # ── 运动特征 → skip融合权重 (可学习的通道级门控) ──
        # 对每个尺度的运动特征做 1×1 conv → 与encoder skip做加权融合
        self.fuse_conv3 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.fuse_conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.fuse_conv1 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.fuse_conv0 = nn.Conv2d(256, 256, kernel_size=1, bias=False)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 初始化融合卷积接近恒等 (小权重, 训练中逐步激活)
        for conv in [self.fuse_conv3, self.fuse_conv2, self.fuse_conv1, self.fuse_conv0]:
            nn.init.normal_(conv.weight, mean=0.0, std=0.01)

    def forward(self, frame_pair):
        """提取多尺度运动特征。

        Args:
            frame_pair: [B, 6, H, W] 两帧RGB拼接

        Returns:
            list of [f0, f1, f2, f3]: 4尺度运动特征
            f0: [B, 64, H/4, W/4]   → ResNet layer0 skip
            f1: [B, 64, H/8, W/8]   → ResNet layer1 skip
            f2: [B, 128, H/16, W/16] → ResNet layer2 skip
            f3: [B, 256, H/32, W/32] → ResNet layer3 skip
        """
        x = self.conv1(frame_pair)   # /2
        x = self.conv2(x)            # /4
        f0 = x                       # [B, 64, H/4, W/4]

        x = self.conv3(x)            # /8
        f1 = x                       # [B, 64, H/8, W/8]

        x = self.conv4(x)            # /16
        f2 = x                       # [B, 128, H/16, W/16]

        x = self.conv5(x)            # /32
        f3 = x                       # [B, 256, H/32, W/32]

        return [f0, f1, f2, f3]

    def fuse_skip(self, encoder_skip, motion_feat, level):
        """将运动特征融合到encoder skip连接中。

        Args:
            encoder_skip: [B, C, H, W] ResNet skip特征
            motion_feat:  [B, C, H, W] 同尺度的运动特征
            level: int, 0=layer0(64ch), 1=layer1(64ch), 2=layer2(128ch), 3=layer3(256ch)

        Returns:
            fused: [B, C, H, W] 融合后的特征
        """
        # 上采样运动特征以匹配skip尺寸 (如果不同)
        if motion_feat.shape[-2:] != encoder_skip.shape[-2:]:
            motion_feat = nn.functional.interpolate(
                motion_feat, size=encoder_skip.shape[-2:],
                mode='bilinear', align_corners=False)

        # 通过融合卷积处理运动特征
        fuse_convs = {0: self.fuse_conv3, 1: self.fuse_conv2,
                      2: self.fuse_conv1, 3: self.fuse_conv0}
        motion_gate = fuse_convs[level](motion_feat)

        # 加法融合 (小权重初始化保证训练初期的稳定性)
        return encoder_skip + motion_gate
