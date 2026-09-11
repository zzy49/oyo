"""
MotionNet: ResNet-18 U-Net for per-pixel tissue motion classification.

Input:  (B, 6, H, W) — stacked RGB pair (I_t, I_{t+1})
Output: (B, 1, H, W) — P(static) logit map
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class ConvBlock(nn.Module):
    """Decoder conv block: Conv→BN→ReLU→Conv→BN→ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class MotionNet(nn.Module):
    """ResNet-18 U-Net for tissue motion probability prediction.

    Args:
        pretrained: ignored (kept for API compatibility)
        img_h, img_w: input image size
    """

    def __init__(self, pretrained=False, img_h=384, img_w=512):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w

        # ── Encoder (ResNet-18, 6-channel input) ──
        self.encoder = nn.ModuleDict()

        # Initial: 6→64, stride=2
        self.encoder['0'] = nn.Conv2d(6, 64, kernel_size=7, stride=2,
                                      padding=3, bias=False)
        self.encoder['1'] = nn.BatchNorm2d(64)

        # Layer 1: 64→64, no downsampling
        self.encoder['4'] = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )

        # Layer 2: 64→128, stride=2
        self.encoder['5'] = nn.Sequential(
            BasicBlock(64, 128, stride=2,
                       downsample=nn.Sequential(
                           nn.Conv2d(64, 128, 1, stride=2, bias=False),
                           nn.BatchNorm2d(128))),
            BasicBlock(128, 128, stride=1),
        )

        # Layer 3: 128→256, stride=2
        self.encoder['6'] = nn.Sequential(
            BasicBlock(128, 256, stride=2,
                       downsample=nn.Sequential(
                           nn.Conv2d(128, 256, 1, stride=2, bias=False),
                           nn.BatchNorm2d(256))),
            BasicBlock(256, 256, stride=1),
        )

        # Layer 4: 256→512, stride=2
        self.encoder['7'] = nn.Sequential(
            BasicBlock(256, 512, stride=2,
                       downsample=nn.Sequential(
                           nn.Conv2d(256, 512, 1, stride=2, bias=False),
                           nn.BatchNorm2d(512))),
            BasicBlock(512, 512, stride=1),
        )

        # ── Split conv ──
        self.split_conv = nn.Conv2d(512, 256, kernel_size=1, bias=True)

        # ── Bottleneck ──
        # Sequential [Conv, BN, ReLU, Conv, BN, ReLU] → state_dict at .0, .1, .3, .4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ──
        self.dec4 = ConvBlock(512, 256)   # skip(enc6=256) + upsample(bottleneck=256)
        self.dec3 = ConvBlock(384, 128)   # skip(enc5=128) + upsample(dec4=256)
        self.dec2 = ConvBlock(192, 64)    # skip(enc4=64) + upsample(dec3=128)

        # Sequential: [Upsample, Conv, BN, ReLU, Identity, Conv, BN]
        # → state_dict at .1, .2, .5, .6
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Identity(),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )

        # ── Head ──
        self.head = nn.Sequential(nn.Conv2d(16, 1, kernel_size=1, bias=True))

        if pretrained:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img_pair, pts0=None):
        """Forward pass.

        Args:
            img_pair: (B, 6, H, W) stacked RGB pair
            pts0:     (B, N, 2) sample points (unused in forward, passed for API compat)

        Returns:
            logit_map: (B, 1, H_out, W_out) raw logits (pre-sigmoid)
        """
        # Encoder
        x0 = F.relu(self.encoder['1'](self.encoder['0'](img_pair)))
        x4 = self.encoder['4'](x0)   # 64ch
        x5 = self.encoder['5'](x4)   # 128ch
        x6 = self.encoder['6'](x5)   # 256ch
        x7 = self.encoder['7'](x6)   # 512ch

        # Bottleneck
        b = self.split_conv(x7)       # 512→256
        b = self.bottleneck(b)        # 256ch

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([
            F.interpolate(b, size=x6.shape[-2:], mode='bilinear', align_corners=False),
            x6
        ], dim=1))  # 512→256

        d3 = self.dec3(torch.cat([
            F.interpolate(d4, size=x5.shape[-2:], mode='bilinear', align_corners=False),
            x5
        ], dim=1))  # 384→128

        d2 = self.dec2(torch.cat([
            F.interpolate(d3, size=x4.shape[-2:], mode='bilinear', align_corners=False),
            x4
        ], dim=1))  # 192→64

        d1 = self.dec1(d2)  # Upsample already inside dec1

        logits = self.head(d1)  # (B, 1, H_enc, W_enc)

        return logits

    def sample_weights(self, logit_map, pts2d, valid_mask):
        """Sample P(static) weights at given 2D locations.

        Args:
            logit_map:   (B, 1, H, W) logit map from forward()
            pts2d:       (B, N, 2) points in pixel coordinates [0,W]×[0,H]
            valid_mask:  (B, N) bool mask of valid points

        Returns:
            weights: (B, N) P(static) ∈ [0, 1]
        """
        B, _, H, W = logit_map.shape

        # Normalize pts2d to [-1, 1] for grid_sample
        grid = pts2d.clone()
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        grid = grid.unsqueeze(2)  # (B, N, 1, 2)

        sampled = F.grid_sample(logit_map, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        sampled = sampled.squeeze(1).squeeze(-1)  # (B, N)

        weights = torch.sigmoid(sampled)
        weights = weights * valid_mask.float()

        return weights

    @classmethod
    def load_from_checkpoint(cls, ckpt_path, device='cpu'):
        """Load model from DyEndoVO checkpoint."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('model', ckpt)
        model = cls(pretrained=False)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
