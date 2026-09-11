#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 mdp_v5 模型加载 + 深度预测是否与 ManyDepth 重复."""
import os, sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as pil

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from networks import ResnetEncoder, MotionEncoder
from networks.litemono_decoder import LiteMonoDepthDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 测试图像
WARP_DIR = r'F:\dataset\c1_transverse1_t1_v2\generated\rgb_warped'
frames = sorted([f for f in os.listdir(WARP_DIR) if f.endswith('.png')])
img0 = os.path.join(WARP_DIR, frames[0])
img1 = os.path.join(WARP_DIR, frames[1])
print(f"测试帧: {frames[0]}, {frames[1]}")

# ── 1. 加载 mdp_v5 ──
print("\n" + "=" * 60)
print("加载 mdp_v5 (ResNet18 + LiteMonoDepthDecoder 64bin + MotionEncoder)")
print("=" * 60)
mdp_path = r'C:\Users\Administrator\tmp\mdp_v5\models\weights_19'

encoder = ResnetEncoder(18, False)
sd = torch.load(os.path.join(mdp_path, "encoder.pth"), map_location=device)
for k in ['height', 'width', 'use_stereo']:
    sd.pop(k, None)
missing, unexpected = encoder.load_state_dict(sd, strict=False)
print(f"encoder load: missing={len(missing)}, unexpected={len(unexpected)}")
if unexpected:
    print(f"  unexpected(前5): {unexpected[:5]}")

depth_decoder = LiteMonoDepthDecoder(
    num_ch_enc=np.array([64, 64, 128, 256, 512]),
    scales=range(3), num_output_channels=1, use_skips=True, num_bins=64)
dsd = torch.load(os.path.join(mdp_path, "depth.pth"), map_location=device)
missing_d, unexpected_d = depth_decoder.load_state_dict(dsd, strict=False)
print(f"depth_decoder load: missing={len(missing_d)}, unexpected={len(unexpected_d)}")
if missing_d:
    print(f"  MISSING keys (前10): {missing_d[:10]}")
if unexpected_d:
    print(f"  UNEXPECTED keys (前10): {unexpected_d[:10]}")

motion_encoder = MotionEncoder()
msd = torch.load(os.path.join(mdp_path, "motion_encoder.pth"), map_location=device)
for k in ['height', 'width', 'use_stereo']:
    msd.pop(k, None)
missing_m, unexpected_m = motion_encoder.load_state_dict(msd, strict=False)
print(f"motion_encoder load: missing={len(missing_m)}, unexpected={len(unexpected_m)}")

encoder.eval().to(device)
depth_decoder.eval().to(device)
motion_encoder.eval().to(device)

# ── 2. 预测深度 (无 motion) ──
def pred_depth(enc, dec, path, use_motion=False):
    input_image = pil.open(path).convert('RGB')
    ow, oh = input_image.size
    input_image = input_image.resize((320, 256), pil.LANCZOS)
    tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = enc(tensor)
        if use_motion:
            ctx = pil.open(img1).convert('RGB').resize((320, 256), pil.LANCZOS)
            ctx_t = transforms.ToTensor()(ctx).unsqueeze(0).to(device)
            pair = torch.cat([tensor, ctx_t], dim=1)
            mfeats = motion_encoder(pair)
            for lv in range(4):
                feats[lv] = motion_encoder.fuse_skip(feats[lv], mfeats[lv], lv)
        out = dec(feats)
        nb = getattr(dec, 'num_bins', 0)
        if nb > 0:
            bin_logits = out[("bins", 0)]
            residual = out.get(("residual", 0), None)
            # 简化: 直接取 bin logits 的 argmax 对应深度中心
            bin_centers = np.exp(np.linspace(np.log(1.0), np.log(500.0), nb))
            bin_centers = torch.from_numpy(bin_centers).float().to(device)
            prob = torch.softmax(bin_logits, dim=1)
            d = (prob * bin_centers.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            depth = d
        else:
            disp = out[("disp", 0)]
            depth = 1.0 / (disp + 1e-8)
    return depth.cpu().numpy().squeeze()

# ── 直接检查 bin_logits / residual 统计 ──
print("\n--- mdp_v5 bin_logits / residual 统计 (无 motion) ---")
input_image = pil.open(img0).convert('RGB').resize((320, 256), pil.LANCZOS)
tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
with torch.no_grad():
    feats = encoder(tensor)
    out = depth_decoder(feats)
    bl = out[("bins", 0)]
    res = out[("residual", 0)]
    print(f"  bin_logits: min={bl.min():.4f} max={bl.max():.4f} mean={bl.mean():.4f} std={bl.std():.4f}")
    print(f"  residual:   min={res.min():.4f} max={res.max():.4f} mean={res.mean():.4f} std={res.std():.4f}")
    probs = torch.softmax(bl, dim=1)
    print(f"  softmax max prob (逐像素均值): {probs.max(dim=1)[0].mean():.4f}")
    print(f"  argmax 唯一 bin 数: {len(torch.unique(probs.argmax(dim=1)))}")
    # 真实 depth_from_bins (含 residual)
    from layers import make_bin_centers, depth_from_bins as dfb
    bc = make_bin_centers(1.0, 500.0, 64)
    d_real = dfb(bl, bc, res)
    d_real = d_real.cpu().numpy().squeeze()
    print(f"  真实深度(含residual): median={np.median(d_real):.3f}, min={d_real.min():.3f}, max={d_real.max():.3f}")

print("\n--- mdp_v5 深度预测 (无 motion) ---")
d1 = pred_depth(encoder, depth_decoder, img0, use_motion=False)
print(f"  形状: {d1.shape}, median={np.median(d1):.3f}, mean={d1.mean():.3f}, min={d1.min():.3f}, max={d1.max():.3f}")

print("\n--- mdp_v5 深度预测 (有 motion) ---")
d2 = pred_depth(encoder, depth_decoder, img0, use_motion=True)
print(f"  形状: {d2.shape}, median={np.median(d2):.3f}, mean={d2.mean():.3f}, min={d2.min():.3f}, max={d2.max():.3f}")
print(f"  motion 与无 motion 差异: max|d2-d1|={np.max(np.abs(d2 - d1)):.4f}, 相对={np.mean(np.abs(d2-d1)/(d1+1e-6))*100:.2f}%")

# ── 3. 对比 ManyDepth ──
print("\n" + "=" * 60)
print("加载 ManyDepth 对比")
print("=" * 60)
from networks import DepthDecoder
md_path = r'C:\Users\Administrator\tmp\c3vd_manydepth_multi\models\weights_19'
enc2 = ResnetEncoder(18, False)
sd2 = torch.load(os.path.join(md_path, "mono_encoder.pth"), map_location=device)
for k in ['height', 'width', 'use_stereo']:
    sd2.pop(k, None)
enc2.load_state_dict(sd2)
dec2 = DepthDecoder(enc2.num_ch_enc, scales=range(4), num_bins=0)
dsd2 = torch.load(os.path.join(md_path, "mono_depth.pth"), map_location=device)
dec2.load_state_dict(dsd2, strict=False)
enc2.eval().to(device)
dec2.eval().to(device)

print("\n--- ManyDepth 深度预测 ---")
d3 = pred_depth(enc2, dec2, img0, use_motion=False)
print(f"  形状: {d3.shape}, median={np.median(d3):.3f}, mean={d3.mean():.3f}, min={d3.min():.3f}, max={d3.max():.3f}")

print("\n--- 深度差异对比 ---")
print(f"  mdp_v5(无motion) vs ManyDepth: 相对差异={np.mean(np.abs(d1-d3)/(d3+1e-6))*100:.2f}%")
print(f"  mdp_v5(有motion) vs ManyDepth: 相对差异={np.mean(np.abs(d2-d3)/(d3+1e-6))*100:.2f}%")

print("\n诊断完成")
