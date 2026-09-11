#!/usr/bin/env python
"""检查 dispconv 权重值"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder

DEV = torch.device('cpu')
CKPT_DIR = r'C:\Users\Administrator\tmp\c3vd_md2_full\models\weights_19'

# 检查 checkpoint 中 dispconv 层的值
ckpt = torch.load(os.path.join(CKPT_DIR, 'depth.pth'), map_location='cpu')
for layer in ['10', '11', '12', '13']:
    w = ckpt[f'decoder.{layer}.conv.weight']
    b = ckpt[f'decoder.{layer}.conv.bias']
    print(f'decoder.{layer}.conv: weight mean={w.mean():.6f} std={w.std():.6f} min={w.min():.6f} max={w.max():.6f}')
    print(f'                 bias   mean={b.mean():.6f} std={b.std():.6f} min={b.min():.6f} max={b.max():.6f}')

# 也检查 encoder 关键层
enc_ckpt = torch.load(os.path.join(CKPT_DIR, 'encoder.pth'), map_location='cpu')
for key in ['encoder.conv1.weight', 'encoder.fc.weight']:
    if key in enc_ckpt:
        w = enc_ckpt[key]
        print(f'{key}: mean={w.mean():.6f} std={w.std():.6f}')

# 创建 decoder 并加载, 检查实际 model 参数
enc = ResnetEncoder(18, False)
enc_state = torch.load(os.path.join(CKPT_DIR, 'encoder.pth'), map_location='cpu')
for k in ['height', 'width', 'use_stereo']:
    if k in enc_state: del enc_state[k]
enc.load_state_dict(enc_state)

dec = DepthDecoder(enc.num_ch_enc, scales=range(4))
dec.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'depth.pth'), map_location='cpu'))

print('\n=== Loaded model dispconv params ===')
for i in range(4):
    param = dec.convs[("dispconv", i)]
    w = param.conv.weight.data
    b = param.conv.bias.data
    print(f'scale{i} dispconv: weight mean={w.mean():.6f} bias={b.item():.6f}')
