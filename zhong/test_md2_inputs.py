#!/usr/bin/env python
"""测试: 不同输入下 Monodepth2 的输出"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from layers import disp_to_depth

DEV = torch.device('cuda')
CKPT_DIR = r'C:\Users\Administrator\tmp\c3vd_md2_full\models\weights_19'

enc = ResnetEncoder(18, False)
enc_state = torch.load(os.path.join(CKPT_DIR, 'encoder.pth'), map_location=DEV)
for k in ['height', 'width', 'use_stereo']:
    if k in enc_state: del enc_state[k]
enc.load_state_dict(enc_state)
enc.eval().to(DEV)

dec = DepthDecoder(enc.num_ch_enc, scales=range(4))
dec_state = torch.load(os.path.join(CKPT_DIR, 'depth.pth'), map_location=DEV)
missing, unexpected = dec.load_state_dict(dec_state, strict=False)
print(f'Decoder loaded: {len(missing)} missing, {len(unexpected)} unexpected')
dec.eval().to(DEV)

# Test 1: all zeros
x0 = torch.zeros(1, 3, 256, 320, device=DEV)
with torch.no_grad():
    feats = enc(x0)
    outs = dec(feats)
    d0 = outs[("disp", 0)]
    print(f'Test zeros: disp min={d0.min():.6f}, max={d0.max():.6f}, mean={d0.mean():.6f}')

# Test 2: all ones
x1 = torch.ones(1, 3, 256, 320, device=DEV)
with torch.no_grad():
    feats = enc(x1)
    outs = dec(feats)
    d1 = outs[("disp", 0)]
    print(f'Test ones:  disp min={d1.min():.6f}, max={d1.max():.6f}, mean={d1.mean():.6f}')

# Test 3: random
xr = torch.randn(1, 3, 256, 320, device=DEV)
with torch.no_grad():
    feats = enc(xr)
    outs = dec(feats)
    dr = outs[("disp", 0)]
    print(f'Test randn: disp min={dr.min():.6f}, max={dr.max():.6f}, mean={dr.mean():.6f}')

# Test 4: raw features check
with torch.no_grad():
    feats = enc(xr)
    for i, f in enumerate(feats):
        print(f'  feats[{i}]: {f.shape}, min={f.min():.4f}, max={f.max():.4f}, mean={f.mean():.4f}')
    
    # Trace through decoder manually
    x = feats[-1]
    for i in range(4, -1, -1):
        x = dec.convs[("upconv", i, 0)](x)
        x = [torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')]
        if dec.use_skips and i > 0:
            x += [feats[i - 1]]
        x = torch.cat(x, 1)
        x = dec.convs[("upconv", i, 1)](x)
        if i in dec.scales:
            dispconv_out = dec.convs[("dispconv", i)](x)
            print(f'  scale{i} dispconv: min={dispconv_out.min():.4f}, max={dispconv_out.max():.4f}, mean={dispconv_out.mean():.4f}')
            d = dec.sigmoid(dispconv_out)
            print(f'  scale{i} sigmoid: min={d.min():.6f}, max={d.max():.6f}, mean={d.mean():.6f}')

del enc, dec
torch.cuda.empty_cache()
