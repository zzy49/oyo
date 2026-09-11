#!/usr/bin/env python
"""测试: 用 scales=range(4) 加载 Monodepth2 模型"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
import torchvision.transforms as T
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from layers import disp_to_depth

DEV = torch.device('cuda')
IMG = r'F:/dataset/c1_transverse1_t1_v2/generated/rgb_warped/frame_0000.png'
CKPT_DIR = r'C:\Users\Administrator\tmp\c3vd_md2_full\models\weights_19'

# ═══ 正确的加载: scales=range(4) ═══
enc = ResnetEncoder(18, False)
enc_state = torch.load(os.path.join(CKPT_DIR, 'encoder.pth'), map_location=DEV)
for k in ['height', 'width', 'use_stereo']:
    if k in enc_state: del enc_state[k]
enc.load_state_dict(enc_state)
enc.eval().to(DEV)

dec = DepthDecoder(enc.num_ch_enc, scales=range(4))
dec_state = torch.load(os.path.join(CKPT_DIR, 'depth.pth'), map_location=DEV)
print(f'Loading depth decoder: {len(dec_state)} keys')
missing, unexpected = dec.load_state_dict(dec_state, strict=False)
print(f'Missing: {len(missing)}, Unexpected: {len(unexpected)}')
if missing: print(f'  Missing: {missing[:5]}...')
dec.eval().to(DEV)

# ═══ 预测 ═══
img = Image.open(IMG).convert('RGB')
o_w, o_h = img.size
t = T.Compose([T.Resize((256, 320), Image.LANCZOS), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(img).unsqueeze(0).to(DEV)
with torch.no_grad():
    feats = enc(t)
    outputs = dec(feats)
    
    # Check before sigmoid
    x_before_sigmoid = dec.convs[("dispconv", 0)](feats[0])
    print(f'Before sigmoid: min={x_before_sigmoid.min():.6f}, max={x_before_sigmoid.max():.6f}, mean={x_before_sigmoid.mean():.6f}')
    
    # Apply sigmoid manually
    disp_with_sigmoid = dec.sigmoid(x_before_sigmoid)
    print(f'After sigmoid: min={disp_with_sigmoid.min():.6f}, max={disp_with_sigmoid.max():.6f}, mean={disp_with_sigmoid.mean():.6f}')
    
    disp = outputs[("disp", 0)]
    print(f'Raw disp: min={disp.min():.6f}, max={disp.max():.6f}, mean={disp.mean():.6f}')
    d_up = torch.nn.functional.interpolate(disp, (o_h, o_w), mode='bilinear', align_corners=False)
    _, depth = disp_to_depth(d_up, 1.0, 500.0)
    d = depth.cpu().numpy().squeeze()
    print(f'Depth: min={d.min():.3f}, max={d.max():.3f}, mean={d.mean():.3f}, median={np.median(d):.3f}')

del enc, dec
torch.cuda.empty_cache()
