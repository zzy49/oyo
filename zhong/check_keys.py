#!/usr/bin/env python
"""对比 DepthDecoder(scales=[0]) vs DepthDecoder(scales=range(4)) 的 state_dict keys"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder

enc = ResnetEncoder(18, False)

print("=== scales=range(4) ===")
dec_full = DepthDecoder(enc.num_ch_enc, scales=range(4))
sd_full = dec_full.state_dict()
for k in sorted(sd_full.keys()):
    print(f"  {k}: {sd_full[k].shape}")

print(f"\n=== scales=[0] ===")
dec_s0 = DepthDecoder(enc.num_ch_enc, scales=[0])
sd_s0 = dec_s0.state_dict()
for k in sorted(sd_s0.keys()):
    print(f"  {k}: {sd_s0[k].shape}")

# Check checkpoint
import torch
ckpt = torch.load(r'C:\Users\Administrator\tmp\c3vd_md2_full\models\weights_19\depth.pth', map_location='cpu')
print(f"\n=== checkpoint ===")
for k in sorted(ckpt.keys()):
    print(f"  {k}: {ckpt[k].shape}")

# Compare
print(f"\n=== Key match ===")
dec_keys = set(sd_full.keys())
ckpt_keys = set(ckpt.keys())
print(f"Decoder keys: {len(dec_keys)}")
print(f"Checkpoint keys: {len(ckpt_keys)}")
print(f"Match: {len(dec_keys & ckpt_keys)}")
print(f"Only decoder: {dec_keys - ckpt_keys}")
print(f"Only checkpoint: {ckpt_keys - dec_keys}")
