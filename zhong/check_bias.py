#!/usr/bin/env python
"""检查 upconv bias 值"""
import torch
d = torch.load(r'C:\Users\Administrator\tmp\c3vd_md2_full\models\weights_19\depth.pth', map_location='cpu')
print('=== Upconv biases (decoder.0-9) ===')
for i in range(10):
    k = f'decoder.{i}.conv.conv.bias'
    if k in d:
        v = d[k]
        print(f'  decoder.{i}: mean={v.mean():.4f}, min={v.min():.4f}, max={v.max():.4f}')
    else:
        # try alternate key format
        for alt in d:
            if f'decoder.{i}' in alt and 'bias' in alt:
                print(f'  decoder.{i} [{alt}]: mean={d[alt].mean():.4f}')

print('\n=== Dispconv biases (decoder.10-13) ===')
for i in range(10, 14):
    k = f'decoder.{i}.conv.bias'
    if k in d:
        v = d[k]
        print(f'  decoder.{i}: mean={v.mean():.4f}')
