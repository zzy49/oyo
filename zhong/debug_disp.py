#!/usr/bin/env python
"""直接检查 decoder 的输出 disp 值"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from PIL import Image
import torchvision.transforms as T

DEV = torch.device('cuda')
IMG = r'F:/dataset/c1_transverse1_t1_v2/generated/rgb_warped/frame_0000.png'

# Load just Monodepth2 via ModelManager, check raw disp
from v6_pipeline.utils import ModelManager
from layers import disp_to_depth

mgr = ModelManager(model_path=r'C:\Users\Administrator\tmp\c3vd_md2_full\models\weights_19', device=DEV)
enc, dec, _, _ = mgr.load_model()
print(f'decoder.num_bins = {getattr(dec, "num_bins", "N/A")}')
print(f'decoder type: {type(dec).__name__}')

img = Image.open(IMG).convert('RGB')
print(f'Image size: {img.size}, mode: {img.mode}')
img_arr = np.array(img)
print(f'Image min/max/mean: {img_arr.min()}/{img_arr.max()}/{img_arr.mean():.1f}')

t = T.ToTensor()(img.resize((320, 256), Image.LANCZOS)).unsqueeze(0).to(DEV)
print(f'Tensor shape: {t.shape}, min={t.min():.3f}, max={t.max():.3f}')

with torch.no_grad():
    feats = enc(t)
    print(f'Features: {len(feats)} scales')
    for i, f in enumerate(feats):
        print(f'  scale{i}: {f.shape}, min={f.min():.4f}, max={f.max():.4f}, mean={f.mean():.4f}')
    
    outputs = dec(feats)
    print(f'\nDecoder output keys: {list(outputs.keys())}')
    
    for key in outputs:
        v = outputs[key]
        print(f'  {key}: {v.shape}, min={v.min():.6f}, max={v.max():.6f}, mean={v.mean():.6f}')
    
    # Check the disp (before disp_to_depth)
    if ("disp", 0) in outputs:
        raw_disp = outputs[("disp", 0)]
        print(f'\nRaw disp (after sigmoid): min={raw_disp.min():.6f}, max={raw_disp.max():.6f}, mean={raw_disp.mean():.6f}')
        
        # Apply disp_to_depth manually
        sd, depth = disp_to_depth(raw_disp, 1.0, 500.0)
        print(f'Scaled disp: min={sd.min():.6f}, max={sd.max():.6f}, mean={sd.mean():.6f}')
        print(f'Depth: min={depth.min():.4f}, max={depth.max():.4f}, mean={depth.mean():.4f}')
    elif ("bins", 0) in outputs:
        print('\nBinary head output')
        bin_logits = outputs[("bins", 0)]
        print(f'Bin logits shape: {bin_logits.shape}')
        residual = outputs.get(("residual", 0), None)
        print(f'Residual: {residual.shape if residual is not None else "None"}')

del enc, dec
torch.cuda.empty_cache()
