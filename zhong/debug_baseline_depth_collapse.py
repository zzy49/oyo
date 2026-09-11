#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断: 三个基线模型深度输出是否坍缩为恒定值 (解释 AbsRel 完全相同现象)。"""
import os, sys, glob
import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

from v6_pipeline.utils import predict_depth
from test_v6_multi_model_enhanced import load_model, MODEL_CONFIGS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('设备:', device)

frames = sorted(glob.glob('F:/dataset/c1_transverse1_t1_v1/generated/rgb_warped/*.png'))
img = frames[0]
print('测试图:', img, ' 共', len(frames), '帧\n')

for cfg in MODEL_CONFIGS:
    name = cfg['name']
    if not os.path.isdir(cfg['path']):
        print(f'{name}: 目录不存在, 跳过')
        continue
    try:
        enc, dec, mot = load_model(cfg, device)
        nb = getattr(dec, 'num_bins', 0)
        d = predict_depth(img, enc, dec, device, feed_h=256, feed_w=320,
                          min_depth=1.0, max_depth=500.0, num_bins=nb)
        d = np.asarray(d, dtype=np.float64)
        flat = d.ravel()
        cv = flat.std() / (flat.mean() + 1e-6)
        print(f'{name} (num_bins={nb}):')
        print(f'   shape={d.shape}  min={d.min():.4f}  max={d.max():.4f}  '
              f'mean={flat.mean():.4f}  median={np.median(flat):.4f}  std={flat.std():.4f}')
        print(f'   变异系数 std/mean={cv:.6f}  (接近0=恒定/坍缩)')
        print(f'   前8像素值: {np.round(flat[:8], 4)}')
        del enc, dec, mot
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        print(f'{name}: ERROR {e}')
        traceback.print_exc()
