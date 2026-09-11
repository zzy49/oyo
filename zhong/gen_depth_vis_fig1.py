#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成论文图1所需的深度图可视化子图（真值 + 4 方法，4 序列第一帧）。

输出目录: zhong/depth_vis_compare/
每个序列一个子目录，内含:
    rgb.png          输入 RGB
    gt.png           真值深度
    monodepth2.png / manydepth.png / litemono.png / ours.png

所有深度图统一使用 TURBO colormap、统一深度显示范围 0-100 mm，
便于横向公平对比（GT 为 uint16 编码 0-100 mm）。
"""

import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

from v6_pipeline.utils import ModelManager, predict_depth
from eval_depth_accuracy import load_manydepth, load_litemono

DEVICE = torch.device('cuda')

GT_SCALE = 100.0 / 65535.0  # uint16 -> mm

SEQS = [
    'c1_transverse1_t1_v2',
    'c1_transverse1_t2_v1',
    'c1_transverse2_t1_v1',
    'c1_transverse1_t1_v1',
]
DATA_ROOT = r'F:\dataset'

# (名称, 模型路径, 类型)  类型: md2 / manydepth / litemono
MODELS = [
    ('monodepth2', r'C:\Users\Administrator\tmp\c3vd_md2_multi\models\weights_19', 'md2'),
    ('manydepth',  r'C:\Users\Administrator\tmp\c3vd_manydepth_multi\models\weights_19', 'manydepth'),
    ('litemono',   r'C:\Users\Administrator\tmp\c3vd_litemono_multi\models\weights_19', 'litemono'),
    ('ours',       r'e:\data1\monodepth2\models\depth', 'md2'),
]

OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'depth_vis_compare')
os.makedirs(OUT_DIR, exist_ok=True)

VMIN, VMAX = 0.0, 100.0  # 统一深度显示范围 (mm)


def save_depth_viz(depth_mm, path, vmin=VMIN, vmax=VMAX):
    d = np.asarray(depth_mm, dtype=np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=vmax, neginf=vmin)
    d = np.clip(d, vmin, vmax)
    norm = (d - vmin) / max(vmax - vmin, 1e-6)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(path, vis)


def median_scale(pred, gt):
    """逐帧 median scaling: 将尺度无关预测对齐到 GT 中位数."""
    mask = (gt > 0.5) & (pred > 0.5)
    if mask.sum() > 100:
        s = float(np.median(gt[mask]) / np.median(pred[mask]))
        return pred * s, s
    return pred, 1.0


def load_one(name, path, mtype):
    """加载单个模型, 返回 (encoder, decoder, num_bins)."""
    print(f'  加载模型 {name} ...')
    if mtype == 'manydepth':
        encoder, decoder = load_manydepth(path, DEVICE)
        nb = 0
    elif mtype == 'litemono':
        encoder, decoder = load_litemono(path, DEVICE)
        nb = 0
    else:  # md2 (Monodepth2 / Ours)
        mgr = ModelManager(model_path=path, device=DEVICE)
        encoder, decoder, _, _ = mgr.load_model()
        nb = getattr(decoder, 'num_bins', 0)
        if name == 'monodepth2':
            encoder.depth_range = (2.0, 100.0)  # Monodepth2 训练 min/max
    return encoder, decoder, nb


def main():
    # 1) 保存所有序列的 RGB 与 GT（不依赖模型），并缓存 GT 供 median scaling
    gt_cache = {}
    for seq in SEQS:
        seq_dir = os.path.join(DATA_ROOT, seq)
        rgb_path = os.path.join(seq_dir, 'generated', 'rgb_warped', 'frame_0000.png')
        gt_path = os.path.join(seq_dir, 'depth', '0000_depth.tiff')
        out_seq = os.path.join(OUT_DIR, seq)
        os.makedirs(out_seq, exist_ok=True)

        rgb = cv2.imread(rgb_path)
        if rgb is not None:
            cv2.imwrite(os.path.join(out_seq, 'rgb.png'), rgb)

        gt_mm = np.array(Image.open(gt_path)).astype(np.float32) * GT_SCALE
        gt_cache[seq] = gt_mm
        save_depth_viz(gt_mm, os.path.join(out_seq, 'gt.png'))
        print(f'[{seq}] RGB={rgb.shape if rgb is not None else "?"} '
              f'GT={gt_mm.shape} 范围[{gt_mm.min():.1f}, {gt_mm.max():.1f}]mm')

    # 2) 逐个模型推理 (避免 ModelManager 单例互相覆盖)
    for name, path, mtype in MODELS:
        print(f'\n===== {name} =====')
        encoder, decoder, nb = load_one(name, path, mtype)
        need_scale = (name != 'ours')  # 三基线尺度无关, 逐帧 median scaling; Ours 物理尺度已对齐
        for seq in SEQS:
            rgb_path = os.path.join(DATA_ROOT, seq, 'generated', 'rgb_warped', 'frame_0000.png')
            out_seq = os.path.join(OUT_DIR, seq)
            pred = predict_depth(rgb_path, encoder, decoder, DEVICE,
                                 feed_h=256, feed_w=320,
                                 min_depth=1.0, max_depth=500.0, num_bins=nb)
            if need_scale:
                pred, s = median_scale(pred, gt_cache[seq])
            else:
                s = 1.0
            save_depth_viz(pred, os.path.join(out_seq, f'{name}.png'))
            print(f'  [{seq}] 范围[{pred.min():.1f}, {pred.max():.1f}]mm '
                  f'median={np.median(pred):.1f}mm scale={s:.3f}')
        del encoder, decoder
        torch.cuda.empty_cache()

    print(f'\n完成。输出目录: {OUT_DIR}')
    for root, _, files in os.walk(OUT_DIR):
        for f in sorted(files):
            print(' ', os.path.relpath(os.path.join(root, f), OUT_DIR))


if __name__ == '__main__':
    main()
