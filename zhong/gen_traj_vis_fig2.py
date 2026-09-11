#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成论文图2: 各方法相机轨迹与真值轨迹对比 (Umeyama 对齐)。

序列: c1_transverse1_t1_v2 (与表2一致)
方法: Monodepth2 / ManyDepth / Lite-Mono / Ours
输出目录: zhong/traj_vis_fig2/
    - gt_traj.npy            GT 轨迹 (相机位置 mm)
    - traj_<name>.npy        各方法 Umeyama 对齐后轨迹
    - fig2_summary.json      ATE / scale 汇总
    - fig2_trajectory_3d.png 3D 轨迹对比
    - fig2_trajectory_proj.png XY/XZ/YZ 投影对比
"""

import os
import sys
import json
import numpy as np
import torch
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

import test_v6_dyendovo as tv6
from test_v6_multi_model_enhanced import (
    load_model, SEQ_NAME, SEQ_DIR, DATA_ROOT)
from dyendovo_dataset import load_gt_poses as dyendo_load_gt_poses
from v6_pipeline.c3vd_loader import align_trajectory_umeyama

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda')

# 4 方法 (与表2一致)
MODELS = [
    {'name': 'Monodepth2', 'path': r'C:\Users\Administrator\tmp\c3vd_md2_multi\models\weights_19', 'type': 'md2'},
    {'name': 'ManyDepth',  'path': r'C:\Users\Administrator\tmp\c3vd_manydepth_multi\models\weights_19', 'type': 'manydepth'},
    {'name': 'Lite-Mono',  'path': r'C:\Users\Administrator\tmp\c3vd_litemono_multi\models\weights_19', 'type': 'litemono'},
    {'name': 'Ours',       'path': r'e:\data1\monodepth2\models\depth', 'type': 'md2'},
]

OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'traj_vis_fig2')
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {
    'GT': '#000000',
    'Monodepth2': '#1f77b4',
    'ManyDepth': '#ff7f0e',
    'Lite-Mono': '#2ca02c',
    'Ours': '#d62728',
}
LINESTYLES = {
    'Monodepth2': '-',
    'ManyDepth': '-',
    'Lite-Mono': '-',
    'Ours': '-',
}


def run_one(cfg, device):
    name = cfg['name']
    print(f'\n===== [{name}] =====')
    encoder, depth_decoder, motion_encoder = load_model(cfg, device)
    params = tv6.estimate_pipeline_params(
        SEQ_DIR, encoder, depth_decoder, motion_encoder, device,
        depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')
    vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
        SEQ_NAME, encoder, depth_decoder, device,
        motion_net=None, mode='baseline', max_frames=None,
        motion_encoder=motion_encoder, depth_source='pred',
        loftr_source='cache', data_root=DATA_ROOT,
        pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale')
    del encoder, depth_decoder, motion_encoder
    torch.cuda.empty_cache()
    return np.array(vo_traj, dtype=np.float64), np.array(gt_traj, dtype=np.float64), stats


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}, 序列: {SEQ_NAME}')

    # ── 1) 跑 VO, 对齐并保存轨迹 ──
    trajs = {}
    gt_ref = None
    summary = {}
    for cfg in MODELS:
        vo, gt, stats = run_one(cfg, device)
        if gt_ref is None:
            gt_ref = gt
        aligned, errors, R, scale = align_trajectory_umeyama(vo, gt)
        ate = float(np.sqrt(np.mean(errors ** 2)))
        trajs[cfg['name']] = aligned
        summary[cfg['name']] = {
            'ate_rmse_mm': ate,
            'umeyama_scale': float(scale),
            'n_frames': int(len(aligned)),
            'n_success': stats['n_success'],
            'n_total': stats['n_total'],
        }
        np.save(os.path.join(OUT_DIR, f'traj_{cfg["name"]}.npy'), aligned)
        print(f'  [{cfg["name"]}] ATE={ate:.2f}mm  scale={scale:.4f}  '
              f'frames={len(aligned)}  success={stats["n_success"]}/{stats["n_total"]}')

    np.save(os.path.join(OUT_DIR, 'gt_traj.npy'), gt_ref)
    with open(os.path.join(OUT_DIR, 'fig2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ── 2) 绘图 ──
    n = min(len(gt_ref), min(len(t) for t in trajs.values()))
    gt = gt_ref[:n]

    # 图A: 3D 轨迹
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=COLORS['GT'], linewidth=3,
            label='Ground Truth', alpha=0.95, zorder=10)
    for name in ['Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']:
        t = trajs[name][:n]
        ax.plot(t[:, 0], t[:, 1], t[:, 2], color=COLORS[name], linewidth=1.8,
                linestyle=LINESTYLES[name], alpha=0.9,
                label=f'{name}  (ATE {summary[name]["ate_rmse_mm"]:.1f} mm)')
    ax.scatter(*gt[0], c='green', s=80, marker='o', zorder=20, label='Start')
    ax.scatter(*gt[-1], c='red', s=80, marker='s', zorder=20, label='End')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Camera Trajectory Comparison ({SEQ_NAME})', fontsize=12)
    ax.legend(fontsize=8, loc='best')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, 'fig2_trajectory_3d.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {p1}')

    # 图B: XY / XZ / YZ 投影
    fig2, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    projs = [('XY', 0, 1), ('XZ', 0, 2), ('YZ', 1, 2)]
    for axp, (title, i, j) in zip(axes, projs):
        axp.plot(gt[:, i], gt[:, j], color=COLORS['GT'], linewidth=2.5,
                 label='Ground Truth', alpha=0.95, zorder=10)
        for name in ['Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']:
            t = trajs[name][:n]
            axp.plot(t[:, i], t[:, j], color=COLORS[name], linewidth=1.5,
                     alpha=0.85, label=name)
        axp.scatter(gt[0, i], gt[0, j], c='green', s=60, marker='o', zorder=20)
        axp.scatter(gt[-1, i], gt[-1, j], c='red', s=60, marker='s', zorder=20)
        axp.set_xlabel(['X', 'X', 'Y'][projs.index((title, i, j))] + ' (mm)')
        axp.set_ylabel(['Y', 'Z', 'Z'][projs.index((title, i, j))] + ' (mm)')
        axp.set_title(f'{title} Projection')
        axp.grid(True, alpha=0.3)
        axp.set_aspect('equal', adjustable='datalim')
        axp.legend(fontsize=8)
    plt.suptitle(f'Trajectory Projections ({SEQ_NAME})', fontsize=12)
    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, 'fig2_trajectory_proj.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {p2}')

    print(f'\n输出目录: {OUT_DIR}')
    for f in sorted(os.listdir(OUT_DIR)):
        print(' ', f)


if __name__ == '__main__':
    main()
