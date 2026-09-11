#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GT vs Ours 两配置 3D 轨迹叠加对比 (per-track).

正确口径:
  同一批 track (Ours 输出的运动 track), 两条轨迹:
    GT:   GT深度(tiff) + GT位姿(pose.txt)   → 反投影 3D 轨迹 (黑实线)
    Ours: Ours深度 + Ours位姿               → JSON 内 x/y/z 轨迹 (红虚线)
  起点各自归零, 直观对比运动幅度/方向.

排序: 按 Ours 总位移降序取 Top-15 (聚焦极端运动点).

输入:
  zhong/ours_rerun/c1_transverse1_t1_v2_baseline_motion_trajectories.json
  F:/dataset/c1_transverse1_t1_v2/depth/{fi:04d}_depth.tiff
  F:/dataset/c1_transverse1_t1_v2/pose.txt

输出:
  zhong/ours_rerun/analysis/per_track_gt_vs_ours/
"""

import os
import sys
import json
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = r'e:\data1\monodepth2'
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

from eval_pipeline_motion_vs_gt import backproject_to_world, load_gt_poses_orderF

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'e:\data1\monodepth2\zhong'
SEQ_DIR = r'F:\dataset\c1_transverse1_t1_v2'
SEQ = 'c1_transverse1_t1_v2'
GT_SCALE = 100.0 / 65535.0

TRAJ_JSON = os.path.join(BASE, 'ours_rerun',
                         f'{SEQ}_baseline_motion_trajectories.json')
OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis', 'per_track_gt_vs_ours')
os.makedirs(OUT_DIR, exist_ok=True)

N_TOP = 15
COLS = 3

COL_GT = '#111111'
COL_OURS = '#d62728'


def ours_xyz(t):
    pts = []
    for fr in t['frames']:
        if fr.get('x_mm') is None or fr.get('y_mm') is None or fr.get('z_mm') is None:
            pts.append(None)
        else:
            pts.append([fr['x_mm'], fr['y_mm'], fr['z_mm']])
    return pts


def main():
    with open(TRAJ_JSON) as f:
        trajs = json.load(f)
    print(f'Ours 运动 track 数: {len(trajs)}')

    # GT 位姿
    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    gt_w2c = [np.linalg.inv(T) for T in gt_c2w]
    n_gt = len(gt_w2c)

    # 按帧分组反投影 GT
    frame_reqs = defaultdict(list)
    for t in trajs:
        tid = t['track_id']
        for fr in t['frames']:
            fi = int(fr['frame'])
            if fi < n_gt:
                frame_reqs[fi].append((tid, float(fr['u']), float(fr['v'])))

    gt_pts = {t['track_id']: [] for t in trajs}
    for fi in sorted(frame_reqs.keys()):
        d = np.array(Image.open(os.path.join(SEQ_DIR, 'depth',
                                             f'{fi:04d}_depth.tiff'))
                     ).astype(np.float32) * GT_SCALE
        T = gt_w2c[fi]
        for tid, u, v in frame_reqs[fi]:
            gt_pts[tid].append(backproject_to_world(u, v, d, T))

    # 排序: 按 Ours 位移降序取 Top-K
    order = sorted(trajs, key=lambda t: -t['total_displacement_mm'])[:N_TOP]

    print(f'Top{N_TOP} (按 Ours 位移, 单位 mm):')
    for t in order:
        tid = t['track_id']
        gt_valid = [p for p in gt_pts[tid] if p is not None]
        gt_disp = float(np.linalg.norm(gt_valid[-1] - gt_valid[0])) \
            if len(gt_valid) >= 2 else 0.0
        print(f'  track {tid:>6}: Ours={t["total_displacement_mm"]:6.2f}  '
              f'GT={gt_disp:6.2f}  n_frames={t["n_frames"]}')

    # 绘图
    rows = (N_TOP + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 5.4, rows * 4.4),
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten() if rows * COLS > 1 else [axes]

    for idx, t in enumerate(order):
        ax = axes[idx]
        tid = t['track_id']

        # GT 轨迹
        gvalid = np.array([p for p in gt_pts[tid] if p is not None])
        if len(gvalid) >= 2:
            g0 = gvalid[0]
            gdisp = float(np.linalg.norm(gvalid[-1] - gvalid[0]))
            ax.plot(*(gvalid - g0).T, color=COL_GT, lw=2.4, alpha=0.95,
                    label=f'GT ({gdisp:.1f}mm)')
            ax.scatter(0, 0, 0, color=COL_GT, s=60, marker='o', zorder=6,
                       depthshade=False, edgecolors='k', linewidths=0.5)
            ax.scatter(*(gvalid[-1] - g0), color=COL_GT, s=90, marker='X',
                       zorder=6, depthshade=False, edgecolors='k', linewidths=0.5)

        # Ours 轨迹
        opts = ours_xyz(t)
        ovalid = np.array([p for p in opts if p is not None])
        if len(ovalid) >= 2:
            o0 = ovalid[0]
            odisp = float(t['total_displacement_mm'])
            ax.plot(*(ovalid - o0).T, color=COL_OURS, ls='--', lw=1.4, alpha=0.85,
                    label=f'Ours ({odisp:.1f}mm)')
            ax.scatter(*(ovalid[-1] - o0), color=COL_OURS, s=45, marker='*',
                       zorder=5, depthshade=False, edgecolors='k', linewidths=0.4)

        ax.set_title(f'Track {tid}', fontsize=9, fontweight='bold')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper left')

    for idx in range(len(order), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'{SEQ}: 同一批 track 两配置反投影对比\n'
        f'(GT深度+GT位姿 vs Ours深度+Ours位姿, 起点各自归零, '
        f'Top{N_TOP} by Ours位移, ○=起点 X/*=终点)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'per_track_gt_vs_ours.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_png}')


if __name__ == '__main__':
    main()
