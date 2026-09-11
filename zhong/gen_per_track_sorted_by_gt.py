#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按 GT 位移排序的 GT vs Ours 3D 轨迹叠加对比 (per-track).

口径 (关键决策):
  同一批 track, 两条轨迹都用「GT 相机位姿」反投影, 消除 VO 位姿误差:
    GT:   GT深度(tiff)  + GT位姿  → 反投影 (黑实线)
    Ours: Ours深度(npz) + GT位姿  → 反投影 (红虚线)
  起点各自归零. 这样两轨迹在相同 GT 坐标系下, 差异纯粹来自深度估计.

排序: 按 GT 净位移降序取 Top-15 (聚焦真实运动最大的点, 看两轨迹一致性).

输入:
  zhong/ours_rerun/c1_transverse1_t1_v2_baseline_motion_trajectories.json
  F:/dataset/c1_transverse1_t1_v2/baseline_depth_maps.npz  (Ours 预测深度, ×scale)
  F:/dataset/c1_transverse1_t1_v2/baseline_chain_data.npz  (global_depth_scale)
  F:/dataset/c1_transverse1_t1_v2/depth/{fi:04d}_depth.tiff (GT 深度)
  F:/dataset/c1_transverse1_t1_v2/pose.txt                  (GT 位姿)

输出:
  zhong/ours_rerun/analysis/per_track_sorted_by_gt/
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
DEPTH_NPZ = os.path.join(SEQ_DIR, 'baseline_depth_maps.npz')
CHAIN_NPZ = os.path.join(SEQ_DIR, 'baseline_chain_data.npz')

OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis', 'per_track_sorted_by_gt')
os.makedirs(OUT_DIR, exist_ok=True)

N_TOP = 15
COLS = 3

COL_GT = '#111111'
COL_OURS = '#d62728'


def disp_of(plist):
    valid = [p for p in plist if p is not None]
    if len(valid) < 2:
        return 0.0
    return float(np.linalg.norm(valid[-1] - valid[0]))


def main():
    with open(TRAJ_JSON) as f:
        trajs = json.load(f)
    print(f'Ours 运动 track 数: {len(trajs)}')

    # Ours 深度 ÷global_scale 恢复原始尺度
    chain = np.load(CHAIN_NPZ, allow_pickle=True)
    global_scale = float(chain['global_depth_scale'])
    depth_npz = np.load(DEPTH_NPZ, allow_pickle=True)
    depth_ours = {k: depth_npz[k] / global_scale for k in depth_npz.files}
    print(f'global_scale={global_scale:.4f}, Ours 深度帧: {len(depth_ours)}')

    # GT 位姿 (cam_to_world -> world_to_cam)
    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    poses_gt = [np.linalg.inv(T) for T in gt_c2w]
    n_gt = len(poses_gt)

    # 对每个 track: GT轨迹(用GT位姿+GT深度) 和 Ours轨迹(用GT位姿+Ours深度)
    gt_traj = {}     # tid -> list of (3,) or None
    ours_traj = {}   # tid -> list of (3,) or None
    for t in trajs:
        tid = t['track_id']
        gt_traj[tid] = []
        ours_traj[tid] = []
        for fr in t['frames']:
            fi = int(fr['frame'])
            u, v = float(fr['u']), float(fr['v'])
            if fi >= n_gt:
                gt_traj[tid].append(None)
                ours_traj[tid].append(None)
                continue
            T = poses_gt[fi]
            d_ours = depth_ours.get(f'{fi:04d}')
            ours_traj[tid].append(
                backproject_to_world(u, v, d_ours, T) if d_ours is not None else None)
            d_gt = np.array(Image.open(os.path.join(
                SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))).astype(np.float32) * GT_SCALE
            gt_traj[tid].append(backproject_to_world(u, v, d_gt, T))

    # 按 GT 位移降序排序
    gt_disp = {tid: disp_of(gt_traj[tid]) for tid in gt_traj}
    order = sorted(trajs, key=lambda t: -gt_disp[t['track_id']])[:N_TOP]

    print(f'\nTop{N_TOP} (按 GT 位移排序, 单位 mm):')
    print(f'  {"track":>7} | {"GT":>7} | {"Ours(GT位姿)":>12} | {"Ours(VO位姿)":>13} | 帧数')
    for t in order:
        tid = t['track_id']
        print(f'  {tid:>7} | {gt_disp[tid]:7.2f} | {disp_of(ours_traj[tid]):12.2f} | '
              f'{t["total_displacement_mm"]:13.2f} | {t["n_frames"]}')

    # 绘图
    rows = (N_TOP + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 5.4, rows * 4.4),
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten() if rows * COLS > 1 else [axes]

    for idx, t in enumerate(order):
        ax = axes[idx]
        tid = t['track_id']

        # GT 轨迹
        gvalid = np.array([p for p in gt_traj[tid] if p is not None])
        if len(gvalid) >= 2:
            g0 = gvalid[0]
            ax.plot(*(gvalid - g0).T, color=COL_GT, lw=2.4, alpha=0.95,
                    label=f'GT ({gt_disp[tid]:.1f}mm)')
            ax.scatter(0, 0, 0, color=COL_GT, s=60, marker='o', zorder=6,
                       depthshade=False, edgecolors='k', linewidths=0.5)
            ax.scatter(*(gvalid[-1] - g0), color=COL_GT, s=90, marker='X',
                       zorder=6, depthshade=False, edgecolors='k', linewidths=0.5)

        # Ours 轨迹 (GT 位姿反投影)
        ovalid = np.array([p for p in ours_traj[tid] if p is not None])
        if len(ovalid) >= 2:
            o0 = ovalid[0]
            odisp = disp_of(ours_traj[tid])
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
        f'{SEQ}: 同一批 track 两配置对比 (均用 GT 位姿反投影)\n'
        f'(GT深度 vs Ours深度, 起点各自归零, Top{N_TOP} by GT位移, ○=起点 X/*=终点)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'per_track_sorted_by_gt.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_png}')


if __name__ == '__main__':
    main()
