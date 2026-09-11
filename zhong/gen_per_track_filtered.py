#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""过滤 GT 深度单帧跳变噪声后, GT vs Ours 轨迹叠加对比 (per-track).

GT 深度(tiff 渲染深度)在边缘/遮挡区域存在单帧跳变噪声(如 86↔33mm 来回跳),
会虚增 GT 反投影位移。本脚本:
  1. 对每条 track 统计 GT 深度的最大相邻帧跳变 max|ΔZ|
  2. 剔除 max|ΔZ| > MAX_GT_JUMP 的噪声 track
  3. 剩余 track 按 GT 位移降序取 Top-K, 出两轨迹叠加图

口径:
  GT:   GT深度 + GT位姿 (黑实线)
  Ours: Ours深度 + GT位姿 (红虚线, 消除 VO 位姿误差)
  起点各自归零, 差异纯粹来自深度估计.

输入:
  zhong/ours_rerun/c1_transverse1_t1_v2_baseline_motion_trajectories.json
  F:/dataset/c1_transverse1_t1_v2/baseline_depth_maps.npz
  F:/dataset/c1_transverse1_t1_v2/baseline_chain_data.npz
  F:/dataset/c1_transverse1_t1_v2/depth/{fi:04d}_depth.tiff
  F:/dataset/c1_transverse1_t1_v2/pose.txt

输出:
  zhong/ours_rerun/analysis/per_track_filtered/
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

OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis', 'per_track_filtered')
os.makedirs(OUT_DIR, exist_ok=True)

N_TOP = 15
COLS = 3
MAX_GT_JUMP = 40.0   # GT 深度相邻帧最大跳变阈值 (mm), 超过视为噪声

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

    chain = np.load(CHAIN_NPZ, allow_pickle=True)
    global_scale = float(chain['global_depth_scale'])
    depth_npz = np.load(DEPTH_NPZ, allow_pickle=True)
    depth_ours = {k: depth_npz[k] / global_scale for k in depth_npz.files}

    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    poses_gt = [np.linalg.inv(T) for T in gt_c2w]
    n_gt = len(poses_gt)

    # 收集所有 track 需要的帧, 按帧读 GT 深度缓存
    frame_needed = set()
    for t in trajs:
        for fr in t['frames']:
            fi = int(fr['frame'])
            if fi < n_gt:
                frame_needed.add(fi)
    gt_depth_frames = {}
    for fi in sorted(frame_needed):
        gt_depth_frames[fi] = np.array(Image.open(os.path.join(
            SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))).astype(np.float32) * GT_SCALE
    print(f'GT 深度帧已缓存: {len(gt_depth_frames)}')

    # 逐 track 统计 GT 深度最大相邻帧跳变 + 反投影轨迹
    jump_stats = []
    gt_traj = {}
    ours_traj = {}
    for t in trajs:
        tid = t['track_id']
        gt_traj[tid] = []
        ours_traj[tid] = []
        zs = []
        for fr in t['frames']:
            fi = int(fr['frame'])
            u, v = float(fr['u']), float(fr['v'])
            ui, vi = int(round(u)), int(round(v))
            if fi >= n_gt:
                gt_traj[tid].append(None)
                ours_traj[tid].append(None)
                continue
            T = poses_gt[fi]
            d_gt = gt_depth_frames[fi]
            gt_traj[tid].append(backproject_to_world(u, v, d_gt, T))
            d_ours = depth_ours.get(f'{fi:04d}')
            ours_traj[tid].append(
                backproject_to_world(u, v, d_ours, T) if d_ours is not None else None)
            zs.append(d_gt[vi, ui])
        # 最大相邻帧跳变
        max_jump = 0.0
        for i in range(1, len(zs)):
            max_jump = max(max_jump, abs(zs[i] - zs[i - 1]))
        jump_stats.append((tid, max_jump))

    jumps = np.array([j for _, j in jump_stats])
    print(f'\nGT 深度相邻帧最大跳变分布: '
          f'min={jumps.min():.1f} median={np.median(jumps):.1f} '
          f'mean={jumps.mean():.1f} max={jumps.max():.1f}')
    n_noise = int((jumps > MAX_GT_JUMP).sum())
    print(f'噪声 track (max|ΔZ| > {MAX_GT_JUMP}mm): {n_noise} / {len(trajs)}')

    # 过滤
    noisy_ids = {tid for tid, j in jump_stats if j > MAX_GT_JUMP}
    clean = [t for t in trajs if t['track_id'] not in noisy_ids]
    print(f'过滤后 track 数: {len(clean)}')

    # 按 GT 位移降序
    gt_disp = {tid: disp_of(gt_traj[tid]) for tid in gt_traj}
    order = sorted(clean, key=lambda t: -gt_disp[t['track_id']])[:N_TOP]

    print(f'\nTop{N_TOP} (过滤后按 GT 位移排序, 单位 mm):')
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

        gvalid = np.array([p for p in gt_traj[tid] if p is not None])
        if len(gvalid) >= 2:
            g0 = gvalid[0]
            ax.plot(*(gvalid - g0).T, color=COL_GT, lw=2.4, alpha=0.95,
                    label=f'GT ({gt_disp[tid]:.1f}mm)')
            ax.scatter(0, 0, 0, color=COL_GT, s=60, marker='o', zorder=6,
                       depthshade=False, edgecolors='k', linewidths=0.5)
            ax.scatter(*(gvalid[-1] - g0), color=COL_GT, s=90, marker='X',
                       zorder=6, depthshade=False, edgecolors='k', linewidths=0.5)

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
        f'{SEQ}: 过滤 GT 深度单帧噪声后两配置对比 (均用 GT 位姿反投影)\n'
        f'(剔除 max|ΔZ|>{MAX_GT_JUMP:.0f}mm, Top{N_TOP} by GT位移, ○=起点 X/*=终点)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'per_track_filtered.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_png}')


if __name__ == '__main__':
    main()
