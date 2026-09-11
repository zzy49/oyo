#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GT vs Ours 轨迹对比（过滤阶跃 + 逐帧median scaling 组合预处理）.

目的: 让两条轨迹在视觉上尽可能重合 (用户偏好).

预处理管线:
  1. Ours 深度 ÷global_scale 恢复原始尺度
  2. Ours 深度逐帧 median scaling 对齐 GT 深度 (全局中位数, 消除尺度/漂移)
  3. 反投影后, 剔除 GT 深度相邻帧 Z 阶跃 > 阈值 的伪影 track
  4. 剩余按 GT 位移排序 Top15 出图
"""

import os, sys, json
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
TRAJ_JSON = os.path.join(BASE, 'ours_rerun', f'{SEQ}_baseline_motion_trajectories.json')
DEPTH_NPZ = os.path.join(SEQ_DIR, 'baseline_depth_maps.npz')
CHAIN_NPZ = os.path.join(SEQ_DIR, 'baseline_chain_data.npz')
OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis', 'per_track_final')
os.makedirs(OUT_DIR, exist_ok=True)

N_TOP = 15
COLS = 3
MAX_GT_JUMP_MM = 15.0
COL_GT = '#111111'
COL_OURS = '#d62728'


def disp_of(plist):
    valid = [p for p in plist if p is not None]
    if len(valid) < 2:
        return 0.0
    return float(np.linalg.norm(valid[-1] - valid[0]))


def main():
    trajs = json.load(open(TRAJ_JSON))
    print(f'Ours 运动 track 数: {len(trajs)}')

    chain = np.load(CHAIN_NPZ, allow_pickle=True)
    gs = float(chain['global_depth_scale'])
    npz = np.load(DEPTH_NPZ, allow_pickle=True)
    keys = sorted(npz.files)
    n_frames = len(keys)
    H, W = npz[keys[0]].shape
    print(f'global_scale={gs:.4f}, 深度帧={n_frames}, 尺寸={H}x{W}')

    # 加载 GT 深度 (全帧, 一次)
    print('加载 GT 深度 tiff ...')
    gt_stack = np.zeros((n_frames, H, W), dtype=np.float32)
    for i, k in enumerate(keys):
        fi = int(k)
        gt_stack[i] = np.array(Image.open(os.path.join(
            SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))).astype(np.float32) * GT_SCALE

    # Ours 深度
    ours_stack = np.stack([npz[k] / gs for k in keys], axis=0)

    # 逐帧 median scaling
    print('Ours 逐帧 median scaling 对齐 GT ...')
    scales = []
    for i in range(n_frames):
        g = gt_stack[i]
        o = ours_stack[i]
        mask = (g > 0.5) & (o > 0.5)
        if mask.sum() > 100:
            s = float(np.median(g[mask]) / np.median(o[mask]))
            if np.isfinite(s) and s > 0:
                ours_stack[i] = o * s
                scales.append(s)
    print(f'  scale: mean={np.mean(scales):.4f}, std={np.std(scales):.4f}')

    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    poses_gt = [np.linalg.inv(T) for T in gt_c2w]
    n_gt = len(poses_gt)

    # 反投影 + 阶跃过滤
    recs = []
    for t in trajs:
        tid = t['track_id']
        gt_pts, ou_pts, z_seq = [], [], []
        for fr in t['frames']:
            fi = int(fr['frame'])
            if fi >= n_frames or fi >= n_gt:
                continue
            u, v = float(fr['u']), float(fr['v'])
            T = poses_gt[fi]
            d_gt = gt_stack[fi]
            d_ou = ours_stack[fi]
            p_gt = backproject_to_world(u, v, d_gt, T)
            p_ou = backproject_to_world(u, v, d_ou, T)
            if p_gt is None or p_ou is None:
                continue
            gt_pts.append(p_gt)
            ou_pts.append(p_ou)
            z_seq.append(float(d_gt[int(np.clip(round(v), 0, H-1)),
                                    int(np.clip(round(u), 0, W-1))]))
        if len(gt_pts) < 2:
            continue
        z = np.array(z_seq)
        max_jump = float(np.max(np.abs(np.diff(z)))) if len(z) >= 2 else 0.0
        recs.append((t, gt_pts, ou_pts, max_jump))

    print(f'有效 track: {len(recs)}')
    keep = [r for r in recs if r[3] <= MAX_GT_JUMP_MM]
    print(f'过滤 GT Z 阶跃 > {MAX_GT_JUMP_MM}mm: 剔除 {len(recs)-len(keep)}, 保留 {len(keep)}')

    def top_stats(rlist):
        srt = sorted(rlist, key=lambda r: -disp_of(r[1]))[:N_TOP]
        diffs = [abs(disp_of(r[1]) - disp_of(r[2])) for r in srt]
        return np.mean(diffs), srt

    mean_after, top_after = top_stats(keep)
    print(f'Top{N_TOP} 平均 |GT-Ours| 差距 (过滤+scaling后) = {mean_after:.2f}mm')

    # 出图
    rows = (N_TOP + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 5.4, rows * 4.4),
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten() if rows * COLS > 1 else [axes]

    for idx, (t, gt_pts, ou_pts, max_jump) in enumerate(top_after):
        ax = axes[idx]
        gv = np.array(gt_pts)
        ov = np.array(ou_pts)
        if len(gv) >= 2:
            g0 = gv[0]
            ax.plot(*(gv - g0).T, color=COL_GT, lw=2.4, alpha=0.95,
                    label=f'GT ({disp_of(gt_pts):.1f}mm)')
            ax.scatter(0, 0, 0, color=COL_GT, s=60, marker='o', zorder=6,
                       depthshade=False, edgecolors='k', linewidths=0.5)
            ax.scatter(*(gv[-1] - g0), color=COL_GT, s=90, marker='X',
                       zorder=6, depthshade=False, edgecolors='k', linewidths=0.5)
        if len(ov) >= 2:
            o0 = ov[0]
            ax.plot(*(ov - o0).T, color=COL_OURS, ls='--', lw=1.4, alpha=0.85,
                    label=f'Ours ({disp_of(ou_pts):.1f}mm)')
            ax.scatter(*(ov[-1] - o0), color=COL_OURS, s=45, marker='*',
                       zorder=5, depthshade=False, edgecolors='k', linewidths=0.4)
        ax.set_title(f'Track {t["track_id"]}', fontsize=9, fontweight='bold')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper left')

    for idx in range(len(top_after), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'{SEQ}: GT vs Ours 轨迹 (GT位姿反投影, 起点归零)\n'
        f'Ours逐帧median scaling对齐GT + 过滤GT深度阶跃>{MAX_GT_JUMP_MM}mm伪影, Top{N_TOP}',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'per_track_final.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
