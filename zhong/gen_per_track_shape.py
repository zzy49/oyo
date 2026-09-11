#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GT vs Ours 轨迹对比（过滤阶跃 + 逐track位移对齐）.

目的: 让两条轨迹视觉上重合 (用户偏好: 预处理使轨迹一致).

管线:
  1. Ours 深度 ÷global_scale
  2. 反投影 (均用 GT 位姿), 剔除 GT 深度相邻帧 Z 阶跃 > 阈值 的伪影 track
  3. 每条 track: 起点归零后, Ours 轨迹做各向同性缩放 (× GT位移/Ours位移)
     使首末点长度对齐, 只比较轨迹形状/方向
  4. 按 GT 位移排序 Top15 出图
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
OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis', 'per_track_shape')
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
    chain = np.load(CHAIN_NPZ, allow_pickle=True)
    gs = float(chain['global_depth_scale'])
    npz = np.load(DEPTH_NPZ, allow_pickle=True)
    keys = sorted(npz.files)
    n_frames = len(keys)
    H, W = npz[keys[0]].shape

    print('加载 GT 深度 tiff ...')
    gt_stack = np.zeros((n_frames, H, W), dtype=np.float32)
    for i, k in enumerate(keys):
        fi = int(k)
        gt_stack[i] = np.array(Image.open(os.path.join(
            SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))).astype(np.float32) * GT_SCALE

    ours_stack = np.stack([npz[k] / gs for k in keys], axis=0)

    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    poses_gt = [np.linalg.inv(T) for T in gt_c2w]
    n_gt = len(poses_gt)

    recs = []
    for t in trajs:
        gt_pts, ou_pts, z_seq = [], [], []
        for fr in t['frames']:
            fi = int(fr['frame'])
            if fi >= n_frames or fi >= n_gt:
                continue
            u, v = float(fr['u']), float(fr['v'])
            T = poses_gt[fi]
            p_gt = backproject_to_world(u, v, gt_stack[fi], T)
            p_ou = backproject_to_world(u, v, ours_stack[fi], T)
            if p_gt is None or p_ou is None:
                continue
            gt_pts.append(p_gt)
            ou_pts.append(p_ou)
            z_seq.append(float(gt_stack[fi][int(np.clip(round(v), 0, H-1)),
                                           int(np.clip(round(u), 0, W-1))]))
        if len(gt_pts) < 2:
            continue
        z = np.array(z_seq)
        max_jump = float(np.max(np.abs(np.diff(z)))) if len(z) >= 2 else 0.0
        if max_jump <= MAX_GT_JUMP_MM:
            recs.append((t, np.array(gt_pts), np.array(ou_pts), max_jump))

    print(f'有效 track (阶跃≤{MAX_GT_JUMP_MM}mm): {len(recs)}')

    # 对齐: 起点归零 + Ours 缩放至 GT 位移
    aligned = []
    for t, gt_pts, ou_pts, mj in recs:
        g0, o0 = gt_pts[0], ou_pts[0]
        g = gt_pts - g0
        o = ou_pts - o0
        dg = float(np.linalg.norm(g[-1]))
        do = float(np.linalg.norm(o[-1]))
        if do > 1e-6:
            s = dg / do
        else:
            s = 0.0
        o_scaled = o * s
        aligned.append((t, g, o_scaled, mj, dg, do))

    # 排序 Top15
    aligned.sort(key=lambda r: -r[4])
    top = aligned[:N_TOP]

    # 逐点平均距离 (对齐后)
    dists = []
    for t, g, o, mj, dg, do in top:
        n = min(len(g), len(o))
        d = np.linalg.norm(g[:n] - o[:n], axis=1).mean()
        dists.append(d)
    print(f'Top{N_TOP} 对齐后逐点平均距离: mean={np.mean(dists):.2f}mm, median={np.median(dists):.2f}mm')

    # 出图
    rows = (N_TOP + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 5.4, rows * 4.4),
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten() if rows * COLS > 1 else [axes]

    for idx, (t, g, o, mj, dg, do) in enumerate(top):
        ax = axes[idx]
        ax.plot(*g.T, color=COL_GT, lw=2.4, alpha=0.95, label=f'GT ({dg:.1f}mm)')
        ax.scatter(0, 0, 0, color=COL_GT, s=60, marker='o', zorder=6,
                   depthshade=False, edgecolors='k', linewidths=0.5)
        ax.scatter(*g[-1], color=COL_GT, s=90, marker='X', zorder=6,
                   depthshade=False, edgecolors='k', linewidths=0.5)
        ax.plot(*o.T, color=COL_OURS, ls='--', lw=1.6, alpha=0.85,
                label=f'Ours×{dg/max(do,1e-6):.2f} ({do:.1f}mm)')
        ax.scatter(*o[-1], color=COL_OURS, s=45, marker='*', zorder=5,
                   depthshade=False, edgecolors='k', linewidths=0.4)
        ax.set_title(f'Track {t["track_id"]}', fontsize=9, fontweight='bold')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper left')

    for idx in range(len(top), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'{SEQ}: GT vs Ours 轨迹形状对比\n'
        f'(均GT位姿反投影, 起点归零, Ours按GT位移等比缩放, 过滤阶跃>{MAX_GT_JUMP_MM}mm, Top{N_TOP})',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'per_track_shape.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
