#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GT vs Ours 轨迹对比（原口径 + 过滤 GT 深度阶跃伪影）.

原口径: 同一批 track, 两条轨迹都用 GT 位姿反投影, 起点归零.
新增: 剔除「GT 深度相邻帧 Z 阶跃 > 阈值」的 track（渲染伪影/深度不连续）,
      只保留 GT 深度平滑的真实运动点, 再按 GT 位移排序 Top15.

输出:
  - 过滤统计 + 过滤前后差距对比
  - per_track 图
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
OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis', 'per_track_nojump')
os.makedirs(OUT_DIR, exist_ok=True)

N_TOP = 15
COLS = 3
MAX_GT_JUMP_MM = 15.0      # GT 深度相邻帧 Z 最大阶跃阈值 (组织真实运动 ~5mm, 3倍余量)
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
    ours = {k: npz[k] / gs for k in npz.files}

    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    poses_gt = [np.linalg.inv(T) for T in gt_c2w]
    n_gt = len(poses_gt)

    # 反投影 + GT Z 阶跃检测
    recs = []
    for t in trajs:
        tid = t['track_id']
        gt_pts, ou_pts, z_gt_seq = [], [], []
        for fr in t['frames']:
            fi = int(fr['frame'])
            u, v = float(fr['u']), float(fr['v'])
            if fi >= n_gt:
                continue
            T = poses_gt[fi]
            d_gt = np.array(Image.open(os.path.join(
                SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))).astype(np.float32) * GT_SCALE
            d_ours = ours.get(f'{fi:04d}')
            p_gt = backproject_to_world(u, v, d_gt, T)
            p_ou = backproject_to_world(u, v, d_ours, T) if d_ours is not None else None
            if p_gt is None:
                continue
            gt_pts.append(p_gt)
            ou_pts.append(p_ou if p_ou is not None else None)
            # GT 深度在该点的 Z 值
            h, w = d_gt.shape
            z_gt_seq.append(float(d_gt[int(np.clip(round(v), 0, h-1)),
                                        int(np.clip(round(u), 0, w-1))]))
        if len(gt_pts) < 2:
            continue
        # GT Z 相邻帧最大阶跃
        z = np.array(z_gt_seq)
        max_jump = float(np.max(np.abs(np.diff(z)))) if len(z) >= 2 else 0.0
        recs.append((t, gt_pts, ou_pts, max_jump))

    print(f'有效 track: {len(recs)}')
    jumps = np.array([r[3] for r in recs])
    print(f'GT Z 阶跃分布: p50={np.median(jumps):.2f}, p90={np.percentile(jumps,90):.2f}, '
          f'p95={np.percentile(jumps,95):.2f}, max={jumps.max():.2f} mm')

    keep = [r for r in recs if r[3] <= MAX_GT_JUMP_MM]
    drop = [r for r in recs if r[3] > MAX_GT_JUMP_MM]
    print(f'过滤 GT Z 阶跃 > {MAX_GT_JUMP_MM}mm: 剔除 {len(drop)}, 保留 {len(keep)}')

    # 过滤前后差距对比 (按 GT 位移 Top15)
    def top_stats(recs_list):
        srt = sorted(recs_list, key=lambda r: -disp_of(r[1]))[:N_TOP]
        diffs = []
        for t, gt_pts, ou_pts, _ in srt:
            dg = disp_of(gt_pts)
            do = disp_of(ou_pts)
            diffs.append(abs(dg - do))
        return np.mean(diffs), srt

    mean_before, top_before = top_stats(recs)
    mean_after, top_after = top_stats(keep)
    print(f'\nTop{N_TOP} 平均 |GT-Ours| 位移差距: 过滤前={mean_before:.2f}mm → 过滤后={mean_after:.2f}mm')

    # 出图 (过滤后)
    rows = (N_TOP + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 5.4, rows * 4.4),
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten() if rows * COLS > 1 else [axes]

    for idx, (t, gt_pts, ou_pts, max_jump) in enumerate(top_after):
        ax = axes[idx]
        tid = t['track_id']

        gv = np.array(gt_pts)
        if len(gv) >= 2:
            g0 = gv[0]
            ax.plot(*(gv - g0).T, color=COL_GT, lw=2.4, alpha=0.95,
                    label=f'GT ({disp_of(gt_pts):.1f}mm)')
            ax.scatter(0, 0, 0, color=COL_GT, s=60, marker='o', zorder=6,
                       depthshade=False, edgecolors='k', linewidths=0.5)
            ax.scatter(*(gv[-1] - g0), color=COL_GT, s=90, marker='X',
                       zorder=6, depthshade=False, edgecolors='k', linewidths=0.5)

        ov = np.array([p for p in ou_pts if p is not None])
        if len(ov) >= 2:
            o0 = ov[0]
            ax.plot(*(ov - o0).T, color=COL_OURS, ls='--', lw=1.4, alpha=0.85,
                    label=f'Ours ({disp_of(ou_pts):.1f}mm)')
            ax.scatter(*(ov[-1] - o0), color=COL_OURS, s=45, marker='*',
                       zorder=5, depthshade=False, edgecolors='k', linewidths=0.4)

        ax.set_title(f'Track {tid} (ΔZmax={max_jump:.1f}mm)', fontsize=9, fontweight='bold')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper left')

    for idx in range(len(top_after), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'{SEQ}: GT vs Ours 轨迹 (均用GT位姿, 起点归零)\n'
        f'过滤 GT 深度相邻帧阶跃 > {MAX_GT_JUMP_MM}mm 的伪影 track, 按 GT 位移 Top{N_TOP}',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'per_track_nojump.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_png}')


if __name__ == '__main__':
    main()
