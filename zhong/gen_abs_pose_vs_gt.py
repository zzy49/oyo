#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运动点绝对 3D 轨迹 vs GT 轨迹 对比可视化（论文表3同口径）.

口径（与 eval_pipeline_motion_paper_metric.py 完全一致）:
  - 深度图 ÷global_scale 恢复原始物理尺度
  - 同一条 track, 用同一深度, 分别用 GT 位姿 / VO 位姿 反投影
  - GT 轨迹(红) = Ours深度 + GT 位姿反投影
  - VO 轨迹(蓝) = Ours深度 + VO 位姿反投影
  - 两条轨迹差距 = 位姿误差在 3D 空间的体现 (~1mm, 合理)
  - GT 运动点: gt_disp >= 1mm

输出:
  1. per_track 图: Top15 运动点, 每条两条绝对轨迹叠加
  2. 散点图: GT位姿位移 vs VO位姿位移 (相关系数 r)
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

from eval_pipeline_motion_vs_gt import (
    chain_tracks, backproject_to_world, load_gt_poses_orderF,
)

DATA_ROOT = r'F:\dataset'
SEQ = 'c1_transverse1_t1_v2'
SEQ_DIR = os.path.join(DATA_ROOT, SEQ)
OUT = os.path.join(PROJECT_DIR, 'zhong', 'abs_pose_vs_gt')
os.makedirs(OUT, exist_ok=True)

GT_STATIC_THRESH_MM = 1.0


def main():
    chain_path = os.path.join(SEQ_DIR, 'baseline_chain_data.npz')
    abs_poses_path = os.path.join(SEQ_DIR, 'baseline_abs_poses.npy')
    depth_path = os.path.join(SEQ_DIR, 'baseline_depth_maps.npz')

    chain = np.load(chain_path, allow_pickle=True)
    depth_npz = np.load(depth_path, allow_pickle=True)
    abs_poses = np.load(abs_poses_path)          # (N,4,4) world_to_cam (VO)
    n_pairs = int(chain['n_pairs'])
    all_k0 = [chain[f'k0_{i:04d}'] for i in range(n_pairs)]
    all_k1 = [chain[f'k1_{i:04d}'] for i in range(n_pairs)]
    chain_dist_thresh = float(chain['chain_dist_thresh']) if 'chain_dist_thresh' in chain else 3.0
    global_scale = float(chain['global_depth_scale']) if 'global_depth_scale' in chain else 1.0
    depth_src = str(chain['depth_source']) if 'depth_source' in chain else 'pred'

    depth_maps = {k: depth_npz[k] for k in depth_npz.files}
    if depth_src != 'gt' and global_scale != 1.0:
        depth_maps = {k: v / global_scale for k, v in depth_maps.items()}
    print(f"深度尺度恢复: ÷global_scale={global_scale:.4f}")

    gt_poses_c2w = load_gt_poses_orderF(SEQ_DIR)
    gt_poses_w2c = [np.linalg.inv(T) for T in gt_poses_c2w]
    n_gt = len(gt_poses_w2c)
    n_vo = len(abs_poses)
    print(f"GT位姿={n_gt}, VO位姿={n_vo}")

    tracks = chain_tracks(all_k0, all_k1, dist_thresh=chain_dist_thresh)
    print(f"总 tracks={len(tracks)}")

    # 每条 track: 完整轨迹反投影 (GT位姿 vs VO位姿)
    recs = []   # (gt_disp, vo_disp, gt_traj[N,3], vo_traj[N,3], tid)
    for t in tracks:
        frames = t['frames']
        if len(frames) < 5:
            continue
        gt_pts, vo_pts = [], []
        ok = True
        for (fi, u, v) in frames:
            if fi >= n_gt or fi >= n_vo:
                ok = False
                break
            dm = depth_maps.get(f'{fi:04d}')
            p_gt = backproject_to_world(u, v, dm, gt_poses_w2c[fi])
            p_vo = backproject_to_world(u, v, dm, abs_poses[fi])
            if p_gt is None or p_vo is None:
                ok = False
                break
            gt_pts.append(p_gt)
            vo_pts.append(p_vo)
        if not ok or len(gt_pts) < 2:
            continue
        gt_traj = np.array(gt_pts)
        vo_traj = np.array(vo_pts)
        gt_disp = float(np.linalg.norm(gt_traj[-1] - gt_traj[0]))
        vo_disp = float(np.linalg.norm(vo_traj[-1] - vo_traj[0]))
        recs.append((gt_disp, vo_disp, gt_traj, vo_traj, t['id']))

    print(f"有效 track={len(recs)}")
    gt_disps = np.array([r[0] for r in recs])
    vo_disps = np.array([r[1] for r in recs])

    # 散点图
    corr = float(np.corrcoef(gt_disps, vo_disps)[0, 1])
    moving = [r for r in recs if r[0] >= GT_STATIC_THRESH_MM]
    print(f"GT运动点(gt>={GT_STATIC_THRESH_MM}mm)={len(moving)}, 相关系数 r={corr:.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(gt_disps, vo_disps, s=4, alpha=0.3, c='#555555', rasterized=True)
    maxd = max(gt_disps.max(), vo_disps.max())
    ax.plot([0, maxd], [0, maxd], 'r--', linewidth=1.2, alpha=0.6)
    ax.set_xlabel('GT位姿反投影位移 (mm)')
    ax.set_ylabel('VO位姿反投影位移 (mm)')
    ax.set_title(f'{SEQ}: 运动点位移 GT位姿 vs VO位姿\nr={corr:.3f}, n={len(recs)}')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    p_scatter = os.path.join(OUT, 'gt_vo_disp_scatter.png')
    fig.savefig(p_scatter, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', p_scatter)

    # per-track 图: 选 GT 运动点, 按 gt_disp 排序取 Top15
    moving.sort(key=lambda r: -r[0])
    top_n = min(15, len(moving))
    top = moving[:top_n]

    cols, rows = 3, (top_n + 2) // 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5.5),
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for idx, (gt_disp, vo_disp, gt_traj, vo_traj, tid) in enumerate(top):
        ax = axes[idx]
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], '-', color='#e74c3c',
                linewidth=1.8, alpha=0.9, label=f'GT位姿 ({gt_disp:.1f}mm)')
        ax.plot(vo_traj[:, 0], vo_traj[:, 1], vo_traj[:, 2], '--', color='#3498db',
                linewidth=1.8, alpha=0.9, label=f'VO位姿 ({vo_disp:.1f}mm)')
        ax.scatter(*gt_traj[0], c='#e74c3c', s=30, marker='o', zorder=5)
        ax.scatter(*gt_traj[-1], c='#e74c3c', s=50, marker='s', zorder=5)
        ax.scatter(*vo_traj[0], c='#3498db', s=30, marker='o', zorder=5)
        ax.scatter(*vo_traj[-1], c='#3498db', s=50, marker='s', zorder=5)
        ax.set_title(f'Track {tid}\nGT位移={gt_disp:.1f}mm', fontsize=9, fontweight='bold')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend(fontsize=6, loc='upper left')

    for idx in range(top_n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'{SEQ}: 运动点绝对3D轨迹 — Ours深度 + GT位姿(红) vs VO位姿(蓝)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p_track = os.path.join(OUT, 'per_track_abs_pose_vs_gt.png')
    fig.savefig(p_track, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', p_track)

    # 汇总
    print('\n' + '=' * 60)
    print(f"运动点绝对位姿对比 (论文口径) 完成")
    print(f"  相关系数 r = {corr:.3f}")
    print(f"  GT运动点 = {len(moving)}")
    mask = gt_disps >= GT_STATIC_THRESH_MM
    print(f"  GT位移中位 = {np.median(gt_disps[mask]):.2f}mm" if mask.sum() else "  -")
    print(f"  VO位移中位 = {np.median(vo_disps[mask]):.2f}mm" if mask.sum() else "  -")
    print(f"输出目录: {OUT}")


if __name__ == '__main__':
    main()
