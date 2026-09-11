#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""所有运动点 (全程) VO vs GT 轨迹对比可视化。

复用 test_v6_dyendovo 的反投影 / 逐帧对齐函数, 将 baseline 管线的
全部运动 track 画出 VO 轨迹与 GT 轨迹叠加对比, 不再只取 top_k。

输出 (zhong/all_tracks_vs_gt/):
    - all_tracks_overlay_3d.png      所有运动点 VO(蓝) vs GT(绿) 3D 叠加
    - all_tracks_disp_scatter.png    全量 VO vs GT 位移散点 (r 相关系数)
    - all_tracks_disp_error_hist.png 位移误差直方图
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import test_v6_dyendovo as tv6

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SEQ_NAME = 'c1_transverse1_t1_v2'
MODE = 'baseline'
SEQ_DIR = os.path.join(r'F:\dataset', SEQ_NAME)
OUT_DIR = os.path.join(PROJECT_DIR, 'motion_output')
FIG_DIR = os.path.join(PROJECT_DIR, 'zhong', 'all_tracks_vs_gt')
os.makedirs(FIG_DIR, exist_ok=True)


def load_depth_maps(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return {int(k): data[k] for k in data.files}


def main():
    # ── 1) 加载管线输出 ──
    abs_poses = np.load(os.path.join(SEQ_DIR, f'{MODE}_abs_poses.npy'))   # (N,4,4)
    depth_maps = load_depth_maps(os.path.join(SEQ_DIR, f'{MODE}_depth_maps.npz'))
    chain = np.load(os.path.join(SEQ_DIR, f'{MODE}_chain_data.npz'), allow_pickle=True)
    global_scale = float(chain['global_depth_scale'])
    K = tv6.K_ORIG  # compute_track_displacement 内部固定用 K_ORIG

    with open(os.path.join(OUT_DIR, f'{SEQ_NAME}_{MODE}_tracks.json')) as f:
        td = json.load(f)
    tracks = [{'id': t['id'],
               'frames': [(int(fi), float(u), float(v)) for fi, u, v in t['frames']]}
              for t in td['tracks']]
    moving_ids = set(td['moving_ids'])
    moving_tracks = [(t, 0) for t in tracks if t['id'] in moving_ids]
    print(f'总 tracks: {len(tracks)}, 运动 tracks: {len(moving_tracks)}')

    # ── 2) 深度尺度恢复 (÷global_scale, 与 evaluate_pipeline 一致) ──
    for k in depth_maps:
        depth_maps[k] = depth_maps[k] / global_scale
    print(f'深度尺度恢复: ÷global_scale={global_scale:.4f}')

    # ── 3) GT 位姿 ──
    gt_poses = tv6.load_gt_poses(SEQ_DIR)          # cam_to_world
    n = min(len(gt_poses), len(abs_poses))
    gt_align = gt_poses[:n]
    abs_align = list(abs_poses[:n])
    depth_align = {k: v for k, v in depth_maps.items() if k < n}
    gt_w2c = [np.linalg.inv(T) for T in gt_align]  # world_to_cam

    # 逐帧 VO→GT 刚性变换 (与 visualize_motion_vs_gt 一致)
    vo_to_gt = tv6.compute_per_frame_vo_to_gt(gt_align, abs_align)

    def align_vo_point(p_vo, track):
        if p_vo is None:
            return None
        mid = track['frames'][len(track['frames']) // 2][0]
        if mid not in vo_to_gt:
            return None
        R, t = vo_to_gt[mid]
        return R @ p_vo + t

    # ── 4) 收集所有运动点轨迹 ──
    all_vo, all_gt, all_vo_raw = [], [], []
    vo_disps, gt_disps = [], []
    for track, _ in moving_tracks:
        p_vo_raw, d_vo = tv6.compute_track_displacement(track, depth_align, abs_align, K)
        p_gt, d_gt = tv6.compute_track_displacement(track, depth_align, gt_w2c, K)
        p_vo_aligned = [align_vo_point(p, track) for p in p_vo_raw]
        vo_pts = np.array([p for p in p_vo_aligned if p is not None])
        vo_pts_raw = np.array([p for p in p_vo_raw if p is not None])
        gt_pts = np.array([p for p in p_gt if p is not None])
        if len(vo_pts) >= 2 and len(gt_pts) >= 2:
            all_vo.append(vo_pts)
            all_gt.append(gt_pts)
            all_vo_raw.append(vo_pts_raw)
            vo_disps.append(d_vo)
            gt_disps.append(d_gt)

    vo_disps = np.array(vo_disps)
    gt_disps = np.array(gt_disps)
    err = np.abs(vo_disps - gt_disps)
    print(f'有效运动轨迹: {len(all_vo)}')

    # ── 5) 图0: 运动点绝对位姿全程图 (VO 世界坐标, 不对齐) ──
    # 这正是 run_motion_pipeline 的 save_motion_trajectories 保存的
    # 每帧 x_mm/y_mm/z_mm 绝对位姿 (VO 位姿 + VO 深度反投影), 全程全部运动点。
    fig0 = plt.figure(figsize=(12, 9))
    ax0 = fig0.add_subplot(111, projection='3d')
    norm = plt.Normalize(0, np.percentile(vo_disps, 95))
    cmap = plt.cm.viridis
    for vo_pts, d in zip(all_vo_raw, vo_disps):
        ax0.plot(vo_pts[:, 0], vo_pts[:, 1], vo_pts[:, 2],
                 color=cmap(norm(d)), alpha=0.35, linewidth=0.9)
        ax0.scatter(*vo_pts[0], color=cmap(norm(d)), s=8, marker='s')
        ax0.scatter(*vo_pts[-1], color=cmap(norm(d)), s=8, marker='*')
    ax0.set_xlabel('X (mm)')
    ax0.set_ylabel('Y (mm)')
    ax0.set_zlabel('Z (mm)')
    ax0.set_title(f'运动点绝对位姿全程 ({len(all_vo_raw)} tracks, VO世界坐标)', fontsize=13)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig0.colorbar(sm, ax=ax0, shrink=0.5, label='Displacement (mm)')
    ax0.view_init(elev=25, azim=-60)
    plt.tight_layout()
    p0 = os.path.join(FIG_DIR, 'all_tracks_abs_pose_3d.png')
    plt.savefig(p0, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {p0}')

    # ── 6) 图1: 所有运动点叠加总览 (3D, VO 对齐到 GT 坐标系) ──
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    for vo_pts in all_vo:
        ax.plot(vo_pts[:, 0], vo_pts[:, 1], vo_pts[:, 2],
                color='steelblue', alpha=0.10, linewidth=0.7)
    for gt_pts in all_gt:
        ax.plot(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2],
                color='green', alpha=0.10, linewidth=0.7)
    # 图例占位 (画一条粗线代表)
    ax.plot([], [], [], color='steelblue', linewidth=2, label=f'VO ({len(all_vo)} tracks)')
    ax.plot([], [], [], color='green', linewidth=2, label=f'GT ({len(all_gt)} tracks)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'All Moving Tracks: VO vs GT ({SEQ_NAME}, {MODE})', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, 'all_tracks_overlay_3d.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {p1}')

    # ── 6) 图2: 全量位移散点 ──
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 6))
    max_val = max(vo_disps.max(), gt_disps.max()) * 1.1
    ax2.scatter(gt_disps, vo_disps, c='steelblue', alpha=0.35, s=12, edgecolors='none')
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='Ideal (VO=GT)')
    r = np.corrcoef(gt_disps, vo_disps)[0, 1]
    coeffs = np.polyfit(gt_disps, vo_disps, 1)
    fit_x = np.linspace(0, max_val, 100)
    ax2.plot(fit_x, np.polyval(coeffs, fit_x), 'g-', linewidth=1.5,
             label=f'Fit: VO={coeffs[0]:.3f}×GT+{coeffs[1]:.1f}')
    ax2.set_xlabel('GT Displacement (mm)')
    ax2.set_ylabel('VO Displacement (mm)')
    ax2.set_title(f'VO vs GT Displacement (n={len(vo_disps)}, r={r:.3f})')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # 图3: 位移误差直方图
    ax3.hist(err, bins=60, color='tomato', edgecolor='white', alpha=0.85)
    ax3.axvline(np.median(err), color='red', linestyle='--',
                label=f'Median: {np.median(err):.2f} mm')
    ax3.axvline(np.mean(err), color='blue', linestyle='--',
                label=f'Mean: {np.mean(err):.2f} mm')
    ax3.set_xlabel('|VO - GT| Displacement Error (mm)')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Displacement Error Distribution (n={len(err)})')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, 'all_tracks_disp_scatter.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {p2}')

    # ── 汇总统计 ──
    print('\n===== 汇总 =====')
    print(f'运动轨迹数: {len(all_vo)}')
    print(f'VO 位移:   mean={vo_disps.mean():.2f}  median={np.median(vo_disps):.2f}  max={vo_disps.max():.2f} mm')
    print(f'GT 位移:   mean={gt_disps.mean():.2f}  median={np.median(gt_disps):.2f}  max={gt_disps.max():.2f} mm')
    print(f'|VO-GT|:   mean={err.mean():.2f}  median={np.median(err):.2f}  RMSE={np.sqrt((err**2).mean()):.2f} mm')
    print(f'相关系数 r = {r:.4f}')
    print(f'\n输出目录: {FIG_DIR}')


if __name__ == '__main__':
    main()
