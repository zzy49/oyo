#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GT校准 vs 配置C 对比可视化.
生成: 3D轨迹对比、各轴时序、ATE误差时序、误差分布、运动点3D轨迹.
"""

import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

OUT_DIR = r'e:\data1\monodepth2\zhong\zero_gt_compare\viz'
os.makedirs(OUT_DIR, exist_ok=True)

BASE = r'e:\data1\monodepth2\zhong\zero_gt_compare'
DATA_ROOT = r'F:\dataset'
SEQ = 'c1_transverse1_t1_v2'

# ============================================================
# 1. 加载数据
# ============================================================

def load_gt_traj():
    raw = np.loadtxt(os.path.join(DATA_ROOT, SEQ, 'pose.txt'), delimiter=',')
    return raw.reshape(raw.shape[0], 4, 4)[:, 3, :3]

def align_traj(est, gt):
    est, gt = np.array(est, np.float64), np.array(gt, np.float64)
    n = min(len(est), len(gt)); est, gt = est[:n], gt[:n]
    em, gm = est.mean(0), gt.mean(0)
    ec, gc = est - em, gt - gm
    U, _, Vt = np.linalg.svd(ec.T @ gc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[-1] *= -1; R = Vt.T @ U.T
    s = np.trace(R @ ec.T @ gc) / np.trace(ec.T @ ec)
    if s < 1e-6: s = 1.0
    aligned = (s * (R @ est.T)).T + gm - (R @ em * s)
    return aligned, np.linalg.norm(aligned - gt, axis=1), R, s

gt = load_gt_traj()
vo_gt = np.load(os.path.join(BASE, 'config_gt', 'vo_traj_scaled.npy'))
vo_c = np.load(os.path.join(BASE, 'config_C', 'vo_traj_scaled.npy'))

al_gt, err_gt, R_gt, s_gt = align_traj(vo_gt, gt)
al_c, err_c, R_c, s_c = align_traj(vo_c, gt)

n = min(len(al_gt), len(al_c), len(gt))
al_gt, err_gt = al_gt[:n], err_gt[:n]
al_c, err_c = al_c[:n], err_c[:n]
gt_n = gt[:n]

print(f"GT校准 - ATE RMSE: {np.sqrt(np.mean(err_gt**2)):.2f}mm, Umeyama s: {s_gt:.4f}")
print(f"配置C  - ATE RMSE: {np.sqrt(np.mean(err_c**2)):.2f}mm, Umeyama s: {s_c:.4f}")
print(f"差异: {np.sqrt(np.mean(err_c**2)) - np.sqrt(np.mean(err_gt**2)):+.2f}mm")

# ============================================================
# 2. 图1: 3D轨迹对比 + 各轴时序 (4面板)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 左上: 3D 轨迹
ax = axes[0, 0]
ax.plot(gt_n[:, 0], gt_n[:, 1], 'k-', linewidth=1.5, alpha=0.5, label='GT')
ax.plot(al_gt[:, 0], al_gt[:, 1], 'b-', linewidth=1.2, label=f'GT校准 (ATE={np.sqrt(np.mean(err_gt**2)):.2f}mm)')
ax.plot(al_c[:, 0], al_c[:, 1], 'r--', linewidth=1.2, label=f'配置C (ATE={np.sqrt(np.mean(err_c**2)):.2f}mm)')
ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
ax.set_title(f'{SEQ} - XY 轨迹 (Top View)')
ax.legend(fontsize=8); ax.axis('equal'); ax.grid(True, alpha=0.3)

# 右上: XZ 轨迹
ax = axes[0, 1]
ax.plot(gt_n[:, 0], gt_n[:, 2], 'k-', linewidth=1.5, alpha=0.5, label='GT')
ax.plot(al_gt[:, 0], al_gt[:, 2], 'b-', linewidth=1.2, label='GT校准')
ax.plot(al_c[:, 0], al_c[:, 2], 'r--', linewidth=1.2, label='配置C')
ax.set_xlabel('X (mm)'); ax.set_ylabel('Z (mm)')
ax.set_title('XZ 轨迹 (Side View)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 左下: 各轴时序
ax = axes[1, 0]
frames = np.arange(n)
for i, (c, name) in enumerate(zip([0,1,2], ['X','Y','Z'])):
    ax.plot(frames, al_gt[:, c] - gt_n[:, c], '-', alpha=0.6, linewidth=0.8,
            color=['#1f77b4','#ff7f0e','#2ca02c'][i], label=f'GT校准 {name}')
    ax.plot(frames, al_c[:, c] - gt_n[:, c], '--', alpha=0.6, linewidth=0.8,
            color=['#d62728','#9467bd','#8c564b'][i], label=f'配置C {name}')
ax.set_xlabel('Frame'); ax.set_ylabel('Error (mm)')
ax.set_title('Per-Axis Error Over Time')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

# 右下: ATE 时序
ax = axes[1, 1]
ax.plot(frames, err_gt, 'b-', alpha=0.7, linewidth=0.8, label=f'GT校准 (mean={np.mean(err_gt):.2f})')
ax.plot(frames, err_c, 'r--', alpha=0.7, linewidth=0.8, label=f'配置C (mean={np.mean(err_c):.2f})')
ax.set_xlabel('Frame'); ax.set_ylabel('ATE (mm)')
ax.set_title('ATE Error Per Frame')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
path1 = os.path.join(OUT_DIR, 'trajectory_compare.png')
fig.savefig(path1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[OK] {path1}")

# ============================================================
# 3. 图2: 误差分布对比
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: ATE 误差分布直方图
ax = axes[0, 0]
bins = np.linspace(0, max(err_gt.max(), err_c.max()), 40)
ax.hist(err_gt, bins=bins, alpha=0.6, color='b', label=f'GT校准 (mean={np.mean(err_gt):.2f}mm)')
ax.hist(err_c, bins=bins, alpha=0.6, color='r', label=f'配置C (mean={np.mean(err_c):.2f}mm)')
ax.set_xlabel('ATE Error (mm)'); ax.set_ylabel('Count')
ax.set_title('ATE Error Distribution')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# 右上: Per-axis error 箱线图
ax = axes[0, 1]
diff_gt = al_gt - gt_n
diff_c = al_c - gt_n
positions = [0, 1, 2, 4, 5, 6]
data = [diff_gt[:, 0], diff_gt[:, 1], diff_gt[:, 2],
        diff_c[:, 0], diff_c[:, 1], diff_c[:, 2]]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax.set_xticks([1, 5]); ax.set_xticklabels(['GT校准', '配置C'])
ax.set_ylabel('Error (mm)')
ax.set_title('Per-Axis Error Distribution (Box Plot)')
# legend
from matplotlib.patches import Patch
ax.legend([Patch(facecolor='#1f77b4', alpha=0.6), Patch(facecolor='#ff7f0e', alpha=0.6),
           Patch(facecolor='#2ca02c', alpha=0.6)], ['X', 'Y', 'Z'], fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 左下: 误差散点 (GT校准 vs 配置C per-frame)
ax = axes[1, 0]
ax.scatter(err_gt, err_c, s=8, alpha=0.6, c=frames, cmap='viridis')
max_e = max(err_gt.max(), err_c.max())
ax.plot([0, max_e], [0, max_e], 'k--', alpha=0.3)
ax.set_xlabel('GT校准 ATE (mm)'); ax.set_ylabel('配置C ATE (mm)')
ax.set_title('Per-Frame ATE: GT校准 vs 配置C')
ax.axis('equal'); ax.grid(True, alpha=0.3)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Frame')

# 右下: 误差差值 (配置C - GT校准)
ax = axes[1, 1]
delta_err = err_c - err_gt
ax.bar(frames, delta_err, width=1.0, color=['#d62728' if d > 0 else '#2ca02c' for d in delta_err], alpha=0.7)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_xlabel('Frame'); ax.set_ylabel('Delta ATE (mm)')
ax.set_title(f'ATE Diff (配置C - GT校准): mean={np.mean(delta_err):+.2f}mm')
ax.grid(True, alpha=0.3)

plt.tight_layout()
path2 = os.path.join(OUT_DIR, 'error_compare.png')
fig.savefig(path2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[OK] {path2}")

# ============================================================
# 4. 图3: 运动点 3D 轨迹对比
# ============================================================

def load_motion_tracks(config_dir):
    """加载运动点轨迹 (JSON)."""
    traj_path = os.path.join(BASE, config_dir,
                             f'{SEQ}_baseline_motion_trajectories.json')
    with open(traj_path, 'r') as f:
        trajectories = json.load(f)
    return trajectories

def load_tracks_data(config_dir):
    """加载 tracks 分类数据."""
    track_path = os.path.join(BASE, config_dir,
                              f'{SEQ}_baseline_tracks.json')
    with open(track_path, 'r') as f:
        return json.load(f)

def load_abs_poses(config_dir):
    """加载 4x4 绝对位姿."""
    return np.load(os.path.join(BASE, config_dir, 'baseline_abs_poses.npy'))

try:
    traj_gt = load_motion_tracks('config_gt')
    traj_c = load_motion_tracks('config_C')
    tracks_gt = load_tracks_data('config_gt')
    tracks_c = load_tracks_data('config_C')
    poses_gt = load_abs_poses('config_gt')
    poses_c = load_abs_poses('config_C')

    # 建立 track_id → trajectory 的映射
    tid_to_traj_gt = {t['track_id']: t for t in traj_gt}
    tid_to_traj_c = {t['track_id']: t for t in traj_c}

    # 找共同 track_id
    common_ids = sorted(set(tid_to_traj_gt.keys()) & set(tid_to_traj_c.keys()))
    print(f"共同运动 track: {len(common_ids)}")

    if len(common_ids) > 0:
        # 采样一些 track 做可视化
        n_viz = min(30, len(common_ids))
        sample_ids = common_ids[::max(1, len(common_ids) // n_viz)][:n_viz]

        fig = plt.figure(figsize=(18, 12))
        # 3x2 grid: each row = one track, left=3D view, right=per-axis
        # Actually let's do a simpler layout: 6 rows x 5 cols = 30 tracks side by side
        
        # Layout: 3 rows, 2 columns of 3D plots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18),
                                 subplot_kw={'projection': '3d'})
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx >= len(sample_ids):
                ax.set_visible(False)
                continue
                
            tid = sample_ids[idx]
            tg = tid_to_traj_gt[tid]
            tc = tid_to_traj_c[tid]
            
            # Extract 3D points
            pts_g = np.array([[f['x_mm'], f['y_mm'], f['z_mm']] for f in tg['frames'] if f])
            pts_c = np.array([[f['x_mm'], f['y_mm'], f['z_mm']] for f in tc['frames'] if f])
            
            if len(pts_g) >= 2:
                ax.plot(pts_g[:, 0], pts_g[:, 1], pts_g[:, 2], 'b-', linewidth=1.5, label='GT校准')
                ax.scatter(pts_g[0, 0], pts_g[0, 1], pts_g[0, 2], c='b', s=30, marker='o')
                ax.scatter(pts_g[-1, 0], pts_g[-1, 1], pts_g[-1, 2], c='b', s=50, marker='s')
            if len(pts_c) >= 2:
                ax.plot(pts_c[:, 0], pts_c[:, 1], pts_c[:, 2], 'r--', linewidth=1.5, label='配置C')
                ax.scatter(pts_c[0, 0], pts_c[0, 1], pts_c[0, 2], c='r', s=30, marker='o')
                ax.scatter(pts_c[-1, 0], pts_c[-1, 1], pts_c[-1, 2], c='r', s=50, marker='s')
            
            ax.set_title(f'Track {tid}', fontsize=9)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.legend(fontsize=7)

        plt.suptitle(f'{SEQ} - 运动点3D轨迹对比 (GT校准 vs 配置C)', fontsize=14, y=0.98)
        plt.tight_layout()
        path3 = os.path.join(OUT_DIR, 'motion_3d_compare.png')
        fig.savefig(path3, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] {path3}")

        # 运动点位移统计对比
        def compute_displacements(trajectories):
            disps = []
            for t in trajectories:
                frames = t['frames']
                if len(frames) < 2: continue
                valid = [f for f in frames if f]
                if len(valid) < 2: continue
                p0 = np.array([valid[0]['x_mm'], valid[0]['y_mm'], valid[0]['z_mm']])
                pn = np.array([valid[-1]['x_mm'], valid[-1]['y_mm'], valid[-1]['z_mm']])
                disps.append(np.linalg.norm(pn - p0))
            return np.array(disps)

        disp_gt = compute_displacements(traj_gt)
        disp_c = compute_displacements(traj_c)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        bins = np.linspace(0, max(disp_gt.max(), disp_c.max()), 50)
        ax.hist(disp_gt, bins=bins, alpha=0.6, color='b', label=f'GT校准 (median={np.median(disp_gt):.1f}mm)')
        ax.hist(disp_c, bins=bins, alpha=0.6, color='r', label=f'配置C (median={np.median(disp_c):.1f}mm)')
        ax.set_xlabel('Displacement (mm)'); ax.set_ylabel('Count')
        ax.set_title('Motion Point Displacement Distribution')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        # Scatter: GT校準 vs 配置C per-track displacement
        common_disp_gt, common_disp_c = [], []
        for tid in common_ids:
            tg = tid_to_traj_gt[tid]
            tc = tid_to_traj_c[tid]
            fg = [f for f in tg['frames'] if f]
            fc = [f for f in tc['frames'] if f]
            if len(fg) < 2 or len(fc) < 2: continue
            pg0 = np.array([fg[0]['x_mm'], fg[0]['y_mm'], fg[0]['z_mm']])
            pgn = np.array([fg[-1]['x_mm'], fg[-1]['y_mm'], fg[-1]['z_mm']])
            pc0 = np.array([fc[0]['x_mm'], fc[0]['y_mm'], fc[0]['z_mm']])
            pcn = np.array([fc[-1]['x_mm'], fc[-1]['y_mm'], fc[-1]['z_mm']])
            common_disp_gt.append(np.linalg.norm(pgn - pg0))
            common_disp_c.append(np.linalg.norm(pcn - pc0))

        common_disp_gt = np.array(common_disp_gt)
        common_disp_c = np.array(common_disp_c)
        ax.scatter(common_disp_gt, common_disp_c, s=8, alpha=0.5)
        max_d = max(common_disp_gt.max(), common_disp_c.max())
        ax.plot([0, max_d], [0, max_d], 'k--', alpha=0.3)
        ratio = np.median(common_disp_c / (common_disp_gt + 1e-6))
        ax.set_xlabel('GT校准 Displacement (mm)'); ax.set_ylabel('配置C Displacement (mm)')
        ax.set_title(f'Per-Track Displacement (median C/GT ratio={ratio:.3f})')
        ax.axis('equal'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path4 = os.path.join(OUT_DIR, 'motion_disp_compare.png')
        fig.savefig(path4, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] {path4}")

except Exception as e:
    print(f"[WARN] Motion track visualization failed: {e}")
    import traceback; traceback.print_exc()

# ============================================================
# 5. Summary
# ============================================================
print(f"\n{'='*60}")
print(f"可视化文件:")
for p in [path1, path2]:
    print(f"  {p}")
if 'path3' in dir(): print(f"  {path3}")
if 'path4' in dir(): print(f"  {path4}")
print(f"所有文件保存在: {OUT_DIR}")
