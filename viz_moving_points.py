"""
可视化运动组织点 3D 轨迹 — 校准对比版
对比 Config A (GT校准) vs Config B (--no_gt) vs GT 相机轨迹

用法: python viz_moving_points.py
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SEQ_DIR = r'F:\dataset\c1_transverse1_t1_v2'
A_DIR = r'e:\data1\monodepth2\zhong\config_a'
B_DIR = r'e:\data1\monodepth2\zhong\config_b'
OUT_DIR = r'e:\data1\monodepth2\zhong'


# ── 数据加载 ──
def load_gt_traj():
    raw = np.loadtxt(os.path.join(SEQ_DIR, 'pose.txt'), delimiter=',')
    poses = raw.reshape(raw.shape[0], 4, 4)
    return poses[:, 3, :3]  # 列主序, translation 在 row3

def load_vo_traj(config_dir):
    poses = np.load(os.path.join(config_dir, 'baseline_abs_poses.npy'))
    return poses[:, :3, 3]

def umeyama(X, Y):
    """Umeyama 对齐: X→Y, 返回 (aligned_X, (R, t, s))"""
    n = X.shape[0]
    mu_x, mu_y = X.mean(0), Y.mean(0)
    X_c, Y_c = X - mu_x, Y - mu_y
    sigma_x = np.sum(X_c ** 2) / n
    S = (Y_c.T @ X_c) / n
    U, s_vec, Vt = np.linalg.svd(S)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = U @ Vt
    s = np.trace(np.diag(s_vec)) / sigma_x if sigma_x > 1e-10 else 1.0
    t = mu_y - s * R @ mu_x
    return s * (X @ R.T) + t, (R, t, s)

def load_motion_trajs(config_dir):
    import glob
    files = glob.glob(os.path.join(config_dir, '*_baseline_motion_trajectories.json'))
    with open(files[0]) as f:
        return json.load(f)


# ── 加载与对齐 ──
gt = load_gt_traj()
vo_a = load_vo_traj(A_DIR)
vo_b = load_vo_traj(B_DIR)

# 对齐帧数
n = min(len(gt), len(vo_a), len(vo_b))
gt, vo_a, vo_b = gt[:n], vo_a[:n], vo_b[:n]

# Umeyama 对齐 VO→GT
vo_a_aligned, (Ra, ta, sa) = umeyama(vo_a, gt)
vo_b_aligned, (Rb, tb, sb) = umeyama(vo_b, gt)

ate_a = np.sqrt(np.mean(np.linalg.norm(vo_a_aligned - gt, axis=1)**2))
ate_b = np.sqrt(np.mean(np.linalg.norm(vo_b_aligned - gt, axis=1)**2))
print(f"ATE: A={ate_a:.2f}mm, B={ate_b:.2f}mm")

# 加载运动轨迹 (flat list)
mt_a = load_motion_trajs(A_DIR)
mt_b = load_motion_trajs(B_DIR)

# 按位移排序
trajs_a = sorted(mt_a, key=lambda x: -x['total_displacement_mm'])
trajs_b = sorted(mt_b, key=lambda x: -x['total_displacement_mm'])

# 将 point 3D 坐标从 VO 空间变换到 GT 空间
def transform_traj_points(traj, R, t, s):
    """对每个 frame 的 (x_mm,y_mm,z_mm) 应用 Umeyama 变换."""
    result = []
    for frame in traj['frames']:
        if frame is None:
            result.append(None)
        else:
            p = np.array([frame['x_mm'], frame['y_mm'], frame['z_mm']])
            p_gt = s * (R @ p) + t
            result.append(tuple(p_gt))
    return result

# 构建变换后的轨迹数据
def build_aligned_trajs(trajs_list, R, t, s):
    out = []
    for tr in trajs_list:
        pts = transform_traj_points(tr, R, t, s)
        frames_with_pts = []
        for i, f in enumerate(tr['frames']):
            if f is None or pts[i] is None:
                continue
            frames_with_pts.append((pts[i], f.get('frame', i)))
        if len(frames_with_pts) >= 2:
            out.append({
                'track_id': tr['track_id'],
                'total_displacement_mm': tr['total_displacement_mm'],
                'n_frames': tr['n_frames'],
                'points': frames_with_pts,
            })
    return out

trajs_a_aligned = build_aligned_trajs(trajs_a, Ra, ta, sa)
trajs_b_aligned = build_aligned_trajs(trajs_b, Rb, tb, sb)

print(f"Aligned tracks: A={len(trajs_a_aligned)}, B={len(trajs_b_aligned)}")


# ═══════════════════════════════════════
# 图1: 运动点 3D 轨迹对比 (两配置并列)
# ═══════════════════════════════════════
fig = plt.figure(figsize=(24, 11))

# 子图1: Config A — Top 50 运动轨迹 + GT 相机
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
# GT 相机轨迹
ax1.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', linewidth=2.5, label='GT Camera', alpha=0.8)
ax1.scatter(*gt[0], c='green', s=80, marker='o', zorder=5)
ax1.scatter(*gt[-1], c='red', s=80, marker='s', zorder=5)

# Top 50 运动点
top_n = min(50, len(trajs_a_aligned))
cmap = plt.cm.YlOrRd
for i, t in enumerate(trajs_a_aligned[:top_n]):
    xs = [p[0][0] for p in t['points']]
    ys = [p[0][1] for p in t['points']]
    zs = [p[0][2] for p in t['points']]
    c = cmap(t['total_displacement_mm'] / max(trajs_a_aligned[0]['total_displacement_mm'], 1))
    ax1.plot(xs, ys, zs, linewidth=0.8, alpha=0.8, color=c)
    ax1.scatter(xs[0], ys[0], zs[0], c=[c], s=10, marker='o')
    ax1.scatter(xs[-1], ys[-1], zs[-1], c=[c], s=15, marker='s')

ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
ax1.set_title(f'Config A (GT calib): Top {top_n} Moving Tracks\n'
              f'n={len(trajs_a_aligned)}, ATE={ate_a:.2f}mm')
ax1.legend(fontsize=8)

# 子图2: Config B — Top 50 运动轨迹 + GT 相机
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', linewidth=2.5, label='GT Camera', alpha=0.8)
ax2.scatter(*gt[0], c='green', s=80, marker='o', zorder=5)
ax2.scatter(*gt[-1], c='red', s=80, marker='s', zorder=5)

top_n = min(50, len(trajs_b_aligned))
for i, t in enumerate(trajs_b_aligned[:top_n]):
    xs = [p[0][0] for p in t['points']]
    ys = [p[0][1] for p in t['points']]
    zs = [p[0][2] for p in t['points']]
    c = cmap(t['total_displacement_mm'] / max(trajs_b_aligned[0]['total_displacement_mm'], 1))
    ax2.plot(xs, ys, zs, linewidth=0.8, alpha=0.8, color=c)
    ax2.scatter(xs[0], ys[0], zs[0], c=[c], s=10, marker='o')
    ax2.scatter(xs[-1], ys[-1], zs[-1], c=[c], s=15, marker='s')

ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
ax2.set_title(f'Config B (no_gt): Top {top_n} Moving Tracks\n'
              f'n={len(trajs_b_aligned)}, ATE={ate_b:.2f}mm')
ax2.legend(fontsize=8)

plt.suptitle(f'Moving Tissue 3D Tracks: A (GT calib) vs B (no_gt)\n'
             f'c1_transverse1_t1_v2  |  color = total displacement (red=more)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save1 = os.path.join(OUT_DIR, 'moving_points_compare_3d.png')
plt.savefig(save1, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {save1}')


# ═══════════════════════════════════════
# 图2: 运动向量 (起点→终点) 两配置对比
# ═══════════════════════════════════════
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(22, 16),
    subplot_kw={'projection': '3d'})

for ax, title_prefix, trajs_a_list, trajs_b_list in [
    (ax1, 'Top 10', trajs_a_aligned[:10], trajs_b_aligned[:10]),
    (ax2, 'Top 10', None, trajs_b_aligned[:10]),
    (ax3, 'Start→End (top 200)', trajs_a_aligned[:200], None),
    (ax4, 'Start→End (top 200)', None, trajs_b_aligned[:200]),
]:
    # 绘制 GT 相机轨迹
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', linewidth=1.5, alpha=0.4)
    ax.scatter(*gt[0], c='green', s=40, marker='o')
    ax.scatter(*gt[-1], c='red', s=40, marker='s')

    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')

# 左上: Config A Top 10 轨迹线
for i, t in enumerate(trajs_a_aligned[:10]):
    xs = [p[0][0] for p in t['points']]
    ys = [p[0][1] for p in t['points']]
    zs = [p[0][2] for p in t['points']]
    c = cmap(i / 10)
    ax1.plot(xs, ys, zs, linewidth=1.5, alpha=0.9, color=c,
             label=f"#{i+1} ({t['total_displacement_mm']:.0f}mm)")
ax1.set_title(f'Config A (GT calib): Top 10 Tracks + GT')
ax1.legend(fontsize=6, loc='upper left')

# 右上: Config B Top 10 轨迹线
for i, t in enumerate(trajs_b_aligned[:10]):
    xs = [p[0][0] for p in t['points']]
    ys = [p[0][1] for p in t['points']]
    zs = [p[0][2] for p in t['points']]
    c = cmap(i / 10)
    ax2.plot(xs, ys, zs, linewidth=1.5, alpha=0.9, color=c,
             label=f"#{i+1} ({t['total_displacement_mm']:.0f}mm)")
ax2.set_title(f'Config B (no_gt): Top 10 Tracks + GT')
ax2.legend(fontsize=6, loc='upper left')

# 左下: Config A 起点→终点向量
starts_a, ends_a = [], []
for t in trajs_a_aligned[:200]:
    pts = t['points']
    starts_a.append(pts[0][0]); ends_a.append(pts[-1][0])
ax3.scatter([s[0] for s in starts_a], [s[1] for s in starts_a], [s[2] for s in starts_a],
            c='blue', alpha=0.3, s=2, label='Start')
ax3.scatter([s[0] for s in ends_a], [s[1] for s in ends_a], [s[2] for s in ends_a],
            c='red', alpha=0.5, s=5, label='End')
for i in range(min(30, len(starts_a))):
    ax3.plot([starts_a[i][0], ends_a[i][0]], [starts_a[i][1], ends_a[i][1]],
             [starts_a[i][2], ends_a[i][2]], 'gray', linewidth=0.5, alpha=0.4)
ax3.set_title(f'Config A (GT calib): Motion Vectors (start→end, top 200)')
ax3.legend(fontsize=8)

# 右下: Config B 起点→终点向量
starts_b, ends_b = [], []
for t in trajs_b_aligned[:200]:
    pts = t['points']
    starts_b.append(pts[0][0]); ends_b.append(pts[-1][0])
ax4.scatter([s[0] for s in starts_b], [s[1] for s in starts_b], [s[2] for s in starts_b],
            c='blue', alpha=0.3, s=2, label='Start')
ax4.scatter([s[0] for s in ends_b], [s[1] for s in ends_b], [s[2] for s in ends_b],
            c='red', alpha=0.5, s=5, label='End')
for i in range(min(30, len(starts_b))):
    ax4.plot([starts_b[i][0], ends_b[i][0]], [starts_b[i][1], ends_b[i][1]],
             [starts_b[i][2], ends_b[i][2]], 'gray', linewidth=0.5, alpha=0.4)
ax4.set_title(f'Config B (no_gt): Motion Vectors (start→end, top 200)')
ax4.legend(fontsize=8)

plt.suptitle('Motion Track Detail: Top 10 + Vectors  |  All aligned to GT frame',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save2 = os.path.join(OUT_DIR, 'moving_points_compare_detail.png')
plt.savefig(save2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {save2}')


# ═══════════════════════════════════════
# 图3: 位移统计分布对比
# ═══════════════════════════════════════
fig3, axes = plt.subplots(2, 3, figsize=(20, 10))

def plot_displacement_stats(ax_row, trajs_list, label, color):
    disps = np.array([t['total_displacement_mm'] for t in trajs_list])
    n_frames = np.array([t['n_frames'] for t in trajs_list])

    ax = ax_row[0]
    ax.hist(disps, bins=60, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(np.median(disps), color='black', linestyle='--', linewidth=2,
               label=f'Median={np.median(disps):.1f}mm')
    ax.axvline(np.mean(disps), color='blue', linestyle='--', linewidth=1.5,
               label=f'Mean={np.mean(disps):.1f}mm')
    ax.set_xlabel('Total Displacement (mm)')
    ax.set_ylabel('Count')
    ax.set_title(f'{label}: Displacement (n={len(disps)})')
    ax.legend(fontsize=7)

    ax = ax_row[1]
    ax.hist(n_frames, bins=40, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Track Length (frames)')
    ax.set_ylabel('Count')
    ax.set_title(f'{label}: Track Length')
    ax.axvline(np.median(n_frames), color='black', linestyle='--',
               label=f'Median={np.median(n_frames):.0f}frames')
    ax.legend(fontsize=7)

    ax = ax_row[2]
    ax.scatter(n_frames, disps, c=color, alpha=0.3, s=4)
    ax.set_xlabel('Track Length (frames)')
    ax.set_ylabel('Total Displacement (mm)')
    ax.set_title(f'{label}: Frames vs Displacement')

plot_displacement_stats([axes[0, 0], axes[0, 1], axes[0, 2]],
                        trajs_a_aligned, 'Config A (GT calib)', '#2ca02c')
plot_displacement_stats([axes[1, 0], axes[1, 1], axes[1, 2]],
                        trajs_b_aligned, 'Config B (no_gt)', '#d62728')

plt.suptitle('Moving Track Statistics: A (GT calib) vs B (no_gt)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save3 = os.path.join(OUT_DIR, 'moving_points_compare_stats.png')
plt.savefig(save3, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {save3}')


# ── 统计打印 ──
for label, trajs in [('Config A (GT calib)', trajs_a_aligned), ('Config B (no_gt)', trajs_b_aligned)]:
    disps = [t['total_displacement_mm'] for t in trajs]
    frames = [t['n_frames'] for t in trajs]
    print(f'\n=== {label} ===')
    print(f'  运动轨迹数: {len(trajs)}')
    print(f'  位移范围: {min(disps):.1f} ~ {max(disps):.1f} mm')
    print(f'  中位位移: {np.median(disps):.1f} mm')
    print(f'  平均位移: {np.mean(disps):.1f} mm')
    print(f'  帧数范围: {min(frames)} ~ {max(frames)}')

print(f'\n✓ All visualizations generated in: {OUT_DIR}')
print(f'  1. {save1}')
print(f'  2. {save2}')
print(f'  3. {save3}')
