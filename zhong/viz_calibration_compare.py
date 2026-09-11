"""
校准配置对比可视化: A (GT校准) vs B (--no_gt) on c1_transverse1_t1_v2
生成 3 张图: 3D轨迹对比、逐帧ATE、运动检测对比
"""
import json, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import pearsonr

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

def load_tracks(config_dir):
    import glob
    files = glob.glob(os.path.join(config_dir, '*_baseline_tracks.json'))
    with open(files[0]) as f:
        return json.load(f)

def load_motion_trajs(config_dir):
    import glob
    files = glob.glob(os.path.join(config_dir, '*_baseline_motion_trajectories.json'))
    with open(files[0]) as f:
        return json.load(f)

def compute_track_displacements(tracks_data):
    """从 tracks 计算每条的图像坐标位移，返回 (static_disps, moving_disps)."""
    moving_ids = set(tracks_data['moving_ids'])
    static_disp, moving_disp = [], []
    for t in tracks_data['tracks']:
        frames = t['frames']
        if len(frames) < 2:
            continue
        f0, fn = frames[0], frames[-1]
        du = float(fn[1]) - float(f0[1])
        dv = float(fn[2]) - float(f0[2])
        d = np.sqrt(du**2 + dv**2)
        if t['id'] in moving_ids:
            moving_disp.append(d)
        else:
            static_disp.append(d)
    return np.array(static_disp, dtype=float), np.array(moving_disp, dtype=float)


# ── 加载 ──
gt_all = load_gt_traj()
vo_a_raw = load_vo_traj(A_DIR)
vo_b_raw = load_vo_traj(B_DIR)

# 对齐帧数
n = min(len(gt_all), len(vo_a_raw), len(vo_b_raw))
gt = gt_all[:n]
vo_a_raw = vo_a_raw[:n]
vo_b_raw = vo_b_raw[:n]

# Umeyama 对齐到 GT (用于 ATE 计算和可视化)
vo_a_aligned, (Ra, ta, sa) = umeyama(vo_a_raw, gt)
vo_b_aligned, (Rb, tb, sb) = umeyama(vo_b_raw, gt)

err_a = np.linalg.norm(vo_a_aligned - gt, axis=1)
err_b = np.linalg.norm(vo_b_aligned - gt, axis=1)

ate_a_rmse = np.sqrt(np.mean(err_a**2))
ate_b_rmse = np.sqrt(np.mean(err_b**2))

tracks_a = load_tracks(A_DIR)
tracks_b = load_tracks(B_DIR)
mt_a = load_motion_trajs(A_DIR)
mt_b = load_motion_trajs(B_DIR)

# ── 计算 per-track 位移 (用于直方图) ──
_sd_a, _md_a = compute_track_displacements(tracks_a)
_sd_b, _md_b = compute_track_displacements(tracks_b)

print(f"ATE A: {ate_a_rmse:.2f} mm, ATE B: {ate_b_rmse:.2f} mm")
print(f"Config A: {tracks_a['n_total']} tracks ({tracks_a['n_moving']} moving)")
print(f"Config B: {tracks_b['n_total']} tracks ({tracks_b['n_moving']} moving)")


# ═══════════════════════════════════════════════════
# 图1: 3D 轨迹对比 (VO A vs VO B vs GT)
# ═══════════════════════════════════════════════════
fig1 = plt.figure(figsize=(22, 8))

# 子图1: 3D 轨迹
ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
ax1.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', linewidth=2.5, label='GT', alpha=0.9)
ax1.plot(vo_a_aligned[:, 0], vo_a_aligned[:, 1], vo_a_aligned[:, 2],
         '-', color='#2ca02c', linewidth=1.8, label=f'A (GT calib) ATE={ate_a_rmse:.1f}mm')
ax1.plot(vo_b_aligned[:, 0], vo_b_aligned[:, 1], vo_b_aligned[:, 2],
         '--', color='#d62728', linewidth=1.8, label=f'B (no_gt) ATE={ate_b_rmse:.1f}mm')
ax1.scatter(*gt[0], c='black', s=60, marker='o', zorder=5)
ax1.scatter(*gt[-1], c='black', s=80, marker='s', zorder=5)
ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
ax1.set_title('3D Trajectory Comparison')
ax1.legend(fontsize=9, loc='upper left')

# 子图2: X-Y 平面 (俯视)
ax2 = fig1.add_subplot(1, 3, 2)
ax2.plot(gt[:, 0], gt[:, 1], 'k-', linewidth=2.5, label='GT', alpha=0.9)
ax2.plot(vo_a_aligned[:, 0], vo_a_aligned[:, 1], '-', color='#2ca02c', linewidth=1.5, label=f'A (GT calib)')
ax2.plot(vo_b_aligned[:, 0], vo_b_aligned[:, 1], '--', color='#d62728', linewidth=1.5, label=f'B (no_gt)')
ax2.scatter(*gt[0, :2], c='green', s=60, marker='o', zorder=5)
ax2.scatter(*gt[-1, :2], c='red', s=60, marker='s', zorder=5)
ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)')
ax2.set_title('Top-Down View (X-Y)')
ax2.legend(fontsize=8)
ax2.set_aspect('equal')

# 子图3: X-Z 平面 (侧面)
ax3 = fig1.add_subplot(1, 3, 3)
ax3.plot(gt[:, 0], gt[:, 2], 'k-', linewidth=2.5, label='GT', alpha=0.9)
ax3.plot(vo_a_aligned[:, 0], vo_a_aligned[:, 2], '-', color='#2ca02c', linewidth=1.5, label=f'A (GT calib)')
ax3.plot(vo_b_aligned[:, 0], vo_b_aligned[:, 2], '--', color='#d62728', linewidth=1.5, label=f'B (no_gt)')
ax3.scatter(gt[0, 0], gt[0, 2], c='green', s=60, marker='o', zorder=5)
ax3.scatter(gt[-1, 0], gt[-1, 2], c='red', s=60, marker='s', zorder=5)
ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Z (mm)')
ax3.set_title('Side View (X-Z)')
ax3.legend(fontsize=8)
ax3.set_aspect('equal')

plt.suptitle(f'c1_transverse1_t1_v2: Calibration Comparison\n'
             f'ATE: A={ate_a_rmse:.2f}mm  B={ate_b_rmse:.2f}mm  '
             f'(diff: {ate_b_rmse-ate_a_rmse:+.2f}mm, {(ate_b_rmse/ate_a_rmse-1)*100:+.1f}%)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save1 = os.path.join(OUT_DIR, 'calib_compare_trajectory.png')
plt.savefig(save1, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {save1}')


# ═══════════════════════════════════════════════════
# 图2: 逐帧 ATE + 位移分布对比
# ═══════════════════════════════════════════════════
fig2, axes = plt.subplots(2, 2, figsize=(18, 10))

# 左上: 逐帧 ATE
ax = axes[0, 0]
frames = np.arange(n)
ax.plot(frames, err_a, '-', color='#2ca02c', linewidth=1.2, alpha=0.7, label='A (GT calib)')
ax.plot(frames, err_b, '-', color='#d62728', linewidth=1.2, alpha=0.7, label='B (no_gt)')
ax.axhline(ate_a_rmse, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.8)
ax.axhline(ate_b_rmse, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.8)
ax.fill_between(frames, 0, err_a - err_b, where=(err_a < err_b),
                color='#2ca02c', alpha=0.15, label='A better')
ax.fill_between(frames, 0, err_b - err_a, where=(err_b < err_a),
                color='#d62728', alpha=0.15, label='B better')
ax.set_xlabel('Frame')
ax.set_ylabel('ATE (mm)')
ax.set_title(f'Per-Frame ATE (RMSE: A={ate_a_rmse:.1f}, B={ate_b_rmse:.1f} mm)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 右上: 误差差值 (B-A)
ax = axes[0, 1]
diff = err_b - err_a
colors = ['#d62728' if d > 0 else '#2ca02c' for d in diff]
ax.bar(frames, diff, color=colors, width=1.0, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Frame')
ax.set_ylabel('ATE Difference B-A (mm)')
ax.set_title(f'Per-Frame Error Difference (B-A): mean={np.mean(diff):+.2f} mm')
ax.grid(True, alpha=0.3)

# 左下: 静止点 / 运动点位移分布 (Config A)
ax = axes[1, 0]
if len(_sd_a) > 0:
    ax.hist(_sd_a, bins=50, color='#1f77b4', alpha=0.5, label=f'Static (n={len(_sd_a)}, med={np.median(_sd_a):.1f}px)', density=True)
if len(_md_a) > 0:
    ax.hist(_md_a, bins=50, color='#ff7f0e', alpha=0.5, label=f'Moving (n={len(_md_a)}, med={np.median(_md_a):.1f}px)', density=True)
ax.set_xlabel('Displacement (px)')
ax.set_ylabel('Density')
ax.set_title(f'Config A (GT calib): Displacement Distribution')
ax.legend(fontsize=8)
all_a = np.concatenate([_sd_a, _md_a]) if len(_sd_a) and len(_md_a) else np.array([0])
ax.set_xlim(0, np.percentile(all_a, 99))

# 右下: 静止点 / 运动点位移分布 (Config B)
ax = axes[1, 1]
if len(_sd_b) > 0:
    ax.hist(_sd_b, bins=50, color='#1f77b4', alpha=0.5, label=f'Static (n={len(_sd_b)}, med={np.median(_sd_b):.1f}px)', density=True)
if len(_md_b) > 0:
    ax.hist(_md_b, bins=50, color='#ff7f0e', alpha=0.5, label=f'Moving (n={len(_md_b)}, med={np.median(_md_b):.1f}px)', density=True)
ax.set_xlabel('Displacement (px)')
ax.set_ylabel('Density')
ax.set_title(f'Config B (no_gt): Displacement Distribution')
ax.legend(fontsize=8)
all_b = np.concatenate([_sd_b, _md_b]) if len(_sd_b) and len(_md_b) else np.array([0])
ax.set_xlim(0, np.percentile(all_b, 99))

plt.suptitle('Per-Frame Error & Motion Detection Distributions', fontsize=13, fontweight='bold')
plt.tight_layout()
save2 = os.path.join(OUT_DIR, 'calib_compare_error_dist.png')
plt.savefig(save2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {save2}')


# ═══════════════════════════════════════════════════
# 图3: 运动点 3D 轨迹对比 (Top 30 moving tracks)
# ═══════════════════════════════════════════════════
fig3 = plt.figure(figsize=(22, 9))

# 子图1: Config A - 运动轨迹 + 相机路径
ax1 = fig3.add_subplot(1, 2, 1, projection='3d')
ax1.plot(vo_a_aligned[:, 0], vo_a_aligned[:, 1], vo_a_aligned[:, 2],
         'k-', linewidth=1.5, alpha=0.5, label='Camera Path')
trajs_a = sorted(mt_a, key=lambda x: -x['total_displacement_mm'])
top_n = min(30, len(trajs_a))
cmap = plt.cm.YlOrRd
for i, t in enumerate(trajs_a[:top_n]):
    pts = [(f['x_mm'], f['y_mm'], f['z_mm']) for f in t['frames'] if f]
    if len(pts) < 2:
        continue
    xs, ys, zs = zip(*pts)
    c = cmap(i / top_n)
    ax1.plot(xs, ys, zs, linewidth=1.0, alpha=0.7, color=c)
ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
ax1.set_title(f'Config A (GT calib): Top {top_n} Moving Tracks\n'
              f'n_moving={len(trajs_a)}, max_disp={trajs_a[0]["total_displacement_mm"]:.1f}mm')

# 子图2: Config B - 运动轨迹 + 相机路径
ax2 = fig3.add_subplot(1, 2, 2, projection='3d')
ax2.plot(vo_b_aligned[:, 0], vo_b_aligned[:, 1], vo_b_aligned[:, 2],
         'k-', linewidth=1.5, alpha=0.5, label='Camera Path')
trajs_b = sorted(mt_b, key=lambda x: -x['total_displacement_mm'])
top_n = min(30, len(trajs_b))
for i, t in enumerate(trajs_b[:top_n]):
    pts = [(f['x_mm'], f['y_mm'], f['z_mm']) for f in t['frames'] if f]
    if len(pts) < 2:
        continue
    xs, ys, zs = zip(*pts)
    c = cmap(i / top_n)
    ax2.plot(xs, ys, zs, linewidth=1.0, alpha=0.7, color=c)
ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
ax2.set_title(f'Config B (no_gt): Top {top_n} Moving Tracks\n'
              f'n_moving={len(trajs_b)}, max_disp={trajs_b[0]["total_displacement_mm"]:.1f}mm')

plt.suptitle('Motion Track Comparison: Config A (GT calib) vs Config B (no_gt)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save3 = os.path.join(OUT_DIR, 'calib_compare_motion_3d.png')
plt.savefig(save3, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {save3}')


# ═══════════════════════════════════════════════════
# 图4: 综合仪表板 — 四象限指标对比
# ═══════════════════════════════════════════════════
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_names = ['ATE RMSE\n(mm)', 'Static RMSE\n(px)', 'Moving RMSE\n(px)', 'Correlation r']
metrics_a = [ate_a_rmse,
             np.sqrt(np.mean(_sd_a**2)) if len(_sd_a) else 0,
             np.sqrt(np.mean(_md_a**2)) if len(_md_a) else 0,
             pearsonr(
                 np.concatenate([_sd_a, _md_a]),
                 np.concatenate([np.zeros(len(_sd_a)), np.ones(len(_md_a))]))[0]
             if len(_sd_a) and len(_md_a) else 0]
metrics_b = [ate_b_rmse,
             np.sqrt(np.mean(_sd_b**2)) if len(_sd_b) else 0,
             np.sqrt(np.mean(_md_b**2)) if len(_md_b) else 0,
             pearsonr(
                 np.concatenate([_sd_b, _md_b]),
                 np.concatenate([np.zeros(len(_sd_b)), np.ones(len(_md_b))]))[0]
             if len(_sd_b) and len(_md_b) else 0]

x = np.arange(len(metrics_names))
width = 0.35

# 左上: 柱状图对比
ax = axes[0, 0]
bars_a = ax.bar(x - width/2, metrics_a, width, color='#2ca02c', alpha=0.85, label='A (GT calib)')
bars_b = ax.bar(x + width/2, metrics_b, width, color='#d62728', alpha=0.85, label='B (no_gt)')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=9)
ax.set_ylabel('Value')
ax.set_title('Key Metrics Comparison')
ax.legend(fontsize=10)
for bar, val in zip(bars_a, metrics_a):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_a)*0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='#2ca02c')
for bar, val in zip(bars_b, metrics_b):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics_b)*0.02,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8, color='#d62728')
ax.grid(True, alpha=0.2, axis='y')

# 右上: 相对变化 %
ax = axes[0, 1]
rel_changes = [(b - a) / a * 100 if a != 0 else 0 for a, b in zip(metrics_a, metrics_b)]
bar_colors = ['#d62728' if c > 0 else '#2ca02c' for c in rel_changes]
bars = ax.bar(x, rel_changes, width*1.2, color=bar_colors, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=9)
ax.axhline(0, color='black', linewidth=1)
ax.set_ylabel('Relative Change B vs A (%)')
ax.set_title('Metric Change (B relative to A)')
for bar, val in zip(bars, rel_changes):
    y_pos = bar.get_height() + (0.8 if val >= 0 else -2.5)
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.1f}%',
            ha='center', va='bottom' if val >=0 else 'top', fontsize=9, fontweight='bold',
            color='#d62728' if val > 0 else '#2ca02c')
ax.grid(True, alpha=0.2, axis='y')

# 左下: 散点图 — Config A vs B 静止/运动中位
ax = axes[1, 0]
for i, label in enumerate(['Static Median', 'Moving Median']):
    if i == 0:
        ma = np.median(_sd_a) if len(_sd_a) else 0
        mb = np.median(_sd_b) if len(_sd_b) else 0
    else:
        ma = np.median(_md_a) if len(_md_a) else 0
        mb = np.median(_md_b) if len(_md_b) else 0
    ax.scatter(ma, mb, s=200, c=['#1f77b4', '#ff7f0e'][i],
               edgecolors='black', linewidth=1.5, zorder=5)
    ax.annotate(label, (ma, mb), textcoords="offset points", xytext=(10, 5), fontsize=9)
lim = max(metrics_a[1]*1.5, metrics_b[1]*1.5, metrics_a[2]*1.5, metrics_b[2]*1.5)
ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.5, label='y=x')
ax.set_xlabel('Config A (GT calib) px')
ax.set_ylabel('Config B (no_gt) px')
ax.set_title('Static/Moving Median Displacement: A vs B')
ax.legend()
ax.grid(True, alpha=0.2)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)

# 右下: 汇总表格
ax = axes[1, 1]
ax.axis('off')
table_data = [
    ['Metric', 'A (GT calib)', 'B (no_gt)', 'Delta'],
    ['ATE RMSE', f'{ate_a_rmse:.2f} mm', f'{ate_b_rmse:.2f} mm', f'{ate_b_rmse-ate_a_rmse:+.2f}'],
    ['Static Median', f'{np.median(_sd_a) if len(_sd_a) else 0:.1f} px',
     f'{np.median(_sd_b) if len(_sd_b) else 0:.1f} px',
     f'{(np.median(_sd_b) if len(_sd_b) else 0)-(np.median(_sd_a) if len(_sd_a) else 0):+.1f}'],
    ['Moving Median', f'{np.median(_md_a) if len(_md_a) else 0:.1f} px',
     f'{np.median(_md_b) if len(_md_b) else 0:.1f} px',
     f'{(np.median(_md_b) if len(_md_b) else 0)-(np.median(_md_a) if len(_md_a) else 0):+.1f}'],
    ['Correlation r', f'{metrics_a[3]:.4f}', f'{metrics_b[3]:.4f}', f'{metrics_b[3]-metrics_a[3]:+.4f}'],
    ['n_static / n_moving', f'{tracks_a["n_total"]-tracks_a["n_moving"]}/{tracks_a["n_moving"]}',
     f'{tracks_b["n_total"]-tracks_b["n_moving"]}/{tracks_b["n_moving"]}', '-'],
]
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.28, 0.24, 0.24, 0.24])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)
for i in range(len(table_data[0])):
    table[0, i].set_facecolor('#404040')
    table[0, i].set_text_props(color='white', fontweight='bold')

plt.suptitle(f'Calibration Comparison Dashboard: c1_transverse1_t1_v2',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save4 = os.path.join(OUT_DIR, 'calib_compare_dashboard.png')
plt.savefig(save4, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {save4}')

print('\n✓ All visualizations generated!')
print(f'  1. {save1}')
print(f'  2. {save2}')
print(f'  3. {save3}')
print(f'  4. {save4}')
