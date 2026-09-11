"""
位姿轨迹详细对比: 3D + 分轴 + 逐帧误差
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


def load_gt_traj():
    raw = np.loadtxt(os.path.join(SEQ_DIR, 'pose.txt'), delimiter=',')
    poses = raw.reshape(raw.shape[0], 4, 4)
    return poses[:, 3, :3]

def load_vo_traj(d):
    return np.load(os.path.join(d, 'baseline_abs_poses.npy'))[:, :3, 3]

def umeyama(X, Y):
    n = X.shape[0]
    mu_x, mu_y = X.mean(0), Y.mean(0)
    X_c, Y_c = X - mu_x, Y - mu_y
    sigma_x = np.sum(X_c**2) / n
    S = (Y_c.T @ X_c) / n
    U, sv, Vt = np.linalg.svd(S)
    R = U @ Vt
    if np.linalg.det(R) < 0: Vt[-1] *= -1; R = U @ Vt
    s = np.trace(np.diag(sv)) / sigma_x if sigma_x > 1e-10 else 1.0
    t = mu_y - s * R @ mu_x
    return s * (X @ R.T) + t, (R, t, s)


# ── 加载 & 对齐 ──
gt = load_gt_traj()
vo_a, vo_b = load_vo_traj(A_DIR), load_vo_traj(B_DIR)
n = min(len(gt), len(vo_a), len(vo_b))
gt, vo_a, vo_b = gt[:n], vo_a[:n], vo_b[:n]

vo_a_aligned, (Ra, ta, sa) = umeyama(vo_a, gt)
vo_b_aligned, (Rb, tb, sb) = umeyama(vo_b, gt)

err_a = np.linalg.norm(vo_a_aligned - gt, axis=1)
err_b = np.linalg.norm(vo_b_aligned - gt, axis=1)
ate_a = np.sqrt(np.mean(err_a**2))
ate_b = np.sqrt(np.mean(err_b**2))

# 分轴误差
err_a_xyz = vo_a_aligned - gt  # (N,3)
err_b_xyz = vo_b_aligned - gt

# 帧间位移 (用于检查轨迹一致性)
step_a = np.linalg.norm(np.diff(vo_a_aligned, axis=0), axis=1)
step_b = np.linalg.norm(np.diff(vo_b_aligned, axis=0), axis=1)
step_gt = np.linalg.norm(np.diff(gt, axis=0), axis=1)

print(f"ATE: A={ate_a:.2f}mm, B={ate_b:.2f}mm")
print(f"Umeyama: A scale={sa:.3f}, B scale={sb:.3f}")
print(f"Mean step size: GT={np.mean(step_gt):.1f}mm, A={np.mean(step_a):.1f}mm, B={np.mean(step_b):.1f}mm")


# ═══════════════════════════════════════
# 图1: 3D 轨迹 + 分轴位置
# ═══════════════════════════════════════
fig1 = plt.figure(figsize=(20, 12))

# 左上: 3D 轨迹
ax3d = fig1.add_subplot(2, 3, (1, 4), projection='3d')
ax3d.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', linewidth=3, label='GT', alpha=0.9, zorder=10)
ax3d.plot(vo_a_aligned[:, 0], vo_a_aligned[:, 1], vo_a_aligned[:, 2],
          '-', color='#2ca02c', linewidth=2, label=f'A (GT calib) ATE={ate_a:.1f}mm')
ax3d.plot(vo_b_aligned[:, 0], vo_b_aligned[:, 1], vo_b_aligned[:, 2],
          '--', color='#d62728', linewidth=2, label=f'B (no_gt) ATE={ate_b:.1f}mm')
# 对应帧连线(GT→A, GT→B) 每5帧
for i in range(0, n, 5):
    ax3d.plot([gt[i, 0], vo_a_aligned[i, 0]], [gt[i, 1], vo_a_aligned[i, 1]],
              [gt[i, 2], vo_a_aligned[i, 2]], color='#2ca02c', linewidth=0.3, alpha=0.3)
    ax3d.plot([gt[i, 0], vo_b_aligned[i, 0]], [gt[i, 1], vo_b_aligned[i, 1]],
              [gt[i, 2], vo_b_aligned[i, 2]], color='#d62728', linewidth=0.3, alpha=0.3)
ax3d.scatter(*gt[0], c='green', s=100, marker='o', zorder=5)
ax3d.scatter(*gt[-1], c='red', s=100, marker='s', zorder=5)
ax3d.set_xlabel('X (mm)'); ax3d.set_ylabel('Y (mm)'); ax3d.set_zlabel('Z (mm)')
ax3d.set_title(f'3D Pose Trajectory  |  Umeyama scales: A={sa:.2f}, B={sb:.2f}')
ax3d.legend(fontsize=9)

frames = np.arange(n)

# 右上: X 轴
ax = fig1.add_subplot(2, 3, 2)
ax.plot(frames, gt[:, 0], 'k-', linewidth=2.5, label='GT')
ax.plot(frames, vo_a_aligned[:, 0], '-', color='#2ca02c', linewidth=1.5, label='A')
ax.plot(frames, vo_b_aligned[:, 0], '--', color='#d62728', linewidth=1.5, label='B')
ax.set_ylabel('X (mm)'); ax.set_title('X Position vs Frame')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 右中: Y 轴
ax = fig1.add_subplot(2, 3, 3)
ax.plot(frames, gt[:, 1], 'k-', linewidth=2.5, label='GT')
ax.plot(frames, vo_a_aligned[:, 1], '-', color='#2ca02c', linewidth=1.5, label='A')
ax.plot(frames, vo_b_aligned[:, 1], '--', color='#d62728', linewidth=1.5, label='B')
ax.set_ylabel('Y (mm)'); ax.set_title('Y Position vs Frame')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 右下: Z 轴
ax = fig1.add_subplot(2, 3, 5)
ax.plot(frames, gt[:, 2], 'k-', linewidth=2.5, label='GT')
ax.plot(frames, vo_a_aligned[:, 2], '-', color='#2ca02c', linewidth=1.5, label='A')
ax.plot(frames, vo_b_aligned[:, 2], '--', color='#d62728', linewidth=1.5, label='B')
ax.set_xlabel('Frame'); ax.set_ylabel('Z (mm)'); ax.set_title('Z Position vs Frame')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 右下: 3D 误差 (A vs B)
ax = fig1.add_subplot(2, 3, 6)
ax.plot(frames, err_a, '-', color='#2ca02c', linewidth=1.5, label=f'A ATE={ate_a:.1f}mm')
ax.plot(frames, err_b, '--', color='#d62728', linewidth=1.5, label=f'B ATE={ate_b:.1f}mm')
ax.fill_between(frames, 0, err_a, color='#2ca02c', alpha=0.1)
ax.fill_between(frames, 0, err_b, color='#d62728', alpha=0.1)
ax.set_xlabel('Frame'); ax.set_ylabel('3D Error (mm)')
ax.set_title('Per-Frame 3D Position Error')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle(f'c1_transverse1_t1_v2  Pose Trajectory Comparison\n'
             f'ATE: A={ate_a:.2f}mm  B={ate_b:.2f}mm  (diff={ate_b-ate_a:+.2f}mm, {(ate_b/ate_a-1)*100:+.1f}%)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
s1 = os.path.join(OUT_DIR, 'traj_detail_3d_axes.png')
plt.savefig(s1, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {s1}')


# ═══════════════════════════════════════
# 图2: 分轴误差 + 帧间位移
# ═══════════════════════════════════════
fig2, axes = plt.subplots(2, 3, figsize=(22, 10))

axis_names = ['X', 'Y', 'Z']
colors_axis = ['#d62728', '#2ca02c', '#1f77b4']

# 上行: 分轴误差 (A)
for j, (ax, aname, c) in enumerate(zip(axes[0], axis_names, colors_axis)):
    ax.plot(frames, err_a_xyz[:, j], '-', color=c, linewidth=1.2, alpha=0.9)
    ax.axhline(0, color='black', linewidth=0.5)
    rms = np.sqrt(np.mean(err_a_xyz[:, j]**2))
    ax.axhline(rms, color=c, linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'RMS={rms:.1f}mm')
    ax.axhline(-rms, color=c, linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_ylabel(f'{aname} Error (mm)')
    ax.set_title(f'Config A (GT calib): {aname} Error')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

# 下行: 分轴误差 (B)
for j, (ax, aname, c) in enumerate(zip(axes[1], axis_names, colors_axis)):
    ax.plot(frames, err_b_xyz[:, j], '-', color=c, linewidth=1.2, alpha=0.9)
    ax.axhline(0, color='black', linewidth=0.5)
    rms = np.sqrt(np.mean(err_b_xyz[:, j]**2))
    ax.axhline(rms, color=c, linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'RMS={rms:.1f}mm')
    ax.axhline(-rms, color=c, linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Frame'); ax.set_ylabel(f'{aname} Error (mm)')
    ax.set_title(f'Config B (no_gt): {aname} Error')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

plt.suptitle(f'Per-Axis Position Error  |  A RMS: X={np.sqrt(np.mean(err_a_xyz[:,0]**2)):.1f} '
             f'Y={np.sqrt(np.mean(err_a_xyz[:,1]**2)):.1f} Z={np.sqrt(np.mean(err_a_xyz[:,2]**2)):.1f} mm  |  '
             f'B RMS: X={np.sqrt(np.mean(err_b_xyz[:,0]**2)):.1f} '
             f'Y={np.sqrt(np.mean(err_b_xyz[:,1]**2)):.1f} Z={np.sqrt(np.mean(err_b_xyz[:,2]**2)):.1f} mm',
             fontsize=12, fontweight='bold')
plt.tight_layout()
s2 = os.path.join(OUT_DIR, 'traj_detail_per_axis_error.png')
plt.savefig(s2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {s2}')


# ═══════════════════════════════════════
# 图3: 帧间位移对比 (验证轨迹平滑性)
# ═══════════════════════════════════════
fig3, axes = plt.subplots(2, 2, figsize=(16, 10))

# 左上: 帧间步长 A vs GT
ax = axes[0, 0]
ax.plot(frames[:-1], step_gt, 'k-', linewidth=1.5, alpha=0.7, label='GT')
ax.plot(frames[:-1], step_a, '-', color='#2ca02c', linewidth=1.2, alpha=0.8, label=f'A (GT calib)')
ax.set_ylabel('Step Size (mm)')
ax.set_title(f'Frame-to-Frame Step: GT vs A\nmean: GT={np.mean(step_gt):.1f}mm, A={np.mean(step_a):.1f}mm')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 右上: 帧间步长 B vs GT
ax = axes[0, 1]
ax.plot(frames[:-1], step_gt, 'k-', linewidth=1.5, alpha=0.7, label='GT')
ax.plot(frames[:-1], step_b, '--', color='#d62728', linewidth=1.2, alpha=0.8, label=f'B (no_gt)')
ax.set_ylabel('Step Size (mm)')
ax.set_title(f'Frame-to-Frame Step: GT vs B\nmean: GT={np.mean(step_gt):.1f}mm, B={np.mean(step_b):.1f}mm')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 左下: 累积位移
ax = axes[1, 0]
cum_gt = np.cumsum(step_gt)
cum_a = np.cumsum(step_a)
cum_b = np.cumsum(step_b)
ax.plot(frames[:-1], cum_gt, 'k-', linewidth=2, label=f'GT (total={cum_gt[-1]:.0f}mm)')
ax.plot(frames[:-1], cum_a, '-', color='#2ca02c', linewidth=1.5, label=f'A (total={cum_a[-1]:.0f}mm)')
ax.plot(frames[:-1], cum_b, '--', color='#d62728', linewidth=1.5, label=f'B (total={cum_b[-1]:.0f}mm)')
ax.set_xlabel('Frame'); ax.set_ylabel('Cumulative Distance (mm)')
ax.set_title('Cumulative Trajectory Length')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 右下: 汇总表
ax = axes[1, 1]
ax.axis('off')
# 分轴 RMSE
rms_a_xyz = [np.sqrt(np.mean(err_a_xyz[:, j]**2)) for j in range(3)]
rms_b_xyz = [np.sqrt(np.mean(err_b_xyz[:, j]**2)) for j in range(3)]
diff_xyz = [rms_b_xyz[j] - rms_a_xyz[j] for j in range(3)]

stats = [
    ['Metric', 'A (GT calib)', 'B (no_gt)', 'Diff'],
    ['ATE RMSE', f'{ate_a:.2f} mm', f'{ate_b:.2f} mm', f'{ate_b-ate_a:+.2f} mm'],
    ['Umeyama Scale', f'{sa:.4f}', f'{sb:.4f}', f'{sb-sa:+.4f}'],
    ['X Error RMS', f'{rms_a_xyz[0]:.2f} mm', f'{rms_b_xyz[0]:.2f} mm', f'{diff_xyz[0]:+.2f} mm'],
    ['Y Error RMS', f'{rms_a_xyz[1]:.2f} mm', f'{rms_b_xyz[1]:.2f} mm', f'{diff_xyz[1]:+.2f} mm'],
    ['Z Error RMS', f'{rms_a_xyz[2]:.2f} mm', f'{rms_b_xyz[2]:.2f} mm', f'{diff_xyz[2]:+.2f} mm'],
    ['Mean Step', f'{np.mean(step_a):.1f} mm', f'{np.mean(step_b):.1f} mm', f'{np.mean(step_b)-np.mean(step_a):+.1f} mm'],
    ['Total Path', f'{cum_a[-1]:.0f} mm', f'{cum_b[-1]:.0f} mm', f'{cum_b[-1]-cum_a[-1]:+.0f} mm'],
]
table = ax.table(cellText=stats, cellLoc='center', loc='center', colWidths=[0.28, 0.24, 0.24, 0.24])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 2.2)
for i in range(4):
    table[0, i].set_facecolor('#404040')
    table[0, i].set_text_props(color='white', fontweight='bold')

plt.suptitle('Frame-by-Frame Step Size & Cumulative Distance', fontsize=14, fontweight='bold')
plt.tight_layout()
s3 = os.path.join(OUT_DIR, 'traj_detail_step_cumulative.png')
plt.savefig(s3, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {s3}')


# ── 打印汇总 ──
print(f'\n=== 分轴 RMSE ===')
for j, name in enumerate(['X', 'Y', 'Z']):
    print(f'  {name}: A={rms_a_xyz[j]:.2f}mm  B={rms_b_xyz[j]:.2f}mm  diff={diff_xyz[j]:+.2f}mm')
print(f'\n✓ Done: {OUT_DIR}')
print(f'  1. {s1}')
print(f'  2. {s2}')
print(f'  3. {s3}')
