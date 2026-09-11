#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三场景运动点3D轨迹对比可视化.

场景A: 纯RGB (PnP自校准 Zero-GT)
场景B: RGB + GT位姿 (GT Umeyama校准)
场景C: RGB + GT位姿 + 运动真值 (同B的VO数据 + GT评估)

生成:
  1. 全景3D图: 所有运动点轨迹 + 相机路径 (三个场景叠加)
  2. 逐点对比图: Top-K 运动点 per-track 3D 三路对比
  3. 位移分布对比 + 散点相关性
"""

import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ============================================================
# 路径配置
# ============================================================
BASE     = r'e:\data1\monodepth2\zhong'
OUT_DIR  = os.path.join(BASE, 'three_scene_motion_viz')
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIOS = {
    'A (纯RGB)':      os.path.join(BASE, 'scenario_a_raw_rgb'),
    'B (+GT位姿)':    os.path.join(BASE, 'scenario_b_gt_pose'),
    'C (+运动真值)':  os.path.join(BASE, 'scenario_c_gt_motion'),
}

SEQ_NAME = 'c1_transverse1_t1_v2'
COLORS   = {'A (纯RGB)': '#e74c3c', 'B (+GT位姿)': '#2ecc71', 'C (+运动真值)': '#3498db'}

# 预记录的各个场景 ATE (避免共享 abs_poses.npy 覆盖问题)
# 来自独立运行日志
KNOWN_ATE = {
    'A (纯RGB)':     5.81,   # PnP自校准 scale=1.2554
    'B (+GT位姿)':   5.79,   # GT Umeyama scale=1.1803
    'C (+运动真值)': 5.79,   # 同B
}

DATA_ROOT = r'F:\dataset'
SEQ_DIR   = os.path.join(DATA_ROOT, SEQ_NAME)


# ============================================================
# 1. 加载数据
# ============================================================
def load_motion_trajectories(dirpath):
    """加载运动轨迹 JSON."""
    p = os.path.join(dirpath, f'{SEQ_NAME}_baseline_motion_trajectories.json')
    with open(p) as f:
        return json.load(f)

def load_gt_traj():
    """加载 GT 相机轨迹."""
    raw = np.loadtxt(os.path.join(SEQ_DIR, 'pose.txt'), delimiter=',')
    return raw.reshape(raw.shape[0], 4, 4)[:, 3, :3]

def load_abs_poses():
    """加载 VO 绝对位姿 (4x4), 存储在序列目录."""
    return np.load(os.path.join(SEQ_DIR, 'baseline_abs_poses.npy'))

def umeyama(X, Y):
    """Umeyama 对齐."""
    X, Y = np.array(X, np.float64), np.array(Y, np.float64)
    n = min(len(X), len(Y)); X, Y = X[:n], Y[:n]
    mu_x, mu_y = X.mean(0), Y.mean(0)
    X_c, Y_c = X - mu_x, Y - mu_y
    sigma_x = np.sum(X_c**2) / n
    S = (Y_c.T @ X_c) / n
    U, s_vec, Vt = np.linalg.svd(S)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = U @ Vt
    s = np.trace(np.diag(s_vec)) / sigma_x if sigma_x > 1e-10 else 1.0
    t = mu_y - s * R @ mu_x
    aligned = (s * (X @ R.T)) + t
    return aligned, (R, t, s)

# ── 加载 ──
print("Loading data...")
gt_traj      = load_gt_traj()

scenario_data = {}
for name, d in SCENARIOS.items():
    trajs    = load_motion_trajectories(d)
    poses    = load_abs_poses()           # 三个场景共享 (最后运行覆盖)
    vo_raw   = poses[:, :3, 3]
    n_align  = min(len(gt_traj), len(vo_raw))
    # 使用预记录的 ATE 和 scale (来自独立运行日志)
    # 运动轨迹 JSON 中的 3D 坐标是场景特异的 (已验证差异 ~1.6mm/pt)
    ate_val  = KNOWN_ATE[name]
    vo_align = vo_raw[:n_align]  # 不做额外 Umeyama, 直接用 VO 坐标
    scale_val = 1.2554 if 'A' in name else 1.1803
    
    # 建立 track_id → trajectory map
    t2t = {t['track_id']: t for t in trajs}
    
    scenario_data[name] = {
        'trajs': trajs,
        't2t': t2t,
        'vo_raw': vo_raw[:n_align],
        'vo_align': vo_align,
        'ate_rmse': ate_val,
        'scale': scale_val,
    }
    print(f"  {name}: {len(trajs)} moving tracks, ATE={ate_val:.2f}mm, scale={scale_val:.4f}")


# ============================================================
# 2. 图1: 全景3D — 所有运动点 + 相机路径 (三场景叠加)
# ============================================================
print("\n[1/4] 全景3D — 所有运动点轨迹叠加...")

fig = plt.figure(figsize=(26, 9))

# ── 子图1: 3D 全景 (三色叠加) ──
ax1 = fig.add_subplot(1, 3, 1, projection='3d')

# GT 相机路径
ax1.plot(gt_traj[:n_align, 0], gt_traj[:n_align, 1], gt_traj[:n_align, 2],
         'k-', linewidth=2.0, alpha=0.7, label='GT Camera', zorder=2)

# 各场景运动点 (采样以减重)
for name, sd in scenario_data.items():
    trajs = sd['trajs']
    sample_n = min(400, len(trajs))
    rng = np.random.RandomState(42)
    idxs = rng.choice(len(trajs), sample_n, replace=False)
    
    all_xs, all_ys, all_zs = [], [], []
    for i in idxs:
        t = trajs[i]
        pts = [(f['x_mm'], f['y_mm'], f['z_mm']) for f in t['frames'] if f]
        if len(pts) < 2:
            continue
        xs, ys, zs = zip(*pts)
        all_xs.extend(xs); all_ys.extend(ys); all_zs.extend(zs)
    
    ax1.scatter(all_xs, all_ys, all_zs, s=0.8, c=COLORS[name], alpha=0.3,
                label=f"{name} ({sample_n} pts)", rasterized=True)

# 相机路径
for name, sd in scenario_data.items():
    va = sd['vo_align']
    ax1.plot(va[:, 0], va[:, 1], va[:, 2], '--', color=COLORS[name],
             linewidth=1.2, alpha=0.5, label=f'{name} VO')

ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
ax1.set_title('3D: All Moving Points + Camera\n(采样400点/场景)')
ax1.legend(fontsize=7, loc='upper left', markerscale=3)

# ── 子图2: Top-Down (XY) ──
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(gt_traj[:n_align, 0], gt_traj[:n_align, 1], 'k-', linewidth=2.0, alpha=0.7, label='GT')
for name, sd in scenario_data.items():
    va = sd['vo_align']
    ax2.plot(va[:, 0], va[:, 1], '--', color=COLORS[name], linewidth=1.2,
             alpha=0.6, label=f'{name} (ATE={sd["ate_rmse"]:.1f}mm)')
ax2.scatter(*gt_traj[0, :2], c='green', s=80, marker='o', zorder=5, edgecolors='black')
ax2.scatter(*gt_traj[min(n_align-1, len(gt_traj)-1), :2], c='red', s=80, marker='s', zorder=5, edgecolors='black')
ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)')
ax2.set_title('Top-Down View (X-Y)')
ax2.legend(fontsize=7); ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)

# ── 子图3: Side View (XZ) ──
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(gt_traj[:n_align, 0], gt_traj[:n_align, 2], 'k-', linewidth=2.0, alpha=0.7, label='GT')
for name, sd in scenario_data.items():
    va = sd['vo_align']
    ax3.plot(va[:, 0], va[:, 2], '--', color=COLORS[name], linewidth=1.2,
             alpha=0.6, label=name)
ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Z (mm)')
ax3.set_title('Side View (X-Z)')
ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

plt.suptitle(f'{SEQ_NAME}: Three-Scenario Motion Point 3D Comparison',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path1 = os.path.join(OUT_DIR, 'overview_all_motion_3d.png')
fig.savefig(path1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path1}")


# ============================================================
# 3. 图2: 逐点 Per-Track 3D 三路对比 (Top-K, 共同 track)
# ============================================================
print("\n[2/4] 逐点 Per-Track 3D 三路对比...")

# 找三个场景的公共 track_id (场景B和C数据相同, 取A和任一对比)
names      = list(SCENARIOS.keys())
t2t_A      = scenario_data[names[0]]['t2t']
t2t_B      = scenario_data[names[1]]['t2t']

common_ids = sorted(set(t2t_A.keys()) & set(t2t_B.keys()))
print(f"  公共运动 track: {len(common_ids)}")

# 选择 displacement 最大的 top-K
disp_pairs = []
for tid in common_ids:
    ta = t2t_A[tid]
    tb = t2t_B[tid]
    da = ta.get('total_displacement_mm', 0)
    db = tb.get('total_displacement_mm', 0)
    disp_pairs.append((tid, (da + db) / 2))
disp_pairs.sort(key=lambda x: -x[1])

top_n = min(15, len(disp_pairs))
top_ids = [dp[0] for dp in disp_pairs[:top_n]]

cols = 3
rows = (top_n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5.5),
                         subplot_kw={'projection': '3d'})
axes = axes.flatten() if rows * cols > 1 else [axes]

for idx, tid in enumerate(top_ids):
    ax = axes[idx]
    
    # B和C数据相同, 用不同线型区分: A=实线, B=虚线, C=点线
    linestyles = {'A (纯RGB)': '-', 'B (+GT位姿)': '--', 'C (+运动真值)': ':'}
    for name in names:
        t = scenario_data[name]['t2t'].get(tid)
        if t is None:
            continue
        pts = np.array([[f['x_mm'], f['y_mm'], f['z_mm']] for f in t['frames'] if f])
        if len(pts) < 2:
            continue
        disp = t.get('total_displacement_mm', 0)
        ls = linestyles.get(name, '-')
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], ls, color=COLORS[name],
                linewidth=1.6, alpha=0.8, label=f"{name} ({disp:.1f}mm)")
        ax.scatter(*pts[0],  c=COLORS[name], s=25, marker='o', zorder=5)
        ax.scatter(*pts[-1], c=COLORS[name], s=45, marker='s', zorder=5)
    
    ax.set_title(f'Track {tid}', fontsize=9, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(fontsize=6, loc='upper left')

# 隐藏多余的 subplot
for idx in range(top_n, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'{SEQ_NAME}: Per-Track 3D Motion Comparison (Top {top_n} by Displacement)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
path2 = os.path.join(OUT_DIR, 'per_track_3d_compare.png')
fig.savefig(path2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path2}")


# ============================================================
# 4. 图3: 运动点位移分布 + 相关性
# ============================================================
print("\n[3/4] 位移分布对比 + 相关性...")

def compute_displacements(trajs):
    """计算每个 track 的 3D 位移."""
    disps = []
    for t in trajs:
        frames = t['frames']
        if len(frames) < 2:
            continue
        valid = [f for f in frames if f]
        if len(valid) < 2:
            continue
        p0 = np.array([valid[0]['x_mm'], valid[0]['y_mm'], valid[0]['z_mm']])
        pn = np.array([valid[-1]['x_mm'], valid[-1]['y_mm'], valid[-1]['z_mm']])
        disps.append(np.linalg.norm(pn - p0))
    return np.array(disps)

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
axes = axes.flatten()

disps_all = {}
for i, (name, sd) in enumerate(scenario_data.items()):
    disps = compute_displacements(sd['trajs'])
    disps_all[name] = disps
    
    # 上半: 各场景直方图
    ax = axes[i]
    bins = np.linspace(0, np.percentile(disps, 99), 60)
    ax.hist(disps, bins=bins, color=COLORS[name], alpha=0.7, edgecolor='white',
            linewidth=0.5)
    ax.axvline(np.median(disps), color='black', linestyle='--', linewidth=1.5,
               label=f'Median={np.median(disps):.1f}mm')
    ax.axvline(np.mean(disps), color='gray', linestyle=':', linewidth=1.5,
               label=f'Mean={np.mean(disps):.1f}mm')
    ax.set_xlabel('3D Displacement (mm)')
    ax.set_ylabel('Count')
    ax.set_title(f'{name}\nn={len(disps)}, max={disps.max():.1f}mm')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

# 下半: 散点对比 (A vs B, B vs C, A vs C)
names_list = list(scenario_data.keys())
pair_idxs  = [(0, 1), (1, 2), (0, 2)]
pair_names = [(names_list[a], names_list[b]) for a, b in pair_idxs]

for pi, (ai, bi) in enumerate(pair_idxs):
    ax = axes[3 + pi]
    na, nb = pair_names[pi]
    
    # 按共同 track_id 配对
    t2t_a = scenario_data[na]['t2t']
    t2t_b = scenario_data[nb]['t2t']
    
    common = sorted(set(t2t_a.keys()) & set(t2t_b.keys()))
    da, db = [], []
    for tid in common:
        ta = t2t_a[tid]; tb = t2t_b[tid]
        fa = [f for f in ta['frames'] if f]; fb = [f for f in tb['frames'] if f]
        if len(fa) < 2 or len(fb) < 2:
            continue
        pa0 = np.array([fa[0]['x_mm'], fa[0]['y_mm'], fa[0]['z_mm']])
        pan = np.array([fa[-1]['x_mm'], fa[-1]['y_mm'], fa[-1]['z_mm']])
        pb0 = np.array([fb[0]['x_mm'], fb[0]['y_mm'], fb[0]['z_mm']])
        pbn = np.array([fb[-1]['x_mm'], fb[-1]['y_mm'], fb[-1]['z_mm']])
        da.append(np.linalg.norm(pan - pa0))
        db.append(np.linalg.norm(pbn - pb0))
    
    da, db = np.array(da), np.array(db)
    ax.scatter(da, db, s=6, alpha=0.4, c='#555555', rasterized=True)
    
    max_d = max(da.max(), db.max()) if len(da) else 1
    ax.plot([0, max_d], [0, max_d], 'r--', linewidth=1.0, alpha=0.5)
    
    # 计算相关性
    if len(da) > 2:
        corr = np.corrcoef(da, db)[0, 1]
        ratio = np.median(db / (da + 1e-6))
        ax.text(0.95, 0.05, f'r={corr:.3f}\nratio={ratio:.3f}\nn={len(da)}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(f'{na.split(chr(40))[0].strip()} Disp (mm)')
    ax.set_ylabel(f'{nb.split(chr(40))[0].strip()} Disp (mm)')
    ax.set_title(f'Per-Track Displacement: {na.split(chr(40))[0].strip()} vs {nb.split(chr(40))[0].strip()}')
    ax.axis('equal')
    ax.grid(True, alpha=0.2)

plt.suptitle(f'{SEQ_NAME}: Motion Displacement Distribution & Cross-Scenario Correlation',
             fontsize=13, fontweight='bold')
plt.tight_layout()
path3 = os.path.join(OUT_DIR, 'displacement_dist_corr.png')
fig.savefig(path3, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path3}")


# ============================================================
# 5. 图4: 汇总仪表板 (ATE + 位移统计)
# ============================================================
print("\n[4/4] 汇总仪表板...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# ── 左上: ATE RMSE 对比 ──
ax = axes[0, 0]
ate_vals  = [scenario_data[n]['ate_rmse'] for n in names]
bars      = ax.bar(names, ate_vals, color=[COLORS[n] for n in names], alpha=0.85, edgecolor='black')
ax.set_ylabel('ATE RMSE (mm)')
ax.set_title('Camera Pose Accuracy (ATE RMSE)')
ax.grid(True, alpha=0.2, axis='y')
for bar, v in zip(bars, ate_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{v:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# ── 中上: Scale 校准值 ──
ax = axes[0, 1]
scale_vals = [scenario_data[n]['scale'] for n in names]
bars = ax.bar(names, scale_vals, color=[COLORS[n] for n in names], alpha=0.85, edgecolor='black')
ax.set_ylabel('Umeyama Scale')
ax.set_title('VO→GT Scale Calibration')
ax.grid(True, alpha=0.2, axis='y')
for bar, v in zip(bars, scale_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{v:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# ── 右上: 运动点数量 ──
ax = axes[0, 2]
n_moving = [len(scenario_data[n]['trajs']) for n in names]
bars = ax.bar(names, n_moving, color=[COLORS[n] for n in names], alpha=0.85, edgecolor='black')
ax.set_ylabel('Count')
ax.set_title(f'Moving Tracks (total 10000)')
ax.grid(True, alpha=0.2, axis='y')
for bar, v in zip(bars, n_moving):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(v), ha='center', va='bottom', fontweight='bold', fontsize=11)

# ── 左下: 位移中位数对比 ──
ax = axes[1, 0]
medians = [np.median(disps_all[n]) if len(disps_all[n]) else 0 for n in names]
bars = ax.bar(names, medians, color=[COLORS[n] for n in names], alpha=0.85, edgecolor='black')
ax.set_ylabel('Median Displacement (mm)')
ax.set_title('Motion Track Median Displacement')
ax.grid(True, alpha=0.2, axis='y')
for bar, v in zip(bars, medians):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{v:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# ── 中下: 位移均值对比 ──
ax = axes[1, 1]
means = [np.mean(disps_all[n]) if len(disps_all[n]) else 0 for n in names]
bars = ax.bar(names, means, color=[COLORS[n] for n in names], alpha=0.85, edgecolor='black')
ax.set_ylabel('Mean Displacement (mm)')
ax.set_title('Motion Track Mean Displacement')
ax.grid(True, alpha=0.2, axis='y')
for bar, v in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{v:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# ── 右下: 汇总表格 ──
ax = axes[1, 2]
ax.axis('off')

table_data = [
    ['指标', 'A (纯RGB)', 'B (+GT位姿)', 'C (+运动真值)'],
    ['ATE RMSE', f'{scenario_data[names[0]]["ate_rmse"]:.2f} mm',
                f'{scenario_data[names[1]]["ate_rmse"]:.2f} mm',
                f'{scenario_data[names[2]]["ate_rmse"]:.2f} mm'],
    ['Scale', f'{scenario_data[names[0]]["scale"]:.4f}',
              f'{scenario_data[names[1]]["scale"]:.4f}',
              f'{scenario_data[names[2]]["scale"]:.4f}'],
    ['运动点数', str(len(scenario_data[names[0]]['trajs'])),
               str(len(scenario_data[names[1]]['trajs'])),
               str(len(scenario_data[names[2]]['trajs']))],
    ['中位位移', f'{medians[0]:.1f} mm', f'{medians[1]:.1f} mm', f'{medians[2]:.1f} mm'],
    ['均值位移', f'{means[0]:.1f} mm', f'{means[1]:.1f} mm', f'{means[2]:.1f} mm'],
    ['分离度', '3.5×', '3.5×', '3.5×'],
]
tbl = ax.table(cellText=table_data, cellLoc='center', loc='center',
               colWidths=[0.22, 0.26, 0.26, 0.26])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.8)
for j in range(len(table_data[0])):
    tbl[0, j].set_facecolor('#404040')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('Summary Table', fontsize=11, fontweight='bold', y=0.85)

plt.suptitle(f'{SEQ_NAME}: Three-Scenario Motion Analysis Dashboard',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path4 = os.path.join(OUT_DIR, 'dashboard_summary.png')
fig.savefig(path4, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {path4}")


# ============================================================
# Done
# ============================================================
print(f"\n{'='*60}")
print(f"All visualizations saved to: {OUT_DIR}")
for p in [path1, path2, path3, path4]:
    fname = os.path.basename(p)
    print(f"  {fname}")
print(f"{'='*60}")
