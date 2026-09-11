"""
逐点比对: Config A (GT校准) vs Config B (--no_gt)
按 track_id 匹配，逐帧计算 3D 位置差，可视化对比
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

def load_motion(d):
    import glob
    fs = glob.glob(os.path.join(d, '*_baseline_motion_trajectories.json'))
    return json.load(open(fs[0]))


# ── 加载 & 对齐 ──
gt = load_gt_traj()
vo_a, vo_b = load_vo_traj(A_DIR), load_vo_traj(B_DIR)
n = min(len(gt), len(vo_a), len(vo_b))
gt, vo_a, vo_b = gt[:n], vo_a[:n], vo_b[:n]
_, (Ra, ta, sa) = umeyama(vo_a, gt)
_, (Rb, tb, sb) = umeyama(vo_b, gt)

mt_a = load_motion(A_DIR)
mt_b = load_motion(B_DIR)

# 按 track_id 索引
map_a = {t['track_id']: t for t in mt_a}
map_b = {t['track_id']: t for t in mt_b}
common_ids = sorted(set(map_a.keys()) & set(map_b.keys()))
print(f"A: {len(mt_a)} tracks, B: {len(mt_b)} tracks, Common: {len(common_ids)}")


# ── 逐 track_id 逐帧对比 ──
def transform_pt(x, y, z, R, t, s):
    p = np.array([x, y, z])
    return s * (R @ p) + t

records = []  # [{track_id, disp_a, disp_b, ratio, frame_errors: [(fi, d3d),...]}]

for tid in common_ids:
    track_a, track_b = map_a[tid], map_b[tid]
    fa_frames = {f['frame']: f for f in track_a['frames'] if f}
    fb_frames = {f['frame']: f for f in track_b['frames'] if f}
    common_fi = sorted(set(fa_frames) & set(fb_frames))
    if len(common_fi) < 2:
        continue

    fer = []  # frame errors
    for fi in common_fi:
        pa = transform_pt(fa_frames[fi]['x_mm'], fa_frames[fi]['y_mm'], fa_frames[fi]['z_mm'], Ra, ta, sa)
        pb = transform_pt(fb_frames[fi]['x_mm'], fb_frames[fi]['y_mm'], fb_frames[fi]['z_mm'], Rb, tb, sb)
        d3d = np.linalg.norm(pa - pb)
        fer.append((fi, float(d3d)))

    # 首尾帧位移
    f0_a = transform_pt(fa_frames[common_fi[0]]['x_mm'], fa_frames[common_fi[0]]['y_mm'], fa_frames[common_fi[0]]['z_mm'], Ra, ta, sa)
    fn_a = transform_pt(fa_frames[common_fi[-1]]['x_mm'], fa_frames[common_fi[-1]]['y_mm'], fa_frames[common_fi[-1]]['z_mm'], Ra, ta, sa)
    f0_b = transform_pt(fb_frames[common_fi[0]]['x_mm'], fb_frames[common_fi[0]]['y_mm'], fb_frames[common_fi[0]]['z_mm'], Rb, tb, sb)
    fn_b = transform_pt(fb_frames[common_fi[-1]]['x_mm'], fb_frames[common_fi[-1]]['y_mm'], fb_frames[common_fi[-1]]['z_mm'], Rb, tb, sb)

    disp_a = float(np.linalg.norm(fn_a - f0_a))
    disp_b = float(np.linalg.norm(fn_b - f0_b))
    ratio = disp_b / disp_a if disp_a > 0.01 else 0

    records.append({
        'track_id': tid, 'disp_a': disp_a, 'disp_b': disp_b, 'ratio': ratio,
        'frame_errors': fer,
        'mean_err': float(np.mean([e[1] for e in fer])),
        'max_err': float(np.max([e[1] for e in fer])),
    })

print(f"Matched pairs: {len(records)}")
ratios = np.array([r['ratio'] for r in records])
mean_errs = np.array([r['mean_err'] for r in records])
print(f"Displacement ratio B/A: median={np.median(ratios):.3f}, mean={np.mean(ratios):.3f}")
print(f"Mean 3D point error: median={np.median(mean_errs):.1f}mm, mean={np.mean(mean_errs):.1f}mm")


# ═══════════════════════════════════════
# 图1: 位移散点 A vs B + 误差分布
# ═══════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(20, 6))

# 左: A vs B 位移散点
ax = axes[0]
disp_a_all = np.array([r['disp_a'] for r in records])
disp_b_all = np.array([r['disp_b'] for r in records])
ax.scatter(disp_a_all, disp_b_all, c='#2ca02c', alpha=0.3, s=6, edgecolors='none')
lim = max(disp_a_all.max(), disp_b_all.max()) * 1.05
ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.5, label='y=x')
ax.set_xlabel('Config A (GT calib) displacement (mm)')
ax.set_ylabel('Config B (no_gt) displacement (mm)')
ax.set_title(f'Per-Track Displacement A vs B\n(n={len(records)}, median ratio={np.median(ratios):.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 中: ratio B/A 直方图
ax = axes[1]
ax.hist(np.clip(ratios, 0, 2), bins=60, color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(1.0, color='black', linestyle='--', linewidth=1.5, label='ratio=1 (identical)')
ax.axvline(np.median(ratios), color='blue', linestyle='--', linewidth=1.5, label=f'median={np.median(ratios):.3f}')
ax.set_xlabel('Displacement Ratio B/A')
ax.set_ylabel('Count')
ax.set_title(f'B/A Displacement Ratio Distribution')
ax.legend(fontsize=8)

# 右: 逐点 3D 位置误差分布
ax = axes[2]
ax.hist(np.clip(mean_errs, 0, np.percentile(mean_errs, 99)), bins=60, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(np.median(mean_errs), color='black', linestyle='--', linewidth=1.5, label=f'median={np.median(mean_errs):.1f}mm')
ax.axvline(np.mean(mean_errs), color='red', linestyle='--', linewidth=1.5, label=f'mean={np.mean(mean_errs):.1f}mm')
ax.set_xlabel('Mean 3D Point Error A vs B (mm)')
ax.set_ylabel('Count')
ax.set_title(f'Per-Track Mean 3D Position Error')
ax.legend(fontsize=8)

plt.suptitle('Per-Track Comparison: Config A (GT calib) vs Config B (no_gt)', fontsize=14, fontweight='bold')
plt.tight_layout()
s1 = os.path.join(OUT_DIR, 'per_track_scatter.png')
plt.savefig(s1, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {s1}')


# ═══════════════════════════════════════
# 图2: Top 10 差异最大的轨迹逐帧覆盖
# ═══════════════════════════════════════
records_sorted = sorted(records, key=lambda x: -x['mean_err'])
top_n = min(10, len(records_sorted))

fig2, axes = plt.subplots(2, 5, figsize=(24, 10), subplot_kw={'projection': '3d'})
axes = axes.flatten()

for idx, rec in enumerate(records_sorted[:top_n]):
    ax = axes[idx]
    track_a = map_a[rec['track_id']]
    track_b = map_b[rec['track_id']]
    fa_frames = {f['frame']: f for f in track_a['frames'] if f}
    fb_frames = {f['frame']: f for f in track_b['frames'] if f}
    common_fi = sorted(set(fa_frames) & set(fb_frames))

    # A 轨迹
    pts_a = []
    for fi in common_fi:
        f = fa_frames[fi]
        p = transform_pt(f['x_mm'], f['y_mm'], f['z_mm'], Ra, ta, sa)
        pts_a.append(p)
    xs_a, ys_a, zs_a = zip(*pts_a)

    # B 轨迹
    pts_b = []
    for fi in common_fi:
        f = fb_frames[fi]
        p = transform_pt(f['x_mm'], f['y_mm'], f['z_mm'], Rb, tb, sb)
        pts_b.append(p)
    xs_b, ys_b, zs_b = zip(*pts_b)

    ax.plot(xs_a, ys_a, zs_a, '-o', color='#2ca02c', linewidth=2, markersize=3, label='A (GT calib)')
    ax.plot(xs_b, ys_b, zs_b, '-s', color='#d62728', linewidth=2, markersize=3, label='B (no_gt)')

    # 对应帧连线
    for i in range(len(common_fi)):
        ax.plot([xs_a[i], xs_b[i]], [ys_a[i], ys_b[i]], [zs_a[i], zs_b[i]],
                'gray', linewidth=0.4, alpha=0.5)

    ax.set_title(f'#{idx+1} id={rec["track_id"]}\n'
                 f'A={rec["disp_a"]:.1f} B={rec["disp_b"]:.1f}mm err={rec["mean_err"]:.1f}mm',
                 fontsize=8)
    if idx == 0:
        ax.legend(fontsize=7)

plt.suptitle('Top 10 Largest Discrepancy Tracks: A (green) vs B (red)  |  GT-aligned',
             fontsize=14, fontweight='bold')
plt.tight_layout()
s2 = os.path.join(OUT_DIR, 'per_track_top10_3d.png')
plt.savefig(s2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {s2}')


# ═══════════════════════════════════════
# 图3: Top 5 最小差异 + Top 5 最大差异
# ═══════════════════════════════════════
records_by_ratio = sorted(records, key=lambda x: abs(x['ratio'] - 1.0))

fig3, axes = plt.subplots(2, 5, figsize=(24, 10), subplot_kw={'projection': '3d'})
axes = axes.flatten()

# 上行: ratio 最接近 1.0 的 (最小差异)
for idx, rec in enumerate(records_by_ratio[:5]):
    ax = axes[idx]
    track_a = map_a[rec['track_id']]; track_b = map_b[rec['track_id']]
    fa_frames = {f['frame']: f for f in track_a['frames'] if f}
    fb_frames = {f['frame']: f for f in track_b['frames'] if f}
    common_fi = sorted(set(fa_frames) & set(fb_frames))
    pts_a = [transform_pt(fa_frames[fi]['x_mm'], fa_frames[fi]['y_mm'], fa_frames[fi]['z_mm'], Ra, ta, sa) for fi in common_fi]
    pts_b = [transform_pt(fb_frames[fi]['x_mm'], fb_frames[fi]['y_mm'], fb_frames[fi]['z_mm'], Rb, tb, sb) for fi in common_fi]
    ax.plot(*zip(*pts_a), '-o', color='#2ca02c', linewidth=2, markersize=4, label='A')
    ax.plot(*zip(*pts_b), '-s', color='#d62728', linewidth=2, markersize=4, label='B')
    ax.set_title(f'Best #{idx+1}: id={rec["track_id"]}\nratio={rec["ratio"]:.3f}', fontsize=9)

# 下行: ratio 离 1.0 最远的 (最大差异)
for idx, rec in enumerate(records_by_ratio[-5:]):
    ax = axes[5 + idx]
    track_a = map_a[rec['track_id']]; track_b = map_b[rec['track_id']]
    fa_frames = {f['frame']: f for f in track_a['frames'] if f}
    fb_frames = {f['frame']: f for f in track_b['frames'] if f}
    common_fi = sorted(set(fa_frames) & set(fb_frames))
    pts_a = [transform_pt(fa_frames[fi]['x_mm'], fa_frames[fi]['y_mm'], fa_frames[fi]['z_mm'], Ra, ta, sa) for fi in common_fi]
    pts_b = [transform_pt(fb_frames[fi]['x_mm'], fb_frames[fi]['y_mm'], fb_frames[fi]['z_mm'], Rb, tb, sb) for fi in common_fi]
    ax.plot(*zip(*pts_a), '-o', color='#2ca02c', linewidth=2, markersize=4, label='A')
    ax.plot(*zip(*pts_b), '-s', color='#d62728', linewidth=2, markersize=4, label='B')
    ax.set_title(f'Worst #{idx+1}: id={rec["track_id"]}\nratio={rec["ratio"]:.3f}', fontsize=9)

plt.suptitle('Best (top) vs Worst (bottom) Agreement  |  Green=A  Red=B', fontsize=14, fontweight='bold')
plt.tight_layout()
s3 = os.path.join(OUT_DIR, 'per_track_best_worst_3d.png')
plt.savefig(s3, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {s3}')


# ═══════════════════════════════════════
# 图4: 汇总统计
# ═══════════════════════════════════════
fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上: 位移差 vs ratio
ax = axes[0, 0]
diff = disp_a_all - disp_b_all
sc = ax.scatter(disp_a_all, diff, c=ratios, cmap='RdYlGn', alpha=0.4, s=5, vmin=0, vmax=2)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Config A displacement (mm)')
ax.set_ylabel('Displacement A - B (mm)')
ax.set_title(f'Displacement Difference (A-B)\nmedian diff={np.median(diff):.1f}mm')
plt.colorbar(sc, ax=ax, label='ratio B/A')

# 右上: 误差 vs 帧数
ax = axes[0, 1]
n_frames_arr = np.array([len(r['frame_errors']) for r in records])
sc = ax.scatter(n_frames_arr, mean_errs, c=ratios, cmap='RdYlGn', alpha=0.4, s=5, vmin=0, vmax=2)
ax.set_xlabel('Track length (frames)')
ax.set_ylabel('Mean 3D error (mm)')
ax.set_title('Error vs Track Length')
plt.colorbar(sc, ax=ax, label='ratio B/A')
ax.grid(True, alpha=0.3)

# 左下: ratio 分位数
ax = axes[1, 0]
percentiles = [5, 10, 25, 50, 75, 90, 95]
pvals = [np.percentile(ratios, p) for p in percentiles]
ax.bar(range(len(percentiles)), pvals, color='#d62728', alpha=0.7, edgecolor='black')
ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='ratio=1')
ax.set_xticks(range(len(percentiles)))
ax.set_xticklabels([f'p{p}' for p in percentiles])
ax.set_ylabel('B/A Ratio')
ax.set_title('Displacement Ratio B/A Percentiles')
ax.legend()

# 右下: 汇总表
ax = axes[1, 1]
ax.axis('off')
stats = [
    ['Metric', 'Value'],
    ['Matched tracks', f'{len(records)}'],
    ['Median ratio B/A', f'{np.median(ratios):.3f}'],
    ['Mean ratio B/A', f'{np.mean(ratios):.3f}'],
    ['Std ratio B/A', f'{np.std(ratios):.3f}'],
    ['Median 3D error', f'{np.median(mean_errs):.1f} mm'],
    ['Mean 3D error', f'{np.mean(mean_errs):.1f} mm'],
    ['Max 3D error', f'{np.max(mean_errs):.1f} mm'],
    ['% ratio < 0.5', f'{np.mean(ratios < 0.5)*100:.1f}%'],
    ['% ratio 0.8-1.2', f'{np.mean((ratios > 0.8) & (ratios < 1.2))*100:.1f}%'],
]
table = ax.table(cellText=stats, cellLoc='center', loc='center', colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 2.0)
table[0, 0].set_facecolor('#404040')
table[0, 0].set_text_props(color='white', fontweight='bold')
table[0, 1].set_facecolor('#404040')
table[0, 1].set_text_props(color='white', fontweight='bold')

plt.suptitle('Per-Track Calibration Comparison Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
s4 = os.path.join(OUT_DIR, 'per_track_summary.png')
plt.savefig(s4, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {s4}')

print(f'\n=== Done ===')
print(f'  1. {s1}')
print(f'  2. {s2}')
print(f'  3. {s3}')
print(f'  4. {s4}')
