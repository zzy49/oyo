#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正确尺度下的三场景 per-track 对比图.

旧 scenario 数据未做 ÷global_scale 深度尺度恢复, 位移被整体放大.
本脚本对旧 JSON 坐标做 ÷scale 修正, 重画三线对比图, 证明:
  正确尺度下三场景(A/B/C 同模型自比)依然几乎重合, 且 A/B 会更贴合.

各场景深度 global_scale (来自各自 log.txt):
  A: pnp_scale   -> 2.2390
  B: GT Umeyama  -> 2.3851
  C: 同 B        -> 2.3851
"""

import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

BASE = r'e:\data1\monodepth2\zhong'
OUT  = os.path.join(BASE, 'three_scene_motion_viz_correct')
os.makedirs(OUT, exist_ok=True)

SCALE = {
    'A (纯RGB)':      2.2390,
    'B (+GT位姿)':    2.3851,
    'C (+运动真值)':  2.3851,
}
SRC = {
    'A (纯RGB)':      os.path.join(BASE, 'scenario_a_raw_rgb'),
    'B (+GT位姿)':    os.path.join(BASE, 'scenario_b_gt_pose'),
    'C (+运动真值)':  os.path.join(BASE, 'scenario_c_gt_motion'),
}
COLORS     = {'A (纯RGB)': '#e74c3c', 'B (+GT位姿)': '#2ecc71', 'C (+运动真值)': '#3498db'}
LINESTYLES = {'A (纯RGB)': '-',      'B (+GT位姿)': '--',     'C (+运动真值)': ':'}


def load_corrected(dirpath, scale):
    p = os.path.join(dirpath, 'c1_transverse1_t1_v2_baseline_motion_trajectories.json')
    trajs = json.load(open(p))
    for t in trajs:
        for f in t['frames']:
            if f:
                f['x_mm'] /= scale
                f['y_mm'] /= scale
                f['z_mm'] /= scale
        if 'total_displacement_mm' in t:
            t['total_displacement_mm'] /= scale
    return trajs


data = {}
for name in SRC:
    trajs = load_corrected(SRC[name], SCALE[name])
    data[name] = {t['track_id']: t for t in trajs}
    disps = np.array([t['total_displacement_mm'] for t in trajs])
    print(f"{name}: {len(trajs)} tracks, ÷{SCALE[name]:.4f} -> median={np.median(disps):.2f}mm, max={np.max(disps):.2f}mm")

# 公共 track
common = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
print(f"公共 track: {len(common)}")

# 位移排序 (A+B 平均)
pairs = []
for tid in common:
    da = data['A (纯RGB)'][tid].get('total_displacement_mm', 0)
    db = data['B (+GT位姿)'][tid].get('total_displacement_mm', 0)
    pairs.append((tid, (da + db) / 2))
pairs.sort(key=lambda x: -x[1])
top_n = min(15, len(pairs))
top_ids = [p[0] for p in pairs[:top_n]]

# A-B 逐点平均距离 (验证重合度, 修正前 vs 修正后)
def mean_pairwise_dist(tA, tB):
    fa = [f for f in tA['frames'] if f]
    fb = [f for f in tB['frames'] if f]
    n = min(len(fa), len(fb))
    d = []
    for i in range(n):
        pa = np.array([fa[i]['x_mm'], fa[i]['y_mm'], fa[i]['z_mm']])
        pb = np.array([fb[i]['x_mm'], fb[i]['y_mm'], fb[i]['z_mm']])
        d.append(np.linalg.norm(pa - pb))
    return np.mean(d) if d else 0.0

dists = [mean_pairwise_dist(data['A (纯RGB)'][tid], data['B (+GT位姿)'][tid]) for tid in common]
print(f"A-B 逐点平均距离: median={np.median(dists):.3f}mm, mean={np.mean(dists):.3f}mm (修正后)")

# 画图
cols, rows = 3, (top_n + 2) // 3
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5.5),
                         subplot_kw={'projection': '3d'})
axes = axes.flatten() if rows * cols > 1 else [axes]

for idx, tid in enumerate(top_ids):
    ax = axes[idx]
    for name in data:
        t = data[name].get(tid)
        if not t:
            continue
        pts = np.array([[f['x_mm'], f['y_mm'], f['z_mm']] for f in t['frames'] if f])
        if len(pts) < 2:
            continue
        disp = t.get('total_displacement_mm', 0)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], LINESTYLES[name], color=COLORS[name],
                linewidth=1.6, alpha=0.8, label=f"{name} ({disp:.1f}mm)")
        ax.scatter(*pts[0],  c=COLORS[name], s=25, marker='o', zorder=5)
        ax.scatter(*pts[-1], c=COLORS[name], s=45, marker='s', zorder=5)
    ax.set_title(f'Track {tid}', fontsize=9, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(fontsize=6, loc='upper left')

for idx in range(top_n, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Correct-Scale Per-Track 3D Comparison (÷global_scale)', fontsize=13, fontweight='bold')
plt.tight_layout()
out_p = os.path.join(OUT, 'per_track_3d_compare_correct_scale.png')
fig.savefig(out_p, dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved:', out_p)
