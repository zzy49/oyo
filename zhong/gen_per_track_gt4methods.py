#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正确对比: 同一批 track, 5 种配置(深度+位姿)反投影的 3D 轨迹 per-track 对比.

学习 viz_three_scenario_motion.py 的 per_track_3d_compare 精髓:
  同一批 LoFTR track, 换不同"深度+位姿"配置反投影, 同口径叠加对比.

配置:
  GT:        GT深度(tiff, 物理mm) + GT位姿(pose.txt)
  Monodepth2/ManyDepth/Lite-Mono/Ours: 各自预测深度 + 各自VO位姿
      (直接读 run_motion_pipeline 内置 JSON 的 x_mm/y_mm/z_mm, 即 VO 反投影结果)

流程:
  1. 读四方法 motion_trajectories.json → track_id → frames(u/v/frame) + x/y/z
  2. 公共 track = 四方法交集
  3. 对每个公共 track, 用 GT深度+GT位姿 反投影 → GT轨迹 + GT净位移
  4. 按 GT 净位移排序 Top-K
  5. 网格叠加 (每track一个子图, 5配置起点归零, 直观对比运动幅度/方向)

输出: zhong/per_track_gt4methods/
"""

import os
import sys
import json
from collections import defaultdict

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

OUT_DIR = os.path.join(BASE, 'per_track_gt4methods')
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ['Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']
ALL_CONFIGS = ['GT'] + METHODS
COLORS = {
    'GT': '#111111', 'Monodepth2': '#1f77b4', 'ManyDepth': '#ff7f0e',
    'Lite-Mono': '#2ca02c', 'Ours': '#d62728',
}
LINESTYLES = {'GT': '-', 'Monodepth2': '--', 'ManyDepth': '-.',
              'Lite-Mono': ':', 'Ours': '-'}
MARKERS = {'GT': 'o', 'Monodepth2': 's', 'ManyDepth': '^',
           'Lite-Mono': 'D', 'Ours': '*'}

N_TOP = 15
COLS = 3


def load_method_tracks(name):
    p = os.path.join(BASE, 'motion_vis_fig3_top5', f'method_{name}',
                     f'{SEQ}_{name}_baseline_motion_trajectories.json')
    with open(p) as f:
        trajs = json.load(f)
    return {t['track_id']: t for t in trajs}


def method_traj_xyz(t):
    pts = []
    for fr in t['frames']:
        if fr.get('x_mm') is None or fr.get('y_mm') is None or fr.get('z_mm') is None:
            continue
        pts.append([fr['x_mm'], fr['y_mm'], fr['z_mm']])
    return np.array(pts, dtype=np.float64)


def main():
    # 1. 读四方法 track
    mdata = {n: load_method_tracks(n) for n in METHODS}
    common = sorted(set.intersection(*[set(mdata[n].keys()) for n in METHODS]))
    print(f'四方法公共 track: {len(common)}')

    # 2. GT 位姿 (cam_to_world -> world_to_cam)
    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    gt_w2c = [np.linalg.inv(T) for T in gt_c2w]

    # 3. 按帧分组反投影 GT (逐帧读 tiff, 控制内存)
    ref = mdata[METHODS[0]]
    frame_reqs = defaultdict(list)   # fi -> [(tid, u, v), ...]
    for tid in common:
        for fr in ref[tid]['frames']:
            frame_reqs[int(fr['frame'])].append((tid, float(fr['u']), float(fr['v'])))

    gt_pts = {tid: [] for tid in common}
    for fi in sorted(frame_reqs.keys()):
        d = np.array(Image.open(os.path.join(SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))
                     ).astype(np.float32) * GT_SCALE
        T = gt_w2c[fi]
        for tid, u, v in frame_reqs[fi]:
            gt_pts[tid].append(backproject_to_world(u, v, d, T))

    # 4. GT 净位移 + 排序
    gt_disp = {}
    for tid, plist in gt_pts.items():
        valid = [p for p in plist if p is not None]
        if len(valid) >= 2:
            gt_disp[tid] = float(np.linalg.norm(valid[-1] - valid[0]))
    print(f'GT 反投影有效 track: {len(gt_disp)}')

    # 按四方法平均位移排序 (评估对象是四方法的运动检测)
    avg_disp = {tid: float(np.mean([mdata[n][tid]['total_displacement_mm']
                                    for n in METHODS])) for tid in common}
    order = sorted(common, key=lambda t: -avg_disp[t])[:N_TOP]
    print(f'\n按四方法平均位移排序的 Top{N_TOP} (单位 mm):')
    print(f'  {"track":>8} | {"GT":>7} | ' + ' | '.join(f'{n:>9}' for n in METHODS) + ' | 平均')
    for tid in order:
        row = ' | '.join(f'{mdata[n][tid]["total_displacement_mm"]:9.1f}' for n in METHODS)
        print(f'  {tid:>8} | {gt_disp[tid]:7.2f} | {row} | {avg_disp[tid]:5.1f}')

    # 5. 绘图 (起点归零)
    rows = (N_TOP + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 5.4, rows * 4.4),
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten() if rows * COLS > 1 else [axes]

    for idx, tid in enumerate(order):
        ax = axes[idx]

        # GT 轨迹
        gvalid = np.array([p for p in gt_pts[tid] if p is not None])
        g0 = gvalid[0]
        ax.plot(*(gvalid - g0).T, color=COLORS['GT'], lw=2.4, alpha=0.95,
                label=f'GT ({gt_disp[tid]:.1f}mm)')
        ax.scatter(0, 0, 0, color=COLORS['GT'], s=60, marker='o', zorder=6,
                   depthshade=False, edgecolors='k', linewidths=0.5)
        ax.scatter(*(gvalid[-1] - g0), color=COLORS['GT'], s=90, marker='X', zorder=6,
                   depthshade=False, edgecolors='k', linewidths=0.5)

        # 四方法轨迹
        for n in METHODS:
            pts = method_traj_xyz(mdata[n][tid])
            if len(pts) < 2:
                continue
            disp = float(mdata[n][tid]['total_displacement_mm'])
            ax.plot(*(pts - pts[0]).T, color=COLORS[n], ls=LINESTYLES[n],
                    lw=1.4, alpha=0.85, label=f'{n} ({disp:.1f}mm)')
            ax.scatter(*(pts[-1] - pts[0]), color=COLORS[n], s=45, marker=MARKERS[n],
                       zorder=5, depthshade=False, edgecolors='k', linewidths=0.4)

        ax.set_title(f'Track {tid}', fontsize=9, fontweight='bold')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.5, loc='upper left', ncol=1)

    for idx in range(len(order), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'{SEQ}: 同一批 track 五种配置(深度+位姿)反投影对比\n'
        f'(起点归零, Top{N_TOP} by 四方法平均位移, ○=起点 X=终点)',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'per_track_5config_compare.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_png}')

    # 6. 保存数值汇总
    summary = {
        'seq': SEQ,
        'n_common_tracks': len(common),
        'top_by_method_avg_disp': [
            {'track_id': tid, 'gt_disp_mm': round(gt_disp[tid], 3),
             'method_avg_disp_mm': round(avg_disp[tid], 3),
             **{n: round(float(mdata[n][tid]['total_displacement_mm']), 3)
                for n in METHODS}}
            for tid in order
        ],
    }
    with open(os.path.join(OUT_DIR, 'per_track_5config_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'Saved: {os.path.join(OUT_DIR, "per_track_5config_summary.json")}')

    # 7. 位移散点对比 (全公共 track: GT反投影位移 vs 四方法平均位移)
    valid_tids = [t for t in common if t in gt_disp]
    xs = np.array([gt_disp[t] for t in valid_tids])
    ys = np.array([avg_disp[t] for t in valid_tids])
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    ax2.scatter(xs, ys, s=12, alpha=0.4, c='#555555', rasterized=True)
    ax2.plot([0, max(xs.max(), ys.max())], [0, max(xs.max(), ys.max())],
             'r--', lw=1.2, alpha=0.6, label='y=x')
    ax2.set_xlabel('GT 反投影位移 (mm)')
    ax2.set_ylabel('四方法平均 VO 反投影位移 (mm)')
    ax2.set_title(f'{SEQ}: 同一批 track 位移对比\n'
                  f'(GT深度+GT位姿 vs 四方法深度+位姿, n={len(valid_tids)})')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    # 统计
    ratio = np.median(ys / (xs + 1e-6))
    corr = np.corrcoef(xs, ys)[0, 1] if len(xs) > 2 else 0
    ax2.text(0.97, 0.03,
             f'中位放大倍数={ratio:.1f}×\nPearson r={corr:.3f}\n'
             f'GT中位={np.median(xs):.2f}mm  四方法中位={np.median(ys):.2f}mm',
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    fig2.tight_layout()
    out2 = os.path.join(OUT_DIR, 'disp_scatter_gt_vs_methods.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {out2}')


if __name__ == '__main__':
    main()
