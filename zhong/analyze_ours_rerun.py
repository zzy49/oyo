#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析 test_v6_dyendovo 重跑 Ours 的输出, 做"正确方法"同口径对比.

正确口径:
  同一批 track (Ours 输出的运动 track),
  分别用:
    GT:   GT深度(tiff, 物理mm) + GT位姿(pose.txt)   → 反投影 3D 轨迹
    Ours: Ours预测深度 + Ours VO位姿                → 反投影 3D 轨迹 (JSON 已含)
  对比两者净位移 (首末帧位移 mm).

输入:
  zhong/ours_rerun/c1_transverse1_t1_v2_baseline_motion_trajectories.json
  F:/dataset/c1_transverse1_t1_v2/depth/{fi:04d}_depth.tiff
  F:/dataset/c1_transverse1_t1_v2/pose.txt

输出:
  zhong/ours_rerun/analysis/
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

TRAJ_JSON = os.path.join(BASE, 'ours_rerun',
                         f'{SEQ}_baseline_motion_trajectories.json')
OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # 1. 读 Ours 运动轨迹
    with open(TRAJ_JSON) as f:
        trajs = json.load(f)
    print(f'Ours 运动 track 数: {len(trajs)}')

    ours_disp = np.array([t['total_displacement_mm'] for t in trajs])
    print(f'Ours 位移: median={np.median(ours_disp):.3f}mm '
          f'mean={np.mean(ours_disp):.3f}mm '
          f'max={np.max(ours_disp):.3f}mm')

    # 2. GT 位姿 (cam_to_world -> world_to_cam)
    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    gt_w2c = [np.linalg.inv(T) for T in gt_c2w]
    n_gt = len(gt_w2c)

    # 3. 按帧分组反投影 GT
    frame_reqs = defaultdict(list)   # fi -> [(tid, u, v), ...]
    for t in trajs:
        tid = t['track_id']
        for fr in t['frames']:
            fi = int(fr['frame'])
            if fi < n_gt:
                frame_reqs[fi].append((tid, float(fr['u']), float(fr['v'])))

    gt_pts = {t['track_id']: [] for t in trajs}
    for fi in sorted(frame_reqs.keys()):
        d = np.array(Image.open(os.path.join(SEQ_DIR, 'depth',
                                             f'{fi:04d}_depth.tiff'))
                     ).astype(np.float32) * GT_SCALE
        T = gt_w2c[fi]
        for tid, u, v in frame_reqs[fi]:
            gt_pts[tid].append(backproject_to_world(u, v, d, T))

    # 4. GT 净位移
    gt_disp = {}
    for tid, plist in gt_pts.items():
        valid = [p for p in plist if p is not None]
        if len(valid) >= 2:
            gt_disp[tid] = float(np.linalg.norm(valid[-1] - valid[0]))
    print(f'GT 反投影有效 track: {len(gt_disp)}')

    # 5. 对齐对比
    pairs = [(t['track_id'], ours_disp[i], gt_disp[t['track_id']])
             for i, t in enumerate(trajs) if t['track_id'] in gt_disp]
    xs = np.array([p[2] for p in pairs])   # GT
    ys = np.array([p[1] for p in pairs])   # Ours
    print(f'对齐 track 数: {len(pairs)}')
    print(f'GT  位移: median={np.median(xs):.3f}mm mean={np.mean(xs):.3f}mm')
    print(f'Ours位移: median={np.median(ys):.3f}mm mean={np.mean(ys):.3f}mm')
    ratio = float(np.median(ys / (xs + 1e-6)))
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else 0.0
    mae = float(np.mean(np.abs(ys - xs)))
    print(f'中位放大倍数: {ratio:.2f}x   Pearson r: {corr:.3f}   MAE: {mae:.3f}mm')

    # 6. 散点图
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(xs, ys, s=12, alpha=0.4, c='#555555', rasterized=True)
    lim = max(xs.max(), ys.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', lw=1.2, alpha=0.6, label='y=x')
    ax.set_xlabel('GT 反投影位移 (mm)  [GT深度+GT位姿]')
    ax.set_ylabel('Ours 反投影位移 (mm)  [Ours深度+Ours位姿]')
    ax.set_title(f'{SEQ}: 同一批运动 track 同口径对比 (n={len(pairs)})\n'
                 f'中位放大={ratio:.1f}x  Pearson r={corr:.3f}  MAE={mae:.2f}mm')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, 'gt_vs_ours_scatter.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_png}')

    # 7. 汇总
    summary = {
        'seq': SEQ,
        'n_ours_tracks': int(len(trajs)),
        'n_gt_valid': int(len(gt_disp)),
        'n_aligned': int(len(pairs)),
        'ours_disp': {
            'median_mm': float(np.median(ours_disp)),
            'mean_mm': float(np.mean(ours_disp)),
            'max_mm': float(np.max(ours_disp)),
        },
        'gt_disp': {
            'median_mm': float(np.median(xs)),
            'mean_mm': float(np.mean(xs)),
        },
        'median_ratio': ratio,
        'pearson_r': corr,
        'mae_mm': mae,
    }
    with open(os.path.join(OUT_DIR, 'gt_vs_ours_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'Saved: {os.path.join(OUT_DIR, "gt_vs_ours_summary.json")}')


if __name__ == '__main__':
    main()
