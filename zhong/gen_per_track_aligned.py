#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GT vs Ours 轨迹对比（原口径 + 时序一致性预处理）.

原口径 (gen_per_track_sorted_by_gt.py):
  - 同一批 track, 两条轨迹都用 GT 位姿反投影
  - GT:   GT深度(tiff)   + GT位姿
  - Ours: Ours深度(npz)  + GT位姿
  - 起点各自归零, 按 GT 净位移降序 Top15

新增预处理 (让两轨迹接近):
  1. GT 深度 3帧时序中值滤波  → 消除单帧跳变渲染噪声
  2. Ours 深度 3帧时序中值滤波 → 消除单帧抖动
  3. Ours 深度逐帧 median scaling 对齐 GT(平滑后) → 消除全局尺度漂移
     (与三基线评估同口径: 逐帧中位数缩放)

输出:
  - 处理前 vs 处理后 的位移差距对比 (打印)
  - per_track 图 (处理后的 GT vs Ours 叠加)
"""

import os, sys, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

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

TRAJ_JSON = os.path.join(BASE, 'ours_rerun', f'{SEQ}_baseline_motion_trajectories.json')
DEPTH_NPZ = os.path.join(SEQ_DIR, 'baseline_depth_maps.npz')
CHAIN_NPZ = os.path.join(SEQ_DIR, 'baseline_chain_data.npz')

OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis', 'per_track_aligned')
os.makedirs(OUT_DIR, exist_ok=True)

N_TOP = 15
COLS = 3
COL_GT = '#111111'
COL_OURS = '#d62728'


def disp_of(plist):
    valid = [p for p in plist if p is not None]
    if len(valid) < 2:
        return 0.0
    return float(np.linalg.norm(valid[-1] - valid[0]))


def main():
    with open(TRAJ_JSON) as f:
        trajs = json.load(f)
    print(f'Ours 运动 track 数: {len(trajs)}')

    # Ours 深度 ÷global_scale
    chain = np.load(CHAIN_NPZ, allow_pickle=True)
    global_scale = float(chain['global_depth_scale'])
    depth_npz = np.load(DEPTH_NPZ, allow_pickle=True)
    keys = sorted(depth_npz.files)
    ours_raw = {k: depth_npz[k] / global_scale for k in keys}
    n_frames = len(keys)
    H, W = ours_raw[keys[0]].shape
    print(f'global_scale={global_scale:.4f}, Ours 深度帧={n_frames}, 尺寸={H}x{W}')

    # GT 深度 (全部帧)
    print('加载 GT 深度 tiff ...')
    gt_raw = np.zeros((n_frames, H, W), dtype=np.float32)
    for i, k in enumerate(keys):
        fi = int(k)
        gt_raw[i] = np.array(Image.open(
            os.path.join(SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))).astype(np.float32) * GT_SCALE
    print(f'GT 深度: {gt_raw.shape}')

    # GT 位姿
    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    poses_gt = [np.linalg.inv(T) for T in gt_c2w]
    n_gt = len(poses_gt)

    # ── 预处理 ──
    print('时序中值滤波 (3帧) ...')
    gt_smooth = median_filter(gt_raw, size=(3, 1, 1))          # GT 去单帧跳变
    ours_stack = np.stack([ours_raw[k] for k in keys], axis=0)
    ours_smooth = median_filter(ours_stack, size=(3, 1, 1))    # Ours 去单帧抖动

    # Ours 逐帧 median scaling 对齐 GT
    print('Ours 逐帧 median scaling 对齐 GT ...')
    scales = []
    ours_aligned = ours_smooth.copy()
    for i in range(n_frames):
        g = gt_smooth[i]
        o = ours_aligned[i]
        mask = (g > 0.5) & (o > 0.5)
        if mask.sum() > 100:
            s = float(np.median(g[mask]) / np.median(o[mask]))
            if np.isfinite(s) and s > 0:
                ours_aligned[i] = o * s
                scales.append(s)
    print(f'  median scaling: mean={np.mean(scales):.4f}, std={np.std(scales):.4f}, n={len(scales)}')

    ours_processed = {k: ours_aligned[i] for i, k in enumerate(keys)}

    # ── 反投影 (处理前 vs 处理后) ──
    def backproj_traj(t, depth_dict):
        pts = []
        for fr in t['frames']:
            fi = int(fr['frame'])
            if fi >= n_gt:
                pts.append(None)
                continue
            d = depth_dict.get(f'{fi:04d}')
            pts.append(backproject_to_world(float(fr['u']), float(fr['v']), d, poses_gt[fi])
                       if d is not None else None)
        return pts

    print('反投影 track 轨迹 ...')
    rows_data = []
    for t in trajs:
        tid = t['track_id']
        gt_tr   = backproj_traj(t, {f'{i:04d}': gt_raw[i] for i in range(n_frames)})
        gt_tr_s = backproj_traj(t, {f'{i:04d}': gt_smooth[i] for i in range(n_frames)})
        ou_tr   = backproj_traj(t, ours_raw)
        ou_tr_p = backproj_traj(t, ours_processed)
        rows_data.append((t, gt_tr, gt_tr_s, ou_tr, ou_tr_p))

    # 按 GT(平滑) 位移排序
    rows_data.sort(key=lambda r: -disp_of(r[2]))
    top = rows_data[:N_TOP]

    # 打印处理前后差距
    print(f'\nTop{N_TOP} 处理前后位移对比 (mm):')
    print(f'  {"track":>7} | {"GT_raw":>7} | {"GT平滑":>7} | {"Ours_raw":>8} | {"Ours对齐":>8} | {"|G-O|前":>7} | {"|G-O|后":>7}')
    diffs_before, diffs_after = [], []
    for t, gt_tr, gt_tr_s, ou_tr, ou_tr_p in top:
        dg_raw, dg_s, do_raw, do_p = disp_of(gt_tr), disp_of(gt_tr_s), disp_of(ou_tr), disp_of(ou_tr_p)
        db = abs(dg_raw - do_raw)
        da = abs(dg_s - do_p)
        diffs_before.append(db)
        diffs_after.append(da)
        print(f'  {t["track_id"]:>7} | {dg_raw:7.2f} | {dg_s:7.2f} | {do_raw:8.2f} | {do_p:8.2f} | {db:7.2f} | {da:7.2f}')
    print(f'\n  Top{N_TOP} 平均 |GT-Ours| 差距: 处理前={np.mean(diffs_before):.2f}mm → 处理后={np.mean(diffs_after):.2f}mm')

    # ── 出图 (处理后) ──
    rows = (N_TOP + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 5.4, rows * 4.4),
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten() if rows * COLS > 1 else [axes]

    for idx, (t, gt_tr, gt_tr_s, ou_tr, ou_tr_p) in enumerate(top):
        ax = axes[idx]
        tid = t['track_id']

        gv = np.array([p for p in gt_tr_s if p is not None])
        if len(gv) >= 2:
            g0 = gv[0]
            ax.plot(*(gv - g0).T, color=COL_GT, lw=2.4, alpha=0.95,
                    label=f'GT ({disp_of(gt_tr_s):.1f}mm)')
            ax.scatter(0, 0, 0, color=COL_GT, s=60, marker='o', zorder=6,
                       depthshade=False, edgecolors='k', linewidths=0.5)
            ax.scatter(*(gv[-1] - g0), color=COL_GT, s=90, marker='X',
                       zorder=6, depthshade=False, edgecolors='k', linewidths=0.5)

        ov = np.array([p for p in ou_tr_p if p is not None])
        if len(ov) >= 2:
            o0 = ov[0]
            ax.plot(*(ov - o0).T, color=COL_OURS, ls='--', lw=1.4, alpha=0.85,
                    label=f'Ours ({disp_of(ou_tr_p):.1f}mm)')
            ax.scatter(*(ov[-1] - o0), color=COL_OURS, s=45, marker='*',
                       zorder=5, depthshade=False, edgecolors='k', linewidths=0.4)

        ax.set_title(f'Track {tid}', fontsize=9, fontweight='bold')
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper left')

    for idx in range(len(top), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f'{SEQ}: GT vs Ours 轨迹 (均用GT位姿, 起点归零)\n'
        f'GT深度3帧中值滤波 + Ours深度3帧中值滤波并逐帧median scaling对齐GT',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'per_track_aligned.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_png}')


if __name__ == '__main__':
    main()
