#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 rgb_warped 图上标注运动点的运动方向 (GT vs Ours 对比).

思路:
  运动方向 = 3D 世界位移向量 (P_end - P_start) 投影到背景帧图像.
  - GT    : GT深度(tiff) + GT位姿 反投影
  - Ours  : Ours深度(npz/scale) + GT位姿 反投影  (只比深度, 消除位姿误差)
  两条箭头同点出发(投影后略偏移), 方向/长度差异体现深度估计差异.

输出: 一张总览图 (背景帧 = 中间帧), Top N 运动点各画 GT/Ours 箭头.
"""

import os, sys, json
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
K = np.array([[767.73, 0, 677.74], [0, 767.73, 543.06], [0, 0, 1]], dtype=np.float64)

TRAJ_JSON = os.path.join(BASE, 'ours_rerun', f'{SEQ}_baseline_motion_trajectories.json')
DEPTH_NPZ = os.path.join(SEQ_DIR, 'baseline_depth_maps.npz')
CHAIN_NPZ = os.path.join(SEQ_DIR, 'baseline_chain_data.npz')
RGB_DIR = os.path.join(SEQ_DIR, 'generated', 'rgb_warped')
OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis', 'rgb_motion_arrows')
os.makedirs(OUT_DIR, exist_ok=True)

N_TOP = 10
MAX_GT_JUMP_MM = 15.0
BG_FRAME = 58  # 背景帧 (rgb_warped frame_0058.png)
COL_GT = '#00c853'
COL_OURS = '#d50000'


def project(world_pt, T_w2c):
    ph = T_w2c @ np.array([world_pt[0], world_pt[1], world_pt[2], 1.0])
    X, Y, Z = ph[0], ph[1], ph[2]
    if Z <= 0.5:
        return None
    u = K[0, 0] * X / Z + K[0, 2]
    v = K[1, 1] * Y / Z + K[1, 2]
    return np.array([u, v])


def disp_norm(p0, p1):
    return float(np.linalg.norm(np.array(p1) - np.array(p0)))


def main():
    trajs = json.load(open(TRAJ_JSON))
    chain = np.load(CHAIN_NPZ, allow_pickle=True)
    gs = float(chain['global_depth_scale'])
    npz = np.load(DEPTH_NPZ, allow_pickle=True)
    ours_dep = {int(k): npz[k] / gs for k in npz.files}

    print('加载 GT 深度 + 位姿 ...')
    gt_dep = {}
    for fi in range(117):
        p = os.path.join(SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff')
        gt_dep[fi] = np.array(Image.open(p)).astype(np.float32) * GT_SCALE

    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    w2c = [np.linalg.inv(T) for T in gt_c2w]

    # 反投影 + 过滤
    recs = []
    for t in trajs:
        frs = t['frames']
        if len(frs) < 2:
            continue
        f0, f1 = frs[0], frs[-1]
        fi0, fi1 = int(f0['frame']), int(f1['frame'])
        if fi1 - fi0 < 2:
            continue
        u0, v0 = float(f0['u']), float(f0['v'])
        u1, v1 = float(f1['u']), float(f1['v'])
        if fi1 not in ours_dep:  # Ours 缺 116
            continue
        # GT 深度 Z 阶跃过滤 (伪影)
        zs = []
        for fr in frs:
            fi = int(fr['frame'])
            u = int(np.clip(round(float(fr['u'])), 0, 1349))
            v = int(np.clip(round(float(fr['v'])), 0, 1079))
            zs.append(float(gt_dep[fi][v, u]))
        zs = np.array(zs)
        if len(zs) >= 2 and np.max(np.abs(np.diff(zs))) > MAX_GT_JUMP_MM:
            continue
        # GT 轨迹 (GT深度+GT位姿)
        p0_gt = backproject_to_world(u0, v0, gt_dep[fi0], w2c[fi0])
        p1_gt = backproject_to_world(u1, v1, gt_dep[fi1], w2c[fi1])
        # Ours 轨迹 (Ours深度+GT位姿)
        p0_ou = backproject_to_world(u0, v0, ours_dep[fi0], w2c[fi0])
        p1_ou = backproject_to_world(u1, v1, ours_dep[fi1], w2c[fi1])
        if p0_gt is None or p1_gt is None or p0_ou is None or p1_ou is None:
            continue
        dgt = disp_norm(p0_gt, p1_gt)
        recs.append((t['track_id'], p0_gt, p1_gt, p0_ou, p1_ou, dgt, fi0, fi1))

    recs.sort(key=lambda r: -r[5])
    top = recs[:N_TOP]

    # 投影到背景帧
    T_bg = w2c[BG_FRAME]
    rows = []
    for tid, p0_gt, p1_gt, p0_ou, p1_ou, dgt, fi0, fi1 in top:
        a_gt = project(p0_gt, T_bg); b_gt = project(p1_gt, T_bg)
        a_ou = project(p0_ou, T_bg); b_ou = project(p1_ou, T_bg)
        if a_gt is None or b_gt is None or a_ou is None or b_ou is None:
            continue
        rows.append((tid, a_gt, b_gt, a_ou, b_ou, dgt))

    # 加载背景图
    bg_path = os.path.join(RGB_DIR, f'frame_{BG_FRAME:04d}.png')
    bg = np.array(Image.open(bg_path).convert('RGB'))

    fig, ax = plt.subplots(figsize=(13, 10.4))
    ax.imshow(bg)
    ax.set_xlim(0, 1350)
    ax.set_ylim(1080, 0)

    # 方向统计
    ang_diffs = []
    len_gt, len_ou = [], []
    for i, (tid, a_gt, b_gt, a_ou, b_ou, dgt) in enumerate(rows):
        v_gt = b_gt - a_gt
        v_ou = b_ou - a_ou
        lg = np.linalg.norm(v_gt); lo = np.linalg.norm(v_ou)
        len_gt.append(lg); len_ou.append(lo)
        ang_g = np.degrees(np.arctan2(v_gt[1], v_gt[0]))
        ang_o = np.degrees(np.arctan2(v_ou[1], v_ou[0]))
        d = abs(ang_g - ang_o); d = min(d, 360 - d)
        ang_diffs.append(d)
        # GT 箭头
        ax.arrow(a_gt[0], a_gt[1], v_gt[0], v_gt[1], head_width=14, head_length=18,
                 fc=COL_GT, ec=COL_GT, lw=2.5, alpha=0.9, length_includes_head=True)
        ax.scatter(a_gt[0], a_gt[1], s=40, c=COL_GT, zorder=5,
                   edgecolors='white', linewidths=0.6)
        # Ours 箭头
        ax.arrow(a_ou[0], a_ou[1], v_ou[0], v_ou[1], head_width=14, head_length=18,
                 fc=COL_OURS, ec=COL_OURS, lw=2.5, alpha=0.9, length_includes_head=True)
        ax.scatter(a_ou[0], a_ou[1], s=40, c=COL_OURS, zorder=5,
                   edgecolors='white', linewidths=0.6)
        ax.text(a_gt[0] + 6, a_gt[1] - 6, f'#{i+1}', fontsize=9, color='yellow',
                fontweight='bold', zorder=6)

    ax.legend(handles=[
        plt.Line2D([0], [0], color=COL_GT, lw=3, label='GT 运动方向 (GT深度)'),
        plt.Line2D([0], [0], color=COL_OURS, lw=3, label='Ours 运动方向 (Ours深度)'),
    ], loc='upper right', fontsize=11, framealpha=0.9)

    if ang_diffs:
        print(f'方向角差(度): mean={np.mean(ang_diffs):.1f}, median={np.median(ang_diffs):.1f}, max={np.max(ang_diffs):.1f}')
        print(f'箭头长度(px): GT mean={np.mean(len_gt):.0f}, Ours mean={np.mean(len_ou):.0f}')
        print('逐点方向角差:', [f'{d:.0f}' for d in ang_diffs])

    ax.set_title(
        f'{SEQ}  背景帧={BG_FRAME}: 运动点运动方向对比 (GT深度 vs Ours深度, 均GT位姿反投影)\n'
        f'Top{N_TOP} 运动点, 已过滤 GT 深度阶跃>{MAX_GT_JUMP_MM}mm 伪影',
        fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'rgb_motion_arrows.png')
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_png}')


if __name__ == '__main__':
    main()
