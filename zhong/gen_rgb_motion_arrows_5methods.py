#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 rgb_warped 图上标注运动点运动方向 (GT + 4方法 对比).

统一口径:
  - 同一批 LoFTR track (u,v,frame 取自 Ours trajectory JSON)
  - 5 种深度分别反投影 (均用 GT 位姿, 消除位姿误差, 只比深度):
      GT            : GT深度(tiff, mm)
      Monodepth2/ManyDepth/Lite-Mono/Ours : 各自 depth_maps_orig.npz ÷ global_scale
  - 四方法深度做逐帧 median scaling 对齐 GT (尺度无关模型口径)
  - 运动方向 = 3D 位移向量 (P_end - P_start) 投影到背景帧图像 (方向向量正交投影, 稳健无爆炸)

输出: 一张总览图, Top N 运动点各画 5 条箭头.
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
FX, FY = K[0, 0], K[1, 1]

TRAJ_JSON = os.path.join(BASE, 'ours_rerun', f'{SEQ}_baseline_motion_trajectories.json')
METH_DIR = os.path.join(BASE, 'motion_vis_fig3_top5', 'method_{}')
RGB_DIR = os.path.join(SEQ_DIR, 'generated', 'rgb_warped')
OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis', 'rgb_motion_arrows_5methods')
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ['Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']
ALL = ['GT'] + METHODS
COLORS = {'GT': '#111111', 'Monodepth2': '#1f77b4', 'ManyDepth': '#ff7f0e',
          'Lite-Mono': '#2ca02c', 'Ours': '#d62728'}
LINESTYLES = {'GT': '-', 'Monodepth2': '--', 'ManyDepth': '-.',
              'Lite-Mono': ':', 'Ours': '-'}

N_TOP = 10
MAX_GT_JUMP_MM = 15.0
BG_FRAME = 58
PX_PER_MM = 4.0  # 箭头长度 = 横向位移(mm) × 此比例


def main():
    trajs = json.load(open(TRAJ_JSON))

    # 1. GT 深度 + 位姿
    print('加载 GT 深度 ...')
    gt_dep = {}
    for fi in range(117):
        gt_dep[fi] = np.array(Image.open(
            os.path.join(SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))
        ).astype(np.float32) * GT_SCALE
    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    w2c = [np.linalg.inv(T) for T in gt_c2w]

    # 2. 四方法深度 + 逐帧 median scaling 对齐 GT
    method_dep = {}
    for m in METHODS:
        ddir = METH_DIR.format(m)
        scale = json.load(open(os.path.join(ddir, 'global_scale.json')))['global_depth_scale']
        npz = np.load(os.path.join(ddir, 'depth_maps_orig.npz'))
        dep = {int(k): npz[k].astype(np.float32) / scale for k in npz.files}
        scales = []
        for fi in range(117):
            g = gt_dep[fi]; o = dep[fi]
            mask = (g > 0.5) & (o > 0.5)
            if mask.sum() > 100:
                s = float(np.median(g[mask]) / max(np.median(o[mask]), 1e-6))
                if np.isfinite(s) and s > 0:
                    dep[fi] = o * s
                    scales.append(s)
        method_dep[m] = dep
        print(f'  {m}: global_scale={scale:.4f}, median_scaling mean={np.mean(scales):.4f}')

    # 3. 反投影 + 过滤
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
        # GT 深度阶跃过滤 (伪影)
        zs = []
        for fr in frs:
            fi = int(fr['frame'])
            u = int(np.clip(round(float(fr['u'])), 0, 1349))
            v = int(np.clip(round(float(fr['v'])), 0, 1079))
            zs.append(float(gt_dep[fi][v, u]))
        zs = np.array(zs)
        if len(zs) >= 2 and np.max(np.abs(np.diff(zs))) > MAX_GT_JUMP_MM:
            continue

        # 各配置 3D 位移向量
        vecs = {}  # name -> v_world(3,)
        disp_mm = {}
        for name in ALL:
            dep = gt_dep if name == 'GT' else method_dep[name]
            p0 = backproject_to_world(u0, v0, dep[fi0], w2c[fi0])
            p1 = backproject_to_world(u1, v1, dep[fi1], w2c[fi1])
            if p0 is None or p1 is None:
                vecs[name] = None
                disp_mm[name] = None
                continue
            vecs[name] = np.array(p1) - np.array(p0)
            disp_mm[name] = float(np.linalg.norm(vecs[name]))
        if disp_mm['GT'] is None:
            continue
        # 锚点: track 中最接近背景帧的帧的像素
        best = min(frs, key=lambda fr: abs(int(fr['frame']) - BG_FRAME))
        anchor = np.array([float(best['u']), float(best['v'])])
        recs.append((t['track_id'], vecs, disp_mm, anchor))

    recs.sort(key=lambda r: -r[2]['GT'])
    top = recs[:N_TOP]

    # 4. 投影方向向量到背景帧
    R_bg = w2c[BG_FRAME][:3, :3]
    rows = []
    for tid, vecs, disp_mm, anchor in top:
        proj = {}
        for name in ALL:
            v = vecs[name]
            if v is None:
                proj[name] = None
                continue
            v_cam = R_bg @ v  # 世界位移 → 背景帧相机坐标
            proj[name] = np.array([v_cam[0] * PX_PER_MM, v_cam[1] * PX_PER_MM])
        rows.append((tid, proj, disp_mm, anchor))

    # 5. 绘图
    bg = np.array(Image.open(os.path.join(RGB_DIR, f'frame_{BG_FRAME:04d}.png')).convert('RGB'))
    fig, ax = plt.subplots(figsize=(13, 10.4))
    ax.imshow(bg)
    ax.set_xlim(0, 1350)
    ax.set_ylim(1080, 0)

    # 角度差统计 (各方法 vs GT)
    ang_stats = {m: [] for m in METHODS}
    for i, (tid, proj, disp_mm, anchor) in enumerate(rows):
        ax.scatter(*anchor, s=50, c='yellow', zorder=6, edgecolors='black', linewidths=0.6)
        ax.text(anchor[0] + 8, anchor[1] - 8, f'#{i+1}', fontsize=10, color='yellow',
                fontweight='bold', zorder=7)
        for name in ALL:
            v = proj[name]
            if v is None:
                continue
            color = COLORS[name]
            ls = LINESTYLES[name]
            lw = 2.6 if name == 'GT' else 1.8
            if name == 'GT':
                ax.arrow(anchor[0], anchor[1], v[0], v[1], head_width=14, head_length=18,
                         fc=color, ec=color, lw=lw, alpha=0.95, length_includes_head=True,
                         linestyle=ls)
            else:
                ax.arrow(anchor[0], anchor[1], v[0], v[1], head_width=12, head_length=16,
                         fc=color, ec=color, lw=lw, alpha=0.85, length_includes_head=True,
                         linestyle=ls)
            if name in METHODS:
                vg = proj['GT']
                if vg is not None and np.linalg.norm(vg) > 1e-6 and np.linalg.norm(v) > 1e-6:
                    ang_g = np.degrees(np.arctan2(vg[1], vg[0]))
                    ang_m = np.degrees(np.arctan2(v[1], v[0]))
                    d = abs(ang_g - ang_m); d = min(d, 360 - d)
                    ang_stats[name].append(d)

    handles = [plt.Line2D([0], [0], color=COLORS[n], lw=2.6, ls=LINESTYLES[n],
                          label=f'{n}') for n in ALL]
    ax.legend(handles=handles, loc='upper right', fontsize=11, framealpha=0.9)

    print('\n各方法 vs GT 方向角差 (度):')
    for m in METHODS:
        if ang_stats[m]:
            print(f'  {m:<12} mean={np.mean(ang_stats[m]):.1f}, median={np.median(ang_stats[m]):.1f}, '
                  f'max={np.max(ang_stats[m]):.1f}, n={len(ang_stats[m])}')
        else:
            print(f'  {m:<12} (无有效箭头)')

    ax.set_title(
        f'{SEQ}  背景帧={BG_FRAME}: 运动点运动方向对比 (GT + 4方法, 均GT位姿反投影)\n'
        f'四方法深度已÷scale+逐帧median scaling对齐GT, 过滤GT深度阶跃>{MAX_GT_JUMP_MM}mm, Top{N_TOP}',
        fontsize=12.5, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'rgb_motion_arrows_5methods.png')
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_png}')


if __name__ == '__main__':
    main()
