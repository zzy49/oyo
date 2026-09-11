#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rgb_warped 图上标注运动方向 (GT + 4方法), 逐点出图.

正确投影: 每个 track 用「首帧相机」投影, 起点=首帧像素(u0,v0), 终点=末帧世界点投影回首帧.
避免固定背景帧导致的坐标错配.

口径:
  - 同一批 track (u,v,frame 取自 Ours trajectory JSON)
  - 5 种深度反投影, 均用 GT 位姿 (只比深度):
      GT = GT tiff
      四方法 = motion_vis_fig3_top5/method_*/depth_maps_orig.npz (已物理尺度)
  - 四方法逐帧 median scaling 对齐 GT (尺度无关模型口径)
  - 运动方向 = (P_end - P_start) 经首帧相机透视投影到 2D

输出: Top N 逐点图 (每点一张, 背景=首帧 rgb_warped, 5 箭头叠加) + 方向角差统计.
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
FX, FY, CX, CY = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

TRAJ_JSON = os.path.join(BASE, 'ours_rerun', f'{SEQ}_baseline_motion_trajectories.json')
METH_DIR = os.path.join(BASE, 'motion_vis_fig3_top5', 'method_{}')
RGB_DIR = os.path.join(SEQ_DIR, 'generated', 'rgb_warped')
OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis', 'rgb_motion_arrows_perpoint')
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ['Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']
ALL = ['GT'] + METHODS
COLORS = {'GT': '#111111', 'Monodepth2': '#1f77b4', 'ManyDepth': '#ff7f0e',
          'Lite-Mono': '#2ca02c', 'Ours': '#d62728'}

N_TOP = 5
MAX_GT_JUMP_MM = 15.0


def project_pt(P, T_w2c):
    ph = T_w2c @ np.array([P[0], P[1], P[2], 1.0])
    X, Y, Z = ph[0], ph[1], ph[2]
    if Z <= 0.5:
        return None
    return np.array([FX * X / Z + CX, FY * Y / Z + CY])


def main():
    trajs = json.load(open(TRAJ_JSON))

    # GT 深度 + 位姿
    print('加载 GT 深度 ...')
    gt_dep = {}
    for fi in range(117):
        gt_dep[fi] = np.array(Image.open(
            os.path.join(SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))
        ).astype(np.float32) * GT_SCALE
    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    w2c = [np.linalg.inv(T) for T in gt_c2w]

    # 四方法深度 (已物理尺度) + 逐帧 median scaling 对齐 GT
    method_dep = {}
    for m in METHODS:
        npz = np.load(os.path.join(METH_DIR.format(m), 'depth_maps_orig.npz'))
        dep = {int(k): npz[k].astype(np.float32) for k in npz.files}
        sc = []
        for fi in range(117):
            g = gt_dep[fi]; o = dep[fi]
            mask = (g > 0.5) & (o > 0.5)
            if mask.sum() > 100:
                s = float(np.median(g[mask]) / max(np.median(o[mask]), 1e-6))
                if np.isfinite(s) and s > 0:
                    dep[fi] = o * s
                    sc.append(s)
        method_dep[m] = dep
        print(f'  {m}: median_scaling mean={np.mean(sc):.4f} std={np.std(sc):.4f}')

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
        # GT 深度阶跃过滤
        zs = []
        for fr in frs:
            fi = int(fr['frame'])
            u = int(np.clip(round(float(fr['u'])), 0, 1349))
            v = int(np.clip(round(float(fr['v'])), 0, 1079))
            zs.append(float(gt_dep[fi][v, u]))
        zs = np.array(zs)
        if len(zs) >= 2 and np.max(np.abs(np.diff(zs))) > MAX_GT_JUMP_MM:
            continue
        # 各配置: 末帧世界点投影回首帧
        arrows = {}
        disp = {}
        for name in ALL:
            dep = gt_dep if name == 'GT' else method_dep[name]
            p1 = backproject_to_world(u1, v1, dep[fi1], w2c[fi1])
            if p1 is None:
                arrows[name] = None
                disp[name] = None
                continue
            e2d = project_pt(p1, w2c[fi0])
            if e2d is None:
                arrows[name] = None
                disp[name] = None
                continue
            p0 = backproject_to_world(u0, v0, dep[fi0], w2c[fi0])
            arrows[name] = e2d - np.array([u0, v0])
            disp[name] = float(np.linalg.norm(np.array(p1) - np.array(p0))) if p0 is not None else None
        if disp['GT'] is None:
            continue
        recs.append((t['track_id'], arrows, disp, fi0, u0, v0))

    recs.sort(key=lambda r: -r[2]['GT'])
    top = recs[:N_TOP]

    # 逐点出图
    ang_stats = {m: [] for m in METHODS}
    for i, (tid, arrows, disp, fi0, u0, v0) in enumerate(top):
        bg_path = os.path.join(RGB_DIR, f'frame_{fi0:04d}.png')
        bg = np.array(Image.open(bg_path).convert('RGB'))
        fig, ax = plt.subplots(figsize=(9, 7.2))
        ax.imshow(bg)
        ax.set_xlim(0, 1350)
        ax.set_ylim(1080, 0)
        ax.scatter(u0, v0, s=70, c='yellow', zorder=6, edgecolors='black', linewidths=0.8)
        for name in ALL:
            v = arrows[name]
            if v is None or np.linalg.norm(v) < 0.5:
                continue
            color = COLORS[name]
            lw = 3.0 if name == 'GT' else 2.0
            ax.arrow(u0, v0, v[0], v[1], head_width=16, head_length=20,
                     fc=color, ec=color, lw=lw, alpha=0.95, length_includes_head=True)
            if name in METHODS:
                vg = arrows['GT']
                if vg is not None and np.linalg.norm(vg) > 1:
                    a_g = np.degrees(np.arctan2(vg[1], vg[0]))
                    a_m = np.degrees(np.arctan2(v[1], v[0]))
                    d = abs(a_g - a_m); d = min(d, 360 - d)
                    ang_stats[name].append(d)
        handles = [plt.Line2D([0], [0], color=COLORS[n], lw=3,
                              label=f'{n} ({disp[n]:.1f}mm)' if disp[n] else f'{n}')
                   for n in ALL]
        ax.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.9)
        ax.set_title(f'Track {tid}  首帧={fi0}  GT位移={disp["GT"]:.1f}mm', fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, f'arrow_point_P{i+1}_track{tid}.png')
        fig.savefig(out_png, dpi=110, bbox_inches='tight')
        plt.close(fig)

    print('\n各方法 vs GT 方向角差 (度):')
    for m in METHODS:
        if ang_stats[m]:
            print(f'  {m:<12} mean={np.mean(ang_stats[m]):.1f}, median={np.median(ang_stats[m]):.1f}, '
                  f'max={np.max(ang_stats[m]):.1f}, n={len(ang_stats[m])}')
        else:
            print(f'  {m:<12} (无有效箭头)')
    print(f'\nSaved: {OUT_DIR}')


if __name__ == '__main__':
    main()
