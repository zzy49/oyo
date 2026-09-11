#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rgb_warped 总览图: 一张图上标注多个运动点的运动方向 (GT + 4方法).

严谨投影:
  - 背景帧 BG_FRAME 固定
  - 每个 track 取「最接近背景帧的帧 f_a」作锚点帧, 锚点=(u_a, v_a)
  - 箭头 = 末帧世界点 P_end 投影回锚点帧相机 与 锚点 之差
    (各方法 P_end 由各自深度反投影得到, 只比深度, 均 GT 位姿)
  - 只保留锚点帧与背景帧距离 <= NEAR_THRESH 的 track (避免投影失真)

口径同 perpoint 版本: 四方法 depth_maps_orig.npz(已物理尺度) + 逐帧 median scaling 对齐 GT.
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
OUT_DIR = os.path.join(BASE, 'ours_rerun', 'analysis', 'rgb_motion_arrows_overview_noscale')
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ['Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']
ALL = ['GT'] + METHODS
COLORS = {'GT': '#111111', 'Monodepth2': '#1f77b4', 'ManyDepth': '#ff7f0e',
          'Lite-Mono': '#2ca02c', 'Ours': '#d62728'}

N_TOP = 12
MAX_GT_JUMP_MM = 15.0
BG_FRAME = 58
NEAR_THRESH = 20


def project_pt(P, T_w2c):
    ph = T_w2c @ np.array([P[0], P[1], P[2], 1.0])
    X, Y, Z = ph[0], ph[1], ph[2]
    if Z <= 0.5:
        return None
    return np.array([FX * X / Z + CX, FY * Y / Z + CY])


def main():
    trajs = json.load(open(TRAJ_JSON))

    print('加载 GT 深度 ...')
    gt_dep = {}
    for fi in range(117):
        gt_dep[fi] = np.array(Image.open(
            os.path.join(SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))
        ).astype(np.float32) * GT_SCALE
    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    w2c = [np.linalg.inv(T) for T in gt_c2w]

    method_dep = {}
    for m in METHODS:
        npz = np.load(os.path.join(METH_DIR.format(m), 'depth_maps_orig.npz'))
        method_dep[m] = {int(k): npz[k].astype(np.float32) for k in npz.files}
    print('四方法深度加载完成 (不做 median scaling, 各用自身 global_scale 物理尺度)')

    recs = []
    for t in trajs:
        frs = t['frames']
        if len(frs) < 2:
            continue
        f1 = frs[-1]
        fi1 = int(f1['frame'])
        u1, v1 = float(f1['u']), float(f1['v'])
        # 锚点帧 = 最接近背景帧
        fa = min(frs, key=lambda fr: abs(int(fr['frame']) - BG_FRAME))
        fia = int(fa['frame'])
        if abs(fia - BG_FRAME) > NEAR_THRESH:
            continue
        ua, va = float(fa['u']), float(fa['v'])
        # GT 阶跃过滤
        zs = []
        for fr in frs:
            fi = int(fr['frame'])
            u = int(np.clip(round(float(fr['u'])), 0, 1349))
            v = int(np.clip(round(float(fr['v'])), 0, 1079))
            zs.append(float(gt_dep[fi][v, u]))
        zs = np.array(zs)
        if len(zs) >= 2 and np.max(np.abs(np.diff(zs))) > MAX_GT_JUMP_MM:
            continue
        arrows = {}
        disp = {}
        for name in ALL:
            dep = gt_dep if name == 'GT' else method_dep[name]
            p1 = backproject_to_world(u1, v1, dep[fi1], w2c[fi1])
            pa = backproject_to_world(ua, va, dep[fia], w2c[fia])
            if p1 is None or pa is None:
                arrows[name] = None
                disp[name] = None
                continue
            e2d = project_pt(p1, w2c[fia])
            if e2d is None:
                arrows[name] = None
                disp[name] = None
                continue
            arrows[name] = e2d - np.array([ua, va])
            disp[name] = float(np.linalg.norm(np.array(p1) - np.array(pa)))
        if disp['GT'] is None:
            continue
        recs.append((t['track_id'], arrows, disp, ua, va))

    recs.sort(key=lambda r: -r[2]['GT'])
    top = recs[:N_TOP]

    # 总览图
    bg = np.array(Image.open(os.path.join(RGB_DIR, f'frame_{BG_FRAME:04d}.png')).convert('RGB'))
    fig, ax = plt.subplots(figsize=(13, 10.4))
    ax.imshow(bg)
    ax.set_xlim(0, 1350)
    ax.set_ylim(1080, 0)

    ang_stats = {m: [] for m in METHODS}
    for i, (tid, arrows, disp, ua, va) in enumerate(top):
        ax.scatter(ua, va, s=55, c='yellow', zorder=6, edgecolors='black', linewidths=0.7)
        ax.text(ua + 8, va - 8, f'#{i+1}', fontsize=10, color='yellow',
                fontweight='bold', zorder=7)
        for name in ALL:
            v = arrows[name]
            if v is None or np.linalg.norm(v) < 0.5:
                continue
            color = COLORS[name]
            lw = 2.8 if name == 'GT' else 1.8
            ax.arrow(ua, va, v[0], v[1], head_width=14, head_length=18,
                     fc=color, ec=color, lw=lw, alpha=0.95, length_includes_head=True)
            if name in METHODS:
                vg = arrows['GT']
                if vg is not None and np.linalg.norm(vg) > 1 and np.linalg.norm(v) > 1:
                    a_g = np.degrees(np.arctan2(vg[1], vg[0]))
                    a_m = np.degrees(np.arctan2(v[1], v[0]))
                    d = abs(a_g - a_m); d = min(d, 360 - d)
                    ang_stats[name].append(d)

    handles = [plt.Line2D([0], [0], color=COLORS[n], lw=3, label=f'{n}') for n in ALL]
    ax.legend(handles=handles, loc='upper right', fontsize=11, framealpha=0.9)

    print(f'\n各方法 vs GT 方向角差 (度), n={len(ang_stats["Ours"])}:')
    for m in METHODS:
        if ang_stats[m]:
            print(f'  {m:<12} mean={np.mean(ang_stats[m]):.1f}, median={np.median(ang_stats[m]):.1f}, '
                  f'max={np.max(ang_stats[m]):.1f}')
        else:
            print(f'  {m:<12} (无有效箭头)')

    print('\n各方法 3D位移 与 GT 比值 (中位数, 越接近1越好):')
    for m in METHODS:
        ratios = [r[2][m] / r[2]['GT'] for r in top
                  if r[2][m] is not None and r[2]['GT'] is not None and r[2]['GT'] > 1e-6]
        if ratios:
            print(f'  {m:<12} median={np.median(ratios):.2f}, '
                  f'mean={np.mean(ratios):.2f}, min={np.min(ratios):.2f}, max={np.max(ratios):.2f}')

    ax.set_title(
        f'{SEQ}  背景帧={BG_FRAME}: 运动点运动方向总览 (GT + 4方法, 均GT位姿反投影)\n'
        f'锚点帧=各track最接近背景帧的帧, 过滤GT深度阶跃>{MAX_GT_JUMP_MM}mm, Top{N_TOP} (不做median scaling)',
        fontsize=12.5, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'rgb_motion_arrows_overview.png')
    fig.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_png}')


if __name__ == '__main__':
    main()
