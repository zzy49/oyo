#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文图3补充: top5 运动点绝对位姿 —— 每个点一张图。

从 top5_trajs.pkl 读取 GT + 四方法各自的 top5 绝对位姿轨迹
(累计位移排名 P1~P5, 均为 GT 相机位姿反投影, 同一 GT 世界坐标系, 不归零),
每个点生成一张独立 3D 图, 图内叠加 GT / Monodepth2 / ManyDepth / Lite-Mono / Ours 五条轨迹。

输出目录: zhong/motion_vis_fig3_top5/
    - fig3_top5_point_P1.png ... fig3_top5_point_P5.png   每个点一张图
    - fig3_top5_point_grid.png                             5 点横排汇总
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'motion_vis_fig3_top5')
PKL = os.path.join(OUT_DIR, 'top5_trajs.pkl')

METHODS = ['GT', 'Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']
COLORS = {
    'GT': '#333333',
    'Monodepth2': '#1f77b4',
    'ManyDepth': '#ff7f0e',
    'Lite-Mono': '#2ca02c',
    'Ours': '#d62728',
}


def draw_point(ax, traj, color):
    """画一条绝对位姿轨迹: 线 + 起点圆 + 终点星."""
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
            color=color, lw=2.4, alpha=0.95)
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=color, s=70,
               marker='o', depthshade=False, edgecolors='k', linewidths=0.5)
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=color, s=140,
               marker='*', depthshade=False, edgecolors='k', linewidths=0.5)


def fit_axes(ax, pts):
    """按所有点自适应等比例坐标范围."""
    c = pts.mean(axis=0)
    r = max(float((pts.max(axis=0) - pts.min(axis=0)).max()) / 2, 0.5)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def main():
    with open(PKL, 'rb') as f:
        data = pickle.load(f)
    names = [n for n in METHODS if n in data]
    n_top = len(data[names[0]]['trajs'])
    print(f'方法: {names}, 每方法 top{n_top} 点')

    # ── 每个点一张图 ──
    for k in range(n_top):
        fig = plt.figure(figsize=(6.8, 6.8))
        ax = fig.add_subplot(111, projection='3d')
        all_pts = []
        for name in names:
            traj = np.asarray(data[name]['trajs'][k])
            d = data[name]['disps'][k]
            draw_point(ax, traj, COLORS[name])
            ax.plot([], [], [], color=COLORS[name], lw=2.4,
                    label=f'{name} ({d:.1f} mm)')
            all_pts.append(traj)
        fit_axes(ax, np.vstack(all_pts))
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
        ax.set_title(f'P{k + 1} — 运动点绝对位姿轨迹 (GT位姿反投影)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.view_init(elev=20, azim=-60)
        plt.tight_layout()
        p = os.path.join(OUT_DIR, f'fig3_top5_point_P{k + 1}.png')
        plt.savefig(p, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'Saved: {p}')

    # ── 5 点横排汇总 ──
    fig = plt.figure(figsize=(4.6 * n_top, 5.4))
    for k in range(n_top):
        ax = fig.add_subplot(1, n_top, k + 1, projection='3d')
        all_pts = []
        for name in names:
            traj = np.asarray(data[name]['trajs'][k])
            d = data[name]['disps'][k]
            draw_point(ax, traj, COLORS[name])
            all_pts.append(traj)
        fit_axes(ax, np.vstack(all_pts))
        ax.set_title(f'P{k + 1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.tick_params(labelsize=6)
        ax.view_init(elev=20, azim=-60)

    handles = [plt.Line2D([0], [0], color=COLORS[n], lw=2.4, label=n) for n in names]
    fig.legend(handles=handles, loc='lower center', ncol=len(names),
               fontsize=10, frameon=False)
    fig.suptitle('top5 运动点绝对位姿轨迹 (各方法各自累计位移排名, GT位姿反投影)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    pg = os.path.join(OUT_DIR, 'fig3_top5_point_grid.png')
    plt.savefig(pg, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {pg}')

    # ── 打印各点位移 (供核对) ──
    print('\n各点累计位移 (mm):')
    print('      ' + '  '.join(f'{n:<11}' for n in names))
    for k in range(n_top):
        row = '  '.join(f'{data[n]["disps"][k]:<11.2f}' for n in names)
        print(f'P{k + 1}  {row}')

    print(f'\n输出目录: {OUT_DIR}')


if __name__ == '__main__':
    main()
