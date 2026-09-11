#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成论文图3: 组织运动 3D 轨迹可视化 (四方法 + GT, 起终点+位移箭头)。

序列: c1_transverse1_t1_v2 (与表3一致)
方法: GT 真值 / Monodepth2 / ManyDepth / Lite-Mono / Ours

GT 组织运动真值:
    - vertex_static/frame_{i:04d}.npy  顶点世界坐标 (mm, 所有帧相同)
    - motion_gt/frame_{i:04d}.npy      顶点累计位移 (米, 相对 frame0)
    - mask_static/frame_{i:04d}.npy    静止=1 / 运动=0
    运动顶点起终点: start=vertex, end=vertex+motion_last*1000

管线 (四方法) 组织运动轨迹:
    1. estimate_pipeline_params + run_vo_sequence (pnp_scale, F1 关闭)
       → baseline_chain_data.npz (LoFTR k0/k1) + baseline_abs_poses.npy + baseline_depth_maps.npz
    2. 深度 ÷global_scale 恢复原始尺度; 三基线额外 median scaling 对齐 GT
    3. chain_tracks 链式追踪 → classify_by_displacement 分类运动/静止
    4. 运动 track 首末帧 VO 位姿+深度反投影 → 起终点 + 位移矢量

输出目录: zhong/motion_vis_fig3/
    - gt_motion_arrows.npy / motion_arrows_<name>.npy   起终点+位移
    - fig3_summary.json
    - fig3_motion_arrows.png                            5 子图 3D 位移箭头对比
"""

import os
import sys
import json
import numpy as np
import torch
import random
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

import test_v6_dyendovo as tv6
from test_v6_multi_model_enhanced import load_model, SEQ_NAME, SEQ_DIR, DATA_ROOT
from eval_pipeline_motion_vs_gt import (
    chain_tracks, backproject_to_world, load_gt_poses_orderF,
    classify_by_displacement,
)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda')

MODELS = [
    {'name': 'Monodepth2', 'path': r'C:\Users\Administrator\tmp\c3vd_md2_multi\models\weights_19', 'type': 'md2'},
    {'name': 'ManyDepth',  'path': r'C:\Users\Administrator\tmp\c3vd_manydepth_multi\models\weights_19', 'type': 'manydepth'},
    {'name': 'Lite-Mono',  'path': r'C:\Users\Administrator\tmp\c3vd_litemono_multi\models\weights_19', 'type': 'litemono'},
    {'name': 'Ours',       'path': r'e:\data1\monodepth2\models\depth', 'type': 'md2'},
]

OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'motion_vis_fig3')
os.makedirs(OUT_DIR, exist_ok=True)

GT_SCALE = 100.0 / 65535.0
N_ARROWS = 500            # 每个方法采样运动点数量
N_BINS = 5                # 位移分位数分层数 (每层等量采样)
MAX_TRACKS = 15000        # 链式追踪 track 采样上限
ARROW_SCALE = 25.0        # 箭头长度放大因子 (位移 mm × 因子 = 视觉长度)

# 5 子图颜色 (图例/标题用)
PANEL_COLORS = {
    'GT': '#333333',
    'Monodepth2': '#1f77b4',
    'ManyDepth': '#ff7f0e',
    'Lite-Mono': '#2ca02c',
    'Ours': '#d62728',
}


def stratified_sample(dnorm, n_total=N_ARROWS, n_bins=N_BINS):
    """按位移分位数分层等量采样, 返回索引 (展示完整位移梯度)。"""
    if len(dnorm) <= n_total:
        return np.arange(len(dnorm))
    order = np.argsort(dnorm)
    n_per = n_total // n_bins
    keeps = []
    for b in range(n_bins):
        lo = int(b * len(order) / n_bins)
        hi = int((b + 1) * len(order) / n_bins)
        seg = order[lo:hi]
        if len(seg) > n_per:
            seg = np.random.choice(seg, n_per, replace=False)
        keeps.append(seg)
    keep = np.concatenate(keeps)
    keep.sort()
    return keep


def extract_gt_motion():
    """提取 GT 组织运动顶点起终点 + 位移矢量 (mesh 世界系, mm)。"""
    gen_dir = os.path.join(SEQ_DIR, 'generated')
    vertex = np.load(os.path.join(gen_dir, 'vertex_static', 'frame_0000.npy'))  # (N,3) mm
    mask = np.load(os.path.join(gen_dir, 'mask_static', 'frame_0000.npy'))       # (N,) 1=静止 0=运动
    # 末帧 motion_gt (累计位移, 米)
    motion_files = sorted(os.listdir(os.path.join(gen_dir, 'motion_gt')))
    motion_last = np.load(os.path.join(gen_dir, 'motion_gt', motion_files[-1]))  # (N,3) 米
    moving = mask <= 0.5
    start = vertex[moving]                                  # (M,3) mm
    disp = motion_last[moving] * 1000.0                     # (M,3) mm
    end = start + disp
    dnorm = np.linalg.norm(disp, axis=1)
    # 按位移分位数分层采样 (展示完整位移梯度)
    keep = stratified_sample(dnorm)
    print(f'  [GT] 运动顶点={int(moving.sum())} 采样={len(keep)} '
          f'位移 median={np.median(dnorm[keep]):.3f} max={dnorm[keep].max():.3f} mm')
    return start[keep], end[keep], disp[keep]


def prepare_depth_maps(seq_dir, chain, depth_npz, use_median_scale):
    """深度 ÷global_scale 恢复原始尺度; 可选逐帧 median scaling 对齐 GT。"""
    depth_maps = {k: depth_npz[k] for k in depth_npz.files}
    global_scale = float(chain['global_depth_scale']) if 'global_depth_scale' in chain else 1.0
    depth_src = str(chain['depth_source']) if 'depth_source' in chain else 'pred'
    if depth_src != 'gt' and global_scale != 1.0:
        depth_maps = {k: v / global_scale for k, v in depth_maps.items()}
        print(f'  深度尺度恢复: ÷global_scale={global_scale:.4f}')

    if use_median_scale:
        gt_depth_dir = os.path.join(seq_dir, 'depth')
        scales = []
        for k, v in depth_maps.items():
            gt_path = os.path.join(gt_depth_dir, f'{int(k):04d}_depth.tiff')
            if not os.path.exists(gt_path):
                continue
            gt_mm = np.array(Image.open(gt_path)).astype(np.float32) * GT_SCALE
            m = (gt_mm > 0.5) & (v > 0.5)
            if m.sum() > 100:
                s = float(np.median(gt_mm[m]) / np.median(v[m]))
                if np.isfinite(s) and s > 0:
                    depth_maps[k] = v * s
                    scales.append(s)
        if scales:
            print(f'  深度 median scaling: mean_scale={np.mean(scales):.4f} (n={len(scales)})')
    return depth_maps


def sample_tracks(tracks, max_tracks=MAX_TRACKS):
    """采样 track (长 track 优先), 与评估脚本口径一致。"""
    if len(tracks) <= max_tracks:
        return tracks
    long_t = [t for t in tracks if len(t['frames']) >= 5]
    short_t = [t for t in tracks if len(t['frames']) < 5]
    n_long = min(len(long_t), int(max_tracks * 2 / 3))
    n_short = max_tracks - n_long
    sampled = (random.sample(long_t, n_long) if n_long > 0 else []) + \
              (random.sample(short_t, min(n_short, len(short_t)))
               if n_short > 0 and short_t else [])
    return sampled


def extract_vo_motion(name, use_median_scale):
    """跑 VO 并提取运动 track 起终点 + 位移矢量 (VO 世界系, mm)。"""
    seq_dir = SEQ_DIR
    chain_path = os.path.join(seq_dir, 'baseline_chain_data.npz')
    abs_poses_path = os.path.join(seq_dir, 'baseline_abs_poses.npy')
    depth_path = os.path.join(seq_dir, 'baseline_depth_maps.npz')

    chain = np.load(chain_path, allow_pickle=True)
    depth_npz = np.load(depth_path, allow_pickle=True)
    abs_poses = np.load(abs_poses_path)          # (N,4,4) world_to_cam
    n_pairs = int(chain['n_pairs'])
    all_k0 = [chain[f'k0_{i:04d}'] for i in range(n_pairs)]
    all_k1 = [chain[f'k1_{i:04d}'] for i in range(n_pairs)]
    chain_dist_thresh = float(chain['chain_dist_thresh']) if 'chain_dist_thresh' in chain else 3.0

    depth_maps = prepare_depth_maps(seq_dir, chain, depth_npz, use_median_scale)

    # 链式追踪
    tracks = chain_tracks(all_k0, all_k1, dist_thresh=chain_dist_thresh)
    print(f'  总 tracks={len(tracks)}')
    tracks = sample_tracks(tracks)

    # 运动/静止分类
    _, moving_tracks, threshold = classify_by_displacement(
        tracks, depth_maps, abs_poses)
    print(f'  分类阈值={threshold:.2f}mm  运动 tracks={len(moving_tracks)}')

    # 提取运动 track 首末帧反投影 → 起终点
    starts, ends, disps = [], [], []
    for t, _med in moving_tracks:
        frames = t['frames']
        if len(frames) < 2:
            continue
        # 第一个可反投影帧
        p0 = None
        for fi, u, v in frames:
            if fi >= len(abs_poses):
                continue
            key = f'{fi:04d}'
            if key not in depth_maps:
                continue
            p0 = backproject_to_world(u, v, depth_maps[key], abs_poses[fi])
            if p0 is not None:
                break
        # 最后一个可反投影帧
        p1 = None
        for fi, u, v in reversed(frames):
            if fi >= len(abs_poses):
                continue
            key = f'{fi:04d}'
            if key not in depth_maps:
                continue
            p1 = backproject_to_world(u, v, depth_maps[key], abs_poses[fi])
            if p1 is not None:
                break
        if p0 is None or p1 is None:
            continue
        d = p1 - p0
        starts.append(p0)
        ends.append(p1)
        disps.append(d)

    if not starts:
        print(f'  [WARN] {name} 无有效运动 track')
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    starts = np.array(starts)
    ends = np.array(ends)
    disps = np.array(disps)
    dnorm = np.linalg.norm(disps, axis=1)
    # 按位移分位数分层采样 (展示完整位移梯度)
    keep = stratified_sample(dnorm)
    print(f'  运动点={len(starts)} 采样={len(keep)} '
          f'位移 median={np.median(dnorm[keep]):.3f} max={dnorm[keep].max():.3f} mm')
    return starts[keep], ends[keep], disps[keep]


def run_vo(name, cfg):
    """跑 VO 生成 baseline_* 产物。"""
    print(f'\n===== [{name}] 跑 VO =====')
    encoder, depth_decoder, motion_encoder = load_model(cfg, DEVICE)
    params = tv6.estimate_pipeline_params(
        SEQ_DIR, encoder, depth_decoder, motion_encoder, DEVICE,
        depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')
    tv6.run_vo_sequence(
        SEQ_NAME, encoder, depth_decoder, DEVICE,
        motion_net=None, mode='baseline', max_frames=None,
        motion_encoder=motion_encoder, depth_source='pred',
        loftr_source='cache', data_root=DATA_ROOT,
        pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale')
    del encoder, depth_decoder, motion_encoder
    torch.cuda.empty_cache()


def center_points(starts, ends):
    """起点质心移到原点, 终点同步平移 (便于跨方法视觉对比)。"""
    c = starts.mean(axis=0)
    return starts - c, ends - c


def load_arrows():
    """从已保存的 npy 加载起终点/位移 (跳过 VO, 用于 --plot-only)。"""
    names = ['GT', 'Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']
    arrows = {}
    summary = {}
    for name in names:
        fn = 'gt_motion_arrows.npy' if name == 'GT' else f'motion_arrows_{name}.npy'
        arr = np.load(os.path.join(OUT_DIR, fn))          # (3, N, 3)
        s, e, d = arr[0], arr[1], arr[2]
        arrows[name] = {'start': s, 'end': e, 'disp': d}
        summary[name] = {'n': int(len(s)),
                         'disp_median_mm': float(np.median(np.linalg.norm(d, axis=1)))}
    return arrows, summary


def plot_figure(arrows, summary):
    """5 子图: 组织运动位移箭头 (起终点)。"""
    names = ['GT', 'Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']
    # 统一颜色映射范围 (vmin=0, vmax=所有方法位移 98 分位)
    all_d = np.concatenate([np.linalg.norm(arrows[n]['disp'], axis=1) for n in names])
    vmax = float(np.percentile(all_d, 98)) if len(all_d) else 5.0
    vmax = max(vmax, 1.0)
    norm = Normalize(vmin=0, vmax=vmax)
    cmap = cm.get_cmap('turbo')

    # 统一等比例坐标轴范围 (基于放大后的箭头, 保证 5 子图视觉可比)
    all_pts = []
    for name in names:
        starts, ends, disps = arrows[name]['start'], arrows[name]['end'], arrows[name]['disp']
        starts_c, _ = center_points(starts, ends)
        all_pts.append(starts_c)
        all_pts.append(starts_c + disps * ARROW_SCALE)
    all_pts = np.vstack(all_pts)
    lo = all_pts.min(axis=0) - 5
    hi = all_pts.max(axis=0) + 5
    mid = (lo + hi) / 2
    half = float((hi - lo).max()) / 2
    lo = mid - half
    hi = mid + half

    fig = plt.figure(figsize=(22, 5))
    for idx, name in enumerate(names):
        starts, ends, disps = arrows[name]['start'], arrows[name]['end'], arrows[name]['disp']
        starts_c, ends_c = center_points(starts, ends)
        dnorm = np.linalg.norm(disps, axis=1)
        rgba = cmap(norm(dnorm))

        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
        # 位移箭头 (起点 -> 终点, 放大 ARROW_SCALE 倍)
        dx, dy, dz = (disps * ARROW_SCALE).T
        ax.quiver(starts_c[:, 0], starts_c[:, 1], starts_c[:, 2],
                  dx, dy, dz, length=1.0, normalize=False,
                  colors=rgba, linewidth=1.2, arrow_length_ratio=0.18)
        # 起点小点
        ax.scatter(starts_c[:, 0], starts_c[:, 1], starts_c[:, 2],
                   c='#888888', s=5, alpha=0.5, depthshade=False)
        ax.set_title(f'{name}\nN={len(starts)}  中位 {np.median(dnorm):.2f} mm',
                     fontsize=10, color=PANEL_COLORS[name])
        ax.set_xlabel('X (mm)', fontsize=8)
        ax.set_ylabel('Y (mm)', fontsize=8)
        ax.set_zlabel('Z (mm)', fontsize=8)
        ax.tick_params(labelsize=7)
        # 等比例坐标轴
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.view_init(elev=20, azim=-60)

    # colorbar (位移幅度 mm)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.02, pad=0.02, shrink=0.8)
    cbar.set_label('组织运动位移 (mm)', fontsize=10)

    fig.suptitle(f'组织运动 3D 轨迹可视化 (箭头长度 ×{ARROW_SCALE:.0f})  —  {SEQ_NAME}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(OUT_DIR, 'fig3_motion_arrows.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

    # 逐方法独立子图 (便于用户自行排版合成)
    for name in names:
        starts, ends, disps = arrows[name]['start'], arrows[name]['end'], arrows[name]['disp']
        starts_c, ends_c = center_points(starts, ends)
        dnorm = np.linalg.norm(disps, axis=1)
        rgba = cmap(norm(dnorm))
        fig2 = plt.figure(figsize=(6, 6))
        ax2 = fig2.add_subplot(111, projection='3d')
        dx, dy, dz = (disps * ARROW_SCALE).T
        ax2.quiver(starts_c[:, 0], starts_c[:, 1], starts_c[:, 2],
                   dx, dy, dz, length=1.0, normalize=False,
                   colors=rgba, linewidth=1.4, arrow_length_ratio=0.18)
        ax2.scatter(starts_c[:, 0], starts_c[:, 1], starts_c[:, 2],
                    c='#888888', s=8, alpha=0.6, depthshade=False)
        ax2.set_title(f'{name}  (N={len(starts)}, 中位 {np.median(dnorm):.2f} mm)',
                      fontsize=11, color=PANEL_COLORS[name])
        ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
        ax2.set_xlim(lo[0], hi[0]); ax2.set_ylim(lo[1], hi[1]); ax2.set_zlim(lo[2], hi[2])
        ax2.view_init(elev=20, azim=-60)
        fig2.tight_layout()
        p = os.path.join(OUT_DIR, f'fig3_panel_{name}.png')
        fig2.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved: {p}')


def main(plot_only=False):
    if plot_only:
        print(f'[plot-only] 从已有 npy 加载数据并绘图, 序列: {SEQ_NAME}')
        arrows, summary = load_arrows()
        print('\n===== 汇总 =====')
        for n, m in summary.items():
            print(f'  {n:<12} N={m["n"]:>5d}  位移中位数={m["disp_median_mm"]:.3f} mm')
        plot_figure(arrows, summary)
        print(f'\n输出目录: {OUT_DIR}')
        for fn in sorted(os.listdir(OUT_DIR)):
            print(' ', fn)
        return

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print(f'设备: {DEVICE}, 序列: {SEQ_NAME}')

    arrows = {}
    summary = {}

    # ── GT 真值 (不依赖模型) ──
    print('\n===== [GT] 提取组织运动真值 =====')
    s, e, d = extract_gt_motion()
    arrows['GT'] = {'start': s, 'end': e, 'disp': d}
    summary['GT'] = {'n': int(len(s)), 'disp_median_mm': float(np.median(np.linalg.norm(d, axis=1)))}
    np.save(os.path.join(OUT_DIR, 'gt_motion_arrows.npy'),
            np.stack([s, e, d], axis=0))

    # ── 四方法 (跑 VO → 立即提取) ──
    for cfg in MODELS:
        name = cfg['name']
        run_vo(name, cfg)
        print(f'  提取 [{name}] 组织运动轨迹...')
        s, e, d = extract_vo_motion(name, use_median_scale=(name != 'Ours'))
        arrows[name] = {'start': s, 'end': e, 'disp': d}
        summary[name] = {'n': int(len(s)),
                         'disp_median_mm': float(np.median(np.linalg.norm(d, axis=1)))}
        np.save(os.path.join(OUT_DIR, f'motion_arrows_{name}.npy'),
                np.stack([s, e, d], axis=0))

    with open(os.path.join(OUT_DIR, 'fig3_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n===== 汇总 =====')
    for n, m in summary.items():
        print(f'  {n:<12} N={m["n"]:>5d}  位移中位数={m["disp_median_mm"]:.3f} mm')

    # ── 绘图 ──
    plot_figure(arrows, summary)

    print(f'\n输出目录: {OUT_DIR}')
    for fn in sorted(os.listdir(OUT_DIR)):
        print(' ', fn)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='跳过 VO, 从已有 npy 直接绘图')
    args = parser.parse_args()
    main(plot_only=args.plot_only)

