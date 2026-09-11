#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成论文图3(改版): 运动幅度最大的 5 个点的 3D 轨迹对比 (GT + 四方法)。

与 fig3 箭头版不同, 本脚本只取每个方法"净位移最大的 5 个运动点",
绘制它们完整的三维运动轨迹(绝对世界位姿, 不归零),
5 种颜色区分 P1~P5, 5 个子图分别对应 GT / Monodepth2 / ManyDepth / Lite-Mono / Ours。

序列: c1_transverse1_t1_v2 (与表3一致)

口径: GT 与四方法统一取"净位移" = ||p[末帧] - p[首帧]|| (mm), 复用 test_v6_dyendovo 内置链

GT 组织运动真值:
    - vertex_static/frame_0000.npy  顶点世界坐标 (mm)
    - motion_gt/frame_{i:04d}.npy   顶点瞬时位移 (米, 相对静止参考)
    累计路径 = sum ||motion_gt[t+1] - motion_gt[t]|| * 1000  (mm)
    轨迹 = vertex + motion_gt[t]*1000  (mm)

管线(四方法)组织运动轨迹:
    1. run_vo_sequence → chain_data (同时保存 ÷global_scale 后物理尺度深度图)
    2. run_motion_pipeline → 输出运动 track 的 (u,v,frame) + 绝对世界位姿 JSON
    3. 直接读 run_motion_pipeline 内置输出的运动 track 绝对位姿 (VO 位姿反投影)
    4. 按内置净位移 total_displacement_mm 排序取 top5 轨迹

输出目录: zhong/motion_vis_fig3_top5/
    - method_<name>/              每方法 run_motion_pipeline 输出 + depth_maps_orig.npz
    - top5_trajs.pkl              每方法 top5 轨迹 + 位移幅度
    - fig3_top5_summary.json
    - fig3_top5_trajectories.png  5 子图 3D 轨迹对比
    - fig3_top5_panel_<name>.png  逐方法独立子图
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

import test_v6_dyendovo as tv6
from test_v6_multi_model_enhanced import load_model, SEQ_NAME, SEQ_DIR, DATA_ROOT
from eval_pipeline_motion_vs_gt import backproject_to_world, load_gt_poses_orderF

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cuda')

MODELS = [
    {'name': 'Monodepth2', 'path': r'C:\Users\Administrator\tmp\c3vd_md2_multi\models\weights_19', 'type': 'md2'},
    {'name': 'ManyDepth',  'path': r'C:\Users\Administrator\tmp\c3vd_manydepth_multi\models\weights_19', 'type': 'manydepth'},
    {'name': 'Lite-Mono',  'path': r'C:\Users\Administrator\tmp\c3vd_litemono_multi\models\weights_19', 'type': 'litemono'},
    {'name': 'Ours',       'path': r'e:\data1\monodepth2\models\depth', 'type': 'md2'},
]

OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'motion_vis_fig3_top5')
os.makedirs(OUT_DIR, exist_ok=True)

N_TOP = 5                 # 每个方法取位移最大的点数
GT_SCALE = 100.0 / 65535.0  # GT 深度 uint16 → mm

PANEL_COLORS = {
    'GT': '#333333',
    'Monodepth2': '#1f77b4',
    'ManyDepth': '#ff7f0e',
    'Lite-Mono': '#2ca02c',
    'Ours': '#d62728',
}

# P1~P5 轨迹颜色 (按位移从大到小)
TOP5_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']


# ----------------------------------------------------------------------------
# GT 轨迹
# ----------------------------------------------------------------------------
def extract_gt_top5_trajs(n_top=N_TOP):
    """GT: 净位移(末帧-首帧)最大的 n_top 个运动顶点, 返回完整 3D 轨迹 + 净位移(mm)。"""
    gen_dir = os.path.join(SEQ_DIR, 'generated')
    vertex = np.load(os.path.join(gen_dir, 'vertex_static', 'frame_0000.npy'))   # (N,3) mm
    motion_dir = os.path.join(gen_dir, 'motion_gt')
    motion_files = sorted(os.listdir(motion_dir))

    # 净位移 = ||motion[末帧] - motion[首帧]|| (米 → 毫米), 与内置链 total_displacement_mm 口径一致
    motion_0 = np.load(os.path.join(motion_dir, motion_files[0]))
    motion_last = np.load(os.path.join(motion_dir, motion_files[-1]))
    disp = np.linalg.norm(motion_last - motion_0, axis=1) * 1000.0   # (N,) mm

    order = np.argsort(-disp)[:n_top]

    # top5 顶点的完整绝对位姿轨迹 (T,3) mm
    trajs = []
    for i in order:
        pts = np.stack([
            np.load(os.path.join(motion_dir, f))[i] for f in motion_files
        ], axis=0)                                           # (T,3) 米
        trajs.append(vertex[i] + pts * 1000.0)               # (T,3) mm
    disps = [float(disp[i]) for i in order]
    disp_str = ', '.join('%.2f' % d for d in disps)
    print(f'  [GT] 顶点数={len(disp)} top{n_top}净位移={disp_str} mm')
    return trajs, disps


# ----------------------------------------------------------------------------
# 四方法轨迹
# ----------------------------------------------------------------------------
def save_depth_maps_for_method(chain_data, name):
    """保存 ÷global_scale 后的物理尺度深度图到 method_out, 供 GT 位姿反投影复用。"""
    method_out = os.path.join(OUT_DIR, f'method_{name}')
    pp = chain_data.get('pipeline_params', {})
    global_scale = float(pp.get('global_depth_scale', 1.0))
    depth_src = str(chain_data.get('depth_source', 'pred'))
    depth_maps = chain_data['depth_maps']          # key: int, 值 ×global_scale
    if depth_src != 'gt' and global_scale != 1.0:
        depth_maps = {k: v / global_scale for k, v in depth_maps.items()}
    depth_dict = {f'{int(k):04d}': depth_maps[k] for k in sorted(depth_maps.keys())}
    np.savez_compressed(os.path.join(method_out, 'depth_maps_orig.npz'), **depth_dict)
    with open(os.path.join(method_out, 'global_scale.json'), 'w') as f:
        json.dump({'global_depth_scale': global_scale, 'depth_source': depth_src}, f)
    print(f'  物理尺度深度图已保存 (÷gs={global_scale:.4f}): '
          f'{method_out}/depth_maps_orig.npz ({len(depth_dict)} 帧)')


def median_scale_depth(depth_maps, name):
    """三基线: 逐帧 median scaling 对齐 GT 深度(物理毫米尺度); Ours 深度已物理尺度, 跳过。"""
    if name == 'Ours':
        return depth_maps
    gt_depth_dir = os.path.join(SEQ_DIR, 'depth')
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
        print(f'  [{name}] 深度 median scaling 对齐 GT: mean_scale={np.mean(scales):.4f} (n={len(scales)})')
    return depth_maps


def load_top5_from_builtin(name, n_top=N_TOP):
    """直接读 run_motion_pipeline 内置输出的 top5 绝对位姿 (VO 位姿反投影, 净位移口径)。"""
    method_out = os.path.join(OUT_DIR, f'method_{name}')
    json_path = os.path.join(method_out, f'{SEQ_NAME}_{name}_baseline_motion_trajectories.json')
    with open(json_path) as f:
        trajs_json = json.load(f)
    # 按内置净位移 total_displacement_mm 排序取 top5
    trajs_json = sorted(trajs_json, key=lambda t: -float(t['total_displacement_mm']))
    top = trajs_json[:n_top]
    trajs, disps = [], []
    for t in top:
        pts = []
        for fr in t['frames']:
            if fr.get('x_mm') is None or fr.get('y_mm') is None or fr.get('z_mm') is None:
                continue
            pts.append([fr['x_mm'], fr['y_mm'], fr['z_mm']])
        if len(pts) < 2:
            continue
        trajs.append(np.array(pts, dtype=np.float64))
        disps.append(float(t['total_displacement_mm']))
    disp_str = ', '.join('%.2f' % d for d in disps)
    print(f'  [{name}] 内置VO位姿 top{len(trajs)} 净位移={disp_str} mm')
    return trajs, disps


def run_vo_and_analyze(name, cfg):
    """跑 VO, 并复用 test_v6_dyendovo 内置分析输出运动最明显 5 点的绝对位姿。"""
    print(f'\n===== [{name}] 跑 VO =====')
    encoder, depth_decoder, motion_encoder = load_model(cfg, DEVICE)
    params = tv6.estimate_pipeline_params(
        SEQ_DIR, encoder, depth_decoder, motion_encoder, DEVICE,
        depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')
    vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
        SEQ_NAME, encoder, depth_decoder, DEVICE,
        motion_net=None, mode='baseline', max_frames=None,
        motion_encoder=motion_encoder, depth_source='pred',
        loftr_source='cache', data_root=DATA_ROOT,
        pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale')
    del encoder, depth_decoder, motion_encoder
    torch.cuda.empty_cache()

    # ── 保存 ÷global_scale 后的物理尺度深度图 (供 GT 位姿反投影复用) ──
    save_depth_maps_for_method(chain_data, name)

    # ── 复用 test_v6_dyendovo 内置分析: 输出运动点绝对位姿 ──
    method_out = os.path.join(OUT_DIR, f'method_{name}')
    summary = tv6.run_motion_pipeline(chain_data, f'{SEQ_NAME}_{name}', method_out, 'baseline')
    mstats = summary['moving_disp_stats']
    print(f'  运动位移: mean={mstats["mean_mm"]:.1f}mm max={mstats["max_mm"]:.1f}mm')
    return method_out


# ----------------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------------
def plot_top5_trajs(all_trajs, all_disps):
    names = ['GT', 'Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']
    names = [n for n in names if n in all_trajs]

    # 主图: 每子图独立等比例范围 (绝对世界坐标)
    fig = plt.figure(figsize=(4.4 * len(names), 5.8))
    for idx, name in enumerate(names):
        trajs = [np.asarray(t) for t in all_trajs[name]]
        ax = fig.add_subplot(1, len(names), idx + 1, projection='3d')
        for k, traj in enumerate(trajs):
            col = TOP5_COLORS[k]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=col, lw=2.2, alpha=0.95)
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=col, s=50,
                       marker='o', depthshade=False, edgecolors='k', linewidths=0.5)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=col, s=120,
                       marker='*', depthshade=False, edgecolors='k', linewidths=0.5)
        pts = np.vstack(trajs)
        c = pts.mean(axis=0)
        r = float((pts.max(axis=0) - pts.min(axis=0)).max()) / 2
        r = max(r, 0.5)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        disp_str = '  '.join(f'P{k + 1}={all_disps[name][k]:.1f}' for k in range(len(trajs)))
        ax.set_title(f'{name}\n{disp_str}', fontsize=9, color=PANEL_COLORS[name])
        ax.set_xlabel('X (mm)', fontsize=8)
        ax.set_ylabel('Y (mm)', fontsize=8)
        ax.set_zlabel('Z (mm)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=20, azim=-60)

    handles = [plt.Line2D([0], [0], color=TOP5_COLORS[k], lw=2.5, label=f'P{k + 1}')
               for k in range(N_TOP)]
    fig.legend(handles=handles, loc='lower center', ncol=N_TOP, fontsize=10, frameon=False)
    fig.suptitle(f'组织运动净位移最大的 5 个点的 3D 轨迹 (VO绝对位姿, ○=起点 * =终点) — {SEQ_NAME}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    out_path = os.path.join(OUT_DIR, 'fig3_top5_trajectories.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')

    # 逐方法独立子图
    for name in names:
        trajs = [np.asarray(t) for t in all_trajs[name]]
        fig2 = plt.figure(figsize=(6.8, 6.8))
        ax2 = fig2.add_subplot(111, projection='3d')
        for k, traj in enumerate(trajs):
            col = TOP5_COLORS[k]
            ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=col, lw=2.4, alpha=0.95)
            ax2.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=col, s=70,
                        marker='o', depthshade=False, edgecolors='k', linewidths=0.5)
            ax2.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=col, s=150,
                        marker='*', depthshade=False, edgecolors='k', linewidths=0.5)
        pts = np.vstack(trajs)
        c = pts.mean(axis=0)
        r = max(float((pts.max(axis=0) - pts.min(axis=0)).max()) / 2, 0.5)
        ax2.set_xlim(c[0] - r, c[0] + r)
        ax2.set_ylim(c[1] - r, c[1] + r)
        ax2.set_zlim(c[2] - r, c[2] + r)
        disp_str = '  '.join(f'P{k + 1}={all_disps[name][k]:.1f}' for k in range(len(trajs)))
        ax2.set_title(f'{name} — top{N_TOP} 运动轨迹 (mm)\n{disp_str}',
                      fontsize=11, color=PANEL_COLORS[name])
        ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
        ax2.view_init(elev=20, azim=-60)
        fig2.tight_layout()
        p = os.path.join(OUT_DIR, f'fig3_top5_panel_{name}.png')
        fig2.savefig(p, dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved: {p}')


# ----------------------------------------------------------------------------
# 保存 / 加载
# ----------------------------------------------------------------------------
def save_top5(all_trajs, all_disps):
    data = {}
    for name in ['GT', 'Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']:
        if name in all_trajs:
            data[name] = {'trajs': all_trajs[name], 'disps': all_disps[name]}
    with open(os.path.join(OUT_DIR, 'top5_trajs.pkl'), 'wb') as f:
        pickle.dump(data, f)


def load_top5():
    with open(os.path.join(OUT_DIR, 'top5_trajs.pkl'), 'rb') as f:
        data = pickle.load(f)
    trajs = {k: v['trajs'] for k, v in data.items()}
    disps = {k: v['disps'] for k, v in data.items()}
    return trajs, disps


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main(plot_only=False, no_vo=False, vo_only=False, methods=None):
    names = ['GT', 'Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']
    vo_names = ['Monodepth2', 'ManyDepth', 'Lite-Mono', 'Ours']
    if methods:
        vo_names = [m for m in vo_names if m in methods]
    if plot_only:
        print(f'[plot-only] 从 pkl 加载并绘图, 序列: {SEQ_NAME}')
        all_trajs, all_disps = load_top5()
        for n in names:
            if n in all_disps:
                ds = ', '.join('%.2f' % d for d in all_disps[n])
                print(f'  {n:<12} top{N_TOP}累计路径={ds} mm')
        plot_top5_trajs(all_trajs, all_disps)
        print(f'\n输出目录: {OUT_DIR}')
        return

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print(f'设备: {DEVICE}, 序列: {SEQ_NAME}, 方法: {vo_names}')

    all_trajs, all_disps = {}, {}

    # GT (不依赖模型)
    print('\n===== [GT] 提取组织运动真值 top5 轨迹 =====')
    trajs, disps = extract_gt_top5_trajs()
    all_trajs['GT'], all_disps['GT'] = trajs, disps

    if vo_only:
        # 只跑 VO: 保存物理尺度深度图 + 运动 track JSON, 不绘图
        for cfg in MODELS:
            if cfg['name'] in vo_names:
                run_vo_and_analyze(cfg['name'], cfg)
        print('\nVO 完成 (仅保存深度图 + JSON), 未绘图')
        return

    # 四方法 (跑 VO → 保存深度图 → GT 位姿反投影读 top5)
    for cfg in MODELS:
        name = cfg['name']
        if name not in vo_names:
            continue
        if not no_vo:
            run_vo_and_analyze(name, cfg)
        trajs, disps = load_top5_from_builtin(name)
        all_trajs[name], all_disps[name] = trajs, disps

    save_top5(all_trajs, all_disps)

    summary = {n: {'top5_disp_mm': [round(d, 3) for d in all_disps[n]],
                   'max_mm': round(max(all_disps[n]), 3)} for n in all_trajs}
    with open(os.path.join(OUT_DIR, 'fig3_top5_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n===== 汇总 (内置VO绝对位姿, 净位移 mm) =====')
    for n in names:
        if n not in all_disps:
            continue
        ds = ', '.join('%.2f' % d for d in all_disps[n])
        print(f'  {n:<12} top{N_TOP}累计路径={ds} mm')

    plot_top5_trajs(all_trajs, all_disps)

    print(f'\n输出目录: {OUT_DIR}')
    for fn in sorted(os.listdir(OUT_DIR)):
        print(' ', fn)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='跳过 VO, 从已有 pkl 直接绘图')
    parser.add_argument('--no-vo', action='store_true',
                        help='跳过 VO, 从已有 JSON + depth_maps_orig.npz 读 top5 并绘图')
    parser.add_argument('--vo-only', action='store_true',
                        help='只跑 VO 并保存深度图 + JSON, 不绘图')
    parser.add_argument('--methods', default=None,
                        help='逗号分隔方法名 (默认全部四方法)')
    args = parser.parse_args()
    methods = args.methods.split(',') if args.methods else None
    main(plot_only=args.plot_only, no_vo=args.no_vo,
         vo_only=args.vo_only, methods=methods)
