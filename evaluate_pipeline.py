"""独立评估脚本：加载管线输出，运行 GT 对比评估。

用法:
    python evaluate_pipeline.py --seq c1_transverse1_t1_v2 --mode baseline
    python evaluate_pipeline.py --seq c1_transverse1_t1_v2 --mode both

依赖:
    - {seq_dir}/{mode}_abs_poses.npy      → VO 绝对位姿 (N×4×4)
    - {seq_dir}/{mode}_depth_maps.npz     → 预测深度图 {frame_idx: array}
    - {seq_dir}/{mode}_chain_data.npz     → 匹配点对 + K + 参数
    - {out_dir}/{seq}_{mode}_tracks.json  → track 分类结果

输出:
    - {out_dir}/{seq}_{mode}_motion_vs_gt_3d.png      → VO vs GT 轨迹对比
    - {out_dir}/{seq}_{mode}_motion_vs_gt_summary.png → 位移汇总图
    - 控制台报告 (混淆矩阵, Recall/Precision/F1)
"""

import argparse
import os
import sys
import json
import numpy as np

# 导入评估函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_v6_dyendovo import (
    evaluate_motion_vs_gt,
    visualize_motion_vs_gt,
    compute_per_frame_vo_to_gt,
    evaluate_motion_detection_vs_gt,
    evaluate_per_point_vs_gt,
    load_gt_poses,
    align_trajectory_umeyama,
    compute_ate,
)

DATA_ROOT = 'F:/dataset'
DEFAULT_OUT = './motion_output'


def load_depth_maps(npz_path):
    """加载深度图 npz, 返回 {frame_idx: array}."""
    data = np.load(npz_path, allow_pickle=True)
    depth_maps = {}
    for key in data.files:
        depth_maps[int(key)] = data[key]
    return depth_maps


def load_chain_data(npz_path):
    """加载 chain_data, 返回 all_k0, all_k1, all_weights, K, params."""
    data = np.load(npz_path, allow_pickle=True)
    n_pairs = int(data['n_pairs'])
    all_k0 = [data[f'k0_{i:04d}'] for i in range(n_pairs)]
    all_k1 = [data[f'k1_{i:04d}'] for i in range(n_pairs)]
    all_weights = [data[f'w_{i:04d}'] for i in range(n_pairs)]
    K = data['K']
    params = {
        'k_factor': float(data.get('k_factor', 0.5)),
        'chain_dist_thresh': float(data.get('chain_dist_thresh', 3.0)),
        'global_depth_scale': float(data.get('global_depth_scale', 1.0)),
        'depth_source': str(data.get('depth_source', 'pred')),
    }
    return all_k0, all_k1, all_weights, K, params


def load_tracks(json_path):
    """加载 track 分类结果, 返回 (tracks, moving_ids)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    raw_tracks = data['tracks']
    tracks = []
    for t in raw_tracks:
        frames = [(int(fi), float(u), float(v)) for fi, u, v in t['frames']]
        tracks.append({'id': t['id'], 'frames': frames})
    moving_ids = set(data['moving_ids'])
    return tracks, moving_ids


def evaluate_mode(seq_dir, out_dir, seq_name, mode):
    """对单个模式运行完整 GT 评估 (Steps F-I).

    Returns:
        dict with evaluation summary
    """
    print(f"\n{'='*60}")
    print(f"  [评估] {seq_name} ({mode})")
    print(f"{'='*60}")

    # ── 加载数据 ──
    abs_poses_path = os.path.join(seq_dir, f'{mode}_abs_poses.npy')
    depth_path = os.path.join(seq_dir, f'{mode}_depth_maps.npz')
    chain_path = os.path.join(seq_dir, f'{mode}_chain_data.npz')
    tracks_path = os.path.join(out_dir, f'{seq_name}_{mode}_tracks.json')

    for label, p in [('绝对位姿', abs_poses_path), ('深度图', depth_path),
                      ('匹配数据', chain_path), ('track分类', tracks_path)]:
        if not os.path.exists(p):
            print(f"  ⚠ {label} 不存在: {p}")
            return None
        print(f"  加载 {label}: {p}")

    abs_poses = np.load(abs_poses_path)  # (N, 4, 4)
    depth_maps = load_depth_maps(depth_path)
    all_k0, all_k1, all_weights, K, params = load_chain_data(chain_path)
    tracks, moving_ids = load_tracks(tracks_path)

    # ── 深度尺度恢复 ──
    # chain_data.npz 里的深度是 ×global_scale 的轨迹尺度深度;
    # GT 位移评估需要原始深度尺度, 故在此 ÷global_scale 恢复。
    # depth_source=='gt' 时深度未乘 global_scale, 无需恢复。
    global_scale = params.get('global_depth_scale', 1.0)
    depth_src = params.get('depth_source', 'pred')
    if depth_src != 'gt' and global_scale != 1.0:
        for k in depth_maps:
            depth_maps[k] = depth_maps[k] / global_scale
        print(f"  深度尺度恢复: ÷global_scale={global_scale:.4f} (GT 评估用原始深度尺度)")

    # 分离动静 track
    static_tracks = [(t, 0) for t in tracks if t['id'] not in moving_ids]
    moving_tracks = [(t, 0) for t in tracks if t['id'] in moving_ids]

    print(f"  帧数: {len(abs_poses)}, 深度: {len(depth_maps)}, 匹配对: {len(all_k0)}")
    print(f"  Tracks: {len(tracks)} (运动: {len(moving_tracks)}, 静止: {len(static_tracks)})")

    # ── 加载 GT ──
    gt_poses = load_gt_poses(seq_dir)
    if gt_poses is None:
        print("  ⚠ GT 位姿加载失败")
        return None

    n_compare = min(len(gt_poses), len(abs_poses))
    gt_align = gt_poses[:n_compare]
    abs_align = list(abs_poses[:n_compare])
    depth_align = {k: v for k, v in depth_maps.items() if k < n_compare}

    # ── 逐帧 VO→GT 对齐 (替代旧的 Umeyama, 消除八字形发散) ──
    vo_to_gt = compute_per_frame_vo_to_gt(gt_align, abs_align)

    # ── 构建 chain_data (Step I 需要) ──
    chain_data = {
        'all_k0': all_k0,
        'all_k1': all_k1,
        'all_weights': all_weights,
        'depth_maps': depth_align,
        'abs_poses': abs_align,
    }

    # ═══════════════════════════════════════════
    # Step F: GT 位移对比评估
    # ═══════════════════════════════════════════
    print("\n  [F] GT 位姿对比评估...")
    gt_metrics = evaluate_motion_vs_gt(
        tracks, depth_align, abs_align, gt_align, K,
        moving_tracks=moving_tracks)

    # ═══════════════════════════════════════════
    # Step G: VO vs GT 运动轨迹可视化
    # ═══════════════════════════════════════════
    print("\n  [G] VO vs GT 轨迹对比可视化...")
    visualize_motion_vs_gt(
        moving_tracks, static_tracks, depth_align, abs_align,
        gt_align, K,
        out_dir, f'{seq_name}_{mode}', top_k=8, n_scatter=2000)

    # ═══════════════════════════════════════════
    # Step H: GT mask 运动检测评估 (track 级别)
    # ═══════════════════════════════════════════
    print("\n  [H] GT mask 运动检测评估 (track 级别)...")
    mask_metrics = evaluate_motion_detection_vs_gt(
        tracks, static_tracks, moving_tracks,
        depth_align, abs_align, K,
        seq_dir, vo_to_gt_align=vo_to_gt)

    # ═══════════════════════════════════════════
    # Step I: 逐帧逐点 GT mask 运动检测评估
    # ═══════════════════════════════════════════
    print("\n  [I] 逐帧逐点 GT mask 运动检测评估 (k=0, threshold=median)...")
    per_point_metrics = evaluate_per_point_vs_gt(
        chain_data, K, seq_dir, k_factor=0.0,
        vo_to_gt_align=vo_to_gt)

    return {
        'gt_evaluation': gt_metrics,
        'mask_evaluation': mask_metrics,
        'per_point_evaluation': per_point_metrics,
    }


def print_summary(seq_name, results):
    """打印评估汇总报告."""
    print("\n" + "═" * 65)
    print(f"  【 GT 评估汇总: {seq_name} 】")
    print("═" * 65)

    for mode, eval_data in results.items():
        if eval_data is None:
            continue

        gt_eval = eval_data.get('gt_evaluation', {})
        mask_eval = eval_data.get('mask_evaluation', {})
        per_pt = eval_data.get('per_point_evaluation', {})

        print(f"\n  ┌── {mode} ──────────────────────────────────────────────────┐")

        # GT 位姿评估
        static_err = gt_eval.get('static_pose_error_mm', {})
        motion_err = gt_eval.get('moving_displacement_error_mm', {})
        corr = gt_eval.get('correlation')
        print(f"  │  GT 静止点 VO 误差 RMSE:  {static_err.get('rmse', 0):>6.2f} mm                          │")
        print(f"  │  GT 运动点 VO 位移 RMSE:   {motion_err.get('rmse', 0):>6.2f} mm                          │")
        print(f"  │  GT vs VO 位移相关性 r:    {corr:>6.3f}                                   │" if corr else "")

        # Track 级 mask 评估
        if mask_eval:
            print(f"  │─────────────────────────────────────────────────────────│")
            print(f"  │  [Track 级 GT mask]                                              │")
            print(f"  │    Recall:    {mask_eval['recall']:>6.1%}                                         │")
            print(f"  │    Precision: {mask_eval['precision']:>6.1%}                                         │")
            print(f"  │    F1:        {mask_eval['f1']:>6.3f}                                         │")

        # 逐点评估
        if per_pt:
            print(f"  │─────────────────────────────────────────────────────────│")
            print(f"  │  [逐帧逐点 GT mask]                                              │")
            print(f"  │    Recall:    {per_pt['recall']:>6.1%}                                         │")
            print(f"  │    Precision: {per_pt['precision']:>6.1%}                                         │")
            print(f"  │    F1:        {per_pt['f1']:>6.3f}                                         │")
            print(f"  │    评估点:    {per_pt['n_total']:>6d}                                         │")

        print(f"  └─────────────────────────────────────────────────────────┘")


def main():
    parser = argparse.ArgumentParser(description='EndoSLAM 管线独立 GT 评估')
    parser.add_argument('--seq', type=str, default='c1_transverse1_t1_v2',
                        help='序列名')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT,
                        help=f'数据集根目录 (默认: {DATA_ROOT})')
    parser.add_argument('--mode', type=str, default='baseline',
                        choices=['baseline', 'motionnet', 'both'],
                        help='评估模式: baseline/motionnet/both')
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT,
                        help='运动轨迹输出目录 (默认: ./motion_output)')
    args = parser.parse_args()

    seq_dir = os.path.join(args.data_root, args.seq)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isdir(seq_dir):
        print(f"ERROR: 序列目录不存在: {seq_dir}")
        sys.exit(1)

    modes = ['baseline', 'motionnet'] if args.mode == 'both' else [args.mode]
    results = {}
    for m in modes:
        results[m] = evaluate_mode(seq_dir, out_dir, args.seq, m)

    print_summary(args.seq, results)

    # ── 输出文件清单 ──
    print(f"\n  评估输出目录: {out_dir}")
    for m in modes:
        print(f"    {args.seq}_{m}_motion_vs_gt_3d.png")
        print(f"    {args.seq}_{m}_motion_vs_gt_summary.png")


if __name__ == '__main__':
    main()
