"""
run_pipeline.py - 零配置 VO 管线

输入: 只有 RGB 帧的文件夹
自动:
  - 从第一帧尺寸估算 K（fx=fy=max(W,H), cx=W/2, cy=H/2）
  - 深度模型预测
  - 在线 LoFTR 匹配
  - 自适应阈值
  - 帧间深度一致性微调
  - 不做全局尺度校准 (无 GT)

输出:
  - depth/              → 逐帧深度图 (.npy)
  - depth_viz/           → 深度可视化图 (.png, turbo colormap)
  - abs_poses.npy        → 绝对位姿 (N×4×4)
  - abs_poses.txt        → 绝对位姿文本 (每帧16数, 逗号分隔)
  - trajectory_3d.png    → 3D 相机轨迹图
  - chain_data.npz       → 匹配链数据
  - tracks.json          → track 分类结果
  - report.json          → 汇总报告
  - motion_3d.png        → 运动轨迹可视化

用法:
  python run_pipeline.py --input F:/dataset/c1_transverse1_t1_v2/generated/rgb_warped --output ./run_output/v2
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(__file__))
from v6_pipeline.utils import (
    ModelManager, predict_depth, init_loftr_matcher, _match_loftr_pair,
)

# ── 默认配置 ──
DEFAULT_MODEL_PATH = r'.\models\depth'
DEPTH_MIN, DEPTH_MAX = 1.0, 500.0
LOFTR_MAX_DIM = 840
MAX_TRACKS = 10000
CHAIN_DIST_THRESH = 3.0
DISP_K_FACTOR = 0.5
SCALE_CLIP_LO = 0.2
SCALE_CLIP_HI = 5.0
MAX_ROTATION_DEG = 30.0
MAX_TRANSLATION_MM = 100.0


def auto_estimate_K(img_path, fx=None, fy=None, cx=None, cy=None):
    """从第一帧尺寸自动估算内参矩阵 K, 支持手动指定."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {img_path}")
    H, W = img.shape[:2]
    if fx is not None and fy is not None and cx is not None and cy is not None:
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        print(f"  使用指定 K: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f} "
              f"(图像 {W}×{H})")
    else:
        focal = max(W, H)
        K = np.array([
            [focal, 0, W / 2.0],
            [0, focal, H / 2.0],
            [0, 0, 1]
        ], dtype=np.float64)
        print(f"  自动估算 K: fx={focal:.1f}, fy={focal:.1f}, cx={W/2:.1f}, cy={H/2:.1f} "
              f"(图像 {W}×{H})")
    return K, W, H


def rotation_angle_deg(R):
    """旋转矩阵 → 角度(度)."""
    return np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)) * 180 / np.pi


def _filter_by_3d_displacement(pts3d, k0, k1, depth_curr, depth_next, K, k_factor=DISP_K_FACTOR):
    """自适应 3D 位移过滤."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    N = len(pts3d)
    h_d, w_d = depth_curr.shape
    h_dn, w_dn = depth_next.shape

    displacements = np.full(N, np.nan, dtype=np.float64)
    for j in range(N):
        u1, v1 = k1[j]
        ui1 = int(np.clip(u1, 0, w_dn - 1))
        vi1 = int(np.clip(v1, 0, h_dn - 1))
        Z1 = float(depth_next[vi1, ui1])
        if Z1 <= 0.5 or Z1 > DEPTH_MAX or not np.isfinite(Z1):
            continue
        X1 = (u1 - cx) * Z1 / fx
        Y1 = (v1 - cy) * Z1 / fy
        p1 = np.array([X1, Y1, Z1])
        displacements[j] = float(np.linalg.norm(p1 - pts3d[j]))

    valid = ~np.isnan(displacements)
    if valid.sum() < 4:
        return np.ones(N, dtype=bool), {'n_kept': N, 'n_total': N, 'threshold_mm': 0}

    valid_disp = displacements[valid]
    med = float(np.median(valid_disp))
    std = float(np.std(valid_disp))
    threshold = med + k_factor * std
    threshold = max(threshold, 1.0)

    keep = displacements <= threshold
    if keep.sum() < 4:
        sorted_idx = np.argsort(displacements)
        n_keep = min(N, max(4, N // 2))
        keep = np.zeros(N, dtype=bool)
        keep[sorted_idx[:n_keep]] = True

    return keep, {'n_kept': int(keep.sum()), 'n_total': N,
                   'threshold_mm': threshold, 'median_mm': med, 'std_mm': std}


def chain_tracks(all_k0, all_k1, dist_thresh=CHAIN_DIST_THRESH):
    """串联相邻帧匹配点, 形成跨帧 track."""
    n_pairs = len(all_k0)
    tracks = []
    active = {}
    next_id = 0

    for pair_idx in range(n_pairs):
        k0 = all_k0[pair_idx]
        k1 = all_k1[pair_idx]

        if pair_idx == 0:
            for j in range(len(k0)):
                tid = next_id
                next_id += 1
                tracks.append({
                    'id': tid,
                    'frames': [(0, float(k0[j, 0]), float(k0[j, 1])),
                               (1, float(k1[j, 0]), float(k1[j, 1]))],
                    'pair_indices': [(0, j)],
                })
                active[tid] = (1, float(k1[j, 0]), float(k1[j, 1]))
            continue

        if not active:
            for j in range(len(k0)):
                tid = next_id
                next_id += 1
                tracks.append({
                    'id': tid,
                    'frames': [(pair_idx, float(k0[j, 0]), float(k0[j, 1])),
                               (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))],
                    'pair_indices': [(pair_idx, j)],
                })
                active[tid] = (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))
            continue

        active_ids = list(active.keys())
        active_pts = np.array([[active[tid][1], active[tid][2]] for tid in active_ids])

        matched_tids = set()
        new_active = {}

        if len(k0) > 0 and len(active_pts) > 0:
            for j in range(len(k0)):
                pt = k0[j]
                dists = np.sqrt(np.sum((active_pts - pt) ** 2, axis=1))
                min_idx = np.argmin(dists)
                if dists[min_idx] <= dist_thresh:
                    tid = active_ids[min_idx]
                    if tid not in matched_tids:
                        tracks[tid]['frames'].append(
                            (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1])))
                        tracks[tid]['pair_indices'].append((pair_idx, j))
                        new_active[tid] = (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))
                        matched_tids.add(tid)

        for j in range(len(k0)):
            pt = k0[j]
            dists = np.sqrt(np.sum((active_pts - pt) ** 2, axis=1))
            if len(active_pts) == 0 or np.min(dists) > dist_thresh:
                tid = next_id
                next_id += 1
                tracks.append({
                    'id': tid,
                    'frames': [(pair_idx, float(k0[j, 0]), float(k0[j, 1])),
                               (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))],
                    'pair_indices': [(pair_idx, j)],
                })
                new_active[tid] = (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))

        active = new_active

    return tracks


def backproject_to_world(u, v, depth_map, T_world_cam, K):
    """将像素点反投影到世界坐标系."""
    h, w = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ui = int(np.clip(u, 0, w - 1))
    vi = int(np.clip(v, 0, h - 1))
    Z = float(depth_map[vi, ui])
    if Z <= 0.5 or Z > DEPTH_MAX or not np.isfinite(Z):
        return None

    X_cam = (u - cx) * Z / fx
    Y_cam = (v - cy) * Z / fy
    P_cam = np.array([X_cam, Y_cam, Z, 1.0])
    P_world = T_world_cam @ P_cam
    return P_world[:3]


def compute_track_displacement(track, depth_maps, abs_poses, K):
    """计算 track 在世界坐标系的总位移."""
    p_worlds = []
    for fi, u, v in track['frames']:
        if fi in depth_maps:
            pw = backproject_to_world(u, v, depth_maps[fi], abs_poses[fi], K)
            p_worlds.append(pw)
        else:
            p_worlds.append(None)

    valid = [p for p in p_worlds if p is not None]
    if len(valid) < 2:
        return p_worlds, 0.0

    total_disp = 0.0
    for i in range(len(valid) - 1):
        total_disp += np.linalg.norm(valid[i + 1] - valid[i])

    return p_worlds, float(total_disp)


def classify_tracks(tracks, depth_maps, abs_poses, K, k_factor=DISP_K_FACTOR):
    """基于自适应 3D 位移阈值分类 track 为静止/运动."""
    import random

    sample_max = 5000
    if len(tracks) > sample_max:
        random.seed(42)
        sample = random.sample(tracks, sample_max)
    else:
        sample = tracks

    track_medians = []
    for track in sample:
        p_worlds, _ = compute_track_displacement(track, depth_maps, abs_poses, K)
        valid_p = [p for p in p_worlds if p is not None]
        if len(valid_p) < 2:
            track_medians.append(0.0)
            continue
        steps = []
        for i in range(len(valid_p) - 1):
            steps.append(np.linalg.norm(valid_p[i + 1] - valid_p[i]))
        track_medians.append(float(np.median(steps)))

    track_medians = np.array(track_medians)
    med = float(np.median(track_medians))
    std = float(np.std(track_medians))
    threshold = med + k_factor * std

    print(f"  自适应阈值: median={med:.2f}mm, std={std:.2f}mm, "
          f"threshold={threshold:.2f}mm (k={k_factor})")

    static_tracks, moving_tracks = [], []
    for track in tracks:
        p_worlds, _ = compute_track_displacement(track, depth_maps, abs_poses, K)
        valid_p = [p for p in p_worlds if p is not None]
        if len(valid_p) < 2:
            static_tracks.append((track, 0.0))
            continue
        steps = []
        for i in range(len(valid_p) - 1):
            steps.append(np.linalg.norm(valid_p[i + 1] - valid_p[i]))
        med_disp = float(np.median(steps))
        if med_disp > threshold:
            moving_tracks.append((track, med_disp))
        else:
            static_tracks.append((track, med_disp))

    return static_tracks, moving_tracks


def compute_motion_summary(moving_tracks, static_tracks, depth_maps, abs_poses, K):
    """计算位移统计."""
    static_disps = []
    for track, _ in static_tracks:
        _, disp = compute_track_displacement(track, depth_maps, abs_poses, K)
        static_disps.append(disp)

    moving_disps = []
    moving_data = []
    for track, avg_w in moving_tracks:
        p_worlds, disp = compute_track_displacement(track, depth_maps, abs_poses, K)
        moving_disps.append(disp)
        n_valid = sum(1 for p in p_worlds if p is not None)
        moving_data.append({
            'track_id': track['id'],
            'n_frames': n_valid,
            'total_displacement_mm': disp,
        })

    return {
        'n_static_tracks': len(static_tracks),
        'n_moving_tracks': len(moving_tracks),
        'static_disp_stats': {
            'mean_mm': float(np.mean(static_disps)) if static_disps else 0,
            'median_mm': float(np.median(static_disps)) if static_disps else 0,
            'max_mm': float(np.max(static_disps)) if static_disps else 0,
        },
        'moving_disp_stats': {
            'mean_mm': float(np.mean(moving_disps)) if moving_disps else 0,
            'median_mm': float(np.median(moving_disps)) if moving_disps else 0,
            'max_mm': float(np.max(moving_disps)) if moving_disps else 0,
        },
        'top_moving_tracks': sorted(moving_data, key=lambda x: -x['total_displacement_mm'])[:20],
    }


def save_motion_trajectories(moving_tracks, depth_maps, abs_poses, K, out_dir, prefix):
    """保存运动轨迹 JSON."""
    trajectories = []
    for track, med_disp in moving_tracks:
        p_worlds, total_disp = compute_track_displacement(track, depth_maps, abs_poses, K)
        valid_worlds = [p.tolist() if p is not None else None for p in p_worlds]
        trajectories.append({
            'track_id': track['id'],
            'total_displacement_mm': total_disp,
            'median_step_mm': float(med_disp),
            'n_frames': len(valid_worlds),
            'world_positions': valid_worlds,
            'frame_indices': [fi for fi, _, _ in track['frames']],
        })

    save_path = os.path.join(out_dir, f'{prefix}_motion_trajectories.json')
    with open(save_path, 'w') as f:
        json.dump({'trajectories': trajectories, 'n_tracks': len(trajectories)}, f, indent=2)
    print(f"  运动轨迹已保存: {save_path}")


def visualize_motion(moving_tracks, static_tracks, depth_maps, abs_poses, K,
                     out_dir, prefix, top_k=8, n_scatter=2000):
    """可视化运动轨迹."""
    import random
    random.seed(42)

    fig = plt.figure(figsize=(16, 6))

    # ── 3D 轨迹对比 ──
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    # 散点: 静止点 (蓝色)
    static_sample = random.sample(static_tracks,
                                  min(n_scatter, len(static_tracks))) if static_tracks else []
    sx, sy, sz = [], [], []
    for track, _ in static_sample:
        for fi, u, v in track['frames']:
            if fi in depth_maps:
                pw = backproject_to_world(u, v, depth_maps[fi], abs_poses[fi], K)
                if pw is not None:
                    sx.append(pw[0]); sy.append(pw[1]); sz.append(pw[2])
    if sx:
        ax1.scatter(sx, sy, sz, c='blue', alpha=0.15, s=2, label=f'Static ({len(static_tracks)})')

    # 轨迹: 运动点 (红)
    sorted_moving = sorted(moving_tracks, key=lambda x: x[1], reverse=True)
    for i, (track, _) in enumerate(sorted_moving[:top_k]):
        pw_list, _ = compute_track_displacement(track, depth_maps, abs_poses, K)
        vals = [p for p in pw_list if p is not None]
        if len(vals) >= 2:
            xs = [p[0] for p in vals]; ys = [p[1] for p in vals]; zs = [p[2] for p in vals]
            ax1.plot(xs, ys, zs, linewidth=1.5, alpha=0.9, label=f'Track {track["id"]}')
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.set_title('Moving Tissue 3D Trajectories')
    ax1.legend(fontsize=7, loc='upper right')

    # ── 位移直方图 ──
    ax2 = fig.add_subplot(1, 2, 2)
    static_disps = []
    for track, _ in static_tracks:
        _, d = compute_track_displacement(track, depth_maps, abs_poses, K)
        static_disps.append(d)

    moving_disps = []
    for track, _ in moving_tracks:
        _, d = compute_track_displacement(track, depth_maps, abs_poses, K)
        moving_disps.append(d)

    bins = np.linspace(0, max(max(moving_disps) if moving_disps else 10, 20), 50)
    ax2.hist(static_disps, bins=bins, alpha=0.6, color='blue', label=f'Static (n={len(static_disps)})')
    ax2.hist(moving_disps, bins=bins, alpha=0.6, color='red', label=f'Moving (n={len(moving_disps)})')
    ax2.set_xlabel('Total Displacement (mm)'); ax2.set_ylabel('Count')
    ax2.set_title('Static vs Moving Displacement Distribution')
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}_motion_3d.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  可视化已保存: {save_path}")


def compute_ate_umeyama(est_poses, gt_pose_path, K):
    """用 Umeyama 对齐计算 ATE.

    Args:
        est_poses: 估计位姿 (N, 4, 4)
        gt_pose_path: GT 位姿文件路径 (每行 16 数, 4×4 行主序)
        K: 相机内参 (仅用于输出信息)

    Returns:
        dict or None: ATE 结果
    """
    try:
        gt_raw = np.loadtxt(gt_pose_path, delimiter=',')
    except Exception as e:
        print(f"  [ATE] 读取 GT pose 失败: {e}")
        return None

    if gt_raw.ndim == 1:
        gt_raw = gt_raw.reshape(1, -1)

    n_gt = gt_raw.shape[0]
    gt_poses = np.zeros((n_gt, 4, 4), dtype=np.float64)
    for i in range(n_gt):
        gt_poses[i] = gt_raw[i].reshape(4, 4)

    # GT 和估计位姿可能有不同帧数, 取最小帧数对齐
    n_est = est_poses.shape[0]
    n_align = min(n_gt, n_est)
    gt_aligned = gt_poses[:n_align]
    est_aligned = est_poses[:n_align]

    # 提取平移 (GT 格式: 平移在最后一行, 即 row 3, cols 0-2)
    gt_t = gt_aligned[:, 3, :3].copy()
    est_t = est_aligned[:, :3, 3].copy()

    # Umeyama 相似变换对齐 (只对平移做)
    gt_mean = gt_t.mean(axis=0)
    est_mean = est_t.mean(axis=0)
    gt_centered = gt_t - gt_mean
    est_centered = est_t - est_mean

    cov = est_centered.T @ gt_centered / n_align
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    est_var = np.sum(est_centered ** 2) / n_align
    scale = np.sum(S) / est_var if est_var > 1e-10 else 1.0

    # 对齐后的估计轨迹
    est_aligned_t = scale * (est_centered @ R.T) + gt_mean

    # ATE
    errors = np.linalg.norm(est_aligned_t - gt_t, axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))
    ate_median = float(np.median(errors))
    ate_std = float(np.std(errors))

    return {
        'ate_rmse_mm': round(ate_rmse, 2),
        'ate_mean_mm': round(ate_mean, 2),
        'ate_median_mm': round(ate_median, 2),
        'ate_std_mm': round(ate_std, 2),
        'umeyama_scale': round(float(scale), 4),
        'n_gt': n_gt,
        'n_est': n_est,
        'n_aligned': n_align,
    }


def visualize_camera_trajectory(abs_poses, out_dir, prefix='output'):
    """绘制 3D 相机位姿轨迹图."""
    traj = np.array([p[:3, 3] for p in abs_poses])

    fig = plt.figure(figsize=(12, 5))

    # ── 3D 轨迹 ──
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
    ax1.plot(xs, ys, zs, 'b-', linewidth=1.5, alpha=0.8)
    ax1.scatter(xs[0], ys[0], zs[0], c='green', s=80, marker='o', label='Start', zorder=5)
    ax1.scatter(xs[-1], ys[-1], zs[-1], c='red', s=80, marker='s', label='End', zorder=5)
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'3D Camera Trajectory ({len(traj)} frames)')
    ax1.legend(fontsize=9)
    ax1.view_init(elev=25, azim=-60)

    # ── 各轴分量随时间变化 ──
    ax2 = fig.add_subplot(1, 2, 2)
    frames = np.arange(len(traj))
    ax2.plot(frames, xs, 'r-', linewidth=1, label='X', alpha=0.7)
    ax2.plot(frames, ys, 'g-', linewidth=1, label='Y', alpha=0.7)
    ax2.plot(frames, zs, 'b-', linewidth=1, label='Z', alpha=0.7)
    ax2.set_xlabel('Frame'); ax2.set_ylabel('Position (mm)')
    ax2.set_title('Camera Position vs Frame')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    total_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    fig.suptitle(f'Path Length: {total_len:.1f} mm', fontsize=11, y=1.02)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f'{prefix}_trajectory_3d.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  相机轨迹图已保存: {save_path}")


def save_depth_visualizations(depth_maps, out_dir):
    """将深度图保存为可视化的 colormap PNG."""
    viz_dir = os.path.join(out_dir, 'depth_viz')
    os.makedirs(viz_dir, exist_ok=True)

    # 收集所有有效深度值计算全局范围
    all_vals = []
    for fi, dmap in depth_maps.items():
        valid = dmap[(dmap > 0.5) & np.isfinite(dmap)]
        if len(valid) > 0:
            all_vals.extend(valid.ravel().tolist())
    if not all_vals:
        print("  警告: 无有效深度值, 跳过可视化")
        return

    vmin, vmax = np.percentile(all_vals, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1.0

    for fi in sorted(depth_maps.keys()):
        dmap = depth_maps[fi].copy()
        dmap = np.clip(dmap, vmin, vmax)
        dmap_norm = (dmap - vmin) / (vmax - vmin)
        colored = plt.cm.turbo(dmap_norm)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)
        colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(viz_dir, f'{fi:04d}_depth.png')
        cv2.imwrite(save_path, colored)

    print(f"  深度可视化已保存: {viz_dir}/ ({len(depth_maps)} 帧, range [{vmin:.1f}, {vmax:.1f}] mm)")


def export_poses_txt(abs_poses, out_dir, prefix='output'):
    """导出相机绝对位姿为可读文本 (4×4 行主序, 每帧16个数)."""
    save_path = os.path.join(out_dir, f'{prefix}_abs_poses.txt')
    n = len(abs_poses)
    with open(save_path, 'w') as f:
        f.write(f'# {n} frames, 4x4 row-major per line\n')
        for i in range(n):
            flat = abs_poses[i].ravel()
            line = ','.join(f'{v:.6f}' for v in flat)
            f.write(line + '\n')
    print(f"  位姿文本已保存: {save_path} ({n} 帧)")


def run_pipeline(input_dir, output_dir, model_path=None, device=None, max_frames=None,
                  fx=None, fy=None, cx=None, cy=None, pose_path=None):
    """零配置 VO 管线主函数.

    Args:
        input_dir: RGB 帧文件夹
        output_dir: 输出目录
        model_path: monodepth2 模型路径 (None→默认)
        device: torch device (None→auto)
        max_frames: 最大帧数 (None→全部)
        fx, fy, cx, cy: 相机内参 (None→自动估算)
        pose_path: GT 位姿文件路径 (None→跳过 ATE 评估)

    Returns:
        dict: report
    """
    t_start = time.time()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    os.makedirs(output_dir, exist_ok=True)

    # ═══════════════════════════════════════════
    # Step 0: 扫描帧 & 自动估算 K
    # ═══════════════════════════════════════════
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    frames = sorted([f for f in os.listdir(input_dir)
                     if f.lower().endswith(exts)])
    if not frames:
        raise FileNotFoundError(f"未找到图像文件: {input_dir}")
    if max_frames:
        frames = frames[:max_frames + 1]

    n_frames = len(frames)
    n_pairs = n_frames - 1
    print(f"  找到 {n_frames} 帧, {n_pairs} 对匹配")

    first_img = os.path.join(input_dir, frames[0])
    K, img_w, img_h = auto_estimate_K(first_img, fx=fx, fy=fy, cx=cx, cy=cy)

    FX, FY = K[0, 0], K[1, 1]
    CX, CY = K[0, 2], K[1, 2]

    # ═══════════════════════════════════════════
    # Step 1: 加载深度模型
    # ═══════════════════════════════════════════
    print(f"\n[1/4] 加载深度预测模型: {model_path}")
    model_manager = ModelManager(model_path=model_path)
    encoder, depth_decoder, _, motion_encoder = model_manager.load_model()
    if motion_encoder is not None:
        print("  检测到 MotionEncoder, 将使用时序增强深度预测")
    else:
        print("  单帧深度预测模式")

    # ═══════════════════════════════════════════
    # Step 2: 初始化 LoFTR
    # ═══════════════════════════════════════════
    print("\n[2/4] 初始化 LoFTR 在线匹配...")
    matcher = init_loftr_matcher(device=str(device))

    # ═══════════════════════════════════════════
    # Step 3: VO 主循环
    # ═══════════════════════════════════════════
    print(f"\n[3/4] 运行 VO 管线 ({n_pairs} 帧对)...")

    T_abs = np.eye(4, dtype=np.float64)
    abs_poses = [T_abs.copy()]  # 帧0
    depth_maps = {}
    all_k0, all_k1, all_weights = [], [], []
    n_success = 0

    pbar = tqdm(range(n_pairs), desc='VO')
    for fi in pbar:
        img0_path = os.path.join(input_dir, frames[fi])
        img1_path = os.path.join(input_dir, frames[fi + 1])

        # ── LoFTR 匹配 ──
        k0, k1, _ = _match_loftr_pair(
            matcher, img0_path, img1_path,
            device=str(device), max_dim=LOFTR_MAX_DIM)

        if k0 is None or len(k0) < 4:
            abs_poses.append(T_abs.copy())
            pbar.set_postfix({'matches': 0, 'status': 'few'})
            continue

        n_match = len(k0)

        # ── 深度预测 ──
        if fi not in depth_maps:
            if motion_encoder is not None:
                depth = predict_depth(img0_path, encoder, depth_decoder, device,
                                      inverse=False, min_depth=DEPTH_MIN,
                                      max_depth=DEPTH_MAX)
                # MotionEncoder 增强深度 (用下一帧做时序引导)
                # 注意: run_pipeline 简化版, 不传 motion_encoder 给 predict_depth
            else:
                depth = predict_depth(img0_path, encoder, depth_decoder, device,
                                      inverse=False, min_depth=DEPTH_MIN,
                                      max_depth=DEPTH_MAX)
            depth_maps[fi] = depth
        else:
            depth = depth_maps[fi]

        # ── 预测下一帧深度 (用于帧间一致性校正 + 自适应过滤) ──
        depth_next = predict_depth(img1_path, encoder, depth_decoder, device,
                                   inverse=False, min_depth=DEPTH_MIN,
                                   max_depth=DEPTH_MAX)

        # ── 帧间深度尺度一致性校正 ──
        scale_corr = 1.0
        if fi > 0 and n_match >= 20:
            h_d, w_d = depth.shape
            h_dn, w_dn = depth_next.shape
            n_sample = min(n_match, 500)
            sample_idx = np.random.choice(n_match, n_sample, replace=False)
            ratios = []
            for j in sample_idx:
                u0, v0 = k0[j]; u1, v1 = k1[j]
                ui0 = int(np.clip(u0, 0, w_d - 1))
                vi0 = int(np.clip(v0, 0, h_d - 1))
                ui1 = int(np.clip(u1, 0, w_dn - 1))
                vi1 = int(np.clip(v1, 0, h_dn - 1))
                Z0 = float(depth[vi0, ui0])
                Z1 = float(depth_next[vi1, ui1])
                if Z0 > 0.5 and Z1 > 0.5 and np.isfinite(Z0) and np.isfinite(Z1):
                    n0 = np.sqrt(((u0 - CX) / FX) ** 2 + ((v0 - CY) / FY) ** 2 + 1)
                    n1 = np.sqrt(((u1 - CX) / FX) ** 2 + ((v1 - CY) / FY) ** 2 + 1)
                    ratios.append((n1 * Z1) / (n0 * Z0))
            if len(ratios) > 10:
                s_raw = float(np.median(ratios))
                scale_corr = 1.0 / np.clip(s_raw, SCALE_CLIP_LO, SCALE_CLIP_HI)
                depth = depth * scale_corr

        # ── 3D 反投影 ──
        h_d, w_d = depth.shape
        pts3d_list, pts2d_list, k0_list = [], [], []
        for j in range(n_match):
            u, v = k0[j]
            ui = int(np.clip(u, 0, w_d - 1))
            vi = int(np.clip(v, 0, h_d - 1))
            Z = float(depth[vi, ui])
            if Z <= 0.5 or Z > DEPTH_MAX or not np.isfinite(Z):
                continue
            X = (u - CX) * Z / FX
            Y = (v - CY) * Z / FY
            pts3d_list.append([X, Y, Z])
            pts2d_list.append(k1[j])
            k0_list.append(k0[j])

        if len(pts3d_list) < 4:
            abs_poses.append(T_abs.copy())
            pbar.set_postfix({'matches': n_match, 'valid3d': len(pts3d_list), 'status': 'few3d'})
            continue

        n_valid = len(pts3d_list)
        pts3d_np = np.array(pts3d_list, dtype=np.float32)
        pts2d_np = np.array(pts2d_list, dtype=np.float32)
        k0_filter = np.array(k0_list, dtype=np.float32)

        # ── 收集链式追踪数据 ──
        all_k0.append(k0.copy())
        all_k1.append(k1.copy())
        all_weights.append(np.ones(n_valid, dtype=np.float32))

        # ── EPnP (F1 自适应 3D 位移过滤已关闭: 消融证明其破坏全局尺度) ──
        R, t = None, None
        filter_keep = np.ones(n_valid, dtype=bool)
        n_filtered = n_valid

        # 尝试过滤后的 PnP
        for attempt, mask in enumerate([filter_keep, np.ones(n_valid, dtype=bool)]):
            if mask.sum() < 4:
                continue
            pts3d_cv = pts3d_np[mask].reshape(-1, 1, 3).astype(np.float64)
            pts2d_cv = pts2d_np[mask].reshape(-1, 1, 2).astype(np.float64)
            K_cv = K.astype(np.float64)
            try:
                ret, rvec, tvec = cv2.solvePnP(
                    pts3d_cv, pts2d_cv, K_cv, None, flags=cv2.SOLVEPNP_EPNP)
                if ret:
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.ravel()
                    break
            except cv2.error:
                pass

        if R is None or t is None:
            abs_poses.append(T_abs.copy())
            pbar.set_postfix({'matches': n_match, 'valid3d': n_valid, 'status': 'pnp_fail'})
            continue

        # 位姿合理性检查
        angle_deg = rotation_angle_deg(R)
        t_norm = float(np.linalg.norm(t))
        if angle_deg > MAX_ROTATION_DEG or t_norm > MAX_TRANSLATION_MM:
            abs_poses.append(T_abs.copy())
            pbar.set_postfix({'matches': n_match, 'valid3d': n_valid,
                              'angle': f'{angle_deg:.1f}°', 'status': 'rejected'})
            continue

        # 累积位姿
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t
        T_abs = T_abs @ T_rel
        abs_poses.append(T_abs.copy())
        n_success += 1

        pbar.set_postfix({'matches': n_match, 'valid3d': n_valid,
                          'sc': f'{scale_corr:.3f}',
                          'angle': f'{angle_deg:.1f}°', 'ok': n_success})

    # ── 预测最后一帧深度 ──
    last_idx = n_frames - 1
    if last_idx not in depth_maps:
        last_img = os.path.join(input_dir, frames[last_idx])
        depth_maps[last_idx] = predict_depth(
            last_img, encoder, depth_decoder, device,
            inverse=False, min_depth=DEPTH_MIN, max_depth=DEPTH_MAX)

    print(f"\n  成功: {n_success}/{n_pairs}")

    # ═══════════════════════════════════════════
    # Step 4: 保存深度图
    # ═══════════════════════════════════════════
    depth_dir = os.path.join(output_dir, 'depth')
    os.makedirs(depth_dir, exist_ok=True)
    for fi in sorted(depth_maps.keys()):
        np.save(os.path.join(depth_dir, f'{fi:04d}.npy'), depth_maps[fi])
    print(f"\n  深度图已保存: {depth_dir}/ ({len(depth_maps)} 帧)")

    # ── 保存 abs_poses ──
    abs_poses_arr = np.stack(abs_poses, axis=0)
    pose_path_out = os.path.join(output_dir, 'abs_poses.npy')
    np.save(pose_path_out, abs_poses_arr)
    print(f"  位姿已保存: {pose_path_out} ({abs_poses_arr.shape[0]} 帧)")

    # ── ATE 评估 (如有 GT pose) ──
    ate_result = None
    if pose_path and os.path.exists(pose_path):
        print(f"\n  [ATE] 加载 GT 位姿: {pose_path}")
        ate_result = compute_ate_umeyama(abs_poses_arr, pose_path, K)
        if ate_result:
            print(f"  ATE RMSE: {ate_result['ate_rmse_mm']:.2f} mm")
            print(f"  ATE Mean: {ate_result['ate_mean_mm']:.2f} mm")
            print(f"  ATE Median: {ate_result['ate_median_mm']:.2f} mm")
            print(f"  ATE Std: {ate_result['ate_std_mm']:.2f} mm")
            print(f"  Umeyama Scale: {ate_result['umeyama_scale']:.4f}")
            print(f"  Aligned frames: {ate_result['n_aligned']}")

    # ── 保存 chain_data ──
    chain_path = os.path.join(output_dir, 'chain_data.npz')
    chain_save = {'n_pairs': len(all_k0)}
    for i in range(len(all_k0)):
        chain_save[f'k0_{i:04d}'] = all_k0[i]
        chain_save[f'k1_{i:04d}'] = all_k1[i]
        chain_save[f'w_{i:04d}'] = all_weights[i]
    chain_save['K'] = K
    chain_save['k_factor'] = float(DISP_K_FACTOR)
    chain_save['chain_dist_thresh'] = float(CHAIN_DIST_THRESH)
    np.savez_compressed(chain_path, **chain_save)
    print(f"  Chain数据已保存: {chain_path}")

    # ═══════════════════════════════════════════
    # Step 5: 运动轨迹分析
    # ═══════════════════════════════════════════
    print(f"\n[4/4] 运动轨迹分析...")

    # 链式追踪
    print("  [A] 链式追踪...")
    tracks = chain_tracks(all_k0, all_k1, dist_thresh=CHAIN_DIST_THRESH)
    n_long = sum(1 for t in tracks if len(t['frames']) >= 5)
    print(f"  总 tracks: {len(tracks)}, 长度≥5: {n_long}")

    # 采样
    if len(tracks) > MAX_TRACKS:
        import random
        random.seed(42)
        tracks_long = [t for t in tracks if len(t['frames']) >= 5]
        tracks_short = [t for t in tracks if len(t['frames']) < 5]
        n_sample_long = min(len(tracks_long), MAX_TRACKS * 2 // 3)
        n_sample_short = MAX_TRACKS - n_sample_long
        tracks = random.sample(tracks_long, n_sample_long) + \
                 random.sample(tracks_short, min(n_sample_short, len(tracks_short)))
        print(f"  采样至: {len(tracks)} tracks")

    # 分类
    print("\n  [B] 静止/运动分类...")
    static_tracks, moving_tracks = classify_tracks(
        tracks, depth_maps, abs_poses, K, k_factor=DISP_K_FACTOR)
    print(f"  静止: {len(static_tracks)}, 运动: {len(moving_tracks)}")

    # 统计
    print("\n  [C] 位移统计...")
    summary = compute_motion_summary(moving_tracks, static_tracks,
                                     depth_maps, abs_poses, K)
    print(f"  静止点位移: mean={summary['static_disp_stats']['mean_mm']:.1f}mm "
          f"median={summary['static_disp_stats']['median_mm']:.1f}mm")
    print(f"  运动点位移: mean={summary['moving_disp_stats']['mean_mm']:.1f}mm "
          f"median={summary['moving_disp_stats']['median_mm']:.1f}mm "
          f"max={summary['moving_disp_stats']['max_mm']:.1f}mm")

    # 保存 tracks.json
    print("\n  [D] 保存结果...")
    tracks_export = []
    for track in tracks:
        t = track if not isinstance(track, tuple) else track[0]
        tracks_export.append({
            'id': t['id'],
            'frames': [[int(fi), float(u), float(v)] for fi, u, v in t['frames']],
        })
    moving_ids = [t[0]['id'] if isinstance(t, tuple) else t['id'] for t in moving_tracks]
    track_save_path = os.path.join(output_dir, 'tracks.json')
    with open(track_save_path, 'w') as f:
        json.dump({
            'tracks': tracks_export,
            'moving_ids': moving_ids,
            'n_total': len(tracks_export),
            'n_moving': len(moving_ids),
            'static_disp': summary['static_disp_stats'],
            'moving_disp': summary['moving_disp_stats'],
        }, f, indent=2)
    print(f"  tracks.json 已保存: {track_save_path}")

    # 保存运动轨迹
    save_motion_trajectories(moving_tracks, depth_maps, abs_poses, K,
                             output_dir, 'output')

    # 可视化
    print("\n  [E] 可视化...")
    visualize_motion(moving_tracks, static_tracks, depth_maps, abs_poses, K,
                     output_dir, 'output')

    # ═══════════════════════════════════════════
    # 三大核心输出: 深度可视化 + 轨迹图 + 位姿文本
    # ═══════════════════════════════════════════
    print("\n  [F] 深度可视化...")
    save_depth_visualizations(depth_maps, output_dir)

    print("\n  [G] 相机轨迹可视化...")
    visualize_camera_trajectory(abs_poses_arr, output_dir, 'output')

    print("\n  [H] 位姿文本导出...")
    export_poses_txt(abs_poses_arr, output_dir, 'output')

    # ═══════════════════════════════════════════
    # 生成 report.json
    # ═══════════════════════════════════════════
    elapsed = time.time() - t_start
    report = {
        'pipeline': 'run_pipeline (zero-config)',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'input_dir': os.path.abspath(input_dir),
        'output_dir': os.path.abspath(output_dir),
        'camera': {
            'fx': float(FX), 'fy': float(FY),
            'cx': float(CX), 'cy': float(CY),
            'width': img_w, 'height': img_h,
            'estimated': (fx is None),
        },
        'vo': {
            'n_frames': n_frames,
            'n_pairs': n_pairs,
            'n_success': n_success,
            'success_rate': f'{n_success / max(n_pairs, 1) * 100:.1f}%',
        },
        'motion': {
            'n_total_tracks': len(tracks),
            'n_static_tracks': summary['n_static_tracks'],
            'n_moving_tracks': summary['n_moving_tracks'],
            'static_disp_mean_mm': summary['static_disp_stats']['mean_mm'],
            'static_disp_median_mm': summary['static_disp_stats']['median_mm'],
            'moving_disp_mean_mm': summary['moving_disp_stats']['mean_mm'],
            'moving_disp_median_mm': summary['moving_disp_stats']['median_mm'],
            'separation_ratio': (
                summary['moving_disp_stats']['mean_mm'] / summary['static_disp_stats']['mean_mm']
                if summary['static_disp_stats']['mean_mm'] > 0 else None
            ),
        },
        'config': {
            'depth_min': DEPTH_MIN,
            'depth_max': DEPTH_MAX,
            'loftr_max_dim': LOFTR_MAX_DIM,
            'disp_k_factor': DISP_K_FACTOR,
            'max_tracks': MAX_TRACKS,
            'chain_dist_thresh': CHAIN_DIST_THRESH,
            'global_scale_calibration': False,
        },
        'ate': ate_result,
        'elapsed_sec': round(elapsed, 1),
    }

    report_path = os.path.join(output_dir, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  report.json 已保存: {report_path}")

    # ── 终端汇总 ──
    sep = report['motion']['separation_ratio']
    print("\n" + "═" * 65)
    print("  【 零配置管线输出汇总 】")
    print("═" * 65)
    print(f"  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  输出                          │  指标              │  值        │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  帧数                          │  n_frames          │  {n_frames:>5d}    │")
    print(f"  │  VO 成功率                     │  success_rate      │  {n_success}/{n_pairs}     │")
    print(f"  │  运动检测                      │  moving tracks     │  {len(moving_tracks):>5d}    │")
    print(f"  │  静止点位移                    │  median            │  {summary['static_disp_stats']['median_mm']:>5.1f} mm │")
    print(f"  │  运动点位移                    │  median            │  {summary['moving_disp_stats']['median_mm']:>5.1f} mm │")
    if sep:
        print(f"  │  运动/静止分离度               │  ratio             │  {sep:>5.1f}×    │")
    print(f"  │  耗时                          │  elapsed           │  {elapsed:>5.1f} s  │")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    output_files = [
        f'{output_dir}/depth/',
        f'{output_dir}/depth_viz/',
        f'{output_dir}/abs_poses.npy',
        f'{output_dir}/output_abs_poses.txt',
        f'{output_dir}/output_trajectory_3d.png',
        f'{output_dir}/chain_data.npz',
        f'{output_dir}/tracks.json',
        f'{output_dir}/report.json',
        f'{output_dir}/output_motion_trajectories.json',
        f'{output_dir}/output_motion_3d.png',
    ]
    print(f"\n  输出文件:")
    for f in output_files:
        print(f"    {f}")

    return report


def main():
    parser = argparse.ArgumentParser(description='零配置 VO 管线')
    parser.add_argument('--input', type=str, required=True,
                        help='RGB 帧文件夹')
    parser.add_argument('--output', type=str, default='./run_output',
                        help='输出目录 (默认: ./run_output)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'monodepth2 模型路径 (默认: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='最大帧数 (默认: 全部)')
    parser.add_argument('--fx', type=float, default=None,
                        help='相机内参 fx (默认: 自动估算)')
    parser.add_argument('--fy', type=float, default=None,
                        help='相机内参 fy (默认: 自动估算)')
    parser.add_argument('--cx', type=float, default=None,
                        help='相机内参 cx (默认: 自动估算)')
    parser.add_argument('--cy', type=float, default=None,
                        help='相机内参 cy (默认: 自动估算)')
    parser.add_argument('--pose', type=str, default=None,
                        help='GT 位姿文件路径 (用于 ATE 评估, 格式: 每行16个数, 4×4行主序)')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用 CPU')
    args = parser.parse_args()

    device = torch.device('cpu') if args.cpu else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    print("═" * 65)
    print("  run_pipeline · 零配置 VO 管线")
    print("═" * 65)
    print(f"  输入: {args.input}")
    print(f"  输出: {args.output}")
    print(f"  设备: {device}")
    print(f"  模型: {args.model}")

    run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        device=device,
        max_frames=args.max_frames,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        pose_path=args.pose,
    )


if __name__ == '__main__':
    main()
