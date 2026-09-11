"""滑动窗口 Pose Graph Optimization (Relative Pose Consistency)

不依赖3D点深度, 直接优化相对位姿:
- Stay close to PnP estimates (data term)
- Smoothness: constant velocity prior on poses
"""

import numpy as np
import cv2
from scipy.optimize import least_squares


def se3_to_params(R, t):
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([rvec.ravel(), t.ravel()])


def params_to_se3(params):
    rvec = params[:3]
    t = params[3:6]
    R, _ = cv2.Rodrigues(rvec)
    return R, t


def rel_pose_diff(R1, t1, R2, t2):
    """Compute SE(3) difference: log(T1^{-1} @ T2).
    Returns (dR_angle, dR_axis, dt_norm).
    """
    dR = R1.T @ R2
    angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    dt = np.linalg.norm(t1 - t2)
    return angle, dt


def optimize_window_rel(rel_poses_init, data_weight=1.0, smoothness_weight=0.5,
                         max_iters=20):
    """Optimize relative poses in a window to be consistent and smooth.

    Args:
        rel_poses_init: list of (R_i, t_i) for frame pairs i→i+1, length = N
        data_weight: weight for staying close to initial PnP estimates
        smoothness_weight: weight for constant-velocity prior

    Returns:
        list of optimized (R_i, t_i), same length as input
    """
    n_pairs = len(rel_poses_init)
    if n_pairs < 2:
        return rel_poses_init

    # Parameters: 6 DoF per relative pose
    n_params = 6 * n_pairs
    x0 = np.zeros(n_params)
    for i in range(n_pairs):
        R, t = rel_poses_init[i]
        rvec, _ = cv2.Rodrigues(R)
        x0[i * 6:i * 6 + 3] = rvec.ravel()
        x0[i * 6 + 3:i * 6 + 6] = t.ravel()

    def cost_func(params):
        # Reconstruct relative poses
        rel_poses_opt = []
        for i in range(n_pairs):
            rvec = params[i * 6:i * 6 + 3]
            R, _ = cv2.Rodrigues(rvec)
            t = params[i * 6 + 3:i * 6 + 6]
            rel_poses_opt.append((R, t))

        residuals = []

        # 1) Data term: stay close to initial PnP
        for i in range(n_pairs):
            R_opt, t_opt = rel_poses_opt[i]
            R_init, t_init = rel_poses_init[i]

            # Rotation difference (angle)
            dR = R_opt.T @ R_init
            rot_angle = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
            residuals.append(data_weight * rot_angle * 100)  # Scale up for optimizer

            # Translation difference (vector norm)
            dt = np.linalg.norm(t_opt - t_init)
            residuals.append(data_weight * dt * 1000)  # m→mm scale

        # 2) Smoothness: constant velocity (acceleration = 0)
        if smoothness_weight > 0 and n_pairs >= 2:
            for i in range(n_pairs - 1):
                R_i, t_i = rel_poses_opt[i]
                R_i1, t_i1 = rel_poses_opt[i + 1]

                # Penalize change in relative rotation
                dR = R_i.T @ R_i1
                rot_diff = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
                residuals.append(smoothness_weight * rot_diff * 100)

                # Penalize change in relative translation
                dt_diff = np.linalg.norm(t_i - t_i1)
                residuals.append(smoothness_weight * dt_diff * 1000)

        return np.array(residuals)

    result = least_squares(
        cost_func, x0,
        method='trf',
        loss='soft_l1',
        f_scale=1.0,
        max_nfev=max_iters,
        verbose=0,
        ftol=1e-8, xtol=1e-8
    )

    optimized_params = result.x
    optimized_rel = []
    for i in range(n_pairs):
        rvec = optimized_params[i * 6:i * 6 + 3]
        R, _ = cv2.Rodrigues(rvec)
        t = optimized_params[i * 6 + 3:i * 6 + 6]
        optimized_rel.append((R, t))

    return optimized_rel


def run_sliding_window_ba(poses_abs, pts_3d_seq, pts_2d_seq, K,
                          window_size=8, smoothness_weight=1.0,
                          huber_delta=5.0, verbose=False):
    """对整个轨迹运行滑动窗口相对位姿优化.

    Args:
        poses_abs: 绝对位姿列表 [T_4x4, ...], 长度 = N+1 (Frame 0 = I)
        pts_3d_seq: 未使用 (保留接口兼容)
        pts_2d_seq: 未使用 (保留接口兼容)
    """
    # 统一为 (R,t) 列表
    poses_se3 = []
    for p in poses_abs:
        if isinstance(p, np.ndarray) and p.shape == (4, 4):
            poses_se3.append((p[:3, :3], p[:3, 3]))
        else:
            poses_se3.append(p)

    n_frames = len(poses_se3)
    n_pairs = n_frames - 1

    if n_frames <= 2:
        return poses_abs

    # Extract relative poses from absolute poses
    rel_poses = []
    for i in range(n_pairs):
        R_i, t_i = poses_se3[i]
        R_i1, t_i1 = poses_se3[i + 1]
        R_rel = R_i.T @ R_i1
        t_rel = R_i.T @ (t_i1 - t_i)
        rel_poses.append((R_rel, t_rel))

    # Non-overlapping windows: step = window_size - 1
    step = max(1, window_size - 1)
    optimized_rel = [None] * n_pairs

    win_start = 0
    while win_start < n_pairs:
        win_end = min(win_start + window_size, n_pairs)
        win_rel_init = rel_poses[win_start:win_end]

        if len(win_rel_init) < 2:
            break

        try:
            win_rel_opt = optimize_window_rel(
                win_rel_init,
                data_weight=1.0,
                smoothness_weight=smoothness_weight)
        except Exception as e:
            if verbose:
                print(f'  BA window [{win_start}:{win_end}] failed: {e}')
            win_rel_opt = win_rel_init

        for j, rel in enumerate(win_rel_opt):
            idx = win_start + j
            if idx < n_pairs:
                optimized_rel[idx] = rel

        win_start += step

    # Fill gaps with original
    for i in range(n_pairs):
        if optimized_rel[i] is None:
            optimized_rel[i] = rel_poses[i]

    # Reconstruct absolute poses from optimized relative poses
    optimized_se3 = [poses_se3[0]]  # Frame 0 = identity
    for i in range(n_pairs):
        R_prev, t_prev = optimized_se3[-1]
        R_rel, t_rel = optimized_rel[i]
        R_new = R_prev @ R_rel
        t_new = R_prev @ t_rel + t_prev
        optimized_se3.append((R_new, t_new))

    # Convert back to original format
    result = []
    for R, t in optimized_se3:
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        result.append(T)

    return result
