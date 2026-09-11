"""统一 5 帧窗口 ATE 重算脚本（Zhou et al. 2017 的 5-frame pose evaluation 协议）。

与 AF-SfMLearner / Endo-FASt3r 论文 Table 9/10 的口径一致：
对每个连续 5 帧窗口，窗口内做 Umeyama 7 自由度（旋转+平移+尺度）对齐后
计算窗口 ATE（RMSE），再对所有窗口取平均。

输入：
  --rel_npz  预测相对位姿 npz（data 字段 [M,4,4]，逐帧相对位姿 cam_i -> cam_{i+1}）
  --gt_npz   GT 绝对位姿 npz（data 字段 [N,4,4]，绝对位姿 cam -> world）
  --window   窗口帧数，默认 5（论文协议）

输出：
  5 帧窗口 ATE（mean + std）、全轨迹 Umeyama ATE（对照）、RPE-T、RPE-R

用法示例：
  python compute_5frame_ate.py --rel_npz splits/endovis/pred_pose_sq1.npz \
      --gt_npz splits/endovis/gt_poses_sq1.npz
"""
import argparse
import numpy as np


def umeyama(src, dst):
    """Umeyama 7 自由度相似变换对齐（旋转+平移+尺度）。src/dst: Nx3。返回 s, R, t。"""
    n = src.shape[0]
    mu_src = src.mean(0)
    mu_dst = dst.mean(0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = dst_c.T @ src_c / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_src = np.sum(src_c ** 2) / n
    s = np.trace(np.diag(D) @ S) / var_src
    t = mu_dst - s * R @ mu_src
    return s, R, t


def apply_sim3(pts, s, R, t):
    return (s * R @ pts.T).T + t


def compute_ate(gt_xyz, pred_xyz):
    """Umeyama 对齐后的均方根 ATE（米）。返回 rmse, mean, std, scale。"""
    s, R, t = umeyama(pred_xyz, gt_xyz)
    aligned = apply_sim3(pred_xyz, s, R, t)
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    return np.sqrt(np.mean(err ** 2)), np.mean(err), np.std(err), s


def compute_5frame_ate(gt_poses, pred_poses, window=5):
    """5 帧滑动窗口 ATE（论文口径）。gt_poses/pred_poses: [N,4,4] 绝对位姿。

    对每个连续 window 帧窗口做 Umeyama 对齐后算窗口 ATE（RMSE），
    返回所有窗口 ATE 的 mean 与 std（米）。
    """
    n = gt_poses.shape[0]
    if n < window:
        raise ValueError(f"帧数 {n} 少于窗口 {window}，无法计算")
    ate_list = []
    for i in range(0, n - window + 1):
        g = gt_poses[i:i + window]
        p = pred_poses[i:i + window]
        g_xyz = g[:, :3, 3]
        p_xyz = p[:, :3, 3]
        s, R, t = umeyama(p_xyz, g_xyz)
        aligned = apply_sim3(p_xyz, s, R, t)
        err = np.linalg.norm(aligned - g_xyz, axis=1)
        ate_list.append(np.sqrt(np.mean(err ** 2)))
    ate_list = np.array(ate_list)
    return float(ate_list.mean()), float(ate_list.std())


def compute_rpe(gt_poses, pred_poses):
    """帧间相对位姿误差。pred 先做全轨迹 Umeyama 对齐。返回 RPE-T(m), RPE-R(deg)。"""
    n = gt_poses.shape[0]
    gt_xyz = gt_poses[:, :3, 3]
    pred_xyz = pred_poses[:, :3, 3]
    s, R, t = umeyama(pred_xyz, gt_xyz)

    aligned = pred_poses.copy()
    for i in range(n):
        aligned[i, :3, :3] = R @ aligned[i, :3, :3]
        aligned[i, :3, 3] = s * (R @ pred_poses[i, :3, 3]) + t

    te, re = [], []
    for i in range(n - 1):
        d_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        d_pred = np.linalg.inv(aligned[i]) @ aligned[i + 1]
        E = np.linalg.inv(d_pred) @ d_gt
        te.append(np.linalg.norm(d_gt[:3, 3] - d_pred[:3, 3]))
        trace = np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)
        re.append(np.arccos(trace) * 180 / np.pi)
    return float(np.mean(te)), float(np.mean(re))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel_npz", required=True, help="预测相对位姿 npz（data 字段 [M,4,4]）")
    ap.add_argument("--gt_npz", required=True, help="GT 绝对位姿 npz（data 字段 [N,4,4]）")
    ap.add_argument("--window", type=int, default=5, help="窗口帧数（默认 5，论文协议）")
    args = ap.parse_args()

    rel = np.load(args.rel_npz, allow_pickle=True)["data"]
    gt = np.load(args.gt_npz, allow_pickle=True)["data"]
    print(f"预测相对位姿 shape: {rel.shape}, GT 绝对位姿 shape: {gt.shape}")

    # 累积相对位姿 -> 绝对轨迹
    traj = [np.eye(4)]
    c2w = np.eye(4)
    for T in rel:
        c2w = c2w @ T
        traj.append(c2w.copy())
    pred = np.array(traj)

    # 对齐帧数
    n = min(pred.shape[0], gt.shape[0])
    pred = pred[:n]
    gt = gt[:n]
    print(f"对齐后帧数: {n}")

    # 5 帧窗口 ATE（论文口径）
    w_ate_mean, w_ate_std = compute_5frame_ate(gt, pred, args.window)

    # 全轨迹 Umeyama ATE（对照）
    ate_rmse, ate_mean, ate_std, scale = compute_ate(gt[:, :3, 3], pred[:, :3, 3])

    # RPE
    rpe_t, rpe_r = compute_rpe(gt, pred)

    print("\n========== 结果（统一口径） ==========")
    print(f"[论文口径] {args.window} 帧窗口 ATE : {w_ate_mean:.4f} m ({w_ate_mean*1000:.2f} mm) +/- {w_ate_std*1000:.2f} mm")
    print(f"[对照]     全轨迹 Umeyama ATE : {ate_rmse:.4f} m ({ate_rmse*1000:.2f} mm)")
    print(f"[对照]     估计尺度            : {scale:.4f}")
    print(f"[对照]     RPE-T               : {rpe_t:.4f} m ({rpe_t*1000:.2f} mm)")
    print(f"[对照]     RPE-R               : {rpe_r:.4f} deg")
    print("======================================")


if __name__ == "__main__":
    main()
