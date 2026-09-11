"""统一口径：相对位姿 npz + GT 绝对位姿 npz -> Umeyama 7 自由度 ATE + RPE。

用于把 Endo-FASt3r / AF-SfMLearner 等在 SCARED 上已保存的相对位姿，
按与 Ours/BodySLAM 一致的 Umeyama 口径重算 ATE。

用法:
    python recompute_umeyama_ate.py --rel_npz <pred_pose_sq1.npz> --gt_npz <gt_poses_sq1.npz>
"""
import argparse
import numpy as np


def umeyama(src, dst):
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
    s, R, t = umeyama(pred_xyz, gt_xyz)
    aligned = apply_sim3(pred_xyz, s, R, t)
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    return np.sqrt(np.mean(err ** 2)), np.mean(err), np.std(err), s


def compute_rpe(gt_poses, pred_poses):
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

    # 对齐帧数（预测轨迹 N 帧 vs GT N 帧）
    n = min(pred.shape[0], gt.shape[0])
    pred = pred[:n]
    gt = gt[:n]
    print(f"对齐后帧数: {n}")

    gt_xyz = gt[:, :3, 3]
    pred_xyz = pred[:, :3, 3]
    ate_rmse, ate_mean, ate_std, scale = compute_ate(gt_xyz, pred_xyz)
    rpe_t, rpe_r = compute_rpe(gt, pred)

    print("\n========== Umeyama 口径结果 ==========")
    print(f"ATE RMSE : {ate_rmse:.4f} m ({ate_rmse*1000:.2f} mm)")
    print(f"ATE Mean : {ate_mean:.4f} m +/- {ate_std:.4f} m")
    print(f"估计尺度 : {scale:.4f}")
    print(f"RPE-T    : {rpe_t:.4f} m ({rpe_t*1000:.2f} mm)")
    print(f"RPE-R    : {rpe_r:.4f} deg")
    print("======================================")


if __name__ == "__main__":
    main()
