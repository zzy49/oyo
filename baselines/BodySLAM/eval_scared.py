"""BodySLAM CycleVO 在 SCARED 上的位姿评估（Umeyama 7 自由度 ATE 口径）。

流程：
1. 读 test_files_sequence{seq}.txt（folder frame_index side）
2. 生成 RGB 帧路径列表（image_02/data/{frame_index:010d}.png）
3. CycleVO 预测相邻帧相对位姿 -> 累积为绝对轨迹
4. GT 绝对位姿（gt_poses_sq{seq}.npz，注意是绝对位姿非相对）直接使用
5. Umeyama 7 自由度对齐 ATE + RPE-T + RPE-R
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pose_estimation.interface import PoseEstimator  # noqa: E402


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
    """Umeyama 对齐后的均方根 ATE（米）。"""
    s, R, t = umeyama(pred_xyz, gt_xyz)
    aligned = apply_sim3(pred_xyz, s, R, t)
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    return np.sqrt(np.mean(err ** 2)), np.mean(err), np.std(err), s


def compute_rpe(gt_poses, pred_poses):
    """帧间相对位姿误差。pred 先做全局 Umeyama 对齐。
    返回 RPE-T(m), RPE-R(deg)。"""
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="weights/CycleVO.pth")
    ap.add_argument("--data_path",
                    default=r"e:\data1\monodepth2\baselines\Endo_FASt3r\SCARED_Images_Resized")
    ap.add_argument("--split_dir",
                    default=r"e:\data1\monodepth2\baselines\Endo_FASt3r\splits\endovis")
    ap.add_argument("--seq", type=int, default=1, choices=[1, 2])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    # 1. 读 test_files
    split_path = Path(args.split_dir) / f"test_files_sequence{args.seq}.txt"
    items = []
    with open(split_path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            folder = parts[0]
            frame_index = int(parts[1])
            side = parts[2] if len(parts) > 2 else 'l'
            items.append((folder, frame_index, side))

    # 2. 图像路径
    side_map = {'l': 2, 'r': 3}
    imgs = []
    for folder, fid, side in items:
        p = Path(args.data_path) / folder / f"image_0{side_map.get(side, 2)}" / "data" / f"{fid:010d}.png"
        imgs.append(str(p))

    if args.max_frames > 0:
        imgs = imgs[:args.max_frames]
    imgs = imgs[::args.stride]
    print(f"使用 {len(imgs)} 帧（seq={args.seq}, stride={args.stride}）")

    # 3. GT 绝对位姿（gt_poses_sq.npz 的 data 为绝对位姿 cam->world）
    gt_path = Path(args.split_dir) / f"gt_poses_sq{args.seq}.npz"
    gt_all = np.load(gt_path, allow_pickle=True)["data"]  # [N, 4, 4]
    gt = gt_all[::args.stride]
    if args.max_frames > 0:
        gt = gt[:args.max_frames]
    n = min(len(imgs), gt.shape[0])
    imgs = imgs[:n]
    gt = gt[:n]
    print(f"GT 位姿数: {n}")

    # 4. CycleVO 推理
    est = PoseEstimator(args.model_path)
    poses = est.process_sequence(imgs)
    pred = np.array(poses)

    # 4.5 保存相对位姿 npz（供 5 帧窗口 ATE 重算）
    rel = np.array([np.linalg.inv(pred[i]) @ pred[i + 1] for i in range(pred.shape[0] - 1)])
    rel_path = Path(args.split_dir) / f"pred_pose_bodyslam_sq{args.seq}.npz"
    np.savez_compressed(rel_path, data=rel)
    print(f"已保存相对位姿: {rel_path} ({rel.shape})")

    # 5. 评估
    gt_xyz = gt[:, :3, 3]
    pred_xyz = pred[:, :3, 3]
    ate_rmse, ate_mean, ate_std, scale = compute_ate(gt_xyz, pred_xyz)
    rpe_t, rpe_r = compute_rpe(gt, pred)

    print(f"\n================ BodySLAM on SCARED seq{args.seq} ================")
    print(f"ATE RMSE : {ate_rmse:.4f} m ({ate_rmse*1000:.2f} mm)")
    print(f"ATE Mean : {ate_mean:.4f} m +/- {ate_std:.4f} m")
    print(f"估计尺度 : {scale:.4f}")
    print(f"RPE-T    : {rpe_t:.4f} m ({rpe_t*1000:.2f} mm)")
    print(f"RPE-R    : {rpe_r:.4f} deg")
    print("============================================================")


if __name__ == "__main__":
    main()
