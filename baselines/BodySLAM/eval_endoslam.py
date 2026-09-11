"""BodySLAM CycleVO 在 EndoSLAM UnityCam Colon 上的位姿评估。

流程：
1. 读取 GT 绝对位姿 CSV（tX,tY,tZ,rX,rY,rZ,rW,time）
2. 采样 RGB 帧（每 stride 帧取 1 帧）
3. CycleVO 预测相邻帧相对位姿 -> 累积为绝对轨迹
4. 计算 ATE（Umeyama 7 自由度对齐，含尺度）+ RPE-T + RPE-R
"""
import sys
import csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pose_estimation.interface import PoseEstimator  # noqa: E402


# ---------- 位姿工具 ----------
def quat_xyzw_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_gt_poses(csv_path, stride=1):
    """读取 EndoSLAM 位姿 CSV，返回 Nx4x4 绝对位姿（cam->world）。"""
    poses = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % stride != 0:
                continue
            t = np.array([float(row["tX"]), float(row["tY"]), float(row["tZ"])])
            q = np.array([float(row["rX"]), float(row["rY"]),
                          float(row["rZ"]), float(row["rW"])])
            T = np.eye(4)
            T[:3, :3] = quat_xyzw_to_mat(q)
            T[:3, 3] = t
            poses.append(T)
    return np.array(poses)


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
    """帧间相对位姿误差。pred 先做全局 Umeyama 尺度对齐（旋转不变，平移用对齐尺度）。
    返回 RPE-T(m), RPE-R(deg)。"""
    n = gt_poses.shape[0]
    # 用轨迹位置做 Umeyama 得到全局尺度
    gt_xyz = gt_poses[:, :3, 3]
    pred_xyz = pred_poses[:, :3, 3]
    s, R, t = umeyama(pred_xyz, gt_xyz)

    # 对齐后的预测位姿（旋转用 R，平移用 s*R*x + t）
    aligned = pred_poses.copy()
    for i in range(n):
        aligned[i, :3, :3] = R @ aligned[i, :3, :3]
        aligned[i, :3, 3] = s * (R @ pred_poses[i, :3, 3]) + t

    te, re = [], []
    for i in range(n - 1):
        # 相对位姿误差
        d_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        d_pred = np.linalg.inv(aligned[i]) @ aligned[i + 1]
        E = np.linalg.inv(d_pred) @ d_gt
        te.append(np.linalg.norm(d_gt[:3, 3] - d_pred[:3, 3]))
        # 旋转角
        trace = np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)
        re.append(np.arccos(trace) * 180 / np.pi)
    return float(np.mean(te)), float(np.mean(re))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="weights/CycleVO.pth")
    ap.add_argument("--frames_dir", default=r"e:\data1\monodepth2\EndoSLAM\UnityCam\Colon\Frames_jpg")
    ap.add_argument("--gt_csv", default=r"e:\data1\monodepth2\EndoSLAM\UnityCam\Colon\Poses\colon_position_rotation.csv")
    ap.add_argument("--stride", type=int, default=10, help="每 stride 帧取 1 帧")
    ap.add_argument("--max_frames", type=int, default=0, help="最多用多少帧，0=全部")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    imgs = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if args.max_frames > 0:
        imgs = imgs[:args.max_frames * args.stride]
    imgs = imgs[::args.stride]
    print(f"使用 {len(imgs)} 帧（stride={args.stride}）")

    gt = load_gt_poses(args.gt_csv, stride=args.stride)
    if args.max_frames > 0:
        gt = gt[:args.max_frames]
    # 对齐帧数与 GT 行数
    n = min(len(imgs), gt.shape[0])
    imgs = imgs[:n]
    gt = gt[:n]
    print(f"GT 位姿数: {n}")

    est = PoseEstimator(args.model_path)
    poses = est.process_sequence([str(p) for p in imgs])  # N 个 4x4 绝对位姿（累积）
    pred = np.array(poses)

    # 保存相对位姿 + GT 绝对位姿 npz（供 5 帧窗口 ATE 重算）
    scene_dir = Path(args.frames_dir).parent
    scene_short = scene_dir.name
    rel = np.array([np.linalg.inv(pred[i]) @ pred[i + 1] for i in range(pred.shape[0] - 1)])
    np.savez_compressed(scene_dir / f"pred_pose_bodyslam_{scene_short}.npz", data=rel)
    np.savez_compressed(scene_dir / f"gt_poses_{scene_short}.npz", data=gt)
    print(f"已保存: pred_pose_bodyslam_{scene_short}.npz ({rel.shape}) + gt_poses_{scene_short}.npz ({gt.shape[0]},4,4)")

    gt_xyz = gt[:, :3, 3]
    pred_xyz = pred[:, :3, 3]
    ate_rmse, ate_mean, ate_std, scale = compute_ate(gt_xyz, pred_xyz)
    rpe_t, rpe_r = compute_rpe(gt, pred)

    print("\n================ 评估结果 ================")
    print(f"ATE RMSE : {ate_rmse:.4f} m ({ate_rmse*1000:.2f} mm)")
    print(f"ATE Mean : {ate_mean:.4f} m ({ate_mean*1000:.2f} mm) +/- {ate_std:.4f} m")
    print(f"估计尺度 : {scale:.4f}")
    print(f"RPE-T    : {rpe_t:.4f} m ({rpe_t*1000:.2f} mm)")
    print(f"RPE-R    : {rpe_r:.4f} deg")
    print("==========================================")


if __name__ == "__main__":
    main()
