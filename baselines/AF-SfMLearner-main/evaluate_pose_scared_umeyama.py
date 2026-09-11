"""AF-SfMLearner 在 SCARED 上的位姿评估（Umeyama 7 自由度 ATE 口径）。

与官方 evaluate_pose.py 相同的推理流程，但按 Ours/BodySLAM 一致的
Umeyama 7 自由度全轨迹对齐口径计算 ATE + RPE。
"""
import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networks  # noqa: E402
from layers import transformation_from_parameters  # noqa: E402
from utils import readlines  # noqa: E402
from datasets import SCAREDRAWDataset  # noqa: E402


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
    ap.add_argument("--weights_folder", default="weights_extracted/Model_MIA")
    ap.add_argument("--data_path",
                    default=r"e:\data1\monodepth2\baselines\Endo_FASt3r\SCARED_Images_Resized")
    ap.add_argument("--seq", type=int, default=1, choices=[1, 2])
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filenames = readlines(os.path.join("splits", "endovis",
                                       f"test_files_sequence{args.seq}.txt"))
    dataset = SCAREDRAWDataset(args.data_path, filenames, args.height, args.width,
                               [0, 1], 4, is_train=False)
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)

    pose_encoder = networks.ResnetEncoder(18, False, 2)
    pose_encoder.load_state_dict(
        torch.load(os.path.join(args.weights_folder, "pose_encoder.pth"), map_location=device))
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(
        torch.load(os.path.join(args.weights_folder, "pose.pth"), map_location=device))
    pose_encoder.to(device).eval()
    pose_decoder.to(device).eval()

    pred_poses = []
    print("-> Computing pose predictions")
    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.to(device)
            all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)
            features = [pose_encoder(all_color_aug)]
            axisangle, translation = pose_decoder(features)
            pred_poses.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)
    np.savez_compressed(os.path.join("splits", "endovis", f"pred_pose_afsl_sq{args.seq}.npz"),
                        data=pred_poses)
    print(f"预测相对位姿 shape: {pred_poses.shape}")

    # GT 绝对位姿
    gt = np.load(os.path.join("splits", "endovis", f"gt_poses_sq{args.seq}.npz"),
                 allow_pickle=True)["data"]

    # 累积绝对轨迹
    traj = [np.eye(4)]
    c2w = np.eye(4)
    for T in pred_poses:
        c2w = c2w @ T
        traj.append(c2w.copy())
    pred = np.array(traj)

    n = min(pred.shape[0], gt.shape[0])
    pred = pred[:n]
    gt = gt[:n]

    gt_xyz = gt[:, :3, 3]
    pred_xyz = pred[:, :3, 3]
    ate_rmse, ate_mean, ate_std, scale = compute_ate(gt_xyz, pred_xyz)
    rpe_t, rpe_r = compute_rpe(gt, pred)

    print(f"\n========= AF-SfMLearner on SCARED seq{args.seq} (Umeyama) =========")
    print(f"ATE RMSE : {ate_rmse:.4f} m ({ate_rmse*1000:.2f} mm)")
    print(f"ATE Mean : {ate_mean:.4f} m +/- {ate_std:.4f} m")
    print(f"估计尺度 : {scale:.4f}")
    print(f"RPE-T    : {rpe_t:.4f} m ({rpe_t*1000:.2f} mm)")
    print(f"RPE-R    : {rpe_r:.4f} deg")
    print("=============================================================")


if __name__ == "__main__":
    main()
