"""Endo-FASt3r (Reloc3r) 在 EndoSLAM 上的位姿评估（Umeyama 7 自由度 ATE 口径）。

流程：
1. 读 EndoSLAM 图像（Frames_jpg/image_XXXX.jpg），按 stride 采样
2. 读 GT 绝对位姿 CSV（tX,tY,tZ,rX,rY,rZ,rW，单位米）
3. Reloc3rX 预测相邻帧相对位姿（view0=当前帧, view1=下一帧）
4. 累积为绝对轨迹
5. Umeyama 7 自由度对齐 ATE + RPE-T + RPE-R
"""
import sys, os, csv, argparse
import numpy as np
import torch
from PIL import Image
import PIL
import torchvision.transforms as tvf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import networks  # noqa: E402  (Reloc3rX)


def prepare_pil(img, device, size=512):
    """与 evaluate_pose.py 的 prepare_images 等价，直接处理 PIL 图像。"""
    S = max(img.size)
    if S > size:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * size / S)) for x in img.size)
    img = img.resize(new_size, interp)
    W, H = img.size
    cx, cy = W // 2, H // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if W == H:  # square_ok=False
        halfh = 3 * halfw // 4
    img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
    ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return ImgNorm(img)[None].to(device)


def quat_xyzw_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_gt_poses(csv_path, stride=1):
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
    ap.add_argument("--pose_ckpt", default="best_weights/pose.pth")
    ap.add_argument("--reloc3r_ckpt", default="Reloc3r-512.pth")
    ap.add_argument("--data_path", default=r"e:\data1\monodepth2\EndoSLAM")
    ap.add_argument("--scene", default="UnityCam/Colon")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 图像列表
    frames_dir = os.path.join(args.data_path, args.scene, "Frames_jpg")
    imgs = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                   if f.lower().endswith((".jpg", ".png"))])
    if args.max_frames > 0:
        imgs = imgs[:args.max_frames * args.stride]
    imgs = imgs[::args.stride]

    # 2. GT
    csv_name = {"UnityCam/Colon": "colon_position_rotation.csv",
                "UnityCam/Small Intestine": "intestine_position_rotation.csv",
                "UnityCam/Stomach": "stomach_position_rotation.csv"}[args.scene]
    csv_path = os.path.join(args.data_path, args.scene, "Poses", csv_name)
    gt = load_gt_poses(csv_path, stride=args.stride)
    if args.max_frames > 0:
        gt = gt[:args.max_frames]
    n = min(len(imgs), gt.shape[0])
    imgs = imgs[:n]
    gt = gt[:n]
    print(f"场景 {args.scene}: {n} 帧（stride={args.stride}）")

    # 3. 模型
    pose_model = networks.Reloc3rX(args.reloc3r_ckpt)
    pose_model_dict = torch.load(args.pose_ckpt, map_location=device)
    model_dict = pose_model.state_dict()
    pose_model.load_state_dict({k: v for k, v in pose_model_dict.items() if k in model_dict})
    pose_model.to(device).eval()

    # 4. 推理相对位姿
    pred_poses = []
    with torch.no_grad():
        for i in range(n - 1):
            img0 = Image.open(imgs[i]).convert("RGB")
            img1 = Image.open(imgs[i + 1]).convert("RGB")
            view0 = {'img': prepare_pil(img0, device)}
            view1 = {'img': prepare_pil(img1, device)}
            pose2, _ = pose_model(view0, view1)
            pred_poses.append(pose2["pose"].cpu().numpy()[0])

    # 5. 累积绝对轨迹
    traj = [np.eye(4)]
    cam_to_world = np.eye(4)
    for T in pred_poses:
        cam_to_world = cam_to_world @ T
        traj.append(cam_to_world.copy())
    pred = np.array(traj)

    # 5.5 保存相对位姿 + GT 绝对位姿 npz（供 5 帧窗口 ATE 重算）
    scene_short = args.scene.split("/")[-1]
    out_dir = os.path.join(args.data_path, args.scene)
    np.savez_compressed(os.path.join(out_dir, f"pred_pose_endofast3r_{scene_short}.npz"),
                        data=np.array(pred_poses))
    np.savez_compressed(os.path.join(out_dir, f"gt_poses_{scene_short}.npz"), data=gt)
    print(f"已保存: pred_pose_endofast3r_{scene_short}.npz ({len(pred_poses)},4,4) + gt_poses_{scene_short}.npz ({gt.shape[0]},4,4)")

    # 6. 评估
    gt_xyz = gt[:, :3, 3]
    pred_xyz = pred[:, :3, 3]
    ate_rmse, ate_mean, ate_std, scale = compute_ate(gt_xyz, pred_xyz)
    rpe_t, rpe_r = compute_rpe(gt, pred)

    print(f"\n===== Endo-FASt3r on EndoSLAM {args.scene} =====")
    print(f"ATE RMSE : {ate_rmse:.4f} m ({ate_rmse*1000:.2f} mm)")
    print(f"ATE Mean : {ate_mean:.4f} m +/- {ate_std:.4f} m")
    print(f"估计尺度 : {scale:.4f}")
    print(f"RPE-T    : {rpe_t:.4f} m ({rpe_t*1000:.2f} mm)")
    print(f"RPE-R    : {rpe_r:.4f} deg")
    print("==================================================")


if __name__ == "__main__":
    main()
