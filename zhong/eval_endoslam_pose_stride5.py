"""Ours 在 EndoSLAM 上的位姿评估（stride=5 全量，Umeyama 7 自由度 ATE 口径）。

与 BodySLAM/AF-SfMLearner/Endo-FASt3r 在 EndoSLAM 上统一的口径：
- stride=5 取帧，全量（不限制帧对数）
- 图像按文件名排序（与 GT CSV 行一一对应）
- Umeyama 7 自由度 ATE + RPE-T + RPE-R
"""
import os, sys, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from networks import PoseCNN  # noqa: E402
from layers import transformation_from_parameters  # noqa: E402

NATIVE_W, NATIVE_H = 320, 320

K_UNITY = np.array([[156.0418, 0.0, 178.5604],
                    [0.0, 155.7529, 181.8043],
                    [0.0, 0.0, 1.0]], dtype=np.float32)

K_CANON = torch.tensor([
    [320.0, 0.0, 320.0],
    [0.0, 96.0, 96.0],
    [0.0, 0.0, 1.0]
], dtype=torch.float32)

SCENE_POSE_CSV = {
    "UnityCam/Colon": "colon_position_rotation.csv",
    "UnityCam/Small Intestine": "intestine_position_rotation.csv",
    "UnityCam/Stomach": "stomach_position_rotation.csv",
}


def _normalize_images_for_pose(images, K_phys):
    B, C, H, W = images.shape
    device = images.device
    K_can = K_CANON.to(device)
    K_phys_batch = K_phys.unsqueeze(0).to(device) if K_phys.dim() == 2 else K_phys
    y_idx = torch.arange(H, dtype=torch.float32, device=device).view(1, H, 1)
    x_idx = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, W)
    x_can_norm = (x_idx - K_can[0, 2]) / K_can[0, 0]
    y_can_norm = (y_idx - K_can[1, 2]) / K_can[1, 1]
    fx_phys = K_phys_batch[:, 0, 0].view(B, 1, 1)
    fy_phys = K_phys_batch[:, 1, 1].view(B, 1, 1)
    cx_phys = K_phys_batch[:, 0, 2].view(B, 1, 1)
    cy_phys = K_phys_batch[:, 1, 2].view(B, 1, 1)
    u_phys = (fx_phys * x_can_norm + cx_phys).expand(B, H, W)
    v_phys = (fy_phys * y_can_norm + cy_phys).expand(B, H, W)
    grid_x = 2.0 * u_phys / W - 1.0
    grid_y = 2.0 * v_phys / H - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(images, grid, mode="bilinear",
                         padding_mode="border", align_corners=False)


def _convert_pose_canonical_to_physical(axisangle, translation, K_phys):
    device = axisangle.device
    K_can = K_CANON.to(device)
    if translation.dim() == 4:
        translation = translation.squeeze(2)
        axisangle = axisangle.squeeze(2)
    B = axisangle.shape[0]
    K_phys_batch = K_phys.unsqueeze(0).to(device) if K_phys.dim() == 2 else K_phys
    fx_phys = K_phys_batch[:, 0, 0].view(B, 1, 1).clamp(min=50.0)
    fy_phys = K_phys_batch[:, 1, 1].view(B, 1, 1).clamp(min=50.0)
    sx = K_can[0, 0] / fx_phys
    sy = K_can[1, 1] / fy_phys
    translation_phys = translation.clone()
    translation_phys[:, :, 0:1] = translation[:, :, 0:1] * sx
    translation_phys[:, :, 1:2] = translation[:, :, 1:2] * sy
    return axisangle, translation_phys


@torch.no_grad()
def predict_relative_pose(img_path0, img_path1, pose_cnn, K_phys, device):
    def load(path):
        img = Image.open(path).convert("RGB").resize((NATIVE_W, NATIVE_H), Image.LANCZOS)
        return transforms.ToTensor()(img).unsqueeze(0).to(device)

    img0 = load(img_path0)
    img1 = load(img_path1)
    img0_can = _normalize_images_for_pose(img0, K_phys)
    img1_can = _normalize_images_for_pose(img1, K_phys)
    pose_input = torch.cat([img0_can, img1_can], dim=1)
    axisangle, translation = pose_cnn(pose_input)
    axisangle_phys, translation_phys = _convert_pose_canonical_to_physical(
        axisangle, translation, K_phys)
    T = transformation_from_parameters(axisangle_phys, translation_phys, invert=False)
    return T.squeeze(0).cpu().numpy()


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
    ap.add_argument("--weights", default=r"C:\Users\Administrator\tmp\endoslam_finetune\models\weights_4")
    ap.add_argument("--data_path", default=r"e:\data1\monodepth2\EndoSLAM")
    ap.add_argument("--scene", default="UnityCam/Colon")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pose = PoseCNN(num_input_frames=2)
    pose.load_state_dict(torch.load(os.path.join(args.weights, "pose.pth"), map_location=device))
    pose.to(device).eval()

    frames_dir = os.path.join(args.data_path, args.scene, "Frames_jpg")
    imgs = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                   if f.lower().endswith((".jpg", ".png"))])
    if args.max_frames > 0:
        imgs = imgs[:args.max_frames * args.stride]
    imgs = imgs[::args.stride]

    csv_path = os.path.join(args.data_path, args.scene, "Poses", SCENE_POSE_CSV[args.scene])
    gt = load_gt_poses(csv_path, stride=args.stride)
    if args.max_frames > 0:
        gt = gt[:args.max_frames]
    n = min(len(imgs), gt.shape[0])
    imgs = imgs[:n]
    gt = gt[:n]
    print(f"场景 {args.scene}: {n} 帧（stride={args.stride}）")

    K_t = torch.from_numpy(K_UNITY).float()
    pred_poses = []
    with torch.no_grad():
        for i in range(n - 1):
            T = predict_relative_pose(imgs[i], imgs[i + 1], pose, K_t, device)
            pred_poses.append(T)

    traj = [np.eye(4)]
    cam_to_world = np.eye(4)
    for T in pred_poses:
        cam_to_world = cam_to_world @ T
        traj.append(cam_to_world.copy())
    pred = np.array(traj)

    # 保存相对位姿 + GT 绝对位姿 npz（供 5 帧窗口 ATE 重算）
    scene_short = args.scene.split("/")[-1]
    out_dir = os.path.join(args.data_path, args.scene)
    np.savez_compressed(os.path.join(out_dir, f"pred_pose_ours_{scene_short}.npz"),
                        data=np.array(pred_poses))
    np.savez_compressed(os.path.join(out_dir, f"gt_poses_{scene_short}.npz"), data=gt)
    print(f"已保存: pred_pose_ours_{scene_short}.npz ({len(pred_poses)},4,4) + gt_poses_{scene_short}.npz ({gt.shape[0]},4,4)")

    gt_xyz = gt[:, :3, 3]
    pred_xyz = pred[:, :3, 3]
    ate_rmse, ate_mean, ate_std, scale = compute_ate(gt_xyz, pred_xyz)
    rpe_t, rpe_r = compute_rpe(gt, pred)

    print(f"\n============ Ours on EndoSLAM {args.scene} (stride={args.stride}) ============")
    print(f"ATE RMSE : {ate_rmse:.4f} m ({ate_rmse*1000:.2f} mm)")
    print(f"ATE Mean : {ate_mean:.4f} m +/- {ate_std:.4f} m")
    print(f"估计尺度 : {scale:.4f}")
    print(f"RPE-T    : {rpe_t:.4f} m ({rpe_t*1000:.2f} mm)")
    print(f"RPE-R    : {rpe_r:.4f} deg")
    print("=====================================================================")


if __name__ == "__main__":
    main()
