"""评估 EndoSLAM 微调后模型: 深度精度 + 位姿精度。

用法:
    python zhong/eval_endoslam_finetune.py \
        --weights C:/Users/Administrator/tmp/endoslam_finetune/models/weights_4 \
        --split endoslam_full \
        --max_frames 0   # 0=全量 val (6793 帧), >0 时采样前 N 帧

深度评估:
    - 输入 320x320 (EndoSLAM 原生分辨率)
    - 预测深度(毫米) vs GT 深度 (Pixelwise Depths R 通道 ×10 mm)
    - 指标: AbsRel, SqRel, RMSE, logRMSE, a1/a2/a3

位姿评估:
    - PoseCNN 预测相邻帧相对位姿 (K_CANON 归一化, 与训练一致)
    - GT 相对位姿来自 Poses/*_position_rotation.csv (米→毫米)
    - 累积轨迹 + Umeyama 对齐, 输出 ATE(平移 RMSE) + 平均旋转误差(deg)
"""
from __future__ import absolute_import, division, print_function

import os, sys, csv, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

torch.backends.cudnn.benchmark = True

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from networks import ResnetEncoder, DepthDecoder, PoseCNN, MotionEncoder
from layers import make_bin_centers, depth_from_bins, transformation_from_parameters

# EndoSLAM UnityCam 内参 (320x320 原生)
K_UNITY = np.array([[156.0418, 0.0, 178.5604],
                    [0.0, 155.7529, 181.8043],
                    [0.0, 0.0, 1.0]], dtype=np.float32)
NATIVE_W, NATIVE_H = 320, 320
DEPTH_SCALE = 10.0  # EndoSLAM 深度 R 通道: 厘米 -> 毫米

# 与 trainer.py 一致的 K_CANON (位姿标准相机)
K_CANON = torch.tensor([
    [320.0, 0.0, 320.0],
    [0.0, 96.0, 96.0],
    [0.0, 0.0, 1.0]
], dtype=torch.float32)

# 场景 -> pose CSV 文件名 (注意 Small Intestine 含空格)
SCENE_POSE_CSV = {
    "UnityCam/Colon": "colon_position_rotation.csv",
    "UnityCam/Small Intestine": "intestine_position_rotation.csv",
    "UnityCam/Stomach": "stomach_position_rotation.csv",
}


def load_model(weights_folder, device):
    """加载微调模型: ResNet18 + DepthDecoder(64bin) + PoseCNN + MotionEncoder."""
    encoder = ResnetEncoder(18, False)
    state = torch.load(os.path.join(weights_folder, "encoder.pth"), map_location=device)
    for k in ["height", "width", "use_stereo"]:
        state.pop(k, None)
    encoder.load_state_dict(state)
    encoder.eval().to(device)

    depth_state = torch.load(os.path.join(weights_folder, "depth.pth"), map_location=device)
    num_bins = 0
    for k in depth_state.keys():
        if ".10.conv.weight" in k and depth_state[k].shape[0] > 1:
            num_bins = depth_state[k].shape[0]
            break
    if num_bins == 0:
        num_bins = 64
    decoder = DepthDecoder(encoder.num_ch_enc, [0], num_bins=num_bins)
    decoder.load_state_dict(depth_state, strict=False)
    decoder.eval().to(device)

    pose = PoseCNN(num_input_frames=2)
    pose_path = os.path.join(weights_folder, "pose.pth")
    if os.path.isfile(pose_path):
        pose.load_state_dict(torch.load(pose_path, map_location=device))
    pose.eval().to(device)

    motion = None
    motion_path = os.path.join(weights_folder, "motion_encoder.pth")
    if os.path.isfile(motion_path):
        motion = MotionEncoder()
        motion.load_state_dict(torch.load(motion_path, map_location=device))
        motion.eval().to(device)

    return encoder, decoder, pose, motion, num_bins


@torch.no_grad()
def predict_depth(img_path, encoder, decoder, device, num_bins,
                  min_depth=1.0, max_depth=500.0):
    """预测单帧深度 (毫米), 输出 [H, W] numpy."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((NATIVE_W, NATIVE_H), Image.LANCZOS)
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    features = encoder(x)
    outputs = decoder(features)

    bin_logits = outputs[("bins", 0)]
    residual = outputs.get(("residual", 0), None)
    bin_centers = make_bin_centers(min_depth, max_depth, num_bins)
    depth = depth_from_bins(bin_logits, bin_centers, residual)
    return depth.squeeze().cpu().numpy()


def _normalize_images_for_pose(images, K_phys):
    """物理相机图像 -> K_CANON 标准相机空间 (与 trainer.py 方案B 一致)."""
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
    """标准相机位姿 -> 物理相机位姿 (与 trainer.py 一致)."""
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
    """PoseCNN 预测 frame0 -> frame1 相对位姿 [4,4] (物理相机空间, 毫米)."""
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
    # axisangle_phys/translation_phys 形状均为 [B,1,3], 直接传入(保持 Bx1x3, 满足 rot_from_axisangle 要求)
    T = transformation_from_parameters(axisangle_phys, translation_phys, invert=False)
    return T.squeeze(0).cpu().numpy()


# ── GT 位姿读取 ────────────────────────────────────────────
def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def load_gt_poses(csv_path):
    """读取 EndoSLAM pose CSV -> 列表 [(R_world<-cam, t_world)] (米)."""
    poses = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                tx, ty, tz = float(row["tX"]), float(row["tY"]), float(row["tZ"])
                rx, ry, rz, rw = (float(row["rX"]), float(row["rY"]),
                                  float(row["rZ"]), float(row["rW"]))
            except (TypeError, ValueError):
                continue
            R = quat_to_R(rx, ry, rz, rw)
            poses.append((R, np.array([tx, ty, tz])))
    return poses


def gt_relative_pose(gt_poses, i, j):
    """GT 相对位姿: 相机 i -> 相机 j (把 j 的点变换到 i 坐标系). 返回 [4,4] (毫米)."""
    Ri, ti = gt_poses[i]
    Rj, tj = gt_poses[j]
    R_rel = Ri.T @ Rj
    t_rel = Ri.T @ (tj - ti)
    T = np.eye(4)
    T[:3, :3] = R_rel
    T[:3, 3] = t_rel * 1000.0  # 米 -> 毫米
    return T


# ── 指标 ──────────────────────────────────────────────────
def compute_depth_metrics(gt_mm, pred_mm):
    mask = (gt_mm > 1.0) & (gt_mm < 500.0) & (pred_mm > 0) & np.isfinite(pred_mm)
    g = gt_mm[mask]
    p = pred_mm[mask]
    if len(g) < 1000:
        return None
    # 单目尺度不确定, 按 GT 中位数对齐 (标准做法)
    scale = np.median(g) / np.median(p)
    p_s = p * scale
    absrel = np.mean(np.abs(g - p_s) / g)
    sqrel = np.mean((g - p_s) ** 2 / g)
    rmse = np.sqrt(np.mean((g - p_s) ** 2))
    logrmse = np.sqrt(np.mean((np.log(g) - np.log(p_s)) ** 2))
    ratio = np.maximum(g / p_s, p_s / g)
    a1 = np.mean(ratio < 1.25)
    a2 = np.mean(ratio < 1.25 ** 2)
    a3 = np.mean(ratio < 1.25 ** 3)
    return dict(absrel=absrel, sqrel=sqrel, rmse=rmse, logrmse=logrmse,
                a1=a1, a2=a2, a3=a3, scale=scale, n=int(len(g)))


def umeyama_alignment(src, dst):
    """Umeyama 相似变换对齐 src->dst (允许缩放). 返回变换后 src."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = src_c.T @ dst_c / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = np.trace(np.diag(S)) / np.sum(src_c ** 2)
    t = mu_d - scale * R @ mu_s
    return (scale * R @ src.T).T + t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data_path", type=str,
                    default=os.path.join(_project_dir, "EndoSLAM"))
    ap.add_argument("--split", type=str, default="endoslam_full")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="0=全量 val; >0 时每个场景只评估前 N 帧")
    ap.add_argument("--pose_stride", type=int, default=10,
                    help="位姿评估的帧间隔 (EndoSLAM 帧率高, 间隔取帧以累积明显运动)")
    ap.add_argument("--max_pose_pairs", type=int, default=300,
                    help="每个场景位姿评估的最大帧对数")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"  EndoSLAM 微调模型评估: {args.weights}")
    print(f"  Device: {device}")
    print("=" * 70)

    encoder, decoder, pose, motion, num_bins = load_model(args.weights, device)
    print(f"  模型加载成功 (num_bins={num_bins})")

    val_file = os.path.join(_project_dir, "splits", args.split, "val_files.txt")
    with open(val_file) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # 按场景分组
    scene_frames = {}
    for ln in lines:
        parts = ln.rsplit(None, 1)
        scene = parts[0]
        fid = int(parts[1]) if len(parts) == 2 else None
        if fid is None:
            continue
        scene_frames.setdefault(scene, []).append(fid)

    K_t = torch.from_numpy(K_UNITY).float()

    # ── 深度评估 ──────────────────────────────────────
    print("\n[1/2] 深度评估 ...")
    all_depth = []
    for scene, fids in sorted(scene_frames.items()):
        fids = sorted(fids)
        if args.max_frames > 0:
            fids = fids[:args.max_frames]
        rgb_dir = os.path.join(args.data_path, scene, "Frames_jpg")
        depth_dir = os.path.join(args.data_path, scene, "Pixelwise Depths")
        scene_met = []
        for fid in fids:
            rgb_path = os.path.join(rgb_dir, f"image_{fid:04d}.jpg")
            depth_path = os.path.join(depth_dir, f"aov_image_{fid:04d}.png")
            if not (os.path.isfile(rgb_path) and os.path.isfile(depth_path)):
                continue
            gt = np.array(Image.open(depth_path))[:, :, 0].astype(np.float32) * DEPTH_SCALE
            pred = predict_depth(rgb_path, encoder, decoder, device, num_bins)
            m = compute_depth_metrics(gt, pred)
            if m:
                scene_met.append(m)
        if scene_met:
            agg = {k: float(np.mean([m[k] for m in scene_met])) for k in
                   ["absrel", "sqrel", "rmse", "logrmse", "a1", "a2", "a3"]}
            agg["n"] = len(scene_met)
            all_depth.append((scene, agg))
            print(f"    {scene:35s} n={agg['n']:5d}  AbsRel={agg['absrel']:.4f} "
                  f"RMSE={agg['rmse']:.2f}mm  a1={agg['a1']:.4f}")

    if all_depth:
        total = {k: float(np.mean([a[k] for _, a in all_depth])) for k in
                 ["absrel", "sqrel", "rmse", "logrmse", "a1", "a2", "a3"]}
        print(f"    {'(平均)':35s} AbsRel={total['absrel']:.4f} "
              f"RMSE={total['rmse']:.2f}mm  a1={total['a1']:.4f}")

    # ── 位姿评估 ──────────────────────────────────────
    print("\n[2/2] 位姿评估 (PoseCNN vs GT, Umeyama 对齐) ...")
    for scene, fids in sorted(scene_frames.items()):
        csv_path = os.path.join(args.data_path, scene, "Poses",
                                SCENE_POSE_CSV.get(scene, ""))
        if not os.path.isfile(csv_path):
            print(f"    {scene}: 无 pose CSV, 跳过")
            continue
        gt_poses = load_gt_poses(csv_path)
        fids = sorted(fids)
        rgb_dir = os.path.join(args.data_path, scene, "Frames_jpg")

        # 以 stride 取帧对
        pairs = []
        for i in range(0, len(fids) - args.pose_stride, args.pose_stride):
            fi, fj = fids[i], fids[i + args.pose_stride]
            if fj >= len(gt_poses):
                break
            pairs.append((fi, fj))
            if len(pairs) >= args.max_pose_pairs:
                break
        if len(pairs) < 5:
            print(f"    {scene}: 有效帧对不足, 跳过")
            continue

        pred_Ts, gt_Ts = [], []
        for fi, fj in pairs:
            p0 = os.path.join(rgb_dir, f"image_{fi:04d}.jpg")
            p1 = os.path.join(rgb_dir, f"image_{fj:04d}.jpg")
            if not (os.path.isfile(p0) and os.path.isfile(p1)):
                continue
            T_pred = predict_relative_pose(p0, p1, pose, K_t, device)
            T_gt = gt_relative_pose(gt_poses, fi, fj)
            pred_Ts.append(T_pred)
            gt_Ts.append(T_gt)

        if len(pred_Ts) < 5:
            print(f"    {scene}: 可加载帧对不足")
            continue

        # 累积轨迹 (相对位姿连乘)
        def accumulate(Ts):
            traj = [np.zeros(3)]
            T = np.eye(4)
            for Ti in Ts:
                T = T @ Ti
                traj.append(T[:3, 3])
            return np.array(traj)

        pred_traj = accumulate(pred_Ts)
        gt_traj = accumulate(gt_Ts)
        pred_aligned = umeyama_alignment(pred_traj, gt_traj)
        ate = np.sqrt(np.mean(np.sum((pred_aligned - gt_traj) ** 2, axis=1)))

        # 相对旋转误差 (deg)
        rot_errs = []
        for Tp, Tg in zip(pred_Ts, gt_Ts):
            R_err = Tp[:3, :3] @ Tg[:3, :3].T
            ang = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)) * 180 / np.pi
            rot_errs.append(ang)
        rot_mean = float(np.mean(rot_errs))

        print(f"    {scene:35s} 帧对={len(pred_Ts):4d}  ATE={ate:7.2f}mm  "
              f"旋转误差={rot_mean:5.2f}deg")

    print("\n评估完成。 ALL_EVAL_DONE")


if __name__ == "__main__":
    main()
