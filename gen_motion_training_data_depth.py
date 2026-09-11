"""
生成深度+光流训练数据
====================
输入从 RGB patch → 深度 patch + 稠密光流 patch。
输出: 每帧对一个 .npz 文件，含 depth_i, depth_j, flow_x, flow_y, labels 等。
"""
import os, sys, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from v6_pipeline.utils import init_loftr_matcher, _match_loftr_pair
from v6_pipeline.logger import setup_logger

logger = setup_logger(__name__)

# ── Config ──
MAX_SAMPLES_PER_PAIR = 2000
PATCH_SIZE = 16
LOFTR_MAX_DIM = 840
DEPTH_MAX_MM = 200.0          # 深度归一化上限
FLOW_MAX_PX = 2.0             # 光流归一化上限（结肠镜帧间运动小）
K = np.array([[767.73, 0, 677.74], [0, 767.73, 543.06], [0, 0, 1]], dtype=np.float64)


def load_pose_matrix(pose_txt_path, frame_idx):
    with open(pose_txt_path) as f:
        lines = f.readlines()
    if frame_idx >= len(lines):
        return None
    vals = [float(p) for p in lines[frame_idx].strip().split(',')]
    return np.array(vals).reshape(4, 4, order='F')


def load_gt_vertex_masks(gt_dir, frame_idx):
    mask_path = os.path.join(gt_dir, 'generated', 'mask_static', f'frame_{frame_idx:04d}.npy')
    motion_path = os.path.join(gt_dir, 'generated', 'motion_gt', f'frame_{frame_idx:04d}.npy')
    vertex_path = os.path.join(gt_dir, 'generated', 'vertex_static', f'frame_{frame_idx:04d}.npy')
    if not os.path.exists(mask_path):
        return None, None, None
    return np.load(mask_path), np.load(motion_path), np.load(vertex_path)


def load_depth_mm(data_dir, frame_idx):
    """加载 GT 深度 TIFF，转为 mm。C3VDv2: uint16, 0-65535 → 0-100mm."""
    depth_path = os.path.join(data_dir, 'depth', f'{frame_idx:04d}_depth.tiff')
    if not os.path.exists(depth_path):
        return None
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    return depth.astype(np.float32) * (100.0 / 65535.0)


def reproject_to_3d_depth(pts_2d, depth_map, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = depth_map.shape
    valid = depth_map > 0
    depth_median = float(np.median(depth_map[valid])) if np.any(valid) else 50.0

    pts_3d = np.zeros((len(pts_2d), 3), dtype=np.float32)
    valid_mask = np.zeros(len(pts_2d), dtype=bool)
    for k, pt in enumerate(pts_2d):
        u, v = pt
        ui = int(np.clip(np.round(u), 0, w - 1))
        vi = int(np.clip(np.round(v), 0, h - 1))
        Z = depth_map[vi, ui]
        if Z <= 0:
            Z = depth_median
        pts_3d[k] = [(u - cx) * Z / fx, (v - cy) * Z / fy, Z]
        valid_mask[k] = True
    return pts_3d, valid_mask


def classify_by_gt(pts_3d_cam, gt_vertices_world, T_cam_to_world, mask_static, motion_gt):
    """用 GT 位姿匹配 3D 点到 mesh 顶点，获取运动标签。motion_gt 单位：米→毫米。"""
    from scipy.spatial import cKDTree
    R = T_cam_to_world[:3, :3]
    t = T_cam_to_world[:3, 3]
    pts_3d_world = (R @ pts_3d_cam.T).T + t.reshape(1, 3)
    tree = cKDTree(gt_vertices_world)
    dists, idxs = tree.query(pts_3d_world)
    gt_labels = (mask_static[idxs] <= 0.5).astype(np.uint8)
    gt_displacements = np.linalg.norm(motion_gt[idxs], axis=1) * 1000.0
    return gt_labels, gt_displacements, dists


def compute_sequence_stats(data_dir, frame_paths, start_idx, end_idx, pose_txt_path,
                           num_sample_frames=50):
    """对序列采样帧，分别计算 depth 和 残差光流 的 mean/std.

    残差光流 = 总光流(Farneback) - 相机光流(pose+depth重投影)
    这是组织运动的纯净信号。

    返回: (depth_mean, depth_std, residual_flow_mean_x, residual_flow_std_x,
           residual_flow_mean_y, residual_flow_std_y)
    """
    n_frames = end_idx - start_idx
    if n_frames <= 1:
        return 50.0, 30.0, 0.0, 1.0, 0.0, 1.0  # 默认值

    # 均匀采样帧索引
    sample_indices = np.linspace(start_idx, end_idx - 1, min(num_sample_frames, n_frames), dtype=int)
    sample_indices = np.unique(np.clip(sample_indices, 0, len(frame_paths) - 2))

    all_depth_vals = []
    all_rfx_vals = []
    all_rfy_vals = []

    rgb_dir = os.path.join(data_dir, 'rgb')

    for idx in sample_indices:
        # 加载深度
        depth = load_depth_mm(data_dir, idx)
        if depth is None:
            continue
        # 只收集有效深度 (0 < depth < 100mm)
        d_flat = depth.ravel()
        valid_d = d_flat[(d_flat > 0) & (d_flat < 100.0)]
        all_depth_vals.append(valid_d)

        # 加载位姿，计算相对变换
        pose_i = load_pose_matrix(pose_txt_path, idx)
        pose_j = load_pose_matrix(pose_txt_path, idx + 1)
        if pose_i is None or pose_j is None:
            continue
        T_rel = np.linalg.inv(pose_i) @ pose_j

        # 相机光流
        cam_fx, cam_fy = compute_camera_flow_dense(depth, T_rel, K)

        # 总光流 (Farneback)
        rgb_i_path = os.path.join(rgb_dir, f'{idx:04d}.png')
        rgb_j_path = os.path.join(rgb_dir, f'{idx+1:04d}.png')
        if not os.path.exists(rgb_i_path) or not os.path.exists(rgb_j_path):
            continue
        rgb_i = cv2.imread(rgb_i_path)
        rgb_j = cv2.imread(rgb_j_path)
        if rgb_i is None or rgb_j is None:
            continue
        rgb_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2RGB)
        rgb_j = cv2.cvtColor(rgb_j, cv2.COLOR_BGR2RGB)
        total_fx, total_fy = compute_dense_flow(rgb_i, rgb_j)

        # 残差光流 = 总流 - 相机流
        rfx = total_fx - cam_fx
        rfy = total_fy - cam_fy

        # 收集非零残差流像素
        rfx_flat = rfx.ravel()
        rfy_flat = rfy.ravel()
        nonzero = (rfx_flat != 0) | (rfy_flat != 0)
        all_rfx_vals.append(rfx_flat[nonzero])
        all_rfy_vals.append(rfy_flat[nonzero])

    if not all_depth_vals:
        return 50.0, 30.0, 0.0, 1.0, 0.0, 1.0

    # 计算统计量（过滤极端值）
    depth_cat = np.concatenate(all_depth_vals)
    d_lo, d_hi = np.percentile(depth_cat, [1, 99])
    depth_cat = depth_cat[(depth_cat >= d_lo) & (depth_cat <= d_hi)]
    mean_depth = float(np.mean(depth_cat))
    std_depth = float(np.std(depth_cat))

    if all_rfx_vals:
        rfx_cat = np.concatenate(all_rfx_vals)
        rfy_cat = np.concatenate(all_rfy_vals)
        # 分别裁掉 1% / 99% 分位数
        rfx_lo, rfx_hi = np.percentile(rfx_cat, [1, 99])
        rfy_lo, rfy_hi = np.percentile(rfy_cat, [1, 99])
        rfx_cat = rfx_cat[(rfx_cat >= rfx_lo) & (rfx_cat <= rfx_hi)]
        rfy_cat = rfy_cat[(rfy_cat >= rfy_lo) & (rfy_cat <= rfy_hi)]
        mean_rfx = float(np.mean(rfx_cat))
        std_rfx = float(np.std(rfx_cat))
        mean_rfy = float(np.mean(rfy_cat))
        std_rfy = float(np.std(rfy_cat))
    else:
        mean_rfx = std_rfx = mean_rfy = 0.0
        std_rfx = std_rfy = 1.0

    # 安全下限
    if std_depth < 0.01:
        std_depth = 0.01
    if std_rfx < 0.01:
        std_rfx = 0.01
    if std_rfy < 0.01:
        std_rfy = 0.01

    logger.info(f'  序列统计: depth mean={mean_depth:.1f} std={std_depth:.1f}mm, '
                f'residual_flow_x mean={mean_rfx:.2f} std={std_rfx:.2f}px, '
                f'residual_flow_y mean={mean_rfy:.2f} std={std_rfy:.2f}px')
    return mean_depth, std_depth, mean_rfx, std_rfx, mean_rfy, std_rfy


def compute_dense_flow(rgb_i, rgb_j):
    """Farneback 稠密光流：两帧 RGB → flow_x, flow_y (H, W) float32."""
    gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_RGB2GRAY)
    gray_j = cv2.cvtColor(rgb_j, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_i, gray_j, None, 0.5, 5, 25, 5, 7, 1.5,
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow[..., 0], flow[..., 1]  # flow_x, flow_y


def compute_camera_flow_dense(depth_map, T_rel, K):
    """相机运动引起的稠密光流 (全图)。

    用位姿 + 深度重投影：frame i 的每个像素 → 3D → 变换到 frame j 相机坐标系 → 投影 → 位移。

    Args:
        depth_map: (H, W) float32, 单位 mm
        T_rel: 4x4 相对位姿 (i→j)
        K: 3x3 相机内参
    Returns:
        flow_u, flow_v: (H, W) float32, 相机运动引起的像素位移 (px)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    R = T_rel[:3, :3]
    t = T_rel[:3, 3]
    H, W = depth_map.shape

    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    Z = depth_map.astype(np.float64)
    valid = Z > 1e-6

    if valid.sum() == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    # 反投影到 frame i 相机坐标系
    X = (uu[valid] - cx) * Z[valid] / fx
    Y = (vv[valid] - cy) * Z[valid] / fy
    P = np.stack([X, Y, Z[valid]], axis=-1)  # (N, 3)

    # 变换到 frame j 相机坐标系
    P_new = (R @ P.T + t.reshape(3, 1)).T  # (N, 3)
    Z_new = P_new[:, 2]
    ok = Z_new > 1e-6

    # 投影到 frame j 图像平面
    u_proj = np.zeros(len(P), dtype=np.float32)
    v_proj = np.zeros(len(P), dtype=np.float32)
    u_proj[ok] = fx * P_new[ok, 0] / Z_new[ok] + cx
    v_proj[ok] = fy * P_new[ok, 1] / Z_new[ok] + cy

    idxs = np.where(valid.ravel())[0]
    flow_u = np.zeros(H * W, dtype=np.float32)
    flow_v = np.zeros(H * W, dtype=np.float32)
    flow_u[idxs] = u_proj - uu[valid]
    flow_v[idxs] = v_proj - vv[valid]

    return flow_u.reshape(H, W), flow_v.reshape(H, W)


def extract_depth_patches(depth_i, depth_j, pts_i, patch_size=16,
                          mean_depth=50.0, std_depth=30.0):
    """从深度图提取 16×16 patches，序列级归一化: (depth - mean) / std."""
    H, W = depth_i.shape[:2]
    half = patch_size // 2
    N = len(pts_i)
    patches_i = np.zeros((N, patch_size, patch_size), dtype=np.float32)
    patches_j = np.zeros((N, patch_size, patch_size), dtype=np.float32)

    for k in range(N):
        x0 = int(np.clip(np.round(pts_i[k, 0]) - half, 0, W - 1))
        y0 = int(np.clip(np.round(pts_i[k, 1]) - half, 0, H - 1))
        x1 = min(x0 + patch_size, W)
        y1 = min(y0 + patch_size, H)
        x0 = max(0, x1 - patch_size)
        y0 = max(0, y1 - patch_size)

        patches_i[k] = (depth_i[y0:y1, x0:x1] - mean_depth) / std_depth
        patches_j[k] = (depth_j[y0:y1, x0:x1] - mean_depth) / std_depth

    return patches_i, patches_j


def extract_flow_patches(flow_x, flow_y, pts_i, patch_size=16,
                        mean_fx=0.0, std_fx=1.0,
                        mean_fy=0.0, std_fy=1.0):
    """从稠密光流场提取 16×16 patches，序列级归一化: (flow - mean) / std."""
    H, W = flow_x.shape[:2]
    half = patch_size // 2
    N = len(pts_i)
    fx_patches = np.zeros((N, patch_size, patch_size), dtype=np.float32)
    fy_patches = np.zeros((N, patch_size, patch_size), dtype=np.float32)

    for k in range(N):
        px = int(np.clip(np.round(pts_i[k, 0]), 0, W - 1))
        py = int(np.clip(np.round(pts_i[k, 1]), 0, H - 1))
        x0 = max(0, px - half)
        y0 = max(0, py - half)
        x1 = min(W, px + half)
        y1 = min(H, py + half)
        x0 = max(0, x1 - patch_size)
        y0 = max(0, y1 - patch_size)

        fx_patches[k] = (flow_x[y0:y1, x0:x1] - mean_fx) / std_fx
        fy_patches[k] = (flow_y[y0:y1, x0:x1] - mean_fy) / std_fy

    return fx_patches, fy_patches


def process_frame_pairs(data_dir, output_dir, start_idx, end_idx,
                        frame_paths, pose_txt_path, matcher, device,
                        prefix='',
                        mean_depth=50.0, std_depth=30.0,
                        mean_fx=0.0, std_fx=1.0,
                        mean_fy=0.0, std_fy=1.0,
                        stats=None):
    """处理帧对：LoFTR匹配 → GT标签 → 深度patch + 光流patch.

    stats: dict 可选，含 depth_stats=[mean, std], flow_stats=[mean_x, std_x, mean_y, std_y]，
           会写入每个 npz。
    """
    os.makedirs(output_dir, exist_ok=True)

    total_samples = 0
    total_moving = 0
    n_pairs = end_idx - start_idx

    # 预检查 rgb_warped 是否存在
    rgb_warped_dir = os.path.join(data_dir, 'generated', 'rgb_warped')
    use_rgb_warped = os.path.exists(rgb_warped_dir)

    for pair_i, frame_idx in enumerate(range(start_idx, end_idx)):
        if pair_i % 25 == 0:
            logger.info(f'  [{pair_i}/{n_pairs}] frame {frame_idx}...')

        # ── LoFTR 匹配 ──
        pts0, pts1, conf = _match_loftr_pair(
            matcher, frame_paths[frame_idx], frame_paths[frame_idx + 1],
            device=device, max_dim=LOFTR_MAX_DIM)
        if pts0 is None or len(pts0) < 16:
            continue

        pts0 = np.array(pts0, dtype=np.float32)
        pts1 = np.array(pts1, dtype=np.float32)

        # ── 降采样 ──
        n_pts = len(pts0)
        if n_pts > MAX_SAMPLES_PER_PAIR:
            indices = np.random.choice(n_pts, MAX_SAMPLES_PER_PAIR, replace=False)
            pts0 = pts0[indices]
            pts1 = pts1[indices]
            n_pts = MAX_SAMPLES_PER_PAIR

        # ── 加载深度 ──
        depth_i = load_depth_mm(data_dir, frame_idx)
        depth_j = load_depth_mm(data_dir, frame_idx + 1)
        if depth_i is None or depth_j is None:
            continue

        # ── 3D 反投影 + GT 标签 ──
        pts_3d_cam, valid_3d = reproject_to_3d_depth(pts0, depth_i, K)
        gt_mask, gt_motion, gt_vertices = load_gt_vertex_masks(data_dir, frame_idx)
        T_wc_gt = load_pose_matrix(pose_txt_path, frame_idx)
        if gt_mask is None or T_wc_gt is None:
            continue

        gt_labels, gt_disps, vertex_dists = classify_by_gt(
            pts_3d_cam, gt_vertices, T_wc_gt, gt_mask, gt_motion)

        # ── 加载原始 RGB（用于稠密光流，帧间运动更大） ──
        rgb_dir = os.path.join(data_dir, 'rgb')
        rgb_i_path = os.path.join(rgb_dir, f'{frame_idx:04d}.png')
        rgb_j_path = os.path.join(rgb_dir, f'{frame_idx+1:04d}.png')
        if not os.path.exists(rgb_i_path) or not os.path.exists(rgb_j_path):
            # Fallback to rgb_warped
            rgb_i_path = frame_paths[frame_idx]
            rgb_j_path = frame_paths[frame_idx + 1]
        rgb_i = cv2.imread(rgb_i_path)
        rgb_j = cv2.imread(rgb_j_path)
        if rgb_i is None or rgb_j is None:
            continue
        rgb_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2RGB)
        rgb_j = cv2.cvtColor(rgb_j, cv2.COLOR_BGR2RGB)

        # ── 稠密光流（Farneback: 总流 = 相机运动 + 组织运动）──
        flow_x, flow_y = compute_dense_flow(rgb_i, rgb_j)
        
        # ── 相机运动补偿：残差光流 = 总流 - 相机流（纯组织运动信号）──
        T_wc_i = load_pose_matrix(pose_txt_path, frame_idx)
        T_wc_j = load_pose_matrix(pose_txt_path, frame_idx + 1)
        if T_wc_i is not None and T_wc_j is not None:
            T_rel = np.linalg.inv(T_wc_i) @ T_wc_j
            cam_fx, cam_fy = compute_camera_flow_dense(depth_i, T_rel, K)
            flow_x = flow_x - cam_fx  # 残差流 x
            flow_y = flow_y - cam_fy  # 残差流 y

        # 光流坐标需与 LoFTR 点坐标对齐，缩放因子
        H_raw, W_raw = flow_x.shape
        H_warp, W_warp, _ = cv2.imread(frame_paths[0]).shape if frame_paths else (H_raw, W_raw)
        scale_x = W_raw / W_warp if W_warp != W_raw else 1.0
        scale_y = H_raw / H_warp if H_warp != H_raw else 1.0
        if scale_x != 1.0 or scale_y != 1.0:
            flow_x = cv2.resize(flow_x, (W_warp, H_warp)) * scale_x
            flow_y = cv2.resize(flow_y, (W_warp, H_warp)) * scale_y

        # ── 提取深度 patches ──
        depth_patches_i, depth_patches_j = extract_depth_patches(
            depth_i, depth_j, pts0, PATCH_SIZE, mean_depth, std_depth)

        # ── 提取光流 patches ──
        flow_fx, flow_fy = extract_flow_patches(
            flow_x, flow_y, pts0, PATCH_SIZE, mean_fx, std_fx, mean_fy, std_fy)

        # ── 稀疏流特征（LoFTR 匹配的 dx, dy） ──
        flow_dx = pts1[:, 0] - pts0[:, 0]
        flow_dy = pts1[:, 1] - pts0[:, 1]

        # ── 归一化位置 ──
        H, W = rgb_i.shape[:2]
        pos_x = pts0[:, 0] / W
        pos_y = pts0[:, 1] / H

        # ── 保存 ──
        out_name = f'{prefix}_pair_{frame_idx:04d}.npz' if prefix else f'pair_{frame_idx:04d}.npz'
        out_path = os.path.join(output_dir, out_name)
        np.savez_compressed(
            out_path,
            depth_i=depth_patches_i,                     # (N, 16, 16) float32, z-score归一化
            depth_j=depth_patches_j,                     # (N, 16, 16) float32, z-score归一化
            flow_x=flow_fx,                               # (N, 16, 16) float32, z-score归一化
            flow_y=flow_fy,                               # (N, 16, 16) float32, z-score归一化
            flow_dx=flow_dx.astype(np.float16),
            flow_dy=flow_dy.astype(np.float16),
            pos_x=pos_x.astype(np.float16),
            pos_y=pos_y.astype(np.float16),
            labels=gt_labels,
            displacements=gt_disps.astype(np.float32),
            depth_stats=np.array([mean_depth, std_depth], dtype=np.float32),
            flow_stats=np.array([mean_fx, std_fx, mean_fy, std_fy], dtype=np.float32),
        )

        total_samples += n_pts
        total_moving += int(np.sum(gt_labels))

    logger.info(f'  完成: {total_samples:,} 样本, moving={total_moving:,} '
                f'({100*total_moving/max(total_samples,1):.1f}%)')
    return total_samples, total_moving


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='数据集根目录 (含 rgb, depth, pose.txt, generated/)')
    parser.add_argument('--output_dir', required=True, help='输出 npz 目录')
    parser.add_argument('--start_frame', type=int, required=True)
    parser.add_argument('--end_frame', type=int, required=True)
    parser.add_argument('--prefix', default='', help='文件名前缀')
    parser.add_argument('--max_samples', type=int, default=2000)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    start_idx = args.start_frame
    end_idx = args.end_frame

    # 优先用 rgb_warped，fallback 到 rgb
    rgb_dir = os.path.join(data_dir, 'generated', 'rgb_warped')
    if not os.path.exists(rgb_dir):
        rgb_dir = os.path.join(data_dir, 'rgb')
    frame_files = sorted([f for f in os.listdir(rgb_dir)
                          if f.lower().endswith(('.png', '.jpg'))])
    frame_paths = [os.path.join(rgb_dir, f) for f in frame_files]
    pose_txt_path = os.path.join(data_dir, 'pose.txt')

    max_frame = len(frame_paths) - 1
    end_idx = min(end_idx, max_frame)
    start_idx = min(start_idx, max_frame)

    logger.info('=' * 60)
    logger.info(f'深度+光流数据生成(序列级归一化): {data_dir}')
    logger.info(f'  帧数: {len(frame_paths)}, 范围: {start_idx}-{end_idx-1} ({end_idx-start_idx}对)')
    logger.info(f'  输出: {output_dir}')
    logger.info(f'  PATCH_SIZE: {PATCH_SIZE}')
    logger.info('=' * 60)

    # ── 计算序列级归一化统计量 ──
    logger.info('计算序列级归一化统计量...')
    mean_depth, std_depth, mean_fx, std_fx, mean_fy, std_fy = compute_sequence_stats(
        data_dir, frame_paths, start_idx, end_idx, pose_txt_path)
    stats = {
        'depth_stats': [mean_depth, std_depth],
        'flow_stats': [mean_fx, std_fx, mean_fy, std_fy],
    }

    logger.info('初始化 LoFTR...')
    matcher = init_loftr_matcher(device='cuda')

    n_total, n_moving = process_frame_pairs(
        data_dir, output_dir, start_idx, end_idx,
        frame_paths, pose_txt_path, matcher, 'cuda',
        prefix=args.prefix,
        mean_depth=mean_depth, std_depth=std_depth,
        mean_fx=mean_fx, std_fx=std_fx,
        mean_fy=mean_fy, std_fy=std_fy,
        stats=stats)
    logger.info(f'总计: {n_total:,} 样本, {n_moving:,} moving')
    logger.info('数据生成完成!')


if __name__ == '__main__':
    main()
