"""
生成多帧深度+残差光流训练数据 V4
=================================
V3: 2帧窗口 [depth_i, depth_j, flow_x, flow_y] (4通道)
V4: 5帧窗口 [depth_i..depth_i+4 (5ch), res_flow_x_0..3 (4ch), res_flow_y_0..3 (4ch)] (13通道)

让 CNN 学习组织运动的时空传播模式（膨胀环沿 X 轴传播）。

输出: 每5帧窗口一个 .npz 文件，含 5 个深度通道 + 4 对残差光流 + labels 等。
"""
import os, sys, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from v6_pipeline.logger import setup_logger

logger = setup_logger(__name__)

# ── Config ──
MAX_SAMPLES_PER_PAIR = 2000
PATCH_SIZE = 16
WINDOW_SIZE = 5               # 5帧窗口: t, t+1, t+2, t+3, t+4
DEPTH_MAX_MM = 200.0
FLOW_MAX_PX = 2.0
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
    depth_path = os.path.join(data_dir, 'depth', f'{frame_idx:04d}_depth.tiff')
    if not os.path.exists(depth_path):
        return None
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    return depth.astype(np.float32) * (100.0 / 65535.0)


def load_flow(data_dir, frame_idx):
    """读取预计算的稠密残差光流 (已扣除相机运动).

    文件: {data_dir}/generated/flow/frame_{frame_idx:04d}.npy
    格式: (H, W, 2) float32, [..., 0]=flow_x, [..., 1]=flow_y
    """
    flow_path = os.path.join(data_dir, 'generated', 'flow', f'frame_{frame_idx:04d}.npy')
    if not os.path.exists(flow_path):
        return None, None
    flow = np.load(flow_path)  # (H, W, 2)
    return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)


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
    """对序列采样帧，分别计算 depth 和 残差光流 的 mean/std."""
    n_frames = end_idx - start_idx
    if n_frames <= 1:
        return 50.0, 30.0, 0.0, 1.0, 0.0, 1.0

    sample_indices = np.linspace(start_idx, end_idx - 1, min(num_sample_frames, n_frames), dtype=int)
    sample_indices = np.unique(np.clip(sample_indices, 0, len(frame_paths) - 2))

    all_depth_vals = []
    all_rfx_vals = []
    all_rfy_vals = []

    for idx in sample_indices:
        depth = load_depth_mm(data_dir, idx)
        if depth is None:
            continue
        d_flat = depth.ravel()
        valid_d = d_flat[(d_flat > 0) & (d_flat < 100.0)]
        all_depth_vals.append(valid_d)

        rfx, rfy = load_flow(data_dir, idx)
        if rfx is None:
            continue
        rfx_flat = rfx.ravel()
        rfy_flat = rfy.ravel()
        nonzero = (rfx_flat != 0) | (rfy_flat != 0)
        all_rfx_vals.append(rfx_flat[nonzero])
        all_rfy_vals.append(rfy_flat[nonzero])

    if not all_depth_vals:
        return 50.0, 30.0, 0.0, 1.0, 0.0, 1.0

    depth_cat = np.concatenate(all_depth_vals)
    d_lo, d_hi = np.percentile(depth_cat, [1, 99])
    depth_cat = depth_cat[(depth_cat >= d_lo) & (depth_cat <= d_hi)]
    mean_depth = float(np.mean(depth_cat))
    std_depth = float(np.std(depth_cat))

    if all_rfx_vals:
        rfx_cat = np.concatenate(all_rfx_vals)
        rfy_cat = np.concatenate(all_rfy_vals)
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
    gray_i = cv2.cvtColor(rgb_i, cv2.COLOR_RGB2GRAY)
    gray_j = cv2.cvtColor(rgb_j, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray_i, gray_j, None, 0.5, 5, 25, 5, 7, 1.5,
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow[..., 0], flow[..., 1]


def compute_camera_flow_dense(depth_map, T_rel, K):
    """相机运动引起的稠密光流 (全图)."""
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

    X = (uu[valid] - cx) * Z[valid] / fx
    Y = (vv[valid] - cy) * Z[valid] / fy
    P = np.stack([X, Y, Z[valid]], axis=-1)

    P_new = (R @ P.T + t.reshape(3, 1)).T
    Z_new = P_new[:, 2]
    ok = Z_new > 1e-6

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


# ══════════════════════════════════════════════
# V4: 多帧 patch 提取函数
# ══════════════════════════════════════════════

def extract_depth_patches_multi(depths, pts_i, patch_size=16,
                                 mean_depth=50.0, std_depth=30.0):
    """从 5 帧深度图提取 patches.
    
    Args:
        depths: list of 5 arrays, each (H, W) float32
        pts_i: (N, 2) float32, keypoint 坐标 (frame i)
    Returns:
        patches: (N, 5, patch_size, patch_size) float32, z-score 归一化
    """
    H, W = depths[0].shape[:2]
    half = patch_size // 2
    N = len(pts_i)
    n_frames = len(depths)
    patches = np.zeros((N, n_frames, patch_size, patch_size), dtype=np.float32)

    for k in range(N):
        x0 = int(np.clip(np.round(pts_i[k, 0]) - half, 0, W - 1))
        y0 = int(np.clip(np.round(pts_i[k, 1]) - half, 0, H - 1))
        x1 = min(x0 + patch_size, W)
        y1 = min(y0 + patch_size, H)
        x0 = max(0, x1 - patch_size)
        y0 = max(0, y1 - patch_size)

        for f in range(n_frames):
            patches[k, f] = (depths[f][y0:y1, x0:x1] - mean_depth) / std_depth

    return patches  # (N, 5, 16, 16)


def extract_flow_patches_multi(flow_fields_x, flow_fields_y, pts_i, patch_size=16,
                                mean_fx=0.0, std_fx=1.0,
                                mean_fy=0.0, std_fy=1.0):
    """从多对残差光流场提取 patches.
    
    Args:
        flow_fields_x: list of 4 arrays, each (H, W) float32 (残差流 x)
        flow_fields_y: list of 4 arrays, each (H, W) float32 (残差流 y)
        pts_i: (N, 2) float32, keypoint 坐标 (frame i)
    Returns:
        fx_patches: (N, 4, patch_size, patch_size) float32
        fy_patches: (N, 4, patch_size, patch_size) float32
    """
    H, W = flow_fields_x[0].shape[:2]
    half = patch_size // 2
    N = len(pts_i)
    n_pairs = len(flow_fields_x)
    fx_patches = np.zeros((N, n_pairs, patch_size, patch_size), dtype=np.float32)
    fy_patches = np.zeros((N, n_pairs, patch_size, patch_size), dtype=np.float32)

    for k in range(N):
        x0 = int(np.clip(np.round(pts_i[k, 0]) - half, 0, W - 1))
        y0 = int(np.clip(np.round(pts_i[k, 1]) - half, 0, H - 1))
        x1 = min(x0 + patch_size, W)
        y1 = min(y0 + patch_size, H)
        x0 = max(0, x1 - patch_size)
        y0 = max(0, y1 - patch_size)

        for p in range(n_pairs):
            fx_patches[k, p] = (flow_fields_x[p][y0:y1, x0:x1] - mean_fx) / std_fx
            fy_patches[k, p] = (flow_fields_y[p][y0:y1, x0:x1] - mean_fy) / std_fy

    return fx_patches, fy_patches  # each (N, 4, 16, 16)


# ══════════════════════════════════════════════
# V4: 5帧窗口处理流水线
# ══════════════════════════════════════════════

def process_frame_windows(data_dir, output_dir, start_idx, end_idx,
                          frame_paths, pose_txt_path,
                          prefix='',
                          mean_depth=50.0, std_depth=30.0,
                          mean_fx=0.0, std_fx=1.0,
                          mean_fy=0.0, std_fy=1.0,
                          stats=None):
    """处理5帧窗口: goodFeaturesToTrack → GT标签 → 多帧深度/残差光流 patch.

    每个窗口 [i, i+1, i+2, i+3, i+4]:
      - 5 帧深度: depth_i..depth_i+4
      - 4 对残差光流: 从 generated/flow/frame_{i:04d}.npy 读取 (预计算)
      - GT 标签: frame i 的 mask_static
    """
    os.makedirs(output_dir, exist_ok=True)

    total_samples = 0
    total_moving = 0
    n_windows = end_idx - start_idx - WINDOW_SIZE + 1  # 可用的5帧窗口数
    if n_windows <= 0:
        logger.warning(f'  窗口数 <= 0，start={start_idx}, end={end_idx}')
        return 0, 0

    logger.info(f'  帧范围: {start_idx}-{end_idx-1}, 可用窗口: {n_windows}')

    rgb_dir = os.path.join(data_dir, 'rgb')

    for win_i, frame_idx in enumerate(range(start_idx, end_idx - WINDOW_SIZE + 1)):
        if win_i % 25 == 0:
            logger.info(f'  [{win_i}/{n_windows}] window at frame {frame_idx}...')

        # ── Shi-Tomasi 角点检测 (frame i) ──
        img_i = cv2.imread(frame_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
        if img_i is None:
            continue
        corners = cv2.goodFeaturesToTrack(img_i, maxCorners=MAX_SAMPLES_PER_PAIR,
                                           qualityLevel=0.01, minDistance=10, blockSize=9)
        if corners is None or len(corners) < 16:
            continue
        pts0 = corners.reshape(-1, 2).astype(np.float32)
        n_pts = len(pts0)

        # ── 加载 5 帧深度 ──
        depths = []
        for offset in range(WINDOW_SIZE):
            d = load_depth_mm(data_dir, frame_idx + offset)
            if d is None:
                break
            depths.append(d)
        if len(depths) < WINDOW_SIZE:
            continue

        # ── 3D 反投影 + GT 标签 (frame i) ──
        pts_3d_cam, valid_3d = reproject_to_3d_depth(pts0, depths[0], K)
        gt_mask, gt_motion, gt_vertices = load_gt_vertex_masks(data_dir, frame_idx)
        T_wc_gt = load_pose_matrix(pose_txt_path, frame_idx)
        if gt_mask is None or T_wc_gt is None:
            continue

        gt_labels, gt_disps, vertex_dists = classify_by_gt(
            pts_3d_cam, gt_vertices, T_wc_gt, gt_mask, gt_motion)

        # ── 读取 4 对残差光流 (预计算文件) ──
        res_flow_x_list = []
        res_flow_y_list = []

        for offset in range(WINDOW_SIZE - 1):
            fi = frame_idx + offset
            rfx, rfy = load_flow(data_dir, fi)
            if rfx is None:
                break
            res_flow_x_list.append(rfx)
            res_flow_y_list.append(rfy)

        if len(res_flow_x_list) < WINDOW_SIZE - 1:
            continue  # 流文件不完整

        # 缩放光流到 warped 图像空间（与角点对齐）
        H_raw, W_raw = res_flow_x_list[0].shape
        H_warp = img_i.shape[0]
        W_warp = img_i.shape[1]
        scale_x = W_raw / W_warp if W_warp != W_raw else 1.0
        scale_y = H_raw / H_warp if H_warp != H_raw else 1.0
        if scale_x != 1.0 or scale_y != 1.0:
            for p in range(len(res_flow_x_list)):
                res_flow_x_list[p] = cv2.resize(res_flow_x_list[p], (W_warp, H_warp)) * scale_x
                res_flow_y_list[p] = cv2.resize(res_flow_y_list[p], (W_warp, H_warp)) * scale_y

        # ── 提取多帧深度 patches: (N, 5, 16, 16) ──
        depth_patches = extract_depth_patches_multi(
            depths, pts0, PATCH_SIZE, mean_depth, std_depth)

        # ── 提取多对残差光流 patches: (N, 4, 16, 16) × 2 ──
        flow_fx, flow_fy = extract_flow_patches_multi(
            res_flow_x_list, res_flow_y_list, pts0, PATCH_SIZE,
            mean_fx, std_fx, mean_fy, std_fy)

        # ── 稀疏流特征（残差光流在角点位置的值）──
        H_img, W_img = img_i.shape[:2]
        pts_int = np.clip(np.round(pts0).astype(int), 0, [W_img - 1, H_img - 1])
        flow_dx = res_flow_x_list[0][pts_int[:, 1], pts_int[:, 0]]
        flow_dy = res_flow_y_list[0][pts_int[:, 1], pts_int[:, 0]]

        # ── 归一化位置 ──
        pos_x = pts0[:, 0] / W_img
        pos_y = pts0[:, 1] / H_img

        # ── 保存（V4-diff: 帧间差分，11 通道）──
        # 将每个通道单独保存为独立的 key，便于 merge 时按固定顺序堆叠
        # 深度差分: depth_t (基准), depth_{t+1}-depth_t, ..., depth_{t+4}-depth_{t+3}
        # 光流差分: flow_{01}-flow_{12}, flow_{12}-flow_{23}, flow_{23}-flow_{34}
        save_dict = {
            # 5 个深度差分通道
            'depth_0': depth_patches[:, 0],                              # depth_t (基准)
            'depth_1': depth_patches[:, 1] - depth_patches[:, 0],       # Δdepth 0→1
            'depth_2': depth_patches[:, 2] - depth_patches[:, 1],       # Δdepth 1→2
            'depth_3': depth_patches[:, 3] - depth_patches[:, 2],       # Δdepth 2→3
            'depth_4': depth_patches[:, 4] - depth_patches[:, 3],       # Δdepth 3→4
            # 3 个残差光流 x 差分（加速度）
            'flow_x_0': flow_fx[:, 0] - flow_fx[:, 1],                  # Δfx 01→12
            'flow_x_1': flow_fx[:, 1] - flow_fx[:, 2],                  # Δfx 12→23
            'flow_x_2': flow_fx[:, 2] - flow_fx[:, 3],                  # Δfx 23→34
            # 3 个残差光流 y 差分
            'flow_y_0': flow_fy[:, 0] - flow_fy[:, 1],                  # Δfy 01→12
            'flow_y_1': flow_fy[:, 1] - flow_fy[:, 2],                  # Δfy 12→23
            'flow_y_2': flow_fy[:, 2] - flow_fy[:, 3],                  # Δfy 23→34
            # 辅助信息
            'flow_dx': flow_dx.astype(np.float16),
            'flow_dy': flow_dy.astype(np.float16),
            'pos_x': pos_x.astype(np.float16),
            'pos_y': pos_y.astype(np.float16),
            'labels': gt_labels,
            'displacements': gt_disps.astype(np.float32),
            'depth_stats': np.array([mean_depth, std_depth], dtype=np.float32),
            'flow_stats': np.array([mean_fx, std_fx, mean_fy, std_fy], dtype=np.float32),
        }

        out_name = f'{prefix}_window_{frame_idx:04d}.npz' if prefix else f'window_{frame_idx:04d}.npz'
        out_path = os.path.join(output_dir, out_name)
        np.savez_compressed(out_path, **save_dict)

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

    # 优先用 rgb_warped，fallback 到 rgb
    rgb_dir = os.path.join(data_dir, 'generated', 'rgb_warped')
    if not os.path.exists(rgb_dir):
        rgb_dir = os.path.join(data_dir, 'rgb')
    frame_files = sorted([f for f in os.listdir(rgb_dir)
                          if f.lower().endswith(('.png', '.jpg'))])
    frame_paths = [os.path.join(rgb_dir, f) for f in frame_files]
    pose_txt_path = os.path.join(data_dir, 'pose.txt')

    max_frame = len(frame_paths) - 1
    end_idx = min(args.end_frame, max_frame)
    start_idx = min(start_idx, max_frame)

    # 确保有足够的帧形成至少 1 个窗口
    n_windows_possible = end_idx - start_idx - WINDOW_SIZE + 1
    if n_windows_possible <= 0:
        logger.warning(f'  帧数不足以形成5帧窗口，需要至少{WINDOW_SIZE}帧，'
                       f'实际 {end_idx - start_idx} 帧')
        return

    logger.info('=' * 60)
    logger.info(f'V4 多帧深度+残差光流数据生成(序列级归一化): {data_dir}')
    logger.info(f'  帧数: {len(frame_paths)}, 范围: {start_idx}-{end_idx-1}')
    logger.info(f'  窗口大小: {WINDOW_SIZE}帧, 可用窗口: {n_windows_possible}')
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

    n_total, n_moving = process_frame_windows(
        data_dir, output_dir, start_idx, end_idx,
        frame_paths, pose_txt_path,
        prefix=args.prefix,
        mean_depth=mean_depth, std_depth=std_depth,
        mean_fx=mean_fx, std_fx=std_fx,
        mean_fy=mean_fy, std_fy=std_fy,
        stats=stats)
    logger.info(f'总计: {n_total:,} 样本, {n_moving:,} moving')
    logger.info('数据生成完成!')


if __name__ == '__main__':
    main()
