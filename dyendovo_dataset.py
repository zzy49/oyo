"""
EndoSLAM 数据集工具：内参、GT位姿加载、GT深度加载
"""
import os
import numpy as np
import cv2

# EndoSLAM 摄像机内参矩阵 (1350×1080)
K_MAT = np.array([
    [767.73, 0.0, 677.74],
    [0.0, 767.73, 543.06],
    [0.0, 0.0, 1.0]
], dtype=np.float64)


def load_gt_poses(seq_dir):
    """加载 GT 摄像机位姿 (cam_to_world, 4×4).

    pose.txt 格式：每行 16 个逗号分隔的浮点数，row-major 4×4 矩阵。
    EndoSLAM 存储格式: [R | 0; t | 1] (translation 在 row 3, column 3 为 [0,0,0,1]^T)。
    转换为标准 cam_to_world: [R | t; 0 | 1]。

    Args:
        seq_dir: 序列目录路径

    Returns:
        list of (4,4) np.ndarray, 每帧 cam_to_world (标准 [R|t;0|1] 格式)
    """
    pose_path = os.path.join(seq_dir, 'pose.txt')
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(f"GT 位姿文件不存在: {pose_path}")

    poses = []
    with open(pose_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split(',')]
            T = np.array(vals, dtype=np.float64).reshape(4, 4)
            # EndoSLAM 格式 [R|0; t|1] → 标准 cam_to_world [R|t; 0|1]
            T = _endo_to_standard(T)
            poses.append(T)

    if not poses:
        raise ValueError(f"pose.txt 为空: {pose_path}")

    return poses


def _endo_to_standard(T):
    """将 EndoSLAM [R|0; t|1] 格式转为标准 [R|t; 0|1] cam_to_world."""
    T_std = T.copy()
    T_std[:3, 3] = T[3, :3]   # translation: row 3 cols 0-2 → col 3
    T_std[3, :3] = 0          # 清零 row 3
    return T_std


def load_gt_depth(seq_dir, frame_idx):
    """加载 GT 深度图（毫米单位）。

    支持两种格式：
      - {seq_dir}/depth/{frame_idx:04d}_depth.tiff  (uint16, 0-65535 → 0-100mm)
      - {seq_dir}/depth/{frame_idx:04d}_depth.npy   (float32, mm)

    Args:
        seq_dir:   序列目录路径
        frame_idx: 帧索引 (0-based)

    Returns:
        (H, W) float32 ndarray, 深度值 (mm)
    """
    # 尝试 TIFF 格式
    tiff_path = os.path.join(seq_dir, 'depth', f'{frame_idx:04d}_depth.tiff')
    if os.path.isfile(tiff_path):
        depth = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise IOError(f"无法读取: {tiff_path}")
        if depth.dtype == np.uint16:
            # EndoSLAM TIFF: uint16 编码, 0-65535 线性映射到 0-100mm
            depth = depth.astype(np.float32) * (100.0 / 65535.0)
        return depth

    # 尝试 .npy 格式
    npy_path = os.path.join(seq_dir, 'depth', f'{frame_idx:04d}_depth.npy')
    if os.path.isfile(npy_path):
        depth = np.load(npy_path)
        return np.asarray(depth, dtype=np.float32)

    # 尝试 generated/depth/
    gen_path = os.path.join(seq_dir, 'generated', 'depth', f'frame_{frame_idx:04d}.npy')
    if os.path.isfile(gen_path):
        depth = np.load(gen_path)
        return np.asarray(depth, dtype=np.float32)

    raise FileNotFoundError(
        f"GT 深度文件不存在: {seq_dir}/depth/{frame_idx:04d}_depth.tiff "
        f"或 .npy, 或 generated/depth/frame_{frame_idx:04d}.npy"
    )
