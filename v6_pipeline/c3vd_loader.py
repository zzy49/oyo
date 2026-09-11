"""C3VDv2数据集加载器

cecum_t1_a 目录结构:
  每帧: NNN_color.png (RGB), NNNN_depth.tiff (深度真值), NNNN_normals.tiff,
        NNNN_flow.tiff (光流), NNNN_occlusion.png
  pose.txt: 276行, 每行16个逗号分隔浮点数(4x4矩阵row-major)
  camera_intrinsics.txt: 全向相机参数(父目录)

使用: 
  loader = C3VDLoader("E:/data1/monodepth2/xinshuju/cecum_t1_a")
  frames = loader.get_frame_paths()
  gt_poses = loader.load_gt_poses()     # 世界→相机变换
  K = loader.get_intrinsics()           # pinhole近似内参矩阵
  depth_gt = loader.load_depth(frame_idx)
"""

import os
import numpy as np


class C3VDLoader:
    """C3VDv2数据集加载器"""
    
    def __init__(self, data_dir, intrinsics_file=None, rgb_subdir=None, img_suffix='_color.png'):
        """
        Args:
            data_dir: 数据目录路径
            intrinsics_file: camera_intrinsics.txt 路径, 默认在同级父目录
            rgb_subdir: 图像子目录名 (如 'rgb'), None表示图像在data_dir根目录
            img_suffix: 图像文件后缀 (如 '_color.png' 或 '.png')
        """
        self.data_dir = data_dir
        self.rgb_subdir = rgb_subdir
        self.img_suffix = img_suffix
        self.rgb_dir = os.path.join(data_dir, rgb_subdir) if rgb_subdir else data_dir
        if intrinsics_file is None:
            intrinsics_file = os.path.join(os.path.dirname(data_dir), 'camera_intrinsics.txt')
        self.intrinsics_file = intrinsics_file
        self._intrinsics = None
        self._poses = None
        self._frame_paths = None
    
    # ── 内参 ────────────────────────────────────────────
    def get_intrinsics(self):
        """返回 pinhole 近似内参矩阵 (3x3).
        
        C3VDv2 使用全向相机模型 (Kannala-Brandt):
          r(θ) = a0*θ + a1*θ² + a2*θ³ + a3*θ⁴ + a4*θ⁵
        对小角度 r ≈ a0*θ ≈ a0*tan(θ), 故 fx=fy≈a0
        """
        if self._intrinsics is not None:
            return self._intrinsics
        
        params = {}
        with open(self.intrinsics_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(';') or not line:
                    continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    params[key.strip()] = float(val.strip())
        
        fx = params.get('a0', 767.73)
        cx = params.get('cx', 677.74)
        cy = params.get('cy', 543.06)
        self.width = int(params.get('width', 1350))
        self.height = int(params.get('height', 1080))
        
        self._intrinsics = np.array([
            [fx, 0, cx],
            [0, fx, cy],
            [0, 0,  1]
        ], dtype=np.float64)
        
        return self._intrinsics
    
    # ── 图像帧 ──────────────────────────────────────────
    def get_frame_paths(self):
        """返回按编号排序的彩色图像路径列表."""
        if self._frame_paths is not None:
            return self._frame_paths
        
        search_dir = self.rgb_dir
        suffix = self.img_suffix
        is_numeric_prefix = not suffix.startswith('_')
        
        frames = []
        for fname in os.listdir(search_dir):
            if fname.endswith(suffix):
                if not is_numeric_prefix:
                    # "NN_color.png" 格式
                    try:
                        num = int(fname.split('_')[0])
                        frames.append((num, os.path.join(search_dir, fname)))
                    except ValueError:
                        continue
                else:
                    # "NNNN.png" 格式 (4位前导零)
                    try:
                        base = fname[:-len(suffix)]
                        num = int(base)
                        frames.append((num, os.path.join(search_dir, fname)))
                    except ValueError:
                        continue
        
        frames.sort(key=lambda x: x[0])
        self._frame_paths = [p for _, p in frames]
        self._frame_indices = [n for n, _ in frames]
        return self._frame_paths
    
    def get_frame_indices(self):
        """返回帧编号列表 (如 [0, 1, 2, ..., 275])."""
        if self._frame_paths is None:
            self.get_frame_paths()
        return self._frame_indices
    
    @property
    def n_frames(self):
        return len(self.get_frame_paths())
    
    # ── GT 位姿 ─────────────────────────────────────────
    def load_gt_poses(self):
        """加载 pose.txt 中的 4x4 变换矩阵.
        
        pose.txt 格式: 每行16个逗号分隔值, 列优先 4x4 矩阵.
        查找顺序: data_dir/pose.txt → data_dir的父目录/pose.txt
        """
        if self._poses is not None:
            return self._poses
        
        # 先在 data_dir 找, 再在父目录找
        pose_file = os.path.join(self.data_dir, 'pose.txt')
        if not os.path.exists(pose_file):
            # 可能在父目录 (如 rgb/ 子目录的数据)
            parent_pose = os.path.join(os.path.dirname(self.data_dir), 'pose.txt')
            if os.path.exists(parent_pose):
                pose_file = parent_pose
        self._poses = []
        
        with open(pose_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = [float(x) for x in line.split(',')]
                if len(vals) >= 16:
                    raw = np.array(vals[:16], dtype=np.float64).reshape(4, 4)
                    # 列优先→行优先: 旋转部分转置, 平移从末行移到末列
                    T = np.eye(4, dtype=np.float64)
                    T[:3, :3] = raw[:3, :3].T  # R = R_col^T
                    T[:3, 3] = raw[3, :3]     # 平移: t_cam
                    self._poses.append(T)
        
        return self._poses
    
    def get_gt_trajectory(self):
        """返回 GT 相机位置轨迹.
        
        load_gt_poses 已返回标准 T_cam_world = [R, t; 0, 1].
        相机在世界坐标中的位置 = t (矩阵第4列前3行).
        """
        poses = self.load_gt_poses()
        positions = []
        for T in poses:
            positions.append(T[:3, 3])
        return np.array(positions)

    def get_gt_relative_poses(self, frame_i, frame_j):
        """计算帧间相对GT位姿 T_rel = T_i^{-1} @ T_j.
        
        T_world_cam 形式, 故 T_rel 表示 从帧i到帧j的相机运动.
        """
        poses = self.load_gt_poses()
        T_i = poses[frame_i]
        T_j = poses[frame_j]
        return np.linalg.inv(T_i) @ T_j
    
    # ── GT 深度 ─────────────────────────────────────────
    def load_depth(self, frame_idx):
        """加载指定帧的 GT 深度图 (tiff).
        
        C3VDv2 depth 格式:
          - uint16, 0-65535 线性映射到 0-100 mm
          - 转换公式: depth_mm = uint16_value * 100.0 / 65535.0
          参考: https://durrlab.github.io/C3VDv2/
        
        Args:
            frame_idx: 帧编号 (0-based, 对应帧文件名)
        Returns:
            np.ndarray 深度图 (H, W), 单位: 毫米, 范围 0-100mm
        """
        from PIL import Image
        depth_path = os.path.join(self.data_dir, f'{frame_idx:04d}_depth.tiff')
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"深度文件不存在: {depth_path}")
        
        raw = np.array(Image.open(depth_path), dtype=np.float32)
        depth_mm = raw * (100.0 / 65535.0)
        return depth_mm
    
    def load_normals(self, frame_idx):
        """加载指定帧的 GT 法线图."""
        normal_path = os.path.join(self.data_dir, f'{frame_idx:04d}_normals.tiff')
        if not os.path.exists(normal_path):
            raise FileNotFoundError(f"法线文件不存在: {normal_path}")
        
        import cv2
        return cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
    
    def load_occlusion(self, frame_idx):
        """加载遮挡掩码."""
        occ_path = os.path.join(self.data_dir, f'{frame_idx:04d}_occlusion.png')
        if not os.path.exists(occ_path):
            return None
        import cv2
        return cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE)


# ── 度量计算 ────────────────────────────────────────────────
def align_trajectory_umeyama(est_positions, gt_positions):
    """Umeyama 对齐: 估计→GT 的相似变换 (R, t, s).
    
    返回: 对齐后的估计轨迹, 对齐误差数组
    """
    est = np.array(est_positions, dtype=np.float64)
    gt = np.array(gt_positions, dtype=np.float64)
    
    n = min(len(est), len(gt))
    est = est[:n]
    gt = gt[:n]
    
    # 去均值
    est_mean = est.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    est_c = est - est_mean
    gt_c = gt - gt_mean
    
    # SVD
    H = est_c.T @ gt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 尺度
    scale = np.trace(R @ est_c.T @ gt_c) / np.trace(est_c.T @ est_c)
    if scale < 1e-6:
        scale = 1.0
    
    est_aligned = (scale * (R @ est.T)).T + gt_mean - (R @ est_mean * scale)
    
    errors = np.linalg.norm(est_aligned - gt, axis=1)
    return est_aligned, errors, R, scale


def compute_ate(est_aligned, gt_positions):
    """计算绝对轨迹误差 ATE."""
    est = np.array(est_aligned)
    gt = np.array(gt_positions[:len(est)])
    errors = np.linalg.norm(est - gt, axis=1)
    return {
        'mean_mm': float(np.mean(errors)),
        'rmse_mm': float(np.sqrt(np.mean(errors**2))),
        'median_mm': float(np.median(errors)),
        'max_mm': float(np.max(errors)),
        'std_mm': float(np.std(errors)),
    }


def compute_rpe(est_positions, gt_positions, delta=1):
    """计算相对位姿误差 RPE (仅平移部分)."""
    est = np.array(est_positions)
    gt = np.array(gt_positions[:len(est)])
    n = len(est) - delta
    
    est_delta = np.linalg.norm(est[delta:] - est[:n], axis=1)
    gt_delta = np.linalg.norm(gt[delta:] - gt[:n], axis=1)
    errors = np.abs(est_delta - gt_delta)
    
    return {
        'mean_mm': float(np.mean(errors)),
        'rmse_mm': float(np.sqrt(np.mean(errors**2))),
        'median_mm': float(np.median(errors)),
        'max_mm': float(np.max(errors)),
        'std_mm': float(np.std(errors)),
    }
