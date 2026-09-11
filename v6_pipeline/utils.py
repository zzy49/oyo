import os
import ssl
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import PIL.Image as pil
from torchvision import transforms
from scipy.spatial import cKDTree

from networks import ResnetEncoder, DepthDecoder, PoseCNN, MotionEncoder
from layers import disp_to_depth, make_bin_centers, depth_from_bins, transformation_from_parameters
from .logger import setup_logger

# LoFTR模型下载绕过SSL证书验证
ssl._create_default_https_context = ssl._create_unverified_context

logger = setup_logger(__name__)

# ── LoFTR 匹配器全局单例 ──────────────────────────
_loftr_matcher = None
_loftr_device = None

# ── K_CANON 标准相机 (方案B: 统一坐标空间) ──────
K_CANON = torch.tensor([
    [320.0, 0.0,   320.0],
    [0.0,   96.0,  96.0],
    [0.0,   0.0,   1.0]
], dtype=torch.float32)
K_CANON_INV = torch.inverse(K_CANON)

class ModelManager:
    _instance = None
    
    def __new__(cls, model_path=None, device=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._encoder = None
            cls._instance._depth_decoder = None
            cls._instance._pose_cnn = None
            cls._instance._motion_encoder = None
            cls._instance._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._instance._model_path = model_path
        # Allow model_path override
        elif model_path is not None:
            cls._instance._model_path = model_path
            cls._instance._encoder = None  # force reload
            cls._instance._depth_decoder = None
            cls._instance._pose_cnn = None
            cls._instance._motion_encoder = None
        return cls._instance
    
    def load_model(self):
        if self._encoder is None and self._model_path:
            try:
                logger.info(f"加载模型: {self._model_path}")
                self._encoder = ResnetEncoder(18, False)
                state_dict = torch.load(os.path.join(self._model_path, "encoder.pth"),
                                      map_location=self._device)
                for key in ['height', 'width', 'use_stereo']:
                    if key in state_dict:
                        del state_dict[key]
                self._encoder.load_state_dict(state_dict)
                self._encoder.eval().to(self._device)

                # 自动检测是否分类头模型
                depth_state = torch.load(os.path.join(self._model_path, "depth.pth"),
                                         map_location=self._device)
                num_bins = 0
                for key in depth_state.keys():
                    # 检测最后一层 conv 输出通道 = num_bins
                    if '.10.conv.weight' in key and depth_state[key].shape[0] > 1:
                        num_bins = depth_state[key].shape[0]
                        break

                self._depth_decoder = DepthDecoder(self._encoder.num_ch_enc, [0],
                                                   num_bins=num_bins)
                self._depth_decoder.load_state_dict(depth_state, strict=False)
                self._depth_decoder.eval().to(self._device)
                if num_bins > 0:
                    logger.info(f"模型加载成功 (分类头, {num_bins} bins)")
                else:
                    logger.info("模型加载成功")

                # 加载 PoseCNN (如果存在, 架构不匹配时静默跳过)
                pose_path = os.path.join(self._model_path, "pose.pth")
                if os.path.isfile(pose_path):
                    try:
                        logger.info("加载 PoseCNN...")
                        self._pose_cnn = PoseCNN(num_input_frames=2)
                        pose_state = torch.load(pose_path, map_location=self._device)
                        self._pose_cnn.load_state_dict(pose_state)
                        self._pose_cnn.eval().to(self._device)
                        logger.info("PoseCNN 加载成功")
                    except Exception as e:
                        logger.warning(f"PoseCNN 加载失败 (非致命): {e}")
                        self._pose_cnn = None
                else:
                    logger.info("未找到 pose.pth (仅深度模式)")
                    self._pose_cnn = None

                # 加载 MotionEncoder (如果存在)
                motion_path = os.path.join(self._model_path, "motion_encoder.pth")
                if os.path.isfile(motion_path):
                    logger.info("加载 MotionEncoder...")
                    self._motion_encoder = MotionEncoder()
                    motion_state = torch.load(motion_path, map_location=self._device)
                    self._motion_encoder.load_state_dict(motion_state)
                    self._motion_encoder.eval().to(self._device)
                    logger.info("MotionEncoder 加载成功")
                else:
                    self._motion_encoder = None

            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}", exc_info=True)
                raise
        return self._encoder, self._depth_decoder, self._pose_cnn, self._motion_encoder

class DepthCache:
    def __init__(self, max_size=50):
        self.cache = {}
        self.max_size = max_size
        self.lru_order = []
    
    def get(self, frame_idx):
        if frame_idx in self.cache:
            self.lru_order.remove(frame_idx)
            self.lru_order.append(frame_idx)
            return self.cache[frame_idx]
        return None
    
    def put(self, frame_idx, depth_map):
        if len(self.cache) >= self.max_size:
            oldest = self.lru_order.pop(0)
            del self.cache[oldest]
        self.cache[frame_idx] = depth_map
        self.lru_order.append(frame_idx)
    
    def clear(self):
        self.cache.clear()
        self.lru_order.clear()


# ═══════════════════════════════════════════════════════════
# K_CANON 归一化工具: 物理相机 <-> 标准相机
# ═══════════════════════════════════════════════════════════

def normalize_images_for_pose(images, K_phys):
    """将物理相机图像 warp 到标准相机 (K_CANON) 空间。

    不同内参的图像映射到统一虚拟相机, PoseCNN 在统一坐标下预测位姿。

    Args:
        images:     [B, C, H, W] 物理相机图像 (需已在 GPU, 已归一化)
        K_phys:     [3, 3] 物理相机 K 矩阵 (已缩放到 H×W 分辨率)
    Returns:
        [B, C, H, W] 标准相机空间图像
    """
    B, C, H, W = images.shape
    device = images.device

    # 确保 K_phys 是 tensor
    if isinstance(K_phys, np.ndarray):
        K_phys = torch.from_numpy(K_phys).float()
    K_phys_batch = K_phys.unsqueeze(0).to(device)  # [1, 3, 3]

    K_can = K_CANON.to(device)

    # 标准相机像素坐标 → 归一化坐标
    y_idx = torch.arange(H, dtype=torch.float32, device=device).view(1, H, 1)
    x_idx = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, W)
    x_can_norm = (x_idx - K_can[0, 2]) / K_can[0, 0]  # [1, 1, W]
    y_can_norm = (y_idx - K_can[1, 2]) / K_can[1, 1]  # [1, H, 1]

    # 归一化坐标 → 物理相机像素坐标
    fx_phys = K_phys_batch[:, 0, 0].view(B, 1, 1)
    fy_phys = K_phys_batch[:, 1, 1].view(B, 1, 1)
    cx_phys = K_phys_batch[:, 0, 2].view(B, 1, 1)
    cy_phys = K_phys_batch[:, 1, 2].view(B, 1, 1)

    u_phys = (fx_phys * x_can_norm + cx_phys).expand(B, H, W)
    v_phys = (fy_phys * y_can_norm + cy_phys).expand(B, H, W)

    # grid_sample 归一化到 [-1, 1]
    grid_x = 2.0 * u_phys / W - 1.0
    grid_y = 2.0 * v_phys / H - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]

    return F.grid_sample(images, grid, mode='bilinear',
                         padding_mode='border', align_corners=False)


def convert_pose_canonical_to_physical(axisangle, translation, K_phys):
    """将 PoseCNN 输出从标准相机空间转换到物理相机空间。

    旋转 R 不变 (3D 旋转与相机内参无关)。
    平移 t 需缩放: t_phys_i = t_can_i * f_can_i / f_phys_i  (i ∈ {x, y})
    z 分量不变 (深度方向不受 fx/fy 影响)。

    Args:
        axisangle:   [1, 1, 3] 或 [1, 3]
        translation: [1, 1, 3] 或 [1, 3]
        K_phys:      [3, 3] 物理相机内参
    Returns:
        (axisangle, translation_phys)
    """
    device = axisangle.device
    K_can = K_CANON.to(device)

    # 标准化到 [1, 1, 3]
    if axisangle.dim() == 2:
        axisangle = axisangle.unsqueeze(0).unsqueeze(1)
    elif axisangle.dim() == 3 and axisangle.shape[0] == 1 and axisangle.shape[1] == 3:
        axisangle = axisangle.unsqueeze(1)
    if translation.dim() == 2:
        translation = translation.unsqueeze(0).unsqueeze(1)
    elif translation.dim() == 3 and translation.shape[0] == 1 and translation.shape[1] == 3:
        translation = translation.unsqueeze(1)

    fx_phys = K_phys[0, 0]
    fy_phys = K_phys[1, 1]

    sx = K_can[0, 0] / fx_phys
    sy = K_can[1, 1] / fy_phys

    translation_phys = translation.clone()
    translation_phys[..., 0:1] = translation[..., 0:1] * sx
    translation_phys[..., 1:2] = translation[..., 1:2] * sy
    # z 分量保留不变

    return axisangle, translation_phys


def predict_posecnn(img_path0, img_path1, pose_cnn, K_phys, device):
    """使用 PoseCNN 预测两帧之间的相对位姿。

    Args:
        img_path0: frame0 图像路径
        img_path1: frame1 图像路径
        pose_cnn:  PoseCNN 模型
        K_phys:    [3, 3] 物理相机内参
        device:    torch device
    Returns:
        axisangle: [1, 1, 3] 标准相机空间轴角
        translation_phys: [1, 1, 3] 物理相机空间平移 (mm)
        T_rel: [4, 4] 相对位姿矩阵 (物理空间)
    """
    # 加载并预处理图像
    def load_image(path):
        img = pil.open(path).convert('RGB')
        img = img.resize((640, 192), pil.LANCZOS)
        img = transforms.ToTensor()(img).unsqueeze(0).to(device)
        return img

    img0 = load_image(img_path0)
    img1 = load_image(img_path1)

    # K_CANON 归一化
    img0_can = normalize_images_for_pose(img0, K_phys)
    img1_can = normalize_images_for_pose(img1, K_phys)

    # 6 通道帧对 → PoseCNN
    pose_input = torch.cat([img0_can, img1_can], dim=1)
    with torch.no_grad():
        axisangle, translation = pose_cnn(pose_input)

    # 标准相机 → 物理相机
    axisangle_phys, translation_phys = convert_pose_canonical_to_physical(
        axisangle, translation, K_phys)

    # 构造 4×4 矩阵
    T_rel = transformation_from_parameters(
        axisangle_phys[:, 0], translation_phys[:, 0], invert=False)

    return axisangle_phys, translation_phys, T_rel.squeeze(0).squeeze(0).cpu().numpy()


def predict_depth(img_path, encoder, depth_decoder, device, feed_h=256, feed_w=320, inverse=False, min_depth=1.0, max_depth=500.0, num_bins=0):
    try:
        # Lite-Mono 等模型训练时使用固定输入尺寸, 优先按其 feed_size 推理
        if hasattr(encoder, 'feed_size'):
            feed_w, feed_h = encoder.feed_size
        # 按模型训练配置覆盖深度范围 (确保推理范围与训练 min_depth/max_depth 一致)
        if hasattr(encoder, 'depth_range'):
            min_depth, max_depth = encoder.depth_range
        input_image = pil.open(img_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_w, feed_h), pil.LANCZOS)
        input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)

        with torch.no_grad():
            features = encoder(input_tensor.to(device))
            outputs = depth_decoder(features)

            # 自动检测 num_bins (优先用参数, 其次读 decoder 属性)
            nb = num_bins if num_bins > 0 else getattr(depth_decoder, 'num_bins', 0)

            if nb > 0:
                # 分类+残差头
                bin_logits = outputs[("bins", 0)]
                residual = outputs.get(("residual", 0), None)
                bin_centers = make_bin_centers(min_depth, max_depth, nb)
                disp_resized = torch.nn.functional.interpolate(
                    bin_logits, (original_height, original_width),
                    mode="bilinear", align_corners=False)
                if residual is not None:
                    res_resized = torch.nn.functional.interpolate(
                        residual, (original_height, original_width),
                        mode="bilinear", align_corners=False)
                else:
                    res_resized = None
                depth = depth_from_bins(disp_resized, bin_centers, res_resized)
            else:
                # 原始回归头
                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width),
                    mode="bilinear", align_corners=False)
                _, depth = disp_to_depth(disp_resized, min_depth, max_depth)

        result = depth.cpu().numpy().squeeze()
        if inverse:
            result = 1.0 / (result + 1e-8)
        return result
    except Exception as e:
        logger.error(f"深度预测失败 ({img_path}): {str(e)}")
        raise

def predict_depth_with_motion(img_path, context_img_path, encoder, depth_decoder,
                               motion_encoder, device, feed_h=256, feed_w=320,
                               inverse=False, min_depth=1.0, max_depth=500.0):
    """MotionEncoder 增强的深度预测 (两帧推理)

    depth_i = Decoder(Enc(f_i) + MotionEnc(f_i, f_context))
    训练时 MotionEncoder 已学会感知相邻帧间的相机运动对深度的影响。

    Args:
        img_path:         当前帧图像路径 (预测深度的目标帧)
        context_img_path: 上下文帧图像路径 (提供运动信息)
        encoder, depth_decoder, motion_encoder: 模型
        device: torch device
    Returns:
        depth: [H, W] numpy array (mm)
    """
    try:
        # Lite-Mono 等模型训练时使用固定输入尺寸, 优先按其 feed_size 推理
        if hasattr(encoder, 'feed_size'):
            feed_w, feed_h = encoder.feed_size
        # 按模型训练配置覆盖深度范围 (确保推理范围与训练 min_depth/max_depth 一致)
        if hasattr(encoder, 'depth_range'):
            min_depth, max_depth = encoder.depth_range
        input_image = pil.open(img_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_w, feed_h), pil.LANCZOS)
        input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

        # 上下文帧
        ctx_image = pil.open(context_img_path).convert('RGB')
        ctx_image = ctx_image.resize((feed_w, feed_h), pil.LANCZOS)
        ctx_tensor = transforms.ToTensor()(ctx_image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Encoder (已冻结的 Stage3 权重，BN 使用评估模式)
            features = encoder(input_tensor)

            # MotionEncoder: 6通道帧对输入
            frame_pair = torch.cat([input_tensor, ctx_tensor], dim=1)
            motion_feats = motion_encoder(frame_pair)

            # 融合 motion 特征到 encoder skip connections
            for level in range(4):
                features[level] = motion_encoder.fuse_skip(
                    features[level], motion_feats[level], level)

            # DepthDecoder (分类头)
            outputs = depth_decoder(features)

            nb = getattr(depth_decoder, 'num_bins', 0)
            if nb > 0:
                bin_logits = outputs[("bins", 0)]
                residual = outputs.get(("residual", 0), None)
                bin_centers = make_bin_centers(min_depth, max_depth, nb)
                disp_resized = torch.nn.functional.interpolate(
                    bin_logits, (original_height, original_width),
                    mode="bilinear", align_corners=False)
                if residual is not None:
                    res_resized = torch.nn.functional.interpolate(
                        residual, (original_height, original_width),
                        mode="bilinear", align_corners=False)
                else:
                    res_resized = None
                depth = depth_from_bins(disp_resized, bin_centers, res_resized)
            else:
                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(
                    disp, (original_height, original_width),
                    mode="bilinear", align_corners=False)
                _, depth = disp_to_depth(disp_resized, min_depth, max_depth)

        result = depth.cpu().numpy().squeeze()
        if inverse:
            result = 1.0 / (result + 1e-8)
        return result
    except Exception as e:
        logger.error(f"Motion深度预测失败 ({img_path}): {str(e)}")
        raise


def preprocess_depth(depth_map, method='bilateral'):
    if method == 'bilateral':
        depth_8u = (depth_map / depth_map.max() * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(depth_8u, 9, 75, 75)
        return filtered.astype(np.float32) / 255.0 * depth_map.max()
    return depth_map

def reproject_to_3d(pts_2d, depth_map, K, preprocess_method='bilateral'):
    depth_map = preprocess_depth(depth_map, preprocess_method)
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    valid_depth = depth_map[depth_map > 0]
    if len(valid_depth) == 0:
        return np.array([], dtype=np.float32).reshape(0, 3), np.array([], dtype=bool)
    
    depth_median = float(np.median(valid_depth))
    depth_lo, depth_hi = depth_median * 0.1, depth_median * 10.0
    
    h, w = depth_map.shape
    pts_3d = []
    valid_mask = []
    
    for pt in pts_2d:
        u, v = pt
        u_idx = np.clip(int(round(u)), 0, w - 1)
        v_idx = np.clip(int(round(v)), 0, h - 1)
        
        Z = depth_map[v_idx, u_idx]
        if Z <= depth_lo or Z >= depth_hi:
            Z = depth_median
        
        pts_3d.append([(u - cx) * Z / fx, (v - cy) * Z / fy, Z])
        valid_mask.append(True)
    
    return np.array(pts_3d, dtype=np.float32), np.array(valid_mask)

def multi_frame_tracking(frames_paths, frame_idx, n_frames=5):
    if frame_idx + n_frames > len(frames_paths):
        logger.debug(f"多帧追踪: 帧索引越界 (frame_idx={frame_idx}, total={len(frames_paths)})")
        return None
    
    img0 = cv2.imread(frames_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        logger.debug(f"多帧追踪: 无法读取图像 {frames_paths[frame_idx]}")
        return None
    
    pts0 = cv2.goodFeaturesToTrack(img0, maxCorners=500, qualityLevel=0.01, minDistance=7)
    if pts0 is None:
        logger.debug(f"多帧追踪: 未检测到特征点")
        return None
    pts0 = pts0.reshape(-1, 2)
    
    tracks = [pts0.copy()]
    pts_prev = pts0.copy()
    img_prev = img0.copy()
    
    for i in range(1, n_frames):
        img_curr = cv2.imread(frames_paths[frame_idx + i], cv2.IMREAD_GRAYSCALE)
        if img_curr is None:
            logger.debug(f"多帧追踪: 无法读取图像 {frames_paths[frame_idx + i]}")
            return None
        
        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            img_prev, img_curr, pts_prev.reshape(-1, 1, 2), None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        if pts_curr is None:
            logger.debug(f"多帧追踪: LK光流追踪失败")
            return None
        
        valid = status.ravel().astype(bool)
        if np.sum(valid) < 10:
            logger.debug(f"多帧追踪: 有效追踪点太少 ({np.sum(valid)})")
            return None
        
        pts_curr = pts_curr.reshape(-1, 2)
        tracks.append(pts_curr)
        
        img_prev = img_curr
        pts_prev = pts_curr
    
    return np.array(tracks)

def compute_static_points_hybrid(tracks, threshold=2.0, min_ratio=0.05):
    if tracks is None or len(tracks) < 2:
        return 0.0, None, 'failed'
    
    N_frames, N_points, _ = tracks.shape
    
    vote_count_m3 = np.zeros(N_points, dtype=int)
    total_pairs_m3 = 0
    
    for i in range(N_frames):
        for j in range(i + 1, N_frames):
            total_pairs_m3 += 1
            H, mask = cv2.findHomography(
                tracks[i].reshape(-1, 1, 2).astype(np.float32),
                tracks[j].reshape(-1, 1, 2).astype(np.float32),
                cv2.RANSAC, threshold
            )
            if mask is not None:
                vote_count_m3 += mask.ravel()
    
    m3_ratio = float(total_pairs_m3) / (N_frames * (N_frames - 1) / 2) if total_pairs_m3 > 0 else 0
    v_threshold_m3 = max(2, int(m3_ratio * 0.6))
    static_mask_m3 = vote_count_m3 >= v_threshold_m3
    static_ratio_m3 = float(np.sum(static_mask_m3)) / N_points if N_points > 0 else 0.0
    
    if static_ratio_m3 >= min_ratio:
        return static_ratio_m3, static_mask_m3, 'm3_v3'
    
    vote_count_m2 = np.zeros(N_points, dtype=int)
    for i in range(N_frames - 1):
        H, mask = cv2.findHomography(
            tracks[i].reshape(-1, 1, 2).astype(np.float32),
            tracks[i+1].reshape(-1, 1, 2).astype(np.float32),
            cv2.RANSAC, threshold
        )
        if mask is not None:
            vote_count_m2 += mask.ravel()
    
    v_threshold_m2 = max(2, int((N_frames - 1) * 0.5))
    static_mask_m2 = vote_count_m2 >= v_threshold_m2
    static_ratio_m2 = float(np.sum(static_mask_m2)) / N_points if N_points > 0 else 0.0
    
    if static_ratio_m2 >= min_ratio:
        return static_ratio_m2, static_mask_m2, 'm2_v2_fallback'
    
    return 0.0, None, 'failed'

def estimate_pose_pnp_best(pts_3d, pts_2d, K):
    if len(pts_3d) < 4:
        return None, None
    
    z_vals = pts_3d[:, 2]
    z_med = np.median(z_vals)
    valid = (z_vals > z_med * 0.1) & (z_vals < z_med * 10.0)
    if np.sum(valid) >= 4:
        pts_3d_use = pts_3d[valid]
        pts_2d_use = pts_2d[valid]
    else:
        pts_3d_use = pts_3d
        pts_2d_use = pts_2d
    
    configs = [
        {'reproj_err': 8.0,  'flag': cv2.SOLVEPNP_EPNP},
        {'reproj_err': 5.0,  'flag': cv2.SOLVEPNP_EPNP},
        {'reproj_err': 12.0, 'flag': cv2.SOLVEPNP_EPNP},
        {'reproj_err': 8.0,  'flag': cv2.SOLVEPNP_ITERATIVE},
        {'reproj_err': 12.0, 'flag': cv2.SOLVEPNP_ITERATIVE},
        {'reproj_err': 8.0,  'flag': cv2.SOLVEPNP_SQPNP},
        {'reproj_err': 12.0, 'flag': cv2.SOLVEPNP_SQPNP},
    ]
    
    best_result = None
    best_inliers = 0
    
    for cfg in configs:
        try:
            object_points = pts_3d_use.reshape(-1, 1, 3).astype(np.float32)
            image_points = pts_2d_use.reshape(-1, 1, 2).astype(np.float32)
            
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points, K, None,
                iterationsCount=100,
                reprojectionError=cfg['reproj_err'],
                confidence=0.999,
                flags=cfg['flag']
            )
            
            if success and rvec is not None and tvec is not None and inliers is not None:
                n_inliers = len(inliers)
                if n_inliers > best_inliers:
                    best_inliers = n_inliers
                    R, _ = cv2.Rodrigues(rvec)
                    best_result = (R, tvec.ravel())
        except cv2.error:
            continue
    
    return best_result


def estimate_pose_pnp_iterative_static(pts_3d, pts_2d, K,
                                        max_iterations=3,
                                        reproj_threshold_px=3.0,
                                        min_static_ratio=0.15):
    """迭代静止点筛选PnP: 用重投影残差剔除运动组织点.
    
    核心逻辑:
      1. 所有点跑PnP → 初始位姿
      2. 用初始位姿投影3D点, 计算每个点的重投影残差
      3. 残差 > 阈值的点 = 运动组织(组织蠕动导致2D位置偏离刚性投影)
      4. 只用残差小的静止点重新跑PnP
      5. 迭代2-3轮直到静止点比例收敛
    
    Args:
        pts_3d: (N,3) 上一帧的3D点
        pts_2d: (N,2) 当前帧的2D观测
        K: (3,3) 相机内参
        max_iterations: 最大迭代次数
        reproj_threshold_px: 重投影残差阈值(像素), 超过此值的点为运动点
        min_static_ratio: 最少需要的静止点比例,低于此值则回退到全点PnP
    
    Returns:
        (R, t, static_mask, n_iterations, static_ratio)
    """
    if len(pts_3d) < 8:
        R, t = estimate_pose_pnp_best(pts_3d, pts_2d, K)
        if R is None:
            return None, None, None, 0, 0.0
        return R, t, np.ones(len(pts_3d), dtype=bool), 1, 1.0
    
    N = len(pts_3d)
    static_mask = np.ones(N, dtype=bool)
    prev_static_ratio = 1.0
    
    for iteration in range(max_iterations):
        # ── 1. 当前静止点上跑PnP ──
        R, t = estimate_pose_pnp_best(pts_3d[static_mask], pts_2d[static_mask], K)
        if R is None or t is None:
            break
        
        # ── 2. 计算所有点的重投影残差 ──
        # 投影: P_3d → P_cam = R*P_3d + t → project(P_cam)
        P_cam = (R @ pts_3d.T).T + t.reshape(1, 3)
        # 只保留在相机前方的点
        in_front = P_cam[:, 2] > 0.01
        if np.sum(in_front) < 4:
            break
        
        # 透视投影
        u_proj = K[0, 0] * P_cam[:, 0] / P_cam[:, 2] + K[0, 2]
        v_proj = K[1, 1] * P_cam[:, 1] / P_cam[:, 2] + K[1, 2]
        proj_2d = np.stack([u_proj, v_proj], axis=1)
        
        # 重投影残差
        residuals = np.linalg.norm(pts_2d - proj_2d, axis=1)
        residuals[~in_front] = 1e6  # 相机后方的点标记为大残差
        
        # ── 3. 自适应阈值: median + k*MAD ──
        med_res = np.median(residuals[in_front])
        mad_res = np.median(np.abs(residuals[in_front] - med_res))
        threshold = med_res + 3.0 * mad_res  # 3-sigma等价
        threshold = max(threshold, reproj_threshold_px)  # 不低于绝对下限
        
        # ── 4. 标记静止点 ──
        new_static_mask = (residuals <= threshold) & in_front
        new_static_ratio = np.sum(new_static_mask) / N
        
        # ── 5. 收敛检查 ──
        if np.abs(new_static_ratio - prev_static_ratio) < 0.02:
            static_mask = new_static_mask
            prev_static_ratio = new_static_ratio
            break
        
        static_mask = new_static_mask
        prev_static_ratio = new_static_ratio
        
        # 静止点太少, 回退到上一轮
        if np.sum(static_mask) < 4:
            static_mask = np.ones(N, dtype=bool)
            break
    
    # ── 最终PnP (使用最终静止点) ──
    if np.sum(static_mask) >= 4 and np.sum(static_mask) >= N * min_static_ratio:
        R_final, t_final = estimate_pose_pnp_best(pts_3d[static_mask], pts_2d[static_mask], K)
        if R_final is not None:
            final_static_ratio = float(np.sum(static_mask) / N)
            # 诊断: 比较全点PnP vs 静止点PnP的差异
            R_all, t_all = estimate_pose_pnp_best(pts_3d, pts_2d, K)
            if R_all is not None:
                t_diff = np.linalg.norm(t_final.flatten() - t_all.flatten())
                angle_diff = np.arccos(np.clip((np.trace(R_final.T @ R_all) - 1) / 2, -1, 1)) * 180 / np.pi
                import logging
                _log = logging.getLogger('v6_pipeline.utils')
                _log.debug(f'IterPnP: static={final_static_ratio:.2f} '
                          f't_diff={t_diff:.3f}mm angle_diff={angle_diff:.2f}deg')
            return R_final, t_final, static_mask, iteration + 1, final_static_ratio
    
    # 回退: 全点PnP
    R_final, t_final = estimate_pose_pnp_best(pts_3d, pts_2d, K)
    return R_final, t_final, np.ones(N, dtype=bool), 0, 1.0


def estimate_pose_icp(pts_3d_0, pts_3d_1, weights=None,
                      ransac_thresh=3.0, ransac_iter=200,
                      min_inlier_ratio=0.3):
    """SVD刚性点云配准 (ICP 替代 PnP).

    通过两个相机坐标系下同一组3D点的对应关系, 用SVD闭式解
    直接估计帧间刚体变换(R,t), 避免PnP对深度空间偏置的敏感性。

    原理:
      给定 p_i^(0), p_i^(1) 是同一物理点分别在帧0/帧1坐标系下的3D坐标,
      求 R, t 使得 p^(1) ≈ R @ p^(0) + t.

    Args:
        pts_3d_0:  (N,3) 帧0相机坐标系下的3D点
        pts_3d_1:  (N,3) 帧1相机坐标系下的3D点
        weights:   (N,)  各点权重, None 则等权
        ransac_thresh: RANSAC内点距离阈值 (mm, 相机坐标系下)
        ransac_iter:   RANSAC迭代次数
        min_inlier_ratio: 最小内点比例

    Returns:
        (R, t, inlier_mask) or (None, None, None)
        R: (3,3)  旋转矩阵
        t: (3,)    平移向量
        inlier_mask: (N,) bool 内点标记
    """
    N = len(pts_3d_0)
    if N < 3:
        return None, None, None

    p0 = pts_3d_0.astype(np.float64)
    p1 = pts_3d_1.astype(np.float64)

    # 距离预筛选: 对应点间距离不能太大 (相机运动有限, 帧间位移通常<50mm)
    pair_dists = np.linalg.norm(p1 - p0, axis=1)
    med_dist = np.median(pair_dists)
    dist_ok = pair_dists < max(med_dist * 3.0, 30.0)
    if np.sum(dist_ok) < 3:
        dist_ok = np.ones(N, dtype=bool)

    # RANSAC
    best_inliers = None
    best_R = None
    best_t = None
    best_n_inliers = 0

    n_use = np.sum(dist_ok)
    idx_use = np.where(dist_ok)[0]

    for _ in range(ransac_iter):
        if n_use < 3:
            break
        sample_idx = np.random.choice(idx_use, size=3, replace=False)
        R, t = _svd_rigid_transform(p0[sample_idx], p1[sample_idx])
        if R is None:
            continue

        # 计算所有点的残差
        p1_pred = (R @ p0.T).T + t
        residuals = np.linalg.norm(p1 - p1_pred, axis=1)
        inliers = (residuals < ransac_thresh) & dist_ok
        n_inliers = np.sum(inliers)

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inliers = inliers
            best_R = R
            best_t = t

    # 用内点重新精炼
    if best_inliers is not None and best_n_inliers >= max(3, N * min_inlier_ratio):
        if weights is not None:
            w = weights[best_inliers]
            w = w / w.sum()
            R_final, t_final = _svd_rigid_transform_weighted(
                p0[best_inliers], p1[best_inliers], w)
        else:
            R_final, t_final = _svd_rigid_transform(
                p0[best_inliers], p1[best_inliers])
        if R_final is not None:
            return R_final, t_final, best_inliers

    # 回退: 全点 SVD (无RANSAC)
    if weights is not None:
        w = weights / weights.sum()
        R_refined, t_refined = _svd_rigid_transform_weighted(p0, p1, w)
    else:
        R_refined, t_refined = _svd_rigid_transform(p0, p1)
    if R_refined is not None:
        return R_refined, t_refined, np.ones(N, dtype=bool)
    return None, None, None


def _svd_rigid_transform(p0, p1):
    """Kabsch-Umeyama 闭式解: 最小化 ||R*p0+t - p1||^2."""
    centroid0 = p0.mean(axis=0)
    centroid1 = p1.mean(axis=0)
    p0_c = p0 - centroid0
    p1_c = p1 - centroid1

    H = p0_c.T @ p1_c  # 3x3 互协方差矩阵
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 反射校正
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid1 - R @ centroid0
    return R.astype(np.float32), t.astype(np.float32)


def _svd_rigid_transform_weighted(p0, p1, weights):
    """加权 Kabsch-Umeyama 闭式解."""
    centroid0 = np.average(p0, axis=0, weights=weights)
    centroid1 = np.average(p1, axis=0, weights=weights)
    p0_c = p0 - centroid0
    p1_c = p1 - centroid1

    W = np.diag(np.sqrt(weights))
    H = p0_c.T @ W @ W @ p1_c  # H = (W*p0_c)^T @ (W*p1_c)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid1 - R @ centroid0
    return R.astype(np.float32), t.astype(np.float32)


def estimate_pose_icp_single_depth(pts_3d_0, pts_2d_0, pts_2d_1, K,
                                     ransac_thresh=5.0, ransac_iter=200,
                                     min_inlier_ratio=0.3):
    """单帧深度ICP: 仅用depth0构建3D点, depth1只用XY几何约束.

    核心思路: V6深度帧间不一致, 但帧内空间相对关系可靠。
    只用frame0的深度构建3D点, frame1的点先用PnP初始化,
    再通过ICP在3D空间精炼——类似"PnP初始化 + 3D几何精炼"。

    比纯ICP好的地方: 不依赖frame1深度, 只用frame1的2D+XY几何。
    比纯PnP好的地方: 最终位姿由3D整体几何决定, 而非2D重投影。

    Args:
        pts_3d_0:  (N,3) frame0 3D点 (来自depth0)
        pts_2d_0:  (N,2) frame0 2D坐标
        pts_2d_1:  (N,2) frame1 2D坐标 (跟踪结果)
        K:         (3,3) 相机内参

    Returns:
        (R, t, inlier_mask) or (None, None, None)
    """
    N = len(pts_3d_0)
    if N < 4:
        return None, None, None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Step 1: PnP 初始化 (给ICP一个好的起点)
    R_init, t_init = estimate_pose_pnp_best(pts_3d_0, pts_2d_1, K)
    if R_init is None:
        return None, None, None

    # Step 2: 用PnP位姿将frame0的3D点变换到frame1
    pts_3d_1_pnp = (R_init @ pts_3d_0.T).T + t_init  # (N, 3)

    # Step 3: 在frame1中, 只使用XY几何 (Z固定为PnP给出的深度)
    # 构建"单深度3D点": Z = Z_pnp, XY = 从frame1的2D观测反投影
    Z_from_pnp = pts_3d_1_pnp[:, 2]
    Z_clipped = np.clip(Z_from_pnp, 0.001, 10.0)
    X_from_2d = (pts_2d_1[:, 0] - cx) * Z_clipped / fx
    Y_from_2d = (pts_2d_1[:, 1] - cy) * Z_clipped / fy
    pts_3d_1_hybrid = np.stack([X_from_2d, Y_from_2d, Z_clipped], axis=1)

    # Step 4: ICP between pts_3d_0 and pts_3d_1_hybrid
    # 只用XY方向的约束 (Z方向由PnP提供)
    R_refined, t_refined, inlier_mask = estimate_pose_icp(
        pts_3d_0, pts_3d_1_hybrid,
        ransac_thresh=ransac_thresh,
        ransac_iter=ransac_iter,
        min_inlier_ratio=min_inlier_ratio
    )

    if R_refined is not None:
        return R_refined, t_refined, inlier_mask

    # 回退: 纯PnP
    return R_init, t_init, np.ones(N, dtype=bool)


def rotation_angle_deg(R):
    return np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)) * 180 / np.pi


# ═══════════════════════════════════════════════════════════
# LoFTR 特征匹配器 (替代 LK 光流)
# ═══════════════════════════════════════════════════════════

def init_loftr_matcher(device='cuda', pretrained='indoor'):
    """初始化 LoFTR 匹配器 (全局单例)

    Args:
        device: 'cuda' 或 'cpu'
        pretrained: 'outdoor' 或 'indoor'

    Returns:
        kornia LoFTR 模型
    """
    global _loftr_matcher, _loftr_device

    if _loftr_matcher is not None and _loftr_device == device:
        return _loftr_matcher

    from kornia.feature import LoFTR

    logger.info(f'初始化 LoFTR 匹配器 (pretrained={pretrained}, device={device})...')

    _loftr_matcher = LoFTR(pretrained=pretrained)
    _loftr_matcher = _loftr_matcher.to(device)
    _loftr_matcher.eval()
    _loftr_device = device

    logger.info('LoFTR 匹配器初始化完成')
    return _loftr_matcher


def _match_loftr_pair(matcher, img0_path, img1_path, device='cuda',
                      max_dim=840):
    """LoFTR 单帧对匹配

    Args:
        matcher: LoFTR 模型
        img0_path, img1_path: 图像路径
        device: 设备
        max_dim: 图像最大边长 (LoFTR 显存限制)

    Returns:
        kpts0: (N, 2) frame0 关键点 (x, y)
        kpts1: (N, 2) frame1 关键点 (x, y)
        conf:  (N,)  匹配置信度
    """
    # 读取并转为灰度图 (LoFTR 需要单通道)
    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    if img0 is None or img1 is None:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

    img0 = img0.astype(np.float32) / 255.0
    img1 = img1.astype(np.float32) / 255.0

    h, w = img0.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        # 保证尺寸能被 8 整除 (LoFTR 要求)
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        img0 = cv2.resize(img0, (new_w, new_h))
        img1 = cv2.resize(img1, (new_w, new_h))

    # 转为 tensor: (H, W) → (1, H, W)
    img0_t = torch.from_numpy(img0).unsqueeze(0).to(device)
    img1_t = torch.from_numpy(img1).unsqueeze(0).to(device)

    # LoFTR outdoor 预训练使用 ImageNet 灰度均值/标准差
    normalize = transforms.Normalize(mean=[0.485], std=[0.229])
    img0_t = normalize(img0_t).unsqueeze(0)  # (1,1,H,W) → (1,1,H,W)
    img1_t = normalize(img1_t).unsqueeze(0)

    with torch.no_grad():
        result = matcher({"image0": img0_t, "image1": img1_t})

    kpts0 = result['keypoints0'].cpu().numpy()
    kpts1 = result['keypoints1'].cpu().numpy()
    conf = result.get('confidence', None)
    if conf is not None:
        conf = conf.cpu().numpy()
    else:
        conf = np.ones(len(kpts0))

    # 缩放回原始坐标
    if scale != 1.0:
        kpts0 /= scale
        kpts1 /= scale

    return kpts0, kpts1, conf


def loftr_multi_frame_tracking(frames_paths, frame_idx, n_frames=5,
                               device='cuda', max_dim=840,
                               proximity_thresh=3.0):
    """LoFTR 独立匹配多帧追踪

    匹配策略 (独立匹配, 避免链式传播误差):
      1. frame0 与 frame1, frame2, ..., frameN-1 独立匹配
      2. 以 frame0→1 匹配的关键点为基准
      3. 通过 frame0 空间邻近度关联各独立匹配, 构建一致点集
      4. 返回 (N_frames, N_points, 2) 格式 tracks 数组

    Args:
        frames_paths: 图像路径列表
        frame_idx:   起始帧索引
        n_frames:    追踪窗口帧数
        device:      设备
        max_dim:     LoFTR 最大图像边长
        proximity_thresh: frame0 空间邻近关联阈值 (像素)

    Returns:
        tracks: (N_frames, N_points, 2) 或 None
    """
    if frame_idx + n_frames > len(frames_paths):
        logger.debug(f'LoFTR多帧追踪: 帧索引越界')
        return None

    matcher = init_loftr_matcher(device=device)

    # ── 第一对: (frame0, frame1) 作为基准 ──
    k0_ref, k1, _ = _match_loftr_pair(
        matcher, frames_paths[frame_idx],
        frames_paths[frame_idx + 1], device, max_dim)

    if len(k0_ref) < 20:
        logger.debug(f'LoFTR: frame0→1 匹配点不足 ({len(k0_ref)})')
        return None

    all_kpts = [k0_ref, k1]  # frame0, frame1

    # ── frame0 与其他帧独立匹配 ──
    for i in range(2, n_frames):
        k0_i, ki, _ = _match_loftr_pair(
            matcher, frames_paths[frame_idx],
            frames_paths[frame_idx + i], device, max_dim)

        if len(k0_i) < 20:
            logger.debug(f'LoFTR: frame0→{i} 匹配点不足 ({len(k0_i)})')
            return None

        # 通过 frame0 空间邻近度关联: k0_ref ↔ k0_i
        tree = cKDTree(k0_i)
        dists, idxs = tree.query(k0_ref, distance_upper_bound=proximity_thresh)
        valid = np.isfinite(dists)

        if np.sum(valid) < 10:
            logger.debug(f'LoFTR: frame0关联frame{i} 有效点不足 ({np.sum(valid)})')
            return None

        # 裁剪所有已有轨迹
        for j in range(len(all_kpts)):
            all_kpts[j] = all_kpts[j][valid]
        k0_ref = k0_ref[valid]  # 更新基准

        # 新帧关键点
        all_kpts.append(ki[idxs[valid]])

    # ── 统一长度 ──
    min_len = min(len(t) for t in all_kpts)
    if min_len < 10:
        logger.debug(f'LoFTR: 最终公共点太少 ({min_len})')
        return None

    tracks = np.array([t[:min_len] for t in all_kpts], dtype=np.float32)
    return tracks