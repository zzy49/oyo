#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4 模型 VO 对比增强版: 滑动窗口 pnp_scale + 深度边缘增强

增强项:
  1. 滑动窗口 pnp_scale: 不再使用单一全局 scale, 每 10 帧窗口重算局部 scale
  2. 深度边缘增强 (bilateral filtering): RGB 引导的联合双边滤波, 平滑平坦区+锐化边缘

用法:
  python zhong\test_v6_multi_model_enhanced.py
"""

import os, sys, json, time
import numpy as np
import cv2
import torch
import random
import PIL.Image as pil

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import test_v6_dyendovo as tv6
from v6_pipeline.utils import ModelManager, predict_depth, predict_depth_with_motion
from dyendovo_dataset import load_gt_poses as dyendo_load_gt_poses
from networks.litemono_decoder import LiteMonoDepthDecoder

# ═══════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════

SEQ_NAME = 'c1_transverse1_t1_v2'
DATA_ROOT = r'F:\dataset'
SEQ_DIR = os.path.join(DATA_ROOT, SEQ_NAME)
WARPED_RGB_DIR = os.path.join(SEQ_DIR, 'generated', 'rgb_warped')

OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'vo_depth_compare')
os.makedirs(OUT_DIR, exist_ok=True)

_LITEMONO_DIR = r'E:\data1\Lite-Mono-main'

# 滑动窗口大小
WINDOW_SIZE = 10

# 双边滤波参数
BILATERAL_D = 9          # 滤波直径
BILATERAL_SIGMA_COLOR = 75.0   # 颜色空间 sigma
BILATERAL_SIGMA_SPACE = 75.0   # 空间 sigma

MODEL_CONFIGS = [
    {'name': 'Monodepth2', 'path': r'C:\Users\Administrator\tmp\c3vd_md2_multi\models\weights_19', 'type': 'md2'},
    {'name': 'ManyDepth',  'path': r'C:\Users\Administrator\tmp\c3vd_manydepth_multi\models\weights_19', 'type': 'manydepth'},
    {'name': 'Lite-Mono',  'path': r'C:\Users\Administrator\tmp\c3vd_litemono_multi\models\weights_19', 'type': 'litemono'},
    {'name': 'Ours',       'path': r'e:\data1\monodepth2\models\depth', 'type': 'md2'},
    {'name': 'mdp_v5',     'path': r'C:\Users\Administrator\tmp\mdp_v5\models\weights_19', 'type': 'ours_enhanced'},
]


# ═══════════════════════════════════════════════
# 增强功能
# ═══════════════════════════════════════════════

def bilateral_filter_depth(depth_mm, rgb_path, target_size=None):
    """RGB 引导联合双边滤波: 平滑平坦区域, 锐化边缘.

    Args:
        depth_mm: (H, W) 深度图 mm
        rgb_path: RGB 图像路径 (作为引导图)
        target_size: (W, H) 可选, 深度图目标尺寸 (与 RGB 对齐)

    Returns:
        filtered_depth: (H, W) 滤波后深度图 mm
    """
    try:
        rgb = np.array(pil.open(rgb_path).convert('RGB'))
    except Exception:
        return depth_mm  # RGB 加载失败, 返回原深度

    # 确保尺寸一致
    if target_size and (depth_mm.shape[1], depth_mm.shape[0]) != target_size:
        depth_mm = cv2.resize(depth_mm, target_size, interpolation=cv2.INTER_LINEAR)

    if rgb.shape[:2] != depth_mm.shape[:2]:
        if rgb.shape[0] * rgb.shape[1] > depth_mm.shape[0] * depth_mm.shape[1]:
            rgb = cv2.resize(rgb, (depth_mm.shape[1], depth_mm.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        else:
            depth_mm = cv2.resize(depth_mm, (rgb.shape[1], rgb.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

    # 深度图需转为 uint8/float32 用于滤波 (0-255 归一化对深度值不合理, 直接用 float32)
    depth_f32 = depth_mm.astype(np.float32)
    rgb_u8 = rgb.astype(np.uint8)

    # 联合双边滤波
    try:
        filtered = cv2.ximgproc.jointBilateralFilter(
            rgb_u8, depth_f32, BILATERAL_D,
            BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    except AttributeError:
        # opencv-contrib 未安装, 回退到普通双边滤波
        # 归一化深度到 0-255 用于滤波
        d_min, d_max = depth_f32.min(), depth_f32.max()
        if d_max > d_min:
            depth_norm = ((depth_f32 - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_f32, dtype=np.uint8)
        filtered_norm = cv2.bilateralFilter(depth_norm, BILATERAL_D,
                                            BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
        filtered = (filtered_norm.astype(np.float32) / 255.0 * (d_max - d_min) + d_min)

    return filtered


def compute_sliding_window_scales(seq_dir, encoder, depth_decoder, motion_encoder,
                                   device, window_size=10):
    """预跑全序列 PnP, 计算每个窗口的局部 scale.

    返回: list of (frame_idx, local_scale)
    """
    from v6_pipeline.utils import init_loftr_matcher, _match_loftr_pair

    warp_dir = os.path.join(seq_dir, 'generated', 'rgb_warped')
    frames = sorted([f for f in os.listdir(warp_dir) if f.endswith('.png')])

    matcher = init_loftr_matcher(device=str(device))
    depth_cache = {}
    all_steps = []  # (fi, step_mm)

    TARGET_STEP_MM = 0.5  # 对齐 tv6 默认值
    MAX_ROT_DEG = 15.0
    MAX_TRANS_MM = 50.0
    MIN_MATCHES = 4

    nb = getattr(depth_decoder, 'num_bins', 0)
    K_orig = tv6.K_ORIG
    fx, fy = K_orig[0, 0], K_orig[1, 1]
    cx, cy = K_orig[0, 2], K_orig[1, 2]

    scale_w = 512.0 / 1350.0
    scale_h = 384.0 / 1080.0
    K_pnp = K_orig.copy()
    K_pnp[0, 0] *= scale_w; K_pnp[1, 1] *= scale_h
    K_pnp[0, 2] *= scale_w; K_pnp[1, 2] *= scale_h

    for fi in range(len(frames) - 1):
        img0 = os.path.join(warp_dir, frames[fi])
        img1 = os.path.join(warp_dir, frames[fi + 1])

        # LoFTR 匹配 (复用 cache)
        cache_path = os.path.join(seq_dir, 'loftr_cache', f'matches_{fi:04d}.npz')
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            k0, k1 = data['pts0'], data['pts1']
        else:
            k0, k1, _ = _match_loftr_pair(matcher, img0, img1, device=str(device), max_dim=840)
        if k0 is None or len(k0) < MIN_MATCHES:
            continue

        # 深度预测 (仅 frame 0, 用 global_scale=1.0 的原始深度)
        if fi not in depth_cache:
            if motion_encoder is not None:
                raw = predict_depth_with_motion(img0, img1, encoder, depth_decoder,
                                                motion_encoder, device, inverse=False)
            else:
                raw = predict_depth(img0, encoder, depth_decoder, device, num_bins=nb)
            depth_cache[fi] = raw

        depth = depth_cache[fi]

        # 反投影 + PnP
        valid_mask = (depth > 1.0) & (depth < 500.0)
        k0_val = k0[valid_mask[k0[:, 1].astype(int), k0[:, 0].astype(int)]]
        k1_val = k1[valid_mask[k0[:, 1].astype(int), k0[:, 0].astype(int)]]

        # 简化: 取所有匹配点的深度
        pts2d_list, pts3d_list = [], []
        for (u, v), (u1, v1) in zip(k0, k1):
            u_i, v_i = int(round(u)), int(round(v))
            if 0 <= v_i < depth.shape[0] and 0 <= u_i < depth.shape[1]:
                z = depth[v_i, u_i]
                if 1.0 < z < 500.0:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    pts3d_list.append([x, y, z])
                    pts2d_list.append([u1, v1])

        if len(pts3d_list) < 4:
            continue

        pts3d_np = np.array(pts3d_list, dtype=np.float64)
        pts2d_np = np.array(pts2d_list, dtype=np.float64)
        pts2d_np[:, 0] *= scale_w
        pts2d_np[:, 1] *= scale_h

        try:
            ret, rvec, tvec = cv2.solvePnP(
                pts3d_np.reshape(-1, 1, 3), pts2d_np.reshape(-1, 1, 2),
                K_pnp.astype(np.float64), None, flags=cv2.SOLVEPNP_EPNP)
            if ret:
                R, _ = cv2.Rodrigues(rvec)
                angle = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)) * 180 / np.pi)
                trans_norm = float(np.linalg.norm(tvec))
                if angle < MAX_ROT_DEG and trans_norm < MAX_TRANS_MM:
                    all_steps.append((fi, trans_norm))
        except cv2.error:
            continue

    if not all_steps:
        return {0: 1.0, len(frames): 1.0}, depth_cache

    # 按窗口计算局部 scale
    window_scales = {}
    global_median = float(np.median([s[1] for s in all_steps]))
    global_scale = TARGET_STEP_MM / max(global_median, 0.001)

    for w_start in range(0, len(frames) - 1, window_size):
        w_end = min(w_start + window_size, len(frames) - 1)
        w_steps = [s[1] for s in all_steps if w_start <= s[0] < w_end]
        if w_steps:
            w_median = float(np.median(w_steps))
            w_scale = TARGET_STEP_MM / max(w_median, 0.001)
            # 平滑: 混合全局和局部 scale
            window_scales[w_start] = 0.7 * global_scale + 0.3 * w_scale
        else:
            window_scales[w_start] = global_scale

    window_scales[len(frames)] = global_scale  # 兜底

    return window_scales, depth_cache


def get_window_scale(window_scales, frame_idx):
    """根据帧索引获取对应的窗口 scale."""
    keys = sorted(window_scales.keys())
    for i, k in enumerate(keys):
        if frame_idx < k:
            return window_scales[keys[max(0, i - 1)]]
    return window_scales[keys[-1]]


# ═══════════════════════════════════════════════
# 模型加载 (复用原版)
# ═══════════════════════════════════════════════

def load_manydepth_model(model_path, device):
    from networks import ResnetEncoder, DepthDecoder
    encoder = ResnetEncoder(18, False)
    state_dict = torch.load(os.path.join(model_path, "mono_encoder.pth"), map_location=device)
    for key in ['height', 'width', 'use_stereo']:
        if key in state_dict: del state_dict[key]
    encoder.load_state_dict(state_dict)
    encoder.eval().to(device)
    encoder.depth_range = (2.0, 100.0)  # ManyDepth 训练 min_depth/max_depth
    depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(4), num_bins=0)
    depth_state = torch.load(os.path.join(model_path, "mono_depth.pth"), map_location=device)
    depth_decoder.load_state_dict(depth_state, strict=False)
    depth_decoder.eval().to(device)
    return encoder, depth_decoder


def load_litemono_model(model_path, device):
    import importlib.util, importlib.machinery
    _orig_path = sys.path.copy()
    _cached_layers = sys.modules.get('layers')
    litemono_networks = os.path.join(_LITEMONO_DIR, 'networks')
    sys.path.insert(0, _LITEMONO_DIR)
    try:
        loader_layers = importlib.machinery.SourceFileLoader('_litemono_layers', os.path.join(_LITEMONO_DIR, 'layers.py'))
        spec_layers = importlib.util.spec_from_loader('_litemono_layers', loader_layers)
        mod_layers = importlib.util.module_from_spec(spec_layers)
        sys.modules['_litemono_layers'] = mod_layers; sys.modules['layers'] = mod_layers
        spec_layers.loader.exec_module(mod_layers)
        loader_enc = importlib.machinery.SourceFileLoader('_litemono_enc', os.path.join(litemono_networks, 'depth_encoder.py'))
        spec_enc = importlib.util.spec_from_loader('_litemono_enc', loader_enc)
        mod_enc = importlib.util.module_from_spec(spec_enc)
        sys.modules['_litemono_enc'] = mod_enc; spec_enc.loader.exec_module(mod_enc)
        LiteMono = mod_enc.LiteMono
        loader_dec = importlib.machinery.SourceFileLoader('_litemono_dec', os.path.join(litemono_networks, 'depth_decoder.py'))
        spec_dec = importlib.util.spec_from_loader('_litemono_dec', loader_dec)
        mod_dec = importlib.util.module_from_spec(spec_dec)
        sys.modules['_litemono_dec'] = mod_dec; spec_dec.loader.exec_module(mod_dec)
        LitemonoDepthDecoder = mod_dec.DepthDecoder
    finally:
        sys.path = _orig_path
        if _cached_layers is not None: sys.modules['layers'] = _cached_layers
    encoder = LiteMono(model='lite-mono', height=192, width=640)
    encoder.feed_size = (640, 192)  # Lite-Mono 训练输入尺寸 (width, height)
    encoder.depth_range = (2.0, 100.0)  # Lite-Mono 训练 min_depth/max_depth
    state_dict = torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    for key in ['height', 'width', 'use_stereo']:
        if key in state_dict: del state_dict[key]
    encoder.load_state_dict(state_dict)
    encoder.eval().to(device)
    depth_decoder = LitemonoDepthDecoder(num_ch_enc=np.array([48, 80, 128]), scales=range(3))
    depth_state = torch.load(os.path.join(model_path, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(depth_state)
    depth_decoder.eval().to(device)
    return encoder, depth_decoder


def load_ours_enhanced_model(model_path, device):
    """加载 Ours+Enhanced 模型 (ResNet18 encoder + LiteMonoDepthDecoder + MotionEncoder)."""
    from networks import ResnetEncoder, MotionEncoder

    encoder = ResnetEncoder(18, False)
    state_dict = torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    for key in ['height', 'width', 'use_stereo']:
        if key in state_dict:
            del state_dict[key]
    encoder.load_state_dict(state_dict)
    encoder.eval().to(device)

    depth_decoder = LiteMonoDepthDecoder(
        num_ch_enc=np.array([64, 64, 128, 256, 512]),
        scales=range(3), num_output_channels=1, use_skips=True,
        num_bins=64)
    depth_state = torch.load(os.path.join(model_path, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(depth_state, strict=False)
    depth_decoder.eval().to(device)

    motion_encoder = None
    motion_path = os.path.join(model_path, "motion_encoder.pth")
    if os.path.isfile(motion_path):
        motion_encoder = MotionEncoder()
        motion_state = torch.load(motion_path, map_location=device)
        for key in ['height', 'width', 'use_stereo']:
            if key in motion_state:
                del motion_state[key]
        motion_encoder.load_state_dict(motion_state)
        motion_encoder.eval().to(device)

    return encoder, depth_decoder, motion_encoder


def load_model(cfg, device):
    model_type = cfg['type']
    model_path = cfg['path']
    name = cfg['name']
    if model_type == 'manydepth':
        print(f'  [{name}] 加载 ManyDepth 模型...')
        enc, dec = load_manydepth_model(model_path, device)
        return enc, dec, None
    elif model_type == 'litemono':
        print(f'  [{name}] 加载 Lite-Mono 模型...')
        enc, dec = load_litemono_model(model_path, device)
        return enc, dec, None
    elif model_type == 'ours_enhanced':
        print(f'  [{name}] 加载 Ours+Enhanced 模型 (Lite-Mono decoder + temporal consistency)...')
        enc, dec, mot = load_ours_enhanced_model(model_path, device)
        return enc, dec, mot
    else:
        mgr = ModelManager(model_path=model_path, device=device)
        encoder, depth_decoder, _, motion_encoder = mgr.load_model()
        if name == 'Monodepth2':
            encoder.depth_range = (2.0, 100.0)  # Monodepth2 训练 min_depth/max_depth
        has_motion = motion_encoder is not None
        print(f'  [{name}] 加载 ModelManager 模型 (motion_encoder={"有" if has_motion else "无"})')
        return encoder, depth_decoder, motion_encoder


# ═══════════════════════════════════════════════
# RPE 计算
# ═══════════════════════════════════════════════

def rotation_angle_deg(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)) * 180 / np.pi)


def compute_rpe(abs_poses, gt_poses_std):
    n = min(len(abs_poses), len(gt_poses_std))
    rpe_trans, rpe_rot = [], []
    for i in range(n - 1):
        T_cw_gt_i = np.linalg.inv(gt_poses_std[i].T)
        T_cw_gt_i1 = np.linalg.inv(gt_poses_std[i + 1].T)
        dP_gt = np.linalg.inv(T_cw_gt_i) @ T_cw_gt_i1
        T_cw_vo_i = np.linalg.inv(abs_poses[i])
        T_cw_vo_i1 = np.linalg.inv(abs_poses[i + 1])
        dP_vo = np.linalg.inv(T_cw_vo_i) @ T_cw_vo_i1
        trans_err = float(np.linalg.norm(dP_vo[:3, 3] - dP_gt[:3, 3]))
        rpe_trans.append(trans_err)
        dR = dP_vo[:3, :3].T @ dP_gt[:3, :3]
        rot_err = rotation_angle_deg(dR)
        rpe_rot.append(rot_err)
    rpe_trans = np.array(rpe_trans)
    rpe_rot = np.array(rpe_rot)
    return {
        'rpe_t_mean': float(np.mean(rpe_trans)), 'rpe_t_std': float(np.std(rpe_trans)),
        'rpe_t_rmse': float(np.sqrt(np.mean(rpe_trans ** 2))),
        'rpe_r_mean': float(np.mean(rpe_rot)), 'rpe_r_std': float(np.std(rpe_rot)),
        'rpe_r_rmse': float(np.sqrt(np.mean(rpe_rot ** 2))), 'n_rpe': len(rpe_trans),
    }


def compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std):
    ate_result = tv6.evaluate_trajectory(vo_traj, gt_traj, SEQ_NAME, 'baseline')
    abs_poses = chain_data['abs_poses']
    rpe = compute_rpe(abs_poses, gt_poses_std)
    scale_error_pct = abs(ate_result.get('scale', 1.0) - 1.0) * 100.0
    return {
        'ate_mean': ate_result.get('mean_mm', float('nan')),
        'ate_std': ate_result.get('std_mm', float('nan')),
        'ate_rmse': ate_result.get('rmse_mm', float('nan')),
        'ate_median': ate_result.get('median_mm', float('nan')),
        'rpe_t_mean': rpe['rpe_t_mean'], 'rpe_t_std': rpe['rpe_t_std'],
        'rpe_t_rmse': rpe['rpe_t_rmse'], 'rpe_r_mean': rpe['rpe_r_mean'],
        'rpe_r_std': rpe['rpe_r_std'], 'rpe_r_rmse': rpe['rpe_r_rmse'],
        'scale_error_pct': scale_error_pct,
        'umeyama_scale': ate_result.get('scale', float('nan')),
        'n_rpe': rpe['n_rpe'],
    }


# ═══════════════════════════════════════════════
# 增强版 VO 管线
# ═══════════════════════════════════════════════

def run_vo_enhanced(encoder, depth_decoder, motion_encoder, device,
                     use_sliding_window=True, use_bilateral=True):
    """增强版 VO: 滑动窗口 pnp_scale + 双边滤波.

    流程:
      1. 先运行标准 calibration (获取 depth_cache + global_scale)
      2. 如果滑动窗口: 运行 compute_sliding_window_scales 获取窗口 scale
      3. 对 depth_cache 中每帧深度应用对应窗口的 scale
      4. 如果双边滤波: 对每帧深度做 RGB 引导滤波
      5. 用 global_scale=1.0 运行 VO (深度已预缩放)
    """
    # ── 1. 标准校准 (获取 depth_cache) ──
    print('  [增强] 标准 pnp_scale 校准...')
    t0 = time.time()
    vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
        SEQ_NAME, encoder, depth_decoder, device,
        motion_net=None, mode='baseline', max_frames=None,
        motion_encoder=motion_encoder, depth_source='pred',
        loftr_source='cache', data_root=DATA_ROOT,
        pipeline_params=None, no_gt=False, zero_gt_mode='pnp_scale')
    print(f'  标准校准完成: {time.time()-t0:.0f}s')

    # ── 2. 滑动窗口 scale ──
    if use_sliding_window:
        print(f'  [增强] 滑动窗口 pnp_scale (窗口={WINDOW_SIZE}帧)...')
        window_scales, window_depths = compute_sliding_window_scales(
            SEQ_DIR, encoder, depth_decoder, motion_encoder, device, WINDOW_SIZE)
        n_windows = len(window_scales)
        print(f'  [增强] 计算了 {n_windows} 个窗口的局部 scale')

        # 预缩放 depth_cache
        pipeline_params = tv6.estimate_pipeline_params(
            SEQ_DIR, encoder, depth_decoder, motion_encoder, device,
            depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')

        # 用窗口 scale 替代全局 scale
        scaled_cache = {}
        for fi, d in pipeline_params['depth_cache'].items():
            w_scale = get_window_scale(window_scales, fi)
            scaled_cache[fi] = d * w_scale

        # 可选: 双边滤波
        if use_bilateral:
            warp_dir = os.path.join(SEQ_DIR, 'generated', 'rgb_warped')
            frames = sorted([f for f in os.listdir(warp_dir) if f.endswith('.png')])
            print(f'  [增强] 双边滤波 (d={BILATERAL_D}, sigma_c={BILATERAL_SIGMA_COLOR})...')
            for fi in list(scaled_cache.keys()):
                if fi < len(frames):
                    rgb_path = os.path.join(warp_dir, frames[fi])
                    scaled_cache[fi] = bilateral_filter_depth(scaled_cache[fi], rgb_path)

        pipeline_params['depth_cache'] = scaled_cache
        pipeline_params['global_depth_scale'] = 1.0  # 深度已预缩放

        # ── 3. 用增强参数运行 VO ──
        print(f'  [增强] 用滑动窗口+双边滤波运行 VO...')
        t0 = time.time()
        vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
            SEQ_NAME, encoder, depth_decoder, device,
            motion_net=None, mode='baseline', max_frames=None,
            motion_encoder=motion_encoder, depth_source='pred',
            loftr_source='cache', data_root=DATA_ROOT,
            pipeline_params=pipeline_params, no_gt=False, zero_gt_mode='pnp_scale')
        print(f'  增强 VO 完成: {time.time()-t0:.0f}s')

    return vo_traj, gt_traj, stats, chain_data


# ═══════════════════════════════════════════════
# 主流程: 每个模型跑 baseline + enhanced 双轨对比
# ═══════════════════════════════════════════════

def format_delta(base_val, enh_val, lower_is_better=True):
    """格式化变化量: 正值=改善(green), 负值=恶化(red)."""
    if base_val is None or enh_val is None:
        return "   N/A"
    delta = base_val - enh_val
    if lower_is_better:
        sign = "↓" if delta > 0 else "↑"
    else:
        sign = "↑" if delta > 0 else "↓"
    return f"{sign}{abs(delta):.2f}"


def print_model_compare(name, base, enh):
    """打印单个模型的 baseline vs enhanced 对比."""
    print(f"\n  {'─' * 55}")
    print(f"  [{name}] Baseline → Enhanced 对比")
    print(f"  {'─' * 55}")
    print(f"  {'指标':<18} {'Baseline':>16} {'Enhanced':>16} {'变化':>10}")
    print(f"  {'-' * 55}")

    rows = [
        ('ATE (mm)',         base['ate_mean'],        enh['ate_mean'],        True),
        ('ATE Std (mm)',     base['ate_std'],         enh['ate_std'],         True),
        ('RPE-T (mm)',       base['rpe_t_mean'],      enh['rpe_t_mean'],      True),
        ('RPE-T Std (mm)',   base['rpe_t_std'],       enh['rpe_t_std'],       True),
        ('RPE-R (°)',        base['rpe_r_mean'],       enh['rpe_r_mean'],       True),
        ('RPE-R Std (°)',    base['rpe_r_std'],        enh['rpe_r_std'],        True),
        ('Scale Error (%)',  base['scale_error_pct'],  enh['scale_error_pct'],  True),
        ('Success',          base['n_success'],        enh['n_success'],        False),
    ]
    for label, b, e, lower_better in rows:
        delta_str = format_delta(b, e, lower_better)
        print(f"  {label:<18} {b:>16.3f} {e:>16.3f} {delta_str:>10}")
    print(f"  {'─' * 55}")


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"序列: {SEQ_NAME}")
    print(f"对比模式: Baseline (原始) vs Enhanced (滑动窗口+双边滤波)")
    print(f"LoFTR: cache (所有模型共用)\n")

    gt_poses_std = dyendo_load_gt_poses(SEQ_DIR)
    print(f"GT: {len(gt_poses_std)} 帧\n")

    all_baseline = {}
    all_enhanced = {}

    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        print(f"\n{'=' * 65}")
        print(f"  [{name}]")
        print(f"  模型路径: {cfg['path']}")
        print(f"{'=' * 65}")

        if not os.path.isdir(cfg['path']):
            print(f"  模型目录不存在, 跳过\n")
            continue

        encoder, depth_decoder, motion_encoder = load_model(cfg, device)

        # ── Pass 1: Baseline (无任何增强) ──
        print(f"\n  >>> Pass 1/2: Baseline (原始 pnp_scale, 无滤波) <<<")
        t0 = time.time()
        vo_traj_b, gt_traj_b, stats_b, chain_data_b = run_vo_enhanced(
            encoder, depth_decoder, motion_encoder, device,
            use_sliding_window=False, use_bilateral=False)
        elapsed_b = time.time() - t0
        metrics_b = compute_metrics(vo_traj_b, gt_traj_b, chain_data_b, gt_poses_std)
        metrics_b['n_frames'] = len(vo_traj_b)
        metrics_b['n_success'] = stats_b['n_success']
        metrics_b['n_pairs'] = stats_b['n_total']
        metrics_b['elapsed_s'] = round(elapsed_b, 1)
        all_baseline[name] = metrics_b
        print(f"  [Baseline] ATE={metrics_b['ate_mean']:.2f}mm, "
              f"RPE-T={metrics_b['rpe_t_mean']:.3f}mm, "
              f"RPE-R={metrics_b['rpe_r_mean']:.2f}°, "
              f"Scale Err={metrics_b['scale_error_pct']:.1f}%, "
              f"Success={stats_b['n_success']}/{stats_b['n_total']}")

        # ── Pass 2: Enhanced (滑动窗口 + 双边滤波) ──
        print(f"\n  >>> Pass 2/2: Enhanced (滑动窗口 + 双边滤波) <<<")
        t0 = time.time()
        vo_traj_e, gt_traj_e, stats_e, chain_data_e = run_vo_enhanced(
            encoder, depth_decoder, motion_encoder, device,
            use_sliding_window=True, use_bilateral=True)
        elapsed_e = time.time() - t0
        metrics_e = compute_metrics(vo_traj_e, gt_traj_e, chain_data_e, gt_poses_std)
        metrics_e['n_frames'] = len(vo_traj_e)
        metrics_e['n_success'] = stats_e['n_success']
        metrics_e['n_pairs'] = stats_e['n_total']
        metrics_e['elapsed_s'] = round(elapsed_e, 1)
        all_enhanced[name] = metrics_e
        print(f"  [Enhanced] ATE={metrics_e['ate_mean']:.2f}mm, "
              f"RPE-T={metrics_e['rpe_t_mean']:.3f}mm, "
              f"RPE-R={metrics_e['rpe_r_mean']:.2f}°, "
              f"Scale Err={metrics_e['scale_error_pct']:.1f}%, "
              f"Success={stats_e['n_success']}/{stats_e['n_total']}")

        # ── 单模型对比 ──
        print_model_compare(name, metrics_b, metrics_e)

        del encoder, depth_decoder, motion_encoder
        del vo_traj_b, gt_traj_b, chain_data_b, vo_traj_e, gt_traj_e, chain_data_e
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 保存结果 ──
    result_base = os.path.join(OUT_DIR, 'multi_model_results_baseline.json')
    result_enhanced = os.path.join(OUT_DIR, 'multi_model_results_enhanced.json')
    with open(result_base, 'w') as f:
        json.dump(all_baseline, f, indent=2)
    with open(result_enhanced, 'w') as f:
        json.dump(all_enhanced, f, indent=2)
    print(f"\nBaseline 结果: {result_base}")
    print(f"Enhanced 结果: {result_enhanced}")

    # ── 四模型 Baseline vs Enhanced 汇总对比表 ──
    print(f"\n{'=' * 110}")
    print(f"  四模型 Baseline → Enhanced 汇总对比")
    print(f"{'=' * 110}")
    print(f"  {'Model':<12} {'ATE Base':>9} {'ATE Enh':>9} {'ATE Δ':>9}  "
          f"{'RPE-T Base':>11} {'RPE-T Enh':>11} {'RPE-T Δ':>9}  "
          f"{'Scale Base':>11} {'Scale Enh':>11} {'Scale Δ':>9}")
    print(f"  {'-' * 108}")
    for name in [c['name'] for c in MODEL_CONFIGS]:
        if name not in all_baseline or name not in all_enhanced:
            continue
        b = all_baseline[name]
        e = all_enhanced[name]
        ate_d = format_delta(b['ate_mean'], e['ate_mean'], True)
        rpet_d = format_delta(b['rpe_t_mean'], e['rpe_t_mean'], True)
        scale_d = format_delta(b['scale_error_pct'], e['scale_error_pct'], True)
        print(f"  {name:<12} {b['ate_mean']:>8.2f} {e['ate_mean']:>8.2f} {ate_d:>8}  "
              f"{b['rpe_t_mean']:>10.3f} {e['rpe_t_mean']:>10.3f} {rpet_d:>8}  "
              f"{b['scale_error_pct']:>10.1f} {e['scale_error_pct']:>10.1f} {scale_d:>8}")
    print(f"{'=' * 110}")
    print(f"  ↓ = 改善 (数值降低)    ↑ = 恶化 (数值升高)")


if __name__ == '__main__':
    main()
