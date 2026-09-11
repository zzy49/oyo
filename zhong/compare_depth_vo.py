#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一 VO 后端深度模型对比 ── 对齐 test_v6_dyendovo.py 管线.

在 c1_transverse1_t1_v2 上用统一的 LoFTR+EPnP+自适应阈值 VO 后端，
对比不同深度模型的位姿估计精度。

管线: 与 test_v6_dyendovo.py baseline 模式一致
  - dyendovo_dataset.load_gt_poses() 加载 GT (标准 [R^T,C;0,1] 格式)
  - PnP 坐标缩放 (K_PNP, pts2d 缩放)
  - T_abs = T_abs @ T_rel 累积
  - 零配置 pnp_scale 深度校准
  - 自适应 3D 位移过滤 (k=0.5)
  - 帧间深度尺度一致性校正

输出指标:
  - ATE↓  (mm):    绝对轨迹误差 (Umeyama 对齐后 RMSE)
  - RPE-T↓ (mm):   相对平移误差 (帧间)
  - RPE-R↓ (°):    相对旋转误差 (帧间)
  - Scale Error↓ (%): Umeyama 尺度偏差

用法:
  python zhong\compare_depth_vo.py
"""

import os, sys, time, json
import numpy as np
import cv2
import torch
import random
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# Lite-Mono 模型根目录 (用 importlib 动态加载, 避免与项目 networks 包冲突)
_LITEMONO_DIR = r'E:\data1\Lite-Mono-main'

from v6_pipeline.utils import (
    ModelManager, predict_depth, predict_depth_with_motion,
    init_loftr_matcher, _match_loftr_pair,
)
from dyendovo_dataset import load_gt_poses as dyendo_load_gt_poses
from v6_pipeline.c3vd_loader import align_trajectory_umeyama, compute_ate

# ═══════════════════════════════════════════════════════════
# 配置 ── 与 test_v6_dyendovo.py 一致
# ═══════════════════════════════════════════════════════════

SEQ_DIR = r'F:\dataset\c1_transverse1_t1_v2'
RGB_DIR = os.path.join(SEQ_DIR, 'generated', 'rgb_warped')
OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'vo_depth_compare')
os.makedirs(OUT_DIR, exist_ok=True)

# 相机内参 (EndoSLAM 1350×1080)
K_ORIG = np.array([
    [767.73, 0,      677.74],
    [0,      767.73, 543.06],
    [0,      0,      1]
], dtype=np.float64)
FX, FY, CX, CY = K_ORIG[0, 0], K_ORIG[1, 1], K_ORIG[0, 2], K_ORIG[1, 2]
IMG_W_ORIG, IMG_H_ORIG = 1350, 1080

# PnP 坐标缩放 (匹配 eval_dyendovo K_rs)
MOTION_H, MOTION_W = 384, 512
SCALE_W_PNP = MOTION_W / IMG_W_ORIG
SCALE_H_PNP = MOTION_H / IMG_H_ORIG
K_PNP = K_ORIG.copy()
K_PNP[0, 0] *= SCALE_W_PNP
K_PNP[1, 1] *= SCALE_H_PNP
K_PNP[0, 2] *= SCALE_W_PNP
K_PNP[1, 2] *= SCALE_H_PNP

# 深度与匹配
DEPTH_MIN, DEPTH_MAX = 1.0, 500.0
LOFTR_MAX_DIM = 840

# PnP 拒绝阈值
MAX_ROTATION_DEG = 15.0
MAX_TRANSLATION_MM = 50.0

# 自适应位移过滤
DISP_K_FACTOR = 0.5

# 零配置尺度校准
TARGET_STEP_MM = 0.5       # PnP 帧间位移目标 (mm)
N_ESTIMATE_FRAMES = 0       # 预跑帧数 (0=全部, 对齐 test_v6)

# ── 测试模型 ──
MODELS_TO_EVAL = [
    ('Monodepth2', r'C:\Users\Administrator\tmp\c3vd_md2_full\models\weights_19'),
    ('ManyDepth',  r'C:\Users\Administrator\tmp\c3vd_manydepth_full\models\weights_4'),
    ('Lite-Mono',  r'C:\Users\Administrator\tmp\c3vd_litemono_full\models\weights_19'),
    ('Ours',       r'e:\data1\monodepth2\models\depth'),
]

# 统一推理分辨率 (对齐 test_v6_dyendovo.py 默认值)
FEED_H, FEED_W = 256, 320


# ═══════════════════════════════════════════════════════════
# 模型加载 ── ManyDepth / Lite-Mono 需要特殊处理
# ═══════════════════════════════════════════════════════════

def load_manydepth_model(model_path, device):
    """加载 ManyDepth 单帧推理模型 (mono_encoder.pth + mono_depth.pth)."""
    from networks import ResnetEncoder, DepthDecoder

    encoder = ResnetEncoder(18, False)
    state_dict = torch.load(os.path.join(model_path, "mono_encoder.pth"),
                          map_location=device)
    for key in ['height', 'width', 'use_stereo']:
        if key in state_dict:
            del state_dict[key]
    encoder.load_state_dict(state_dict)
    encoder.eval().to(device)

    # ManyDepth 单帧推理输出 sigmoid disparity (非 bin 模式)
    # ManyDepth 训练时 scales=[0,1,2,3]
    depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(4), num_bins=0)
    depth_state = torch.load(os.path.join(model_path, "mono_depth.pth"),
                             map_location=device)
    depth_decoder.load_state_dict(depth_state, strict=False)
    depth_decoder.eval().to(device)

    return encoder, depth_decoder


def load_litemono_model(model_path, device):
    """加载 Lite-Mono 模型 (LiteMono encoder + 自定义 DepthDecoder).

    使用 importlib 动态加载 Lite-Mono 的 networks 模块,
    避免与项目自身的 networks 包冲突.
    注意: 项目 layers.py (upsample(x) 无 mode) 被 v6_pipeline.utils 导入并缓存,
    需要先显式导入 Lite-Mono 的 layers 并覆盖 sys.modules['layers'].
    """
    import importlib.util, importlib.machinery

    _orig_path = sys.path.copy()
    # 保存原 layers 缓存, 先回退 sys.path 让 importlib 正确解析
    _cached_layers = sys.modules.get('layers')

    litemono_networks = os.path.join(_LITEMONO_DIR, 'networks')

    # 显式加载 Lite-Mono 版 layers 并覆盖 sys.modules
    sys.path.insert(0, _LITEMONO_DIR)
    try:
        loader_layers = importlib.machinery.SourceFileLoader(
            '_litemono_layers',
            os.path.join(_LITEMONO_DIR, 'layers.py'))
        spec_layers = importlib.util.spec_from_loader('_litemono_layers', loader_layers)
        mod_layers = importlib.util.module_from_spec(spec_layers)
        sys.modules['_litemono_layers'] = mod_layers
        sys.modules['layers'] = mod_layers  # 覆盖项目版本
        spec_layers.loader.exec_module(mod_layers)

        # 用唯一模块名加载 depth_encoder, 避免与项目 networks 冲突
        loader_enc = importlib.machinery.SourceFileLoader(
            '_litemono_enc',
            os.path.join(litemono_networks, 'depth_encoder.py'))
        spec_enc = importlib.util.spec_from_loader('_litemono_enc', loader_enc)
        mod_enc = importlib.util.module_from_spec(spec_enc)
        sys.modules['_litemono_enc'] = mod_enc
        spec_enc.loader.exec_module(mod_enc)
        LiteMono = mod_enc.LiteMono

        # 加载 depth_decoder (from layers import * 拿到 Lite-Mono 版本)
        loader_dec = importlib.machinery.SourceFileLoader(
            '_litemono_dec',
            os.path.join(litemono_networks, 'depth_decoder.py'))
        spec_dec = importlib.util.spec_from_loader('_litemono_dec', loader_dec)
        mod_dec = importlib.util.module_from_spec(spec_dec)
        sys.modules['_litemono_dec'] = mod_dec
        spec_dec.loader.exec_module(mod_dec)
        LitemonoDepthDecoder = mod_dec.DepthDecoder
    finally:
        sys.path = _orig_path
        # 恢复项目 layers (Lite-Mono depth_decoder 已持有正确函数引用)
        if _cached_layers is not None:
            sys.modules['layers'] = _cached_layers

    encoder = LiteMono(model='lite-mono', height=192, width=640)
    state_dict = torch.load(os.path.join(model_path, "encoder.pth"),
                          map_location=device)
    for key in ['height', 'width', 'use_stereo']:
        if key in state_dict:
            del state_dict[key]
    encoder.load_state_dict(state_dict)
    encoder.eval().to(device)

    depth_decoder = LitemonoDepthDecoder(num_ch_enc=np.array([48, 80, 128]), scales=range(3))
    depth_state = torch.load(os.path.join(model_path, "depth.pth"),
                             map_location=device)
    depth_decoder.load_state_dict(depth_state)
    depth_decoder.eval().to(device)

    return encoder, depth_decoder


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def rotation_angle_deg(R):
    """旋转矩阵 → 角度 (度)."""
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)) * 180 / np.pi)


def _filter_by_3d_displacement(pts3d, k0, k1, depth_curr, depth_next,
                               K, k_factor=DISP_K_FACTOR):
    """自适应 3D 位移过滤 (与 run_pipeline.py 一致)."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    N = len(pts3d)
    h_dn, w_dn = depth_next.shape

    displacements = np.full(N, np.nan, dtype=np.float64)
    for j in range(N):
        u1, v1 = k1[j]
        ui1 = int(np.clip(u1, 0, w_dn - 1))
        vi1 = int(np.clip(v1, 0, h_dn - 1))
        Z1 = float(depth_next[vi1, ui1])
        if Z1 <= 0.5 or Z1 > DEPTH_MAX or not np.isfinite(Z1):
            continue
        X1 = (u1 - cx) * Z1 / fx
        Y1 = (v1 - cy) * Z1 / fy
        p1 = np.array([X1, Y1, Z1])
        displacements[j] = float(np.linalg.norm(p1 - pts3d[j]))

    valid = ~np.isnan(displacements)
    if valid.sum() < 4:
        return np.ones(N, dtype=bool)

    valid_disp = displacements[valid]
    med = float(np.median(valid_disp))
    std = float(np.std(valid_disp))
    threshold = med + k_factor * std
    threshold = max(threshold, 1.0)

    keep = displacements <= threshold
    if keep.sum() < 4:
        sorted_idx = np.argsort(displacements)
        n_keep = min(N, max(4, N // 2))
        keep = np.zeros(N, dtype=bool)
        keep[sorted_idx[:n_keep]] = True

    return keep


def umeyama_alignment(est, gt):
    """Umeyama 相似变换对齐 (仅平移)."""
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]

    est_mean = est.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    est_c = est - est_mean
    gt_c = gt - gt_mean

    cov = est_c.T @ gt_c / n
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    est_var = (est_c ** 2).sum() / n
    scale = np.sum(S) / est_var if est_var > 1e-10 else 1.0

    aligned = scale * (est_c @ R.T) + gt_mean
    errors = np.linalg.norm(aligned - gt, axis=1)
    return aligned, errors, R, float(scale)


def compute_all_metrics(abs_poses, gt_poses_std):
    """计算 ATE, RPE-T, RPE-R, Scale Error.

    Args:
        abs_poses:     list of (4,4) VO 估计 cam→world 4×4 位姿
        gt_poses_std:  list of (4,4), dyendovo_dataset 标准格式 [R^T, C; 0, 1]

    Returns:
        dict: ATE/RPE/Scale 指标
    """
    # ── 提取相机位置 ──
    vo_positions = np.array([p[:3, 3] for p in abs_poses], dtype=np.float64)
    gt_positions = np.array([p[:3, 3] for p in gt_poses_std], dtype=np.float64)

    # 截取最短
    n = min(len(vo_positions), len(gt_positions))
    vo_positions = vo_positions[:n]
    gt_positions = gt_positions[:n]

    # ── ATE (Umeyama 对齐后 RMSE) ──
    aligned, ate_errors, R_align, umeyama_s = umeyama_alignment(vo_positions, gt_positions)
    ate_rmse = float(np.sqrt(np.mean(ate_errors ** 2)))
    ate_mean = float(np.mean(ate_errors))
    ate_std = float(np.std(ate_errors))

    # ── Scale Error ──
    scale_error_pct = abs(umeyama_s - 1.0) * 100.0

    # ── RPE (帧间相对位姿误差) ──
    # GT: [R^T, C; 0, 1] → T_cam_world = [R, C; 0, 1]
    # VO: cam→world (direct)
    # RPE uses T_world_cam = inv(T_cam_world) for both GT and VO
    rpe_trans = []
    rpe_rot = []

    for i in range(n - 1):
        # GT: gt_poses_std[i] = [R_i^T, C_i; 0, 1]
        # T_cam_world_gt = [R_i, C_i; 0, 1] = gt_poses_std[i].T
        # T_world_cam_gt = inv([R_i, C_i; 0, 1])
        T_cw_gt_i = np.linalg.inv(gt_poses_std[i].T)
        T_cw_gt_i1 = np.linalg.inv(gt_poses_std[i + 1].T)
        dP_gt = np.linalg.inv(T_cw_gt_i) @ T_cw_gt_i1

        # VO: cam→world, need to convert to world→cam
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
        'ate_mean': ate_mean,
        'ate_std': ate_std,
        'ate_rmse': ate_rmse,
        'rpe_t_mean': float(np.mean(rpe_trans)),
        'rpe_t_std': float(np.std(rpe_trans)),
        'rpe_t_rmse': float(np.sqrt(np.mean(rpe_trans ** 2))),
        'rpe_r_mean': float(np.mean(rpe_rot)),
        'rpe_r_std': float(np.std(rpe_rot)),
        'rpe_r_rmse': float(np.sqrt(np.mean(rpe_rot ** 2))),
        'scale_error_pct': scale_error_pct,
        'umeyama_scale': umeyama_s,
        'n_frames': n,
        'n_rpe': len(rpe_trans),
    }


# ═══════════════════════════════════════════════════════════
# 零配置尺度校准 (对齐 test_v6: _estimate_scale_zero_gt)
# ═══════════════════════════════════════════════════════════

def estimate_scale_from_pnp(model_name, encoder, depth_decoder, motion_encoder,
                            matcher, frames, device, n_est=N_ESTIMATE_FRAMES):
    """预跑 VO 前 n_est 帧, 统计 PnP 帧间位移中位数反推全局深度尺度.

    与 test_v6_dyendovo._estimate_scale_zero_gt(pnp_scale) 对齐.
    global_scale = TARGET_STEP_MM / median(valid_pnp_steps)
    """
    nb = getattr(depth_decoder, 'num_bins', 0)
    depth_cache = {}
    steps = []
    if n_est <= 0:
        n_est = len(frames) - 1
    n_est = min(n_est, len(frames) - 1)

    pbar = tqdm(range(n_est), desc=f'{model_name} ScaleEst')
    for fi in pbar:
        img0_path = os.path.join(RGB_DIR, frames[fi])
        img1_path = os.path.join(RGB_DIR, frames[fi + 1])

        k0, k1, _ = _match_loftr_pair(
            matcher, img0_path, img1_path,
            device=str(device), max_dim=LOFTR_MAX_DIM)
        if k0 is None or len(k0) < 4:
            continue

        if fi not in depth_cache:
            if motion_encoder is not None:
                raw = predict_depth_with_motion(
                    img0_path, img1_path, encoder, depth_decoder,
                    motion_encoder, device, inverse=False,
                    min_depth=DEPTH_MIN, max_depth=DEPTH_MAX)
            else:
                raw = predict_depth(img0_path, encoder, depth_decoder, device,
                                   feed_h=FEED_H, feed_w=FEED_W,
                                   inverse=False, min_depth=DEPTH_MIN,
                                   max_depth=DEPTH_MAX, num_bins=nb)
            depth_cache[fi] = raw
        depth0 = depth_cache[fi]

        h_d, w_d = depth0.shape
        pts3d, pts2d = [], []
        for j in range(len(k0)):
            u, v = k0[j]
            ui, vi = int(np.clip(u, 0, w_d - 1)), int(np.clip(v, 0, h_d - 1))
            Z = float(depth0[vi, ui])
            if not (0.5 < Z < DEPTH_MAX and np.isfinite(Z)):
                continue
            X = (u - CX) * Z / FX
            Y = (v - CY) * Z / FY
            pts3d.append([X, Y, Z])
            pts2d.append([k1[j][0], k1[j][1]])

        if len(pts3d) < 4:
            pbar.set_postfix({'m': len(k0), '3d': len(pts3d), 'steps': len(steps)})
            continue

        pts3d_np = np.array(pts3d, dtype=np.float32)
        pts2d_np = np.array(pts2d, dtype=np.float32)
        # 缩放 pts2d 到 PnP 空间 (对齐 test_v6)
        pts2d_np[:, 0] *= SCALE_W_PNP
        pts2d_np[:, 1] *= SCALE_H_PNP

        ret, t_norm, angle = False, 0.0, 0.0
        try:
            ret, rvec, tvec = cv2.solvePnP(
                pts3d_np.reshape(-1, 1, 3).astype(np.float64),
                pts2d_np.reshape(-1, 1, 2).astype(np.float64),
                K_PNP.astype(np.float64), None, flags=cv2.SOLVEPNP_EPNP)
            if ret:
                R, _ = cv2.Rodrigues(rvec)
                t_norm = float(np.linalg.norm(tvec))
                angle = rotation_angle_deg(R)
                if angle <= MAX_ROTATION_DEG and t_norm <= MAX_TRANSLATION_MM:
                    steps.append(t_norm)
        except cv2.error:
            ret = False

        pbar.set_postfix({'m': len(k0), '3d': len(pts3d_np),
                          'rot': f'{angle:.1f}°' if ret else 'fail',
                          't': f'{t_norm:.3f}' if ret else '-',
                          'steps': len(steps)})

    if len(steps) < 5:
        print(f'  [ScaleEst] ⚠ 有效 PnP 步数不足 ({len(steps)}), 回退 scale=1.0')
        return 1.0, depth_cache

    steps = np.array(steps)
    lo, hi = np.percentile(steps, [5, 95])
    steps_clean = steps[(steps >= lo) & (steps <= hi)]
    obs_step = float(np.median(steps_clean))
    scale = TARGET_STEP_MM / obs_step if obs_step > 0.001 else 1.0
    print(f'  [ScaleEst] PnP 位移中位数: observed={obs_step:.4f}mm '
          f'→ global_scale={scale:.4f} ({len(steps)} 有效对)')
    return scale, depth_cache


# ═══════════════════════════════════════════════════════════
# VO 管线 (对齐 test_v6: run_vo_sequence baseline 模式)
# ═══════════════════════════════════════════════════════════

def run_vo_for_model(model_name, model_path, frames, device,
                     calibrate_scale=True):
    """对单个深度模型运行 VO 管线.

    与 test_v6_dyendovo.run_vo_sequence(baseline) 对齐.
    """
    # ── 加载模型 ──
    motion_encoder = None
    if model_name == 'Lite-Mono':
        print('  使用 Lite-Mono 架构加载…')
        encoder, depth_decoder = load_litemono_model(model_path, device)
        nb = 0  # Lite-Mono 使用 sigmoid disparity 输出
    elif model_name == 'ManyDepth':
        print('  使用 ManyDepth 单帧推理加载 (mono_*.pth)…')
        encoder, depth_decoder = load_manydepth_model(model_path, device)
        nb = 0  # mono_depth 输出 sigmoid disparity
    else:
        mgr = ModelManager(model_path=model_path, device=device)
        encoder, depth_decoder, _, motion_encoder = mgr.load_model()
        nb = getattr(depth_decoder, 'num_bins', 0)

    has_motion = motion_encoder is not None
    print(f'  推理分辨率: {FEED_H}x{FEED_W}, num_bins={nb}, '
          f'motion_encoder={"有" if has_motion else "无"}')

    # ── 初始化 LoFTR ──
    matcher = init_loftr_matcher(device=str(device))

    # ── 阶段 1: 零配置尺度校准 (pnp_scale, 对齐 test_v6 --no_gt) ──
    if calibrate_scale:
        n_calib = len(frames) - 1 if N_ESTIMATE_FRAMES <= 0 else N_ESTIMATE_FRAMES
        print(f'  [ScaleCalib] 预跑前 {n_calib} 帧 PnP (零配置 pnp_scale)…')
        global_scale, depth_cache = estimate_scale_from_pnp(
            model_name, encoder, depth_decoder, motion_encoder, matcher,
            frames, device, n_est=N_ESTIMATE_FRAMES)
        print(f'  [ScaleCalib] global_scale={global_scale:.4f}, '
              f'已缓存 {len(depth_cache)} 帧深度\n')
    else:
        global_scale = 1.0
        depth_cache = {}

    # ── 阶段 2: 全序列 VO ──
    n_pairs = len(frames) - 1
    T_abs = np.eye(4, dtype=np.float64)  # frame0 → world
    abs_poses = [T_abs.copy()]
    n_success = 0

    pbar = tqdm(range(n_pairs), desc=f'{model_name} VO')
    for fi in pbar:
        img0_path = os.path.join(RGB_DIR, frames[fi])
        img1_path = os.path.join(RGB_DIR, frames[fi + 1])

        # ── LoFTR 匹配 ──
        k0, k1, _ = _match_loftr_pair(
            matcher, img0_path, img1_path,
            device=str(device), max_dim=LOFTR_MAX_DIM)

        if k0 is None or len(k0) < 4:
            abs_poses.append(T_abs.copy())
            pbar.set_postfix({'m': 0})
            continue

        n_match = len(k0)

        # ── 深度预测 + 全局尺度校准 ──
        if fi not in depth_cache:
            if motion_encoder is not None:
                raw = predict_depth_with_motion(
                    img0_path, img1_path, encoder, depth_decoder,
                    motion_encoder, device, inverse=False,
                    min_depth=DEPTH_MIN, max_depth=DEPTH_MAX)
            else:
                raw = predict_depth(img0_path, encoder, depth_decoder, device,
                                   feed_h=FEED_H, feed_w=FEED_W,
                                   inverse=False, min_depth=DEPTH_MIN,
                                   max_depth=DEPTH_MAX, num_bins=nb)
            depth0 = np.clip(raw * global_scale, DEPTH_MIN, DEPTH_MAX)
            depth_cache[fi] = depth0
        else:
            depth0 = depth_cache[fi]

        # 下一帧深度 (单帧预测, 对齐 test_v6)
        raw1 = predict_depth(img1_path, encoder, depth_decoder, device,
                            feed_h=FEED_H, feed_w=FEED_W,
                            inverse=False, min_depth=DEPTH_MIN,
                            max_depth=DEPTH_MAX, num_bins=nb)
        depth1 = np.clip(raw1 * global_scale, DEPTH_MIN, DEPTH_MAX)

        # ── 帧间深度尺度一致性校正 (对齐 test_v6) ──
        h_d, w_d = depth0.shape
        scale_corr = 1.0
        if fi > 0 and n_match >= 20:
            n_sample = min(n_match, 500)
            sample_idx = np.random.choice(n_match, n_sample, replace=False)
            ratios = []
            for j in sample_idx:
                u0, v0 = k0[j]; u1, v1 = k1[j]
                ui0, vi0 = int(np.clip(u0, 0, w_d - 1)), int(np.clip(v0, 0, h_d - 1))
                ui1, vi1 = int(np.clip(u1, 0, w_d - 1)), int(np.clip(v1, 0, h_d - 1))
                Z0 = float(depth0[vi0, ui0])
                Z1 = float(depth1[vi1, ui1])
                if Z0 > 0.5 and Z1 > 0.5 and np.isfinite(Z0) and np.isfinite(Z1):
                    n0 = np.sqrt(((u0 - CX) / FX) ** 2 + ((v0 - CY) / FY) ** 2 + 1)
                    n1 = np.sqrt(((u1 - CX) / FX) ** 2 + ((v1 - CY) / FY) ** 2 + 1)
                    ratios.append((n1 * Z1) / (n0 * Z0))
            if len(ratios) > 10:
                s_raw = float(np.median(ratios))
                scale_corr = 1.0 / np.clip(s_raw, 0.2, 5.0)
                depth = np.clip(depth0 * scale_corr, DEPTH_MIN, DEPTH_MAX)
            else:
                depth = depth0
        else:
            depth = depth0

        # ── 3D 反投影 ──
        pts3d_list = []
        pts2d_list = []
        k0_filter = []
        for j in range(n_match):
            u, v = k0[j]
            ui, vi = int(np.clip(u, 0, w_d - 1)), int(np.clip(v, 0, h_d - 1))
            Z = float(depth[vi, ui])
            if Z <= 0.5 or Z > DEPTH_MAX or not np.isfinite(Z):
                continue
            X = (u - CX) * Z / FX
            Y = (v - CY) * Z / FY
            pts3d_list.append([X, Y, Z])
            pts2d_list.append(k1[j])
            k0_filter.append(k0[j])

        if len(pts3d_list) < 4:
            abs_poses.append(T_abs.copy())
            pbar.set_postfix({'m': n_match, '3d': 0, 'ok': n_success})
            continue

        n_valid = len(pts3d_list)
        pts3d_np = np.array(pts3d_list, dtype=np.float32)
        pts2d_np = np.array(pts2d_list, dtype=np.float32)
        k0_filter = np.array(k0_filter, dtype=np.float32)

        # ── F1 自适应 3D 位移过滤已关闭 (消融证明其破坏全局尺度) ──
        filter_keep = np.ones(n_valid, dtype=bool)
        n_filtered = n_valid

        # PnP 坐标缩放
        pts2d_pnp = pts2d_np.copy()
        pts2d_pnp[:, 0] *= SCALE_W_PNP
        pts2d_pnp[:, 1] *= SCALE_H_PNP

        R, t = None, None
        K_cv = K_PNP.astype(np.float64)

        def try_pnp(p3d, p2d):
            """尝试 cv2.solvePnP EPnP (对齐 test_v6)."""
            try:
                ret, rvec, tvec = cv2.solvePnP(
                    p3d.reshape(-1, 1, 3).astype(np.float64),
                    p2d.reshape(-1, 1, 2).astype(np.float64),
                    K_cv, None, flags=cv2.SOLVEPNP_EPNP)
                if ret:
                    R_out, _ = cv2.Rodrigues(rvec)
                    return R_out, tvec.ravel()
            except cv2.error:
                pass
            return None, None

        if n_filtered >= 4:
            R, t = try_pnp(pts3d_np[filter_keep], pts2d_pnp[filter_keep])

        # 过滤后 PnP 失败 → 回退全量点
        if R is None:
            R, t = try_pnp(pts3d_np, pts2d_pnp)

        if R is None:
            abs_poses.append(T_abs.copy())
            pbar.set_postfix({'m': n_match, '3d': n_valid, 'ok': n_success,
                              'status': 'pnp_fail'})
            continue

        angle = rotation_angle_deg(R)
        trans_norm = float(np.linalg.norm(t))

        if angle > MAX_ROTATION_DEG or trans_norm > MAX_TRANSLATION_MM:
            abs_poses.append(T_abs.copy())
            pbar.set_postfix({'m': n_match, '3d': n_valid,
                              'angle': f'{angle:.1f}°', 'ok': n_success,
                              'status': 'rejected'})
            continue

        # ── 累积位姿 (对齐 test_v6: T_abs = T_abs @ T_rel) ──
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t
        T_abs = T_abs @ T_rel
        abs_poses.append(T_abs.copy())
        n_success += 1

        pbar.set_postfix({'m': n_match, '3d': n_valid,
                          'sc': f'{scale_corr:.3f}',
                          'angle': f'{angle:.1f}°', 'ok': n_success})

    return {
        'abs_poses': abs_poses,
        'n_success': n_success,
        'n_pairs': n_pairs,
        'depth_cache': depth_cache,
        'global_scale': global_scale,
    }


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def main():
    # ── 固定随机种子 (对齐 test_v6) ──
    random.seed(42)
    np.random.seed(42)
    cv2.setRNGSeed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")

    # ── 加载 GT (dyendovo_dataset 标准格式) ──
    print(f"加载 GT 位姿: {SEQ_DIR}")
    gt_poses_std = dyendo_load_gt_poses(SEQ_DIR)
    print(f"  GT: {len(gt_poses_std)} 帧\n")

    # ── 扫描帧 ──
    exts = ('.png', '.jpg', '.jpeg')
    frames = sorted([f for f in os.listdir(RGB_DIR)
                     if f.lower().endswith(exts)])
    print(f"扫描帧: {len(frames)} 图像\n")

    # ── 逐模型评估 ──
    all_results = {}

    for model_name, model_path in MODELS_TO_EVAL:
        print(f"{'=' * 60}")
        print(f"  [{model_name}]")
        print(f"  模型: {model_path}")
        print(f"{'=' * 60}")

        if not os.path.isdir(model_path):
            print(f"  ⚠ 模型目录不存在, 跳过\n")
            continue

        t0 = time.time()
        vo_result = run_vo_for_model(model_name, model_path, frames, device,
                                     calibrate_scale=True)
        elapsed = time.time() - t0

        abs_poses = vo_result['abs_poses']
        n_success = vo_result['n_success']
        n_pairs = vo_result['n_pairs']

        print(f"\n  VO 完成: {n_success}/{n_pairs} 成功, 耗时 {elapsed:.0f}s")

        # ── 计算指标 ──
        metrics = compute_all_metrics(abs_poses, gt_poses_std)
        metrics['elapsed_s'] = round(elapsed, 1)
        metrics['n_success'] = n_success
        metrics['n_pairs'] = n_pairs
        metrics['global_scale'] = vo_result.get('global_scale', 1.0)
        all_results[model_name] = metrics

        print(f"  ATE:  {metrics['ate_mean']:.2f} ± {metrics['ate_std']:.2f} mm  "
              f"(RMSE={metrics['ate_rmse']:.2f})")
        print(f"  RPE-T: {metrics['rpe_t_mean']:.3f} ± {metrics['rpe_t_std']:.3f} mm  "
              f"(RMSE={metrics['rpe_t_rmse']:.3f})")
        print(f"  RPE-R: {metrics['rpe_r_mean']:.4f} ± {metrics['rpe_r_std']:.4f} °  "
              f"(RMSE={metrics['rpe_r_rmse']:.4f})")
        print(f"  Scale Error: {metrics['scale_error_pct']:.2f}%  "
              f"(Umeyama scale={metrics['umeyama_scale']:.4f})")
        print(f"  Global Scale: {metrics['global_scale']:.4f}")
        print(f"  Frames: {metrics['n_frames']}/{n_pairs + 1}  "
              f"(RPE pairs: {metrics['n_rpe']})")

    # ── 汇总表格 ──
    if len(all_results) < 1:
        print("\n无有效结果")
        return

    print(f"\n{'=' * 100}")
    print(f"  VO Backend Comparison ─ c1_transverse1_t1_v2")
    print(f"  Pipeline: test_v6_dyendovo baseline (LoFTR+EPnP+Adaptive 3D Filter)")
    print(f"  Scale Calibration: pnp_scale (target_step={TARGET_STEP_MM}mm)")
    print(f"{'=' * 100}\n")

    model_names = list(all_results.keys())
    col_count = len(model_names)
    header_fmt = "{0:<28}" + "".join(f" | {{{i}:>18}}" for i in range(1, col_count + 1))
    sep = "-" * (28 + 3 + col_count * 21)

    def fmt_row(label, *values):
        return header_fmt.format(label, *values)

    print(fmt_row("Metric", *model_names))
    print(sep)

    vals = [f"{all_results[n]['ate_mean']:.2f} ± {all_results[n]['ate_std']:.2f}" for n in model_names]
    print(fmt_row("ATE↓ (mm) Mean±SD", *vals))

    vals = [f"{all_results[n]['ate_rmse']:.2f}" for n in model_names]
    print(fmt_row("ATE↓ (mm) RMSE", *vals))

    vals = [f"{all_results[n]['rpe_t_mean']:.3f} ± {all_results[n]['rpe_t_std']:.3f}" for n in model_names]
    print(fmt_row("RPE-T↓ (mm) Mean±SD", *vals))

    vals = [f"{all_results[n]['rpe_r_mean']:.4f} ± {all_results[n]['rpe_r_std']:.4f}" for n in model_names]
    print(fmt_row("RPE-R↓ (°) Mean±SD", *vals))

    vals = [f"{all_results[n]['scale_error_pct']:.2f}%" for n in model_names]
    print(fmt_row("Scale Error↓ (%)", *vals))

    vals = [f"{all_results[n]['umeyama_scale']:.4f}" for n in model_names]
    print(fmt_row("Umeyama Scale", *vals))

    vals = [f"{all_results[n]['global_scale']:.4f}" for n in model_names]
    print(fmt_row("Global Scale", *vals))

    vals = [f"{all_results[n]['n_success']}/{all_results[n]['n_pairs']}" for n in model_names]
    print(fmt_row("VO Success Rate", *vals))

    vals = [f"{all_results[n]['elapsed_s']:.0f}s" for n in model_names]
    print(fmt_row("Runtime", *vals))

    print(sep)
    print(f"\n  结果目录: {OUT_DIR}")

    # 保存详细结果
    results_json = {}
    for name, m in all_results.items():
        results_json[name] = {k: v for k, v in m.items()
                              if isinstance(v, (int, float, str, bool))}
    json_path = os.path.join(OUT_DIR, 'compare_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"  结果已保存: {json_path}")


if __name__ == '__main__':
    main()
