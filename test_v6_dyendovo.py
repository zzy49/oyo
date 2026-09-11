"""
test_v6_dyendovo.py
V6管线 + DyEndoVO MotionNet 集成测试 + 运动组织绝对轨迹

管线:
  RGB → 深度预测(monodepth2) → LoFTR匹配 → PnP → 相机位姿
                                                        ↓
                                        运动点链式追踪 → 世界坐标3D轨迹

对比模式:
  --mode baseline:  标准EPnP
  --mode motionnet: MotionNet加权EPnP

测试序列: EndoSLAM c1_transverse1_t1_v2 (117帧)

用法:
  # EndoSLAM 序列 (有 GT)
  python test_v6_dyendovo.py --seq c1_transverse1_t1_v2 --mode baseline

  # 0 配置陌生序列 (无 GT, 自动估算 K)
  python test_v6_dyendovo.py --input_dir /path/to/rgb --mode baseline
"""
import os
import sys
import argparse
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from dyendovo_network import MotionNet
from dyendovo_dataset import K_MAT as ENDOSLAM_K, load_gt_poses, load_gt_depth
from v6_pipeline.utils import (
    ModelManager, predict_depth, predict_depth_with_motion,
    init_loftr_matcher, _match_loftr_pair,
)
from v6_pipeline.c3vd_loader import align_trajectory_umeyama, compute_ate

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
DATA_ROOT = r'F:\dataset'
MODEL_PATH = r'.\models\depth'
MOTIONNET_CKPT = r'.\models\dyendovo\best_model.pth'

# EndoSLAM 内参 (原始 1350×1080)
K_ORIG = ENDOSLAM_K.copy()  # fx=767.73, fy=767.73, cx=677.74, cy=543.06
FX, FY, CX, CY = K_ORIG[0, 0], K_ORIG[1, 1], K_ORIG[0, 2], K_ORIG[1, 2]
IMG_W_ORIG, IMG_H_ORIG = 1350, 1080

# MotionNet 输入尺寸 (与 eval_dyendovo 一致, 用于 PnP 坐标缩放)
MOTION_H, MOTION_W = 384, 512
SCALE_W_PNP = MOTION_W / IMG_W_ORIG  # 512/1350 ≈ 0.379
SCALE_H_PNP = MOTION_H / IMG_H_ORIG  # 384/1080 ≈ 0.356

# PnP 缩放内参 (匹配 eval_dyendovo.py 的 K_rs)
K_PNP = K_ORIG.copy()
K_PNP[0, 0] *= SCALE_W_PNP
K_PNP[1, 1] *= SCALE_H_PNP
K_PNP[0, 2] *= SCALE_W_PNP
K_PNP[1, 2] *= SCALE_H_PNP

# 深度预测
DEPTH_MIN, DEPTH_MAX = 1.0, 500.0
DEPTH_GLOBAL_SCALE = 2.4185  # PnP前全局深度缩放 (使VO轨迹尺度与GT一致)

# LoFTR
LOFTR_MAX_DIM = 840

# 位姿拒绝阈值
MAX_ROTATION_DEG = 15.0
MAX_TRANSLATION = 50.0  # mm, EndoSLAM相机运动较小

# 评估
DEFAULT_SEQ = 'c1_transverse1_t1_v2'
MAX_FRAMES = None  # None=全部


def rotation_angle_deg(R):
    """旋转矩阵 → 角度(度)."""
    return np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)) * 180 / np.pi


def solve_pnp_weighted(pts3d, pts2d, K, weights, weight_thresh=0.5):
    """加权 PnP: 用 MotionNet 权重过滤 + 点复制增强.

    策略:
      1. 按 P(static) > weight_thresh 过滤运动点
      2. 对保留点做点复制 (高权重点复制更多次)
      3. 标准 EPnP + LM 精化

    Args:
        pts3d:    (N, 3) 3D点
        pts2d:    (N, 2) 2D点
        K:        (3, 3) 内参
        weights:  (N,) P(static) [0,1]
        weight_thresh: 低于此值的点被丢弃

    Returns:
        (R, t) 或 (None, None)
    """
    N = len(pts3d)
    if N < 4:
        return None, None

    # Step 1: 按权重过滤
    keep = weights >= weight_thresh
    if keep.sum() < 6:
        # 放宽阈值
        keep = weights >= weight_thresh * 0.5
    if keep.sum() < 4:
        keep = np.ones(N, dtype=bool)  # 回退全部

    p3d_f = pts3d[keep]
    p2d_f = pts2d[keep]
    w_f = weights[keep]

    # Step 2: 点复制 (高权重点复制更多次)
    multiplicities = np.clip((w_f * 5).astype(int), 1, 5)
    p3d_rep = np.repeat(p3d_f, multiplicities, axis=0)
    p2d_rep = np.repeat(p2d_f, multiplicities, axis=0)

    # Step 3: EPnP + RANSAC
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            p3d_rep.reshape(-1, 1, 3).astype(np.float32),
            p2d_rep.reshape(-1, 1, 2).astype(np.float32),
            K.astype(np.float64), None,
            iterationsCount=200,
            reprojectionError=5.0,
            flags=cv2.SOLVEPNP_EPNP)

        if not success or inliers is None or len(inliers) < 4:
            # 回退: 不用复制, 只用过滤后的点
            success, rvec, tvec = cv2.solvePnP(
                p3d_f.reshape(-1, 1, 3).astype(np.float32),
                p2d_f.reshape(-1, 1, 2).astype(np.float32),
                K.astype(np.float64), None,
                flags=cv2.SOLVEPNP_EPNP)
            if not success:
                return None, None
    except cv2.error:
        return None, None

    # Step 4: LM 精化 (用过滤后的原始点, 不用复制点)
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            p3d_f.reshape(-1, 1, 3).astype(np.float32),
            p2d_f.reshape(-1, 1, 2).astype(np.float32),
            K.astype(np.float64), None, rvec, tvec)
    except cv2.error:
        pass

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


# ── 零配置全局尺度估算 ──
def _pred_depth_cached(fi, warp_dir, frames, encoder, depth_decoder, motion_encoder, device):
    """预测深度图并返回 (单帧缓存不跨调用)."""
    img0_path = os.path.join(warp_dir, frames[fi])
    if motion_encoder is not None:
        img1_path = os.path.join(warp_dir, frames[min(fi + 1, len(frames) - 1)])
        return predict_depth_with_motion(
            img0_path, img1_path, encoder, depth_decoder,
            motion_encoder, device, inverse=False,
            min_depth=DEPTH_MIN, max_depth=DEPTH_MAX)
    else:
        return predict_depth(img0_path, encoder, depth_decoder, device,
                             inverse=False, min_depth=DEPTH_MIN, max_depth=DEPTH_MAX)


def _estimate_scale_zero_gt(seq_dir, encoder, depth_decoder, motion_encoder, device,
                             n_estimate_frames, zero_gt_mode,
                             pnp_max_rot_deg, pnp_max_trans,
                             target_depth_mm, target_step_mm):
    """零配置模式: 无需 GT 估算 global_depth_scale.

    三种子模式:
      'raw':         scale=1.0 (当前 --no_gt 行为)
      'depth_prior': 预跑 VO, 计算深度中位数, scale = target/observed
      'pnp_scale':   预跑 VO, 计算 PnP 帧间位移中位数, scale = target/observed

    Returns: (global_scale, depth_cache)
    """
    warp_dir = os.path.join(seq_dir, 'generated', 'rgb_warped')
    if not os.path.isdir(warp_dir):
        print('[零配置] 无 rgb_warped 目录, scale=1.0')
        return 1.0, {}

    frames = sorted([f for f in os.listdir(warp_dir) if f.endswith('.png')])
    n_est = min(n_estimate_frames, len(frames) - 1) if n_estimate_frames > 0 else len(frames) - 1

    if zero_gt_mode == 'raw':
        print(f'[零配置:raw] 预跑前 {n_est} 帧对 VO (仅缓存深度, scale=1.0)...')
        depth_cache = {}
        for fi in range(n_est):
            if fi not in depth_cache:
                depth_cache[fi] = _pred_depth_cached(fi, warp_dir, frames, encoder, depth_decoder, motion_encoder, device)
        print(f'  已缓存 {len(depth_cache)} 帧深度, scale=1.0')
        return 1.0, depth_cache

    elif zero_gt_mode == 'depth_prior':
        print(f'[零配置:depth_prior] 预跑前 {n_est} 帧, 统计深度中位数 → scale = target/median...')
        depth_cache = {}
        all_depths = []
        for fi in range(n_est):
            if fi not in depth_cache:
                depth_cache[fi] = _pred_depth_cached(fi, warp_dir, frames, encoder, depth_decoder, motion_encoder, device)
            d = depth_cache[fi]
            # 取中心 50% 区域避免边缘噪声
            h, w = d.shape
            crop = d[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
            all_depths.append(crop.ravel())
        all_depths = np.concatenate(all_depths)
        # 排除离群值 (1% ~ 99%)
        lo, hi = np.percentile(all_depths, [1, 99])
        valid = all_depths[(all_depths >= lo) & (all_depths <= hi)]
        obs_median = float(np.median(valid))
        scale = target_depth_mm / obs_median if obs_median > 0.1 else 1.0
        print(f'  深度中位数: observed={obs_median:.1f}, target={target_depth_mm:.0f}mm → scale={scale:.4f}')
        print(f'  已缓存 {len(depth_cache)} 帧深度')
        return scale, depth_cache

    elif zero_gt_mode == 'pnp_scale':
        print(f'[零配置:pnp_scale] 预跑前 {n_est} 帧 VO, 统计 PnP 帧间位移 → scale = target/median...')
        depth_cache = {}
        steps = []  # 收集 PnP 帧间位移 (mm)

        # 复用 LoFTR 缓存; 无 GT → 不需 gt_calib
        for fi in range(n_est):
            cache_path = os.path.join(seq_dir, 'loftr_cache', f'matches_{fi:04d}.npz')
            if not os.path.exists(cache_path):
                continue
            d = np.load(cache_path)
            k0, k1 = d['pts0'], d['pts1']
            if len(k0) < 4:
                continue

            if fi not in depth_cache:
                depth_cache[fi] = _pred_depth_cached(fi, warp_dir, frames, encoder, depth_decoder, motion_encoder, device)
            depth = depth_cache[fi]

            h_d, w_d = depth.shape
            pts3d, pts2d = [], []
            for j in range(len(k0)):
                u, v = k0[j]; u1, v1 = k1[j]
                ui, vi = int(np.clip(u, 0, w_d - 1)), int(np.clip(v, 0, h_d - 1))
                Z = float(depth[vi, ui])
                if not (Z > 0.5 and Z < 500 and np.isfinite(Z)):
                    continue
                X = (u - CX) * Z / FX
                Y = (v - CY) * Z / FY
                pts3d.append([X, Y, Z])
                pts2d.append([u1, v1])

            if len(pts3d) < 4:
                continue

            pts3d_np = np.array(pts3d, dtype=np.float32)
            pts2d_np = np.array(pts2d, dtype=np.float32)
            pts2d_np[:, 0] *= SCALE_W_PNP
            pts2d_np[:, 1] *= SCALE_H_PNP

            try:
                ret, rvec, tvec = cv2.solvePnP(
                    pts3d_np.reshape(-1, 1, 3).astype(np.float64),
                    pts2d_np.reshape(-1, 1, 2).astype(np.float64),
                    K_PNP.astype(np.float64), None, flags=cv2.SOLVEPNP_EPNP)
                if ret:
                    R, _ = cv2.Rodrigues(rvec)
                    t_norm = float(np.linalg.norm(tvec))
                    angle = rotation_angle_deg(R)
                    if angle <= pnp_max_rot_deg and t_norm <= pnp_max_trans:
                        steps.append(t_norm)
            except cv2.error:
                pass

        if len(steps) < 5:
            print(f'  ⚠ 有效 PnP 步数不足 ({len(steps)}), 回退 scale=1.0')
            if not depth_cache:
                for fi in range(n_est):
                    if fi not in depth_cache:
                        depth_cache[fi] = _pred_depth_cached(fi, warp_dir, frames, encoder, depth_decoder, motion_encoder, device)
            return 1.0, depth_cache

        steps = np.array(steps)
        lo_s, hi_s = np.percentile(steps, [5, 95])
        steps_clean = steps[(steps >= lo_s) & (steps <= hi_s)]
        obs_step = float(np.median(steps_clean))
        scale = target_step_mm / obs_step if obs_step > 0.001 else 1.0
        print(f'  PnP位移中位数: observed={obs_step:.4f}mm, target={target_step_mm:.2f}mm → scale={scale:.4f}')
        print(f'  有效PnP帧数: {len(steps)}, 已缓存 {len(depth_cache)} 帧深度')
        return scale, depth_cache

    else:
        raise ValueError(f'未知 zero_gt_mode: {zero_gt_mode}')


def estimate_global_scale(seq_dir, encoder, depth_decoder, motion_encoder, device,
                          n_estimate_frames=20, depth_source='pred',
                          loftr_source='cache',
                          pnp_max_rot_deg=None, pnp_max_trans=None,
                          no_gt=False, zero_gt_mode=None,
                          target_depth_mm=50.0, target_step_mm=0.5):
    """
    自动估计全局深度尺度: 前 N 帧预跑 VO → Umeyama 对齐 GT / 零配置估算 → 得 scale。

    同时缓存预测的深度图 (depth_cache), 供主 VO 复用避免重复推理。

    Args:
        seq_dir: 序列目录
        encoder, depth_decoder: monodepth2 模型
        motion_encoder: MotionEncoder (可选)
        device: torch device
        n_estimate_frames: 预跑帧对数 (0 = 全序列)
        depth_source: 'pred' 或 'gt'
        loftr_source: 'online' 或 'cache'
        pnp_max_rot_deg: PnP旋转拒绝阈值 (None→默认)
        pnp_max_trans: PnP位移拒绝阈值 (None→默认)
        no_gt: True → 跳过 Umeyama 对齐, 直接 scale=1.0 (等效 zero_gt_mode='raw')
        zero_gt_mode: 'raw' | 'depth_prior' | 'pnp_scale' (None → 正常GT模式)
        target_depth_mm: depth_prior 模式的目标深度中位数 (mm)
        target_step_mm: pnp_scale 模式的目标帧间位移中位数 (mm)

    Returns:
        (global_scale, depth_cache) where depth_cache is dict[frame_idx→ndarray]
    """
    # backward compat: --no_gt → zero_gt_mode='pnp_scale' (配置 C, 最优零配置方案)
    if no_gt and zero_gt_mode is None:
        zero_gt_mode = 'pnp_scale'

    if pnp_max_rot_deg is None:
        pnp_max_rot_deg = MAX_ROTATION_DEG
    if pnp_max_trans is None:
        pnp_max_trans = MAX_TRANSLATION

    if depth_source == 'gt':
        return 1.0, {}  # GT 深度无需校正, 无需缓存

    # ── 零配置模式: 跳过 Umeyama 校准 ──
    if zero_gt_mode is not None:
        return _estimate_scale_zero_gt(
            seq_dir, encoder, depth_decoder, motion_encoder, device,
            n_estimate_frames, zero_gt_mode,
            pnp_max_rot_deg, pnp_max_trans,
            target_depth_mm, target_step_mm)

    # ── 0 配置模式: 无 GT 位姿则跳过 Umeyama 校准 ──
    pose_path = os.path.join(seq_dir, 'pose.txt')
    if not os.path.isfile(pose_path):
        print('[估计全局尺度] 无 GT 位姿 (pose.txt 不存在), 跳过 Umeyama 校准, scale=1.0')
        return 1.0, {}

    warp_dir = os.path.join(seq_dir, 'generated', 'rgb_warped')
    frames = sorted([f for f in os.listdir(warp_dir) if f.endswith('.png')])
    n_est = min(n_estimate_frames, len(frames) - 1) if n_estimate_frames > 0 else len(frames) - 1

    print(f'[估计全局尺度] 预跑前 {n_est} 帧对 VO (同时缓存深度)...')

    # 预跑 VO + 深度缓存
    vo_calib, gt_calib = [], []
    T_calib = np.eye(4, dtype=np.float64)
    vo_calib.append(T_calib[:3, 3].copy())

    gt_poses = load_gt_poses(seq_dir)
    gt_calib.append(gt_poses[0][:3, 3].copy())

    depth_cache = {}  # frame_idx → depth (mm)

    for fi in range(n_est):
        # LoFTR 加载
        cache_path = os.path.join(seq_dir, 'loftr_cache', f'matches_{fi:04d}.npz')
        if not os.path.exists(cache_path):
            continue
        d = np.load(cache_path)
        k0, k1 = d['pts0'], d['pts1']
        if len(k0) < 4:
            continue

        # 深度预测 (优先用缓存)
        if fi not in depth_cache:
            img0_path = os.path.join(warp_dir, frames[fi])
            if motion_encoder is not None:
                img1_path = os.path.join(warp_dir, frames[min(fi + 1, len(frames) - 1)])
                depth_cache[fi] = predict_depth_with_motion(
                    img0_path, img1_path, encoder, depth_decoder,
                    motion_encoder, device, inverse=False,
                    min_depth=DEPTH_MIN, max_depth=DEPTH_MAX)
            else:
                depth_cache[fi] = predict_depth(
                    img0_path, encoder, depth_decoder, device,
                    inverse=False, min_depth=DEPTH_MIN, max_depth=DEPTH_MAX)

        depth = depth_cache[fi]

        # 3D 反投影 + PnP (scale=1.0)
        h_d, w_d = depth.shape
        pts3d, pts2d = [], []
        for j in range(len(k0)):
            u, v = k0[j]; u1, v1 = k1[j]
            ui, vi = int(np.clip(u, 0, w_d - 1)), int(np.clip(v, 0, h_d - 1))
            Z = float(depth[vi, ui])
            if not (Z > 0.5 and Z < 500 and np.isfinite(Z)):
                continue
            X = (u - CX) * Z / FX
            Y = (v - CY) * Z / FY
            pts3d.append([X, Y, Z])
            pts2d.append([u1, v1])

        if len(pts3d) < 4:
            continue

        pts3d_np = np.array(pts3d, dtype=np.float32)
        pts2d_np = np.array(pts2d, dtype=np.float32)
        pts2d_np[:, 0] *= SCALE_W_PNP
        pts2d_np[:, 1] *= SCALE_H_PNP

        try:
            ret, rvec, tvec = cv2.solvePnP(
                pts3d_np.reshape(-1, 1, 3).astype(np.float64),
                pts2d_np.reshape(-1, 1, 2).astype(np.float64),
                K_PNP.astype(np.float64), None, flags=cv2.SOLVEPNP_EPNP)
            if ret:
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.ravel()
                angle = rotation_angle_deg(R)
                if angle > pnp_max_rot_deg or np.linalg.norm(t) > pnp_max_trans:
                    continue
                T_rel = np.eye(4)
                T_rel[:3, :3] = R; T_rel[:3, 3] = t
                T_calib = T_calib @ T_rel
        except cv2.error:
            pass

        vo_calib.append(T_calib[:3, 3].copy())
        gt_calib.append(gt_poses[fi + 1][:3, 3].copy())

    vo_calib = np.array(vo_calib)
    gt_calib = np.array(gt_calib)

    if len(vo_calib) < 10:
        print(f'  ⚠ VO 帧数不足 ({len(vo_calib)}), 回退 scale=1.0')
        return 1.0, depth_cache

    _, errors, _, umeyama_scale = align_trajectory_umeyama(vo_calib, gt_calib)
    calib_ate = np.sqrt((errors ** 2).mean())
    print(f'  预跑 ATE={calib_ate:.2f}mm, Umeyama scale={umeyama_scale:.4f}')
    print(f'  全局深度尺度: {umeyama_scale:.4f} (depth *= {umeyama_scale:.4f})')
    print(f'  已缓存 {len(depth_cache)} 帧深度, 主 VO 将复用')

    return umeyama_scale, depth_cache


def estimate_pipeline_params(seq_dir, encoder, depth_decoder, motion_encoder, device,
                              depth_source='pred', loftr_source='cache',
                              no_gt=False, zero_gt_mode=None):
    """
    自动估计管线所有自适应参数 (一次预跑, 全面校准)。

    Args:
        no_gt: True (等效 zero_gt_mode='raw'): global_depth_scale=1.0, 禁用帧间校正
        zero_gt_mode: 'raw' | 'depth_prior' | 'pnp_scale' (None → 正常GT模式)

    Returns:
        dict with keys:
          - global_depth_scale: float
          - max_rotation_deg: float (PnP 旋转拒绝阈值)
          - max_translation: float (PnP 位移拒绝阈值, mm)
          - scale_clip_lo: float (帧间校正下限)
          - scale_clip_hi: float (帧间校正上限)
          - chain_dist_thresh: float (链式追踪匹配阈值, px)
          - motion_factor: float (运动分类: median * factor)
          - img_w, img_h: int (实际图像/深度尺寸)
          - depth_cache: dict[frame_idx→ndarray] 复用的深度缓存
          - no_gt: bool (True → 禁用 scale_corr; depth_prior/pnp_scale 下为 False)
          - zero_gt_mode: str (当前使用的零配置模式)
    """
    # backward compat
    if no_gt and zero_gt_mode is None:
        zero_gt_mode = 'pnp_scale'

    # ── 1. 获取全局尺度 + 深度缓存 ──
    global_depth_scale, depth_cache = estimate_global_scale(
        seq_dir, encoder, depth_decoder, motion_encoder, device,
        n_estimate_frames=0, depth_source=depth_source,
        loftr_source=loftr_source, no_gt=no_gt, zero_gt_mode=zero_gt_mode)

    # ── 1.5 确定 no_gt 标志: raw → True (禁用scale_corr), 其他 → False (启�scale_corr) ──
    disable_scale_corr = (zero_gt_mode == 'raw')
    img_w, img_h = 1350, 1080  # EndoSLAM 默认 (如果无缓存)
    if depth_cache:
        sample_depth = next(iter(depth_cache.values()))
        img_h, img_w = sample_depth.shape

    # ── 3. 帧间校正裁剪范围 ──
    # scale < 1.0 (深度需缩小): 允许更低的帧间校正值
    # scale > 1.0 (深度需放大): 允许更高的帧间校正值
    # 注意: s_raw 理想值为 1.0, clip 范围必须围绕 1.0 且 lo < hi;
    # 否则 global_scale 过大时 lo > hi, np.clip 恒返回 hi, 深度被错误缩放
    scale_clip_lo = min(max(0.2, global_depth_scale * 0.4), 0.9)
    scale_clip_hi = max(min(5.0, max(2.0, global_depth_scale * 2.5)), 1.1)

    # ── 4. 链式追踪距离阈值: 基于图像尺寸 ──
    chain_dist_thresh = max(img_w, img_h) / 300.0  # ≈3px for 1000px image

    # ── 5. 运动分类倍数: 默认 2.0 (median + k*std) ──
    k_factor = DISP_K_FACTOR

    # ── 6. PnP 拒绝阈值: 基于图像尺寸的自适应 ──
    # 高分辨率 → 更多 3D 信息 → PnP 更稳定 → 阈值可缩紧
    # 低分辨率 → PnP 稳定性差 → 阈值需放宽
    resolution_factor = min(img_w, img_h) / 640.0  # 以 VGA 为基准
    pnp_max_rot = 15.0 / max(resolution_factor, 0.5)  # 低分辨率放宽
    pnp_max_trans = 50.0 / max(resolution_factor, 0.5)

    params = {
        'global_depth_scale': global_depth_scale,
        'max_rotation_deg': pnp_max_rot,
        'max_translation': pnp_max_trans,
        'scale_clip_lo': scale_clip_lo,
        'scale_clip_hi': scale_clip_hi,
        'chain_dist_thresh': chain_dist_thresh,
        'k_factor': k_factor,
        'img_w': img_w,
        'img_h': img_h,
        # 相机内参 (当前使用 EndoSLAM 默认, 可在 estimate_pipeline_params 中按数据集覆盖)
        'fx': FX, 'fy': FY, 'cx': CX, 'cy': CY,
        'img_w_orig': IMG_W_ORIG, 'img_h_orig': IMG_H_ORIG,
        'K': K_ORIG.copy(),  # (3,3) 相机内参矩阵
        'depth_cache': depth_cache,
        'no_gt': disable_scale_corr,
        'zero_gt_mode': zero_gt_mode,
    }
    return params


def run_vo_sequence(seq_name, encoder, depth_decoder, device,
                    motion_net=None, mode='motionnet', max_frames=None,
                    weight_thresh=0.3, motion_encoder=None,
                    depth_source='pred', loftr_source='online',
                    pipeline_params=None, data_root=None,
                    input_dir=None, output_dir=None, no_gt=False,
                    zero_gt_mode=None,
                    use_displacement_filter=False,
                    use_scale_corr=True,
                    k_factor=None):
    """在单个序列上跑 VO, 返回轨迹和 GT.

    支持两种模式:
      1. EndoSLAM 模式 (seq_name + data_root): 从 generated/rgb_warped 读帧
      2. 0 配置模式 (input_dir): 直接从 RGB 文件夹读帧, 无 GT

    Args:
        seq_name: 序列名 (如 'c1_transverse1_t1_v2'), input_dir 模式可为 None
        encoder, depth_decoder: monodepth2 模型
        device: torch device
        motion_net: MotionNet 模型 (mode='motionnet' 时需要)
        mode: 'baseline' (标准EPnP) 或 'motionnet' (MotionNet加权EPnP)
        max_frames: 最大帧对数
        motion_encoder: MotionEncoder 模型 (可选, 用于两帧时序深度增强)
        depth_source: 'pred' (预测深度) 或 'gt' (GT深度)
        loftr_source: 'online' (在线匹配) 或 'cache' (预计算缓存)
        pipeline_params: dict, estimate_pipeline_params() 的返回值
                         (None → 自动调用 estimate_pipeline_params)
        data_root: 数据集根目录 (None → 使用全局 DATA_ROOT)
        input_dir: 0 配置模式: 直接指定 RGB 帧文件夹
        output_dir: 0 配置模式: 输出目录 (None → 使用 input_dir)

    Returns:
        vo_traj: (N, 3) VO轨迹 (相机位置, mm, frame-0坐标系)
        gt_traj: (N, 3) GT轨迹 (相机位置, mm), 无 GT 时为空数组
        stats: dict with n_success, n_total
        chain_data: dict
    """
    is_zero_config = input_dir is not None

    if is_zero_config:
        # ── 0 配置模式: 直接扫描 RGB 文件夹 ──
        warp_dir = input_dir
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        frames = sorted([f for f in os.listdir(warp_dir) if f.lower().endswith(exts)])
        gt_poses = None
        seq_dir = output_dir if output_dir else input_dir
        loftr_source = 'online'  # 0配置模式强制在线匹配
    else:
        if data_root is None:
            data_root = DATA_ROOT
        seq_dir = os.path.join(data_root, seq_name)
        warp_dir = os.path.join(seq_dir, 'generated', 'rgb_warped')
        frames = sorted([f for f in os.listdir(warp_dir) if f.endswith('.png')])
        gt_poses = load_gt_poses(seq_dir)
        # 对齐 GT 帧数: rgb_warped 可能多于 GT 位姿, 截断到 GT 长度
        if len(frames) > len(gt_poses):
            print(f'  帧数对齐: rgb_warped {len(frames)} 帧 → GT {len(gt_poses)} 帧')
            frames = frames[:len(gt_poses)]

    has_gt = gt_poses is not None

    if is_zero_config:
        os.makedirs(seq_dir, exist_ok=True)

    if max_frames:
        frames = frames[:max_frames + 1]

    n_pairs = len(frames) - 1
    if n_pairs < 1:
        raise ValueError(f'帧数不足: {len(frames)} 帧, 至少需要 2 帧')

    # ── 自动估计管线参数 ──
    if pipeline_params is None:
        print(f'  自动估计管线参数...')
        pipeline_params = estimate_pipeline_params(
            seq_dir, encoder, depth_decoder, motion_encoder, device,
            depth_source=depth_source, loftr_source=loftr_source,
            no_gt=no_gt, zero_gt_mode=zero_gt_mode)

    global_depth_scale = pipeline_params['global_depth_scale']
    cached_depths = pipeline_params['depth_cache']
    pnp_max_rot_deg = pipeline_params['max_rotation_deg']
    pnp_max_trans = pipeline_params['max_translation']
    scale_clip_lo = pipeline_params['scale_clip_lo']
    scale_clip_hi = pipeline_params['scale_clip_hi']
    no_gt_mode = pipeline_params.get('no_gt', False)
    current_mode = pipeline_params.get('zero_gt_mode', None)

    mode_label = {'raw': 'raw (scale=1, no_corr)', 'depth_prior': 'depth_prior', 'pnp_scale': 'pnp_scale'}.get(current_mode, 'GT')
    if no_gt_mode:
        print(f'  参数: 零配置-{mode_label} 模式: global_scale={global_depth_scale:.4f}, 禁用帧间深度校正')
    elif current_mode is not None:
        print(f'  参数: 零配置-{mode_label} 模式: global_scale={global_depth_scale:.4f}, '
              f'PnP rot<{pnp_max_rot_deg:.1f}°, trans<{pnp_max_trans:.1f}mm, '
              f'scale_clip=[{scale_clip_lo:.3f},{scale_clip_hi:.3f}]')
    else:
        print(f'  参数: global_scale={global_depth_scale:.4f}, '
              f'PnP rot<{pnp_max_rot_deg:.1f}°, trans<{pnp_max_trans:.1f}mm, '
              f'scale_clip=[{scale_clip_lo:.3f},{scale_clip_hi:.3f}]')

    # ── 使用 pipeline_params 中的 K (支持 0 配置模式自定义内参) ──
    K_cam = pipeline_params.get('K', K_ORIG)
    fx_cam, fy_cam = K_cam[0, 0], K_cam[1, 1]
    cx_cam, cy_cam = K_cam[0, 2], K_cam[1, 2]
    img_w_cam = pipeline_params.get('img_w_orig', IMG_W_ORIG)
    img_h_cam = pipeline_params.get('img_h_orig', IMG_H_ORIG)

    # PnP 缩放参数 (MotionNet 输入尺寸 384×512)
    scale_w_pnp_local = MOTION_W / img_w_cam
    scale_h_pnp_local = MOTION_H / img_h_cam
    K_pnp_local = K_cam.copy()
    K_pnp_local[0, 0] *= scale_w_pnp_local
    K_pnp_local[1, 1] *= scale_h_pnp_local
    K_pnp_local[0, 2] *= scale_w_pnp_local
    K_pnp_local[1, 2] *= scale_h_pnp_local

    # VO 累积位姿
    T_abs = np.eye(4, dtype=np.float64)
    vo_traj = [T_abs[:3, 3].copy()]
    if has_gt:
        gt_traj = [gt_poses[0][:3, 3].copy()]
    else:
        gt_traj = []

    n_success = 0
    n_total = 0
    depth_ratios = []  # 逐帧 pred/GT 深度比

    # ── 链式追踪数据收集 ──
    all_k0 = []      # list of (N_i, 2)  每对匹配的 pt0 坐标
    all_k1 = []      # list of (N_i, 2)  每对匹配的 pt1 坐标
    all_weights = []  # list of (M_i,)   每对有效3D点的 MotionNet 权重
    depth_maps = {}   # frame_idx → (H,W) 深度图 (mm)
    abs_poses = []    # list of (4,4)   每帧的 T_world_cam

    # 初始化 LoFTR (仅 online 模式)
    matcher = None
    if loftr_source == 'online':
        matcher = init_loftr_matcher(device=str(device))

    if motion_net is not None:
        motion_net.eval()

    pbar = tqdm(range(n_pairs), desc=f'[{mode}] {seq_name}')

    # 帧0的初始位姿
    abs_poses.append(T_abs.copy())

    for fi in pbar:
        n_total += 1

        # ── Step 1: LoFTR 匹配 ──
        img0_path = os.path.join(warp_dir, frames[fi])
        img1_path = os.path.join(warp_dir, frames[fi + 1])

        if loftr_source == 'cache':
            cache_path = os.path.join(seq_dir, 'loftr_cache',
                                      f'matches_{fi:04d}.npz')
            if os.path.exists(cache_path):
                d = np.load(cache_path)
                k0 = d['pts0']
                k1 = d['pts1']
            else:
                k0, k1 = np.empty((0, 2)), np.empty((0, 2))
        else:
            k0, k1, _ = _match_loftr_pair(
                matcher, img0_path, img1_path,
                device=str(device), max_dim=LOFTR_MAX_DIM)

        if k0 is None or len(k0) < 4:
            vo_traj.append(T_abs[:3, 3].copy())
            if has_gt:
                gt_traj.append(gt_poses[fi + 1][:3, 3].copy())
            abs_poses.append(T_abs.copy())  # 帧fi+1: 位姿不变
            pbar.set_postfix({'matches': 0, 'status': 'few_matches'})
            continue

        n_match = len(k0)

        # ── Step 2: 深度获取 ──
        if depth_source == 'gt':
            depth = load_gt_depth(seq_dir, fi)  # GT 深度, mm
        elif cached_depths is not None and fi in cached_depths:
            depth = cached_depths[fi]  # 复用预跑缓存的深度
        elif motion_encoder is not None:
            # MotionEncoder 增强: 两帧推理, 利用时序一致性
            depth = predict_depth_with_motion(img0_path, img1_path,
                                              encoder, depth_decoder,
                                              motion_encoder, device,
                                              inverse=False,
                                              min_depth=DEPTH_MIN,
                                              max_depth=DEPTH_MAX)
        else:
            depth = predict_depth(img0_path, encoder, depth_decoder, device,
                                  inverse=False, min_depth=DEPTH_MIN,
                                  max_depth=DEPTH_MAX)

        # ── Step 2.5: 逐帧深度比 (pred vs GT, 仅在有GT时) ──
        if has_gt:
            gt_d = load_gt_depth(seq_dir, fi)
            depth_ratios.append(float(np.median(depth / (gt_d + 1e-6))))

        # ── Step 2.6: 深度全局尺度校正 (消除 Umeyama scale 膨胀) ──
        if depth_source != 'gt':
            depth = depth * global_depth_scale

        # ── Step 2.6b: 预测下一帧深度 (基线模式 PnP 自适应过滤 + 帧间一致性) ──
        depth_next = None
        if mode == 'baseline' and depth_source != 'gt':
            depth_next = predict_depth(img1_path, encoder, depth_decoder, device,
                                       inverse=False, min_depth=DEPTH_MIN,
                                       max_depth=DEPTH_MAX)
            depth_next = depth_next * global_depth_scale

        # ── Step 2.7: 深度帧间尺度一致性校正 ──
        # 通过匹配点反投影, 抑制帧间深度漂移
        # 注意: 在全局尺度校正之后进行, 对全局尺度后的 depth 做微调
        scale_corr = 1.0
        if fi > 0 and depth_source != 'gt' and n_match >= 20 and not no_gt_mode and use_scale_corr:
            if depth_next is None:
                depth_next = predict_depth(img1_path, encoder, depth_decoder, device,
                                           inverse=False, min_depth=DEPTH_MIN,
                                           max_depth=DEPTH_MAX)
                depth_next = depth_next * global_depth_scale
            h_dn, w_dn = depth_next.shape
            n_sample = min(n_match, 500)
            sample_idx = np.random.choice(n_match, n_sample, replace=False)
            ratios = []
            for j in sample_idx:
                u0, v0 = k0[j]; u1, v1 = k1[j]
                ui0, vi0 = int(np.clip(u0, 0, w_d - 1)), int(np.clip(v0, 0, h_d - 1))
                ui1, vi1 = int(np.clip(u1, 0, w_dn - 1)), int(np.clip(v1, 0, h_dn - 1))
                Z0 = float(depth[vi0, ui0])
                Z1 = float(depth_next[vi1, ui1])
                if Z0 > 0.5 and Z1 > 0.5 and np.isfinite(Z0) and np.isfinite(Z1):
                    n0 = np.sqrt(((u0 - cx_cam) / fx_cam) ** 2 + ((v0 - cy_cam) / fy_cam) ** 2 + 1)
                    n1 = np.sqrt(((u1 - cx_cam) / fx_cam) ** 2 + ((v1 - cy_cam) / fy_cam) ** 2 + 1)
                    ratios.append((n1 * Z1) / (n0 * Z0))
            if len(ratios) > 10:
                s_raw = float(np.median(ratios))
                scale_corr = 1.0 / np.clip(s_raw, scale_clip_lo, scale_clip_hi)
                depth = depth * scale_corr

        # ── Step 3: 深度 → 3D点 ──
        h_d, w_d = depth.shape
        pts3d_list, pts2d_list, k0_list = [], [], []
        for j in range(n_match):
            u, v = k0[j]
            ui, vi = int(np.clip(u, 0, w_d - 1)), int(np.clip(v, 0, h_d - 1))
            Z = float(depth[vi, ui])
            # 过滤无效深度
            if Z <= 0.5 or Z > 500.0 or not np.isfinite(Z):
                continue
            X = (u - cx_cam) * Z / fx_cam
            Y = (v - cy_cam) * Z / fy_cam
            pts3d_list.append([X, Y, Z])
            pts2d_list.append(k1[j])
            k0_list.append(k0[j])

        if len(pts3d_list) < 4:
            vo_traj.append(T_abs[:3, 3].copy())
            if has_gt:
                gt_traj.append(gt_poses[fi + 1][:3, 3].copy())
            pbar.set_postfix({'matches': n_match, 'valid3d': len(pts3d_list),
                              'status': 'few_3d'})
            continue

        n_valid = len(pts3d_list)

        # ── Step 4: MotionNet 权重 (仅 motionnet 模式) ──
        if mode == 'motionnet' and motion_net is not None:
            # 加载并 resize 图像对 → MotionNet 输入
            img_t = cv2.imread(img0_path)
            img_tp1 = cv2.imread(img1_path)
            img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
            img_tp1 = cv2.cvtColor(img_tp1, cv2.COLOR_BGR2RGB)

            h_orig, w_orig = img_t.shape[:2]
            scale_h_m = MOTION_H / h_orig
            scale_w_m = MOTION_W / w_orig

            img_t_rs = cv2.resize(img_t, (MOTION_W, MOTION_H))
            img_tp1_rs = cv2.resize(img_tp1, (MOTION_W, MOTION_H))

            # 6通道堆叠 → (1, 6, H, W)
            img_pair = np.concatenate([img_t_rs, img_tp1_rs], axis=-1)
            img_pair = torch.from_numpy(img_pair).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            # 缩放匹配点坐标到 MotionNet 空间
            pts0_for_motion = k0.copy()
            pts0_for_motion[:, 0] *= scale_w_m
            pts0_for_motion[:, 1] *= scale_h_m

            # 只取有效3D点的索引
            valid_mask_t = torch.ones(1, n_valid, dtype=torch.bool, device=device)

            with torch.no_grad():
                p_map = motion_net(img_pair)

                # 取有效3D点的坐标用于采样
                pts2d_for_sample = torch.from_numpy(
                    np.array([pts0_for_motion[j] for j in range(n_match)
                              if _is_valid_3d(j, k0, depth, h_d, w_d)], dtype=np.float32)
                ).unsqueeze(0).to(device)  # (1, M, 2)

                weights = motion_net.sample_weights(p_map, pts2d_for_sample, valid_mask_t)
                weights_np = weights[0].cpu().numpy()  # (M,)
        else:
            weights_np = None

        # ── Step 5: PnP ──
        pts3d_np = np.array(pts3d_list, dtype=np.float32)
        pts2d_np = np.array(pts2d_list, dtype=np.float32)
        # 缩放 pts2d 到 384x512 (匹配 eval_dyendovo PnP)
        pts2d_pnp = pts2d_np.copy()
        pts2d_pnp[:, 0] *= scale_w_pnp_local
        pts2d_pnp[:, 1] *= scale_h_pnp_local

        # ── 收集链式追踪数据 ──
        all_k0.append(k0.copy())
        all_k1.append(k1.copy())
        depth_maps[fi] = depth.copy()
        if weights_np is not None:
            all_weights.append(weights_np.copy())
        else:
            all_weights.append(np.ones(n_valid, dtype=np.float32))

        if mode == 'motionnet' and weights_np is not None:
            # MotionNet 加权 PnP
            R, t = solve_pnp_weighted(pts3d_np, pts2d_pnp, K_pnp_local,
                                      weights_np, weight_thresh=weight_thresh)
            n_static = int((weights_np >= weight_thresh).sum())
        else:
            # Baseline: 自适应 3D 位移过滤 + 标准 EPnP
            R, t = None, None
            pts3d_pnp = pts3d_np.copy()  # 帧0 3D 点
            pts2d_pnp = pts2d_np.copy()
            pts2d_pnp[:, 0] *= scale_w_pnp_local
            pts2d_pnp[:, 1] *= scale_h_pnp_local

            k0_filter = np.array(k0_list, dtype=np.float32)  # 有效点的 k0 坐标
            k1_filter = pts2d_np  # 有效点的 k1 坐标 (原始分辨率)

            if depth_next is not None and use_displacement_filter:
                # 自适应 3D 位移过滤: median + k*std
                _k = k_factor if k_factor is not None else pipeline_params.get('k_factor', DISP_K_FACTOR)
                filter_keep, fstats = _filter_by_3d_displacement(
                    pts3d_np, k0_filter, k1_filter, depth, depth_next, K_cam,
                    k_factor=_k)
                n_filtered = int(filter_keep.sum())
                if n_filtered >= 4:
                    pts3d_cv = pts3d_np[filter_keep].reshape(-1, 1, 3).astype(np.float64)
                    pts2d_cv = pts2d_pnp[filter_keep].reshape(-1, 1, 2).astype(np.float64)
                    K_cv = K_pnp_local.astype(np.float64)
                    try:
                        ret, rvec, tvec = cv2.solvePnP(
                            pts3d_cv, pts2d_cv, K_cv, None, flags=cv2.SOLVEPNP_EPNP)
                        if ret:
                            R, _ = cv2.Rodrigues(rvec)
                            t = tvec.ravel()
                    except cv2.error:
                        pass
                    if R is not None:
                        n_static = n_filtered
                    else:
                        # PnP 失败 → 回退到全量点
                        pts3d_cv = pts3d_np.reshape(-1, 1, 3).astype(np.float64)
                        pts2d_cv = pts2d_pnp.reshape(-1, 1, 2).astype(np.float64)
                        try:
                            ret, rvec, tvec = cv2.solvePnP(
                                pts3d_cv, pts2d_cv, K_cv, None, flags=cv2.SOLVEPNP_EPNP)
                            if ret:
                                R, _ = cv2.Rodrigues(rvec)
                                t = tvec.ravel()
                        except cv2.error:
                            pass
                        n_static = n_valid
                else:
                    # 过滤后点太少 → 回退到全量点
                    pts3d_cv = pts3d_np.reshape(-1, 1, 3).astype(np.float64)
                    pts2d_cv = pts2d_pnp.reshape(-1, 1, 2).astype(np.float64)
                    K_cv = K_pnp_local.astype(np.float64)
                    try:
                        ret, rvec, tvec = cv2.solvePnP(
                            pts3d_cv, pts2d_cv, K_cv, None, flags=cv2.SOLVEPNP_EPNP)
                        if ret:
                            R, _ = cv2.Rodrigues(rvec)
                            t = tvec.ravel()
                    except cv2.error:
                        pass
                    n_static = n_valid
            else:
                # 无 depth_next (GT深度模式) → 标准 EPnP
                pts3d_cv = pts3d_np.reshape(-1, 1, 3).astype(np.float64)
                pts2d_cv = pts2d_pnp.reshape(-1, 1, 2).astype(np.float64)
                K_cv = K_pnp_local.astype(np.float64)
                try:
                    ret, rvec, tvec = cv2.solvePnP(
                        pts3d_cv, pts2d_cv, K_cv, None, flags=cv2.SOLVEPNP_EPNP)
                    if ret:
                        R, _ = cv2.Rodrigues(rvec)
                        t = tvec.ravel()
                except cv2.error:
                    pass
                n_static = n_valid

        if R is None or t is None:
            vo_traj.append(T_abs[:3, 3].copy())
            if has_gt:
                gt_traj.append(gt_poses[fi + 1][:3, 3].copy())
            abs_poses.append(T_abs.copy())  # 帧fi+1: PnP失败,位姿不变
            pbar.set_postfix({'matches': n_match, 'valid3d': n_valid,
                              'status': 'pnp_fail'})
            continue

        # 位姿合理性检查
        angle_deg = rotation_angle_deg(R)
        t_norm = float(np.linalg.norm(t))
        if angle_deg > pnp_max_rot_deg or t_norm > pnp_max_trans:
            vo_traj.append(T_abs[:3, 3].copy())
            if has_gt:
                gt_traj.append(gt_poses[fi + 1][:3, 3].copy())
            abs_poses.append(T_abs.copy())  # 帧fi+1: 被拒绝,位姿不变
            pbar.set_postfix({'matches': n_match, 'valid3d': n_valid,
                              'angle': f'{angle_deg:.1f}°', 'status': 'rejected'})
            continue

        # ── Step 6: 累积位姿 ──
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = t
        T_abs = T_abs @ T_rel

        vo_traj.append(T_abs[:3, 3].copy())
        if has_gt:
            gt_traj.append(gt_poses[fi + 1][:3, 3].copy())
        abs_poses.append(T_abs.copy())  # 帧fi+1: 成功更新位姿
        n_success += 1

        if mode == 'motionnet':
            pbar.set_postfix({'matches': n_match, 'valid3d': n_valid,
                              'static': n_static, 'angle': f'{angle_deg:.1f}°',
                              'sc': f'{scale_corr:.3f}', 'ok': n_success})
        else:
            pbar.set_postfix({'matches': n_match, 'valid3d': n_valid,
                              'sc': f'{scale_corr:.3f}',
                              'angle': f'{angle_deg:.1f}°', 'ok': n_success})

    vo_traj = np.array(vo_traj)  # mm
    if has_gt and len(gt_traj) > 0:
        gt_traj = np.array(gt_traj)  # mm
    else:
        gt_traj = np.array([]).reshape(0, 3)

    # 保存轨迹供后续分析
    np.save(os.path.join(seq_dir, 'vo_traj_scaled.npy'), vo_traj)
    if has_gt:
        np.save(os.path.join(seq_dir, 'gt_traj.npy'), gt_traj)

    # ── 保存完整 4×4 绝对位姿矩阵 ──
    abs_poses_arr = np.stack(abs_poses, axis=0)  # (N, 4, 4)
    np.save(os.path.join(seq_dir, f'{mode}_abs_poses.npy'), abs_poses_arr)
    print(f"  绝对位姿已保存: {seq_dir}/{mode}_abs_poses.npy ({abs_poses_arr.shape[0]} 帧)")

    # ── 保存深度预测图 ──
    depth_dict = {f'{k:04d}': depth_maps[k] for k in sorted(depth_maps.keys())}
    depth_path = os.path.join(seq_dir, f'{mode}_depth_maps.npz')
    np.savez_compressed(depth_path, **depth_dict)
    print(f"  深度图已保存: {depth_path} ({len(depth_dict)} 帧)")

    # ── 预测最后一帧深度 (用于链式追踪的世界坐标反投影) ──
    n_frames = len(frames)
    last_idx = n_frames - 1
    if last_idx not in depth_maps and depth_source != 'gt' and encoder is not None:
        last_img = os.path.join(warp_dir, frames[last_idx])
        if motion_encoder is not None and last_idx > 0:
            prev_img = os.path.join(warp_dir, frames[last_idx - 1])
            depth_maps[last_idx] = predict_depth_with_motion(
                last_img, prev_img, encoder, depth_decoder,
                motion_encoder, device, inverse=False,
                min_depth=DEPTH_MIN, max_depth=DEPTH_MAX)
        else:
            depth_maps[last_idx] = predict_depth(
                last_img, encoder, depth_decoder, device,
                inverse=False, min_depth=DEPTH_MIN, max_depth=DEPTH_MAX)
    elif last_idx not in depth_maps and depth_source == 'gt' and has_gt:
        depth_maps[last_idx] = load_gt_depth(seq_dir, last_idx)

    # 末帧深度也做全局尺度校正
    if last_idx in depth_maps and depth_source != 'gt':
        depth_maps[last_idx] = depth_maps[last_idx] * global_depth_scale

    # 打包链式追踪数据
    chain_data = {
        'all_k0': all_k0,
        'all_k1': all_k1,
        'all_weights': all_weights,
        'depth_maps': depth_maps,
        'abs_poses': abs_poses,  # list of (4,4) T_world_cam per frame
        'pipeline_params': pipeline_params,  # 自适应参数
        'depth_source': depth_source,  # 深度来源, 用于组织运动尺度恢复判断
    }

    # ── 保存 chain_data 供独立评估脚本使用 ──
    chain_save = {}
    chain_save['n_pairs'] = len(all_k0)
    for i in range(len(all_k0)):
        chain_save[f'k0_{i:04d}'] = all_k0[i]
        chain_save[f'k1_{i:04d}'] = all_k1[i]
        chain_save[f'w_{i:04d}'] = all_weights[i]
    chain_save['K'] = pipeline_params.get('K', np.eye(3))
    chain_save['k_factor'] = float(pipeline_params.get('k_factor', DISP_K_FACTOR))
    chain_save['chain_dist_thresh'] = float(pipeline_params.get('chain_dist_thresh', CHAIN_DIST_THRESH))
    chain_save['global_depth_scale'] = float(pipeline_params.get('global_depth_scale', 1.0))
    chain_save['depth_source'] = depth_source
    chain_path = os.path.join(seq_dir, f'{mode}_chain_data.npz')
    np.savez_compressed(chain_path, **chain_save)
    print(f"  Chain数据已保存: {chain_path} ({len(all_k0)} 帧对)")

    return vo_traj, gt_traj, {'n_success': n_success, 'n_total': n_pairs, 'depth_ratio': float(np.median(depth_ratios)) if depth_ratios else None}, chain_data


def _is_valid_3d(j, k0, depth, h_d, w_d):
    """检查匹配点 j 是否有有效3D深度."""
    u, v = k0[j]
    ui, vi = int(np.clip(u, 0, w_d - 1)), int(np.clip(v, 0, h_d - 1))
    Z = float(depth[vi, ui])
    return Z > 0.5 and Z <= 500.0 and np.isfinite(Z)


# ═══════════════════════════════════════════════════════════
# 运动组织绝对轨迹计算
# ═══════════════════════════════════════════════════════════

CHAIN_DIST_THRESH = 3.0   # 跨帧匹配距离阈值 (像素)
MOTION_STATIC_THRESH = 0.5  # P(static) < 此值 → 运动点
DISP_K_FACTOR = 0.5  # 自适应阈值: median + k*std (PnP/track k=0.5)


def chain_tracks(all_k0, all_k1, dist_thresh=CHAIN_DIST_THRESH):
    """串联相邻帧 LoFTR 匹配, 形成跨帧 track.

    Args:
        all_k0: list of (N_i, 2) — pt0 在帧 i
        all_k1: list of (N_i, 2) — pt1 在帧 i+1
        dist_thresh: 跨帧匹配的最大像素距离

    Returns:
        tracks: list of dict {
            'id': int,
            'frames': [(frame_idx, u, v), ...],  # 该 track 在各帧的坐标
            'pair_indices': [(pair_idx, pt_idx), ...],  # 在各匹配中的索引
        }
    """
    n_pairs = len(all_k0)
    n_frames = n_pairs + 1

    # 初始化: 第0对匹配的每个点都是一个新 track
    tracks = []
    active = {}  # track_id → (frame, u, v)
    next_id = 0

    for pair_idx in range(n_pairs):
        k0 = all_k0[pair_idx]
        k1 = all_k1[pair_idx]

        # 如果这是第一对, 为每个 pt0 创建新 track
        if pair_idx == 0:
            for j in range(len(k0)):
                tid = next_id
                next_id += 1
                tracks.append({
                    'id': tid,
                    'frames': [(0, float(k0[j, 0]), float(k0[j, 1])),
                               (1, float(k1[j, 0]), float(k1[j, 1]))],
                    'pair_indices': [(0, j)],
                })
                active[tid] = (1, float(k1[j, 0]), float(k1[j, 1]))
            continue

        # 构建当前帧 (pair_idx) 的 active 点 KD-tree
        if not active:
            # 没有延续的 track, 为每个 k0 创建新 track
            for j in range(len(k0)):
                tid = next_id
                next_id += 1
                tracks.append({
                    'id': tid,
                    'frames': [(pair_idx, float(k0[j, 0]), float(k0[j, 1])),
                               (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))],
                    'pair_indices': [(pair_idx, j)],
                })
                active[tid] = (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))
            continue

        # 匹配: k0 中的点到 active tracks 的最近点
        active_ids = list(active.keys())
        active_pts = np.array([[active[tid][1], active[tid][2]] for tid in active_ids])

        matched_tids = set()
        new_active = {}

        if len(k0) > 0 and len(active_pts) > 0:
            # 对每个 k0 点, 找最近的 active 点
            for j in range(len(k0)):
                pt = k0[j]
                dists = np.sqrt(np.sum((active_pts - pt) ** 2, axis=1))
                min_idx = np.argmin(dists)
                if dists[min_idx] <= dist_thresh:
                    tid = active_ids[min_idx]
                    if tid not in matched_tids:
                        # 延续 track
                        tracks[tid]['frames'].append(
                            (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1])))
                        tracks[tid]['pair_indices'].append((pair_idx, j))
                        new_active[tid] = (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))
                        matched_tids.add(tid)

        # 未匹配的 k0 点: 创建新 track
        for j in range(len(k0)):
            pt = k0[j]
            dists = np.sqrt(np.sum((active_pts - pt) ** 2, axis=1))
            if len(active_pts) == 0 or np.min(dists) > dist_thresh:
                tid = next_id
                next_id += 1
                tracks.append({
                    'id': tid,
                    'frames': [(pair_idx, float(k0[j, 0]), float(k0[j, 1])),
                               (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))],
                    'pair_indices': [(pair_idx, j)],
                })
                new_active[tid] = (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))

        active = new_active

    return tracks


def backproject_to_world(u, v, depth_map, T_world_cam, K):
    """反投影 2D 点到世界 3D 坐标.

    Args:
        u, v: 像素坐标 (原始分辨率 1350×1080)
        depth_map: (H,W) 深度图 (mm)
        T_world_cam: (4,4) 世界→相机变换
        K: (3,3) 原始内参

    Returns:
        p_world: (3,) 世界坐标 (mm), 或 None
    """
    h, w = depth_map.shape
    ui = int(np.clip(round(u), 0, w - 1))
    vi = int(np.clip(round(v), 0, h - 1))
    Z = float(depth_map[vi, ui])
    if Z <= 0.5 or not np.isfinite(Z):
        return None

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_cam = np.array([(u - cx) * Z / fx, (v - cy) * Z / fy, Z])

    T_cam_world = np.linalg.inv(T_world_cam)
    p_world = T_cam_world[:3, :3] @ p_cam + T_cam_world[:3, 3]
    return p_world


def compute_track_displacement(track, depth_maps, abs_poses, K):
    """计算单个 track 的世界坐标轨迹和总位移.

    Returns:
        p_worlds: list of (3,) or None — 每帧世界坐标 (mm)
        total_disp: float — 总位移量 (mm)
    """
    p_worlds = []
    for frame_idx, u, v in track['frames']:
        if frame_idx not in depth_maps or frame_idx >= len(abs_poses):
            p_worlds.append(None)
            continue
        pw = backproject_to_world(u, v, depth_maps[frame_idx],
                                  abs_poses[frame_idx], K_ORIG)
        p_worlds.append(pw)

    valid = [p for p in p_worlds if p is not None]
    if len(valid) < 2:
        return p_worlds, 0.0

    total_disp = np.linalg.norm(valid[-1] - valid[0])
    return p_worlds, float(total_disp)


def classify_tracks(tracks, all_weights, depth_maps=None, abs_poses=None,
                     motion_thresh=MOTION_STATIC_THRESH,
                     k_factor=None):
    """Hybrid: Track 级分类强制走几何位移, 不用 MotionNet 权重.

    PnP 阶段用 MotionNet 保证精度, Track 分类用位移保证分离度.
    """
    # 混合策略: 只要位移数据可用, 就走几何分类
    if depth_maps is not None and abs_poses is not None:
        print("  (混合模式: Track 分类用自适应 3D 位移, PnP 用 MotionNet)")
        k = k_factor if k_factor is not None else DISP_K_FACTOR
        static_tracks, moving_tracks = _classify_by_displacement(
            tracks, depth_maps, abs_poses, K_ORIG, k_factor=k)
    else:
        # 纯 MotionNet 回退 (无位移数据)
        has_weights = any(len(w) > 0 for w in all_weights) and \
                      any(np.any(w < 0.9) for w in all_weights if len(w) > 0)
        if has_weights:
            static_tracks, moving_tracks = _classify_by_weights(
                tracks, all_weights, motion_thresh)
        else:
            static_tracks, moving_tracks = [], tracks  # 全判运动

    return static_tracks, moving_tracks


def _classify_by_weights(tracks, all_weights, motion_thresh):
    """基于 MotionNet 权重分类."""
    static_tracks, moving_tracks = [], []
    for track in tracks:
        weights_list = []
        for pair_idx, pt_idx in track['pair_indices']:
            if pair_idx < len(all_weights) and pt_idx < len(all_weights[pair_idx]):
                weights_list.append(all_weights[pair_idx][pt_idx])
        avg_w = float(np.mean(weights_list)) if weights_list else 1.0
        if avg_w < motion_thresh:
            moving_tracks.append((track, avg_w))
        else:
            static_tracks.append((track, avg_w))
    return static_tracks, moving_tracks


def _filter_by_3d_displacement(pts3d, k0, k1, depth_curr, depth_next, K, k_factor=DISP_K_FACTOR):
    """用自适应 3D 位移阈值过滤匹配点 (基线模式 PnP 前处理).

    计算每个匹配点在两帧间的 3D 位移 (相机坐标系近似),
    阈值 = median + k*std, 位移超阈值 → 丢弃 (疑似运动点)。

    Args:
        pts3d:      (N, 3) 帧0 3D点 (相机坐标系)
        k0:         (N, 2) 帧0 像素坐标
        k1:         (N, 2) 帧1 像素坐标
        depth_curr: (H, W) 帧0 深度图
        depth_next: (H, W) 帧1 深度图
        K:          (3, 3) 内参
        k_factor:   自适应阈值系数

    Returns:
        keep:  (N,) bool mask
        stats: dict
    """
    N = len(pts3d)
    if N < 4:
        return np.ones(N, dtype=bool), {'n_kept': N, 'n_total': N, 'threshold_mm': 0}

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    h_dn, w_dn = depth_next.shape

    displacements = np.full(N, np.nan, dtype=np.float32)
    for j in range(N):
        u1, v1 = k1[j]
        ui1 = int(np.clip(round(u1), 0, w_dn - 1))
        vi1 = int(np.clip(round(v1), 0, h_dn - 1))
        Z1 = float(depth_next[vi1, ui1])
        if Z1 <= 0.5 or Z1 > 500.0 or not np.isfinite(Z1):
            continue
        X1 = (u1 - cx) * Z1 / fx
        Y1 = (v1 - cy) * Z1 / fy
        p1 = np.array([X1, Y1, Z1])
        displacements[j] = float(np.linalg.norm(p1 - pts3d[j]))

    valid = ~np.isnan(displacements)
    if valid.sum() < 4:
        return np.ones(N, dtype=bool), {'n_kept': N, 'n_total': N, 'threshold_mm': 0}

    valid_disp = displacements[valid]
    med = float(np.median(valid_disp))
    std = float(np.std(valid_disp))
    threshold = med + k_factor * std
    threshold = max(threshold, 1.0)  # 至少 1mm

    keep = displacements <= threshold
    if keep.sum() < 4:
        sorted_idx = np.argsort(displacements)
        n_keep = min(N, max(4, N // 2))
        keep = np.zeros(N, dtype=bool)
        keep[sorted_idx[:n_keep]] = True

    return keep, {'n_kept': int(keep.sum()), 'n_total': N, 'threshold_mm': threshold, 'median_mm': med, 'std_mm': std}


def _classify_by_displacement(tracks, depth_maps, abs_poses, K,
                               k_factor=DISP_K_FACTOR, sample_max=5000):
    """基于自适应 3D 位移阈值分类 track 为静止/运动。

    策略:
      1. 对每个 track, 计算相邻帧间 3D 位移中位数
      2. 全局阈值 = median(所有 track 中位位移) + k * std
      3. track 中位位移 > 阈值 → 运动点
    """
    import random

    # 限制采样以加速阈值估计
    if len(tracks) > sample_max:
        random.seed(42)
        sample = random.sample(tracks, sample_max)
    else:
        sample = tracks

    # 计算每个采样 track 的相邻帧 3D 位移中位数
    track_medians = []
    for track in sample:
        p_worlds, _ = compute_track_displacement(track, depth_maps, abs_poses, K)
        valid_p = [p for p in p_worlds if p is not None]
        if len(valid_p) < 2:
            track_medians.append(0.0)
            continue
        steps = []
        for i in range(len(valid_p) - 1):
            steps.append(np.linalg.norm(valid_p[i + 1] - valid_p[i]))
        track_medians.append(float(np.median(steps)))

    track_medians = np.array(track_medians)
    med = float(np.median(track_medians))
    std = float(np.std(track_medians))
    threshold = med + k_factor * std
    print(f"  自适应阈值: median={med:.2f}mm, std={std:.2f}mm, "
          f"threshold={threshold:.2f}mm (k={k_factor})")

    # 用采样阈值分类全部 track
    static_tracks, moving_tracks = [], []
    for track in tracks:
        p_worlds, _ = compute_track_displacement(track, depth_maps, abs_poses, K)
        valid_p = [p for p in p_worlds if p is not None]
        if len(valid_p) < 2:
            static_tracks.append((track, 0.0))
            continue
        steps = []
        for i in range(len(valid_p) - 1):
            steps.append(np.linalg.norm(valid_p[i + 1] - valid_p[i]))
        med_disp = float(np.median(steps))
        if med_disp > threshold:
            moving_tracks.append((track, med_disp))
        else:
            static_tracks.append((track, med_disp))

    return static_tracks, moving_tracks


def compute_motion_summary(moving_tracks, static_tracks, depth_maps, abs_poses, K):
    """计算所有 track 的位移统计.

    Returns:
        dict with displacement stats
    """
    # 静止点: 理想位移≈0, 实际反映相机位姿误差
    static_disps = []
    for track, _ in static_tracks:
        _, disp = compute_track_displacement(track, depth_maps, abs_poses, K)
        static_disps.append(disp)

    # 运动点: 组织实际位移
    moving_disps = []
    moving_data = []
    for track, avg_w in moving_tracks:
        p_worlds, disp = compute_track_displacement(track, depth_maps, abs_poses, K)
        moving_disps.append(disp)
        n_valid = sum(1 for p in p_worlds if p is not None)
        moving_data.append({
            'track_id': track['id'],
            'n_frames': n_valid,
            'avg_motion_weight': float(avg_w),
            'total_displacement_mm': disp,
        })

    summary = {
        'n_static_tracks': len(static_tracks),
        'n_moving_tracks': len(moving_tracks),
        'static_disp_stats': {
            'mean_mm': float(np.mean(static_disps)) if static_disps else 0,
            'median_mm': float(np.median(static_disps)) if static_disps else 0,
            'max_mm': float(np.max(static_disps)) if static_disps else 0,
        },
        'moving_disp_stats': {
            'mean_mm': float(np.mean(moving_disps)) if moving_disps else 0,
            'median_mm': float(np.median(moving_disps)) if moving_disps else 0,
            'max_mm': float(np.max(moving_disps)) if moving_disps else 0,
        },
        'top_moving_tracks': sorted(moving_data, key=lambda x: -x['total_displacement_mm'])[:20],
    }
    return summary


def evaluate_motion_vs_gt(tracks, depth_maps, abs_poses_vo, gt_poses, K,
                           moving_tracks=None):
    """对比 VO 位移 vs GT 位姿位移.

    逻辑:
      - 用 GT 相机位姿反投影 → GT 世界位移 (真正的运动真值)
      - 用 VO 相机位姿反投影 → VO 世界位移 (估计值)
      - 静止点: GT_disp ≈ 0, VO_disp = 相机位姿误差
      - 运动点: GT_disp > 0, 对比 VO_disp 与 GT_disp 的误差

    Returns:
        dict with error metrics
    """
    import random

    # 如果提供了 moving_tracks, 分别评估静止和运动点
    if moving_tracks is not None:
        all_tracks_eval = tracks
        moving_set = set(t[0]['id'] for t in moving_tracks)
    else:
        all_tracks_eval = random.sample(tracks, min(5000, len(tracks)))
        moving_set = set()

    gt_disps = []
    vo_disps = []
    abs_errors = []
    is_static_list = []

    # GT 位姿是 c2w, 需转 w2c 才能传入 compute_track_displacement
    gt_poses_w2c = [np.linalg.inv(T) for T in gt_poses]

    for track in all_tracks_eval:
        if isinstance(track, tuple):
            track = track[0]

        # GT 位移
        _, gt_disp = compute_track_displacement(track, depth_maps, gt_poses_w2c, K)
        # VO 位移
        _, vo_disp = compute_track_displacement(track, depth_maps, abs_poses_vo, K)

        gt_disps.append(gt_disp)
        vo_disps.append(vo_disp)
        abs_errors.append(abs(vo_disp - gt_disp))
        is_static_list.append(track['id'] not in moving_set)

    gt_disps = np.array(gt_disps)
    vo_disps = np.array(vo_disps)
    abs_errors = np.array(abs_errors)
    is_static_arr = np.array(is_static_list)

    # GT 静止点: GT_disp < 1mm
    static_mask = gt_disps < 1.0
    moving_mask = ~static_mask

    # 如果提供了 MotionNet 分类，交叉验证
    mn_static_mask = is_static_arr
    mn_moving_mask = ~mn_static_mask

    metrics = {
        'n_total': len(all_tracks_eval),
        'n_gt_static': int(static_mask.sum()),
        'n_gt_moving': int(moving_mask.sum()),
        # GT 静止点: VO 估计的位姿误差 (应为0, 实际反映相机ATE)
        'static_pose_error_mm': {
            'mean': float(np.mean(abs_errors[static_mask])) if static_mask.any() else 0,
            'median': float(np.median(abs_errors[static_mask])) if static_mask.any() else 0,
            'rmse': float(np.sqrt(np.mean(abs_errors[static_mask]**2))) if static_mask.any() else 0,
        },
        # GT 运动点: VO 位移 vs GT 位移
        'moving_displacement_error_mm': {
            'mean': float(np.mean(abs_errors[moving_mask])) if moving_mask.any() else 0,
            'median': float(np.median(abs_errors[moving_mask])) if moving_mask.any() else 0,
            'rmse': float(np.sqrt(np.mean(abs_errors[moving_mask]**2))) if moving_mask.any() else 0,
        },
        # GT vs VO 位移相关性
        'correlation': float(np.corrcoef(gt_disps, vo_disps)[0, 1])
                       if len(gt_disps) > 1 else 0,
        # MotionNet 分类 vs GT 静止/运动 一致性
        'motionnet_accuracy': float((mn_static_mask == static_mask).mean()),
    }

    # 打印
    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  GT 位姿评估 (运动真值)                ║")
    print(f"  ╠══════════════════════════════════════════╣")
    print(f"  ║  GT 静止点 (n={metrics['n_gt_static']}):                ║")
    print(f"  ║    VO 位移误差 RMSE:                 ║")
    print(f"  ║      {metrics['static_pose_error_mm']['rmse']:>8.2f} mm (应≈ATE)         ║")
    print(f"  ║  GT 运动点 (n={metrics['n_gt_moving']}):                ║")
    print(f"  ║    VO 位移误差 RMSE:                 ║")
    print(f"  ║      {metrics['moving_displacement_error_mm']['rmse']:>8.2f} mm               ║")
    print(f"  ║  GT vs VO 位移相关: {metrics['correlation']:>8.3f}             ║")
    print(f"  ╚══════════════════════════════════════════╝")

    return metrics


def save_motion_trajectories(moving_tracks, depth_maps, abs_poses, K, out_dir, seq_name):
    """保存运动点世界坐标轨迹到 JSON."""
    os.makedirs(out_dir, exist_ok=True)

    trajectories = []
    for track, avg_w in moving_tracks:
        p_worlds, disp = compute_track_displacement(track, depth_maps, abs_poses, K)
        frames_data = []
        for fi, (frame_idx, u, v) in enumerate(track['frames']):
            pw = p_worlds[fi]
            frames_data.append({
                'frame': int(frame_idx),
                'u': float(u),
                'v': float(v),
                'x_mm': float(pw[0]) if pw is not None else None,
                'y_mm': float(pw[1]) if pw is not None else None,
                'z_mm': float(pw[2]) if pw is not None else None,
            })
        trajectories.append({
            'track_id': track['id'],
            'avg_motion_weight': float(avg_w),
            'total_displacement_mm': float(disp),
            'n_frames': sum(1 for p in p_worlds if p is not None),
            'frames': frames_data,
        })

    out_path = os.path.join(out_dir, f'{seq_name}_motion_trajectories.json')
    with open(out_path, 'w') as f:
        json.dump(trajectories, f, indent=2)
    print(f"  运动轨迹已保存: {out_path} ({len(trajectories)} tracks)")
    return out_path


def visualize_top_trajectories(moving_tracks, depth_maps, abs_poses, K,
                                out_dir, seq_name, top_k=5):
    """3D 可视化位移最大的运动点轨迹."""
    os.makedirs(out_dir, exist_ok=True)

    # 选位移最大的 top_k 个 track
    disp_list = []
    for track, avg_w in moving_tracks:
        _, disp = compute_track_displacement(track, depth_maps, abs_poses, K)
        disp_list.append((disp, track, avg_w))
    disp_list.sort(key=lambda x: -x[0])

    fig = plt.figure(figsize=(14, 5))

    # 子图1: 3D 轨迹
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    colors = plt.cm.jet(np.linspace(0, 1, min(top_k, len(disp_list))))

    for idx, (disp, track, avg_w) in enumerate(disp_list[:top_k]):
        p_worlds, _ = compute_track_displacement(track, depth_maps, abs_poses, K)
        valid_pts = np.array([p for p in p_worlds if p is not None])
        if len(valid_pts) < 2:
            continue
        ax1.plot(valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2],
                 'o-', color=colors[idx], markersize=3, linewidth=1.5,
                 label=f'Track {track["id"]} ({disp:.1f}mm)')
        ax1.scatter(*valid_pts[0], color=colors[idx], s=50, marker='s', edgecolors='k')
        ax1.scatter(*valid_pts[-1], color=colors[idx], s=50, marker='*', edgecolors='k')

    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Top {top_k} 运动点世界3D轨迹')
    ax1.legend(fontsize=7, loc='upper left')

    # 子图2: 位移分布直方图
    ax2 = fig.add_subplot(1, 2, 2)
    all_disps = [d for d, _, _ in disp_list]
    ax2.hist(all_disps, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(np.median(all_disps), color='red', linestyle='--',
                label=f'Median: {np.median(all_disps):.1f}mm')
    ax2.set_xlabel('Total Displacement (mm)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'运动点位移分布 (N={len(all_disps)})')
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{seq_name}_motion_3d.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  可视化已保存: {out_path}")
    return out_path


def visualize_motion_vs_gt(moving_tracks, static_tracks, depth_maps, abs_poses_vo, gt_poses, K,
                            out_dir, seq_name, top_k=5, n_scatter=2000, n_align=2000):
    """VO vs GT 运动点轨迹对比可视化 (静态点云对齐版).

    核心改进:
      - 不再用相机轨迹位置做 Umeyama (仅 ~30 对点, 且无法对齐 6-DoF 朝向)
      - 改用静态组织点云做 Umeyama (数千对点, 精确求解 VO→GT 坐标系刚性变换)
      - 静态点在同一物理位置 → VO坐标 和 GT坐标 的对应关系直接定义了两系变换

    生成两张图:
      图1: 每个运动点独立的 3D 轨迹对比子图网格 (不叠加, 清晰可辨)
           每个子图标题含 track ID、VO位移、GT位移、误差Δ
      图2: 汇总图 — 左: VO vs GT 位移散点图 (r 相关系数) / 右: 前 top_k 位移误差柱状图
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 0: 逐帧相机对齐 (per-frame VO→GT rigid transform) ──
    # 原理: 对于帧 i, VO 和 GT 相机位姿直接给出 VO_world → GT_world 的刚性变换
    #   T_vo_to_gt[i] = gt_poses[i] @ abs_poses_vo[i]
    # 每个 track 使用其中间帧的对齐变换, 避免全局 Umeyama 的旋转偏差
    # 这完全消除了 "八字形" 发散问题 (全局旋转对齐误差在远离 centroid 处被放大)
    gt_poses_w2c = [np.linalg.inv(T) for T in gt_poses]
    vo_to_gt = compute_per_frame_vo_to_gt(gt_poses, abs_poses_vo)
    print(f"  [逐帧对齐] 已计算 {len(vo_to_gt)} 帧的 VO→GT 刚性变换")

    def _align_vo_point_for_track(p_vo, track):
        """用 track 中间帧的 VO→GT 变换对齐 VO 点."""
        if p_vo is None:
            return None
        mid_idx = len(track['frames']) // 2
        mid_frame = track['frames'][mid_idx][0]
        if mid_frame not in vo_to_gt:
            return None
        R_t, t_t = vo_to_gt[mid_frame]
        return R_t @ p_vo + t_t

    # ── 收集所有运动点数据 ──
    # GT 位姿已在上方转为 w2c (gt_poses_w2c)
    all_data = []  # (disp_vo, disp_gt, track, avg_w, p_worlds_vo_aligned, p_worlds_gt)
    for track, avg_w in moving_tracks:
        p_worlds_vo_raw, disp_vo = compute_track_displacement(
            track, depth_maps, abs_poses_vo, K)
        p_worlds_gt, disp_gt = compute_track_displacement(
            track, depth_maps, gt_poses_w2c, K)
        # ── 对齐 VO 3D 点到 GT 坐标系 (逐 track 相机对齐) ──
        p_worlds_vo_aligned = [_align_vo_point_for_track(p, track) for p in p_worlds_vo_raw]
        all_data.append((disp_vo, disp_gt, track, avg_w, p_worlds_vo_aligned, p_worlds_gt,
                         p_worlds_vo_raw))

    # 按 GT 位移排序, 优先选长轨迹 (frames≥5), 短轨迹不可靠
    all_data.sort(key=lambda x: -x[1])
    # 过滤: 只取 frames >= 5 的长轨迹作为 top 候选
    long_tracks = [d for d in all_data if len(d[2]['frames']) >= 5]
    if len(long_tracks) >= top_k:
        top_data = long_tracks[:top_k]
    else:
        # 不够 top_k 个长轨迹时从短轨迹补充
        short_tracks = [d for d in all_data if len(d[2]['frames']) < 5]
        top_data = long_tracks + short_tracks[:top_k - len(long_tracks)]

    n_tracks = len(top_data)
    n_cols = min(4, n_tracks)
    n_rows = (n_tracks + n_cols - 1) // n_cols

    # ═══════════════════════════════════════════
    # 图1: 每个运动点独立 3D 轨迹对比 (不再叠加, 避免成团)
    # ═══════════════════════════════════════════
    fig1 = plt.figure(figsize=(4.2 * n_cols, 3.8 * n_rows))
    fig1.suptitle(f'VO (aligned) vs GT  —  Individual 3D Trajectories\n{seq_name}',
                  fontsize=12, fontweight='bold')

    for idx in range(n_tracks):
        disp_vo, disp_gt, track, _, p_vo_aligned, p_gt, _ = top_data[idx]

        ax = fig1.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        valid_vo = np.array([p for p in p_vo_aligned if p is not None])
        valid_gt = np.array([p for p in p_gt if p is not None])

        # GT: 绿色虚线方块
        if len(valid_gt) >= 2:
            ax.plot(valid_gt[:, 0], valid_gt[:, 1], valid_gt[:, 2],
                    's--', color='green', markersize=3, linewidth=1.3, label='GT')
            ax.plot(valid_gt[:, 0], valid_gt[:, 1], valid_gt[:, 2],
                    color='green', alpha=0.15, linewidth=0.5)  # 投影线辅助
            ax.scatter(*valid_gt[0], color='darkgreen', s=35, marker='s',
                       edgecolors='k', linewidths=0.5, zorder=5)
            ax.scatter(*valid_gt[-1], color='darkgreen', s=45, marker='D',
                       edgecolors='k', linewidths=0.5, zorder=5)

        # VO: 蓝色实线圆点
        if len(valid_vo) >= 2:
            ax.plot(valid_vo[:, 0], valid_vo[:, 1], valid_vo[:, 2],
                    'o-', color='steelblue', markersize=3, linewidth=1.5, label='VO')
            ax.scatter(*valid_vo[0], color='steelblue', s=45, marker='s',
                       edgecolors='k', linewidths=0.5, zorder=5)
            ax.scatter(*valid_vo[-1], color='steelblue', s=55, marker='D',
                       edgecolors='k', linewidths=0.5, zorder=5)

            # 自适应坐标轴范围 (基于当前 track 所有点)
            all_pts = valid_vo
            if len(valid_gt) >= 2:
                all_pts = np.vstack([valid_vo, valid_gt])
            x_range = all_pts[:, 0].max() - all_pts[:, 0].min()
            y_range = all_pts[:, 1].max() - all_pts[:, 1].min()
            z_range = all_pts[:, 2].max() - all_pts[:, 2].min()
            margin = max(5, max(x_range, y_range, z_range) * 0.2)
            ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
            ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
            ax.set_zlim(all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

        # 标题含关键指标
        err_mm = abs(disp_vo - disp_gt)
        ax.set_title(f"T{track['id']}  VO={disp_vo:.1f}  GT={disp_gt:.1f}  Δ={err_mm:.1f}mm",
                     fontsize=8, fontweight='bold')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='upper left')

    plt.tight_layout()
    out_path_3d = os.path.join(out_dir, f'{seq_name}_motion_vs_gt_3d.png')
    fig1.savefig(out_path_3d, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"  独立3D轨迹对比图 (n={n_tracks}) 已保存: {out_path_3d}")

    # ═══════════════════════════════════════════
    # 图2: 位移散点图 + 位移误差柱状图 (汇总)
    # ═══════════════════════════════════════════
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig2.suptitle(f'VO vs GT Displacement Summary — {seq_name}',
                  fontsize=12, fontweight='bold')

    n_sample = min(n_scatter, len(all_data))
    if len(all_data) > n_sample:
        import random
        random.seed(42)
        sampled = random.sample(all_data, n_sample)
    else:
        sampled = all_data

    vo_disps = np.array([d[0] for d in sampled])
    gt_disps = np.array([d[1] for d in sampled])

    max_val = max(vo_disps.max(), gt_disps.max()) * 1.1
    ax3.scatter(gt_disps, vo_disps, c='steelblue', alpha=0.4, s=8, edgecolors='none')
    ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=1, alpha=0.7, label='Ideal (VO=GT)')

    # 线性拟合线
    if len(gt_disps) > 2:
        coeffs = np.polyfit(gt_disps, vo_disps, 1)
        fit_x = np.linspace(0, max_val, 100)
        ax3.plot(fit_x, np.polyval(coeffs, fit_x), 'g-', linewidth=1.5, alpha=0.7,
                 label=f'Fit: VO={coeffs[0]:.3f}×GT+{coeffs[1]:.1f}')
        r = np.corrcoef(gt_disps, vo_disps)[0, 1]
    else:
        coeffs, r = [0, 0], 0

    ax3.set_xlabel('GT Displacement (mm)')
    ax3.set_ylabel('VO Displacement (mm)')
    ax3.set_title(f'VO vs GT Displacement (n={len(sampled)}, r={r:.3f})')
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, max_val); ax3.set_ylim(0, max_val)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # 右: 前 top_k 运动点位移误差柱状图
    track_labels = []
    vo_vals, gt_vals, err_vals = [], [], []
    for idx, (disp_vo, disp_gt, track, _, _, _, _) in enumerate(top_data):
        track_labels.append(f'T{track["id"]}')
        vo_vals.append(disp_vo)
        gt_vals.append(disp_gt)
        err_vals.append(abs(disp_vo - disp_gt))

    x = np.arange(len(track_labels))
    width = 0.3

    bars_gt = ax4.bar(x - width/2, gt_vals, width, color='green', alpha=0.7, label='GT Displacement')
    bars_vo = ax4.bar(x + width/2, vo_vals, width, color='steelblue', alpha=0.7, label='VO Displacement')

    # 在柱状图上标注误差
    for i in range(len(track_labels)):
        ax4.text(x[i], max(gt_vals[i], vo_vals[i]) + 0.5,
                 f'Δ={err_vals[i]:.1f}', ha='center', fontsize=7, color='red', fontweight='bold')

    ax4.set_xticks(x)
    ax4.set_xticklabels(track_labels, fontsize=8)
    ax4.set_ylabel('Displacement (mm)')
    ax4.set_title(f'Top {top_k} Tracks: VO vs GT Displacement')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path_summary = os.path.join(out_dir, f'{seq_name}_motion_vs_gt_summary.png')
    fig2.savefig(out_path_summary, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  位移汇总图 已保存: {out_path_summary}")
    return out_path_3d


def compute_vo_gt_umeyama(static_tracks, depth_maps, abs_poses_vo, gt_poses, K, n_sample=2000):
    """用静态组织点云计算 VO→GT 坐标系的 Umeyama 对齐。

    原理: 同一物理静止点在 VO 和 GT 坐标系有不同的坐标值,
    这些对应关系直接定义了两系之间的刚性变换。

    Returns:
        (R, t, s): p_gt ≈ s * R @ p_vo + t
        或 (np.eye(3), np.zeros(3), 1.0) 如果静态点不足
    """
    import random
    random.seed(42)
    gt_poses_w2c = [np.linalg.inv(T) for T in gt_poses]

    sample = random.sample(static_tracks, min(n_sample, len(static_tracks)))
    vo_pts, gt_pts = [], []
    for track, _ in sample:
        n_frames = len(track['frames'])
        mid_idx = n_frames // 2
        frame_idx, u, v = track['frames'][mid_idx]
        if frame_idx not in depth_maps or frame_idx >= len(abs_poses_vo):
            continue
        p_vo = backproject_to_world(u, v, depth_maps[frame_idx],
                                     abs_poses_vo[frame_idx], K)
        p_gt = backproject_to_world(u, v, depth_maps[frame_idx],
                                     gt_poses_w2c[frame_idx], K)
        if p_vo is not None and p_gt is not None:
            vo_pts.append(p_vo)
            gt_pts.append(p_gt)

    vo_pts = np.array(vo_pts)
    gt_pts = np.array(gt_pts)

    if len(vo_pts) < 10:
        print(f"  ⚠ 静态点不足 ({len(vo_pts)}), 回退到单位对齐")
        return np.eye(3), np.zeros(3), 1.0

    _, _, R, s = align_trajectory_umeyama(vo_pts, gt_pts)
    t = np.mean(gt_pts, axis=0) - s * R @ np.mean(vo_pts, axis=0)
    print(f"  [Umeyama] 静态点对: {len(vo_pts)}, scale={s:.4f}, "
          f"R trace={np.trace(R):.3f}, t=[{t[0]:.1f},{t[1]:.1f},{t[2]:.1f}]")
    return R, t, s


def compute_per_frame_vo_to_gt(gt_poses, abs_poses_vo):
    """逐帧计算 VO 世界坐标 → GT 世界坐标 的刚性变换。

    对于每帧 i: T_vo_to_gt[i] = gt_poses[i] @ abs_poses_vo[i]
    原理: abs_poses_vo[i] 是 world_to_cam_vo, gt_poses[i] 是 cam_to_world_gt,
         组合后直接给出 VO_world → VO_cam → GT_cam → GT_world 的变换。

    Args:
        gt_poses: list of (4,4) GT cam_to_world 位姿 (标准 [R|t;0|1])
        abs_poses_vo: list of (4,4) VO world_to_cam 位姿

    Returns:
        dict {frame_idx: (R, t)}:  p_gt = R @ p_vo + t
    """
    n = min(len(gt_poses), len(abs_poses_vo))
    align = {}
    for i in range(n):
        T = gt_poses[i] @ abs_poses_vo[i]
        align[i] = (T[:3, :3].copy(), T[:3, 3].copy())
    return align


def evaluate_motion_detection_vs_gt(tracks, static_tracks, moving_tracks,
                                     depth_maps, abs_poses, K, seq_dir,
                                     umeyama_align=None, vo_to_gt_align=None):
    """用 GT mask_moving 评估运动检测性能。

    流程:
      1. 加载 coverage_mesh.obj 并构建 KD-tree
      2. 对每个 track 取最后一帧, 反投影到世界坐标
      3. KD-tree 找 mesh 最近顶点
      4. 查 GT mask_moving/frame_XXXX.npy[vertex_idx] → GT 标签
      5. 对比管线分类 (运动/静止) vs GT 标签 (1=运动/0=静止)

    Args:
        vo_to_gt_align: dict {frame_idx: (R, t)} 逐帧 VO→GT 刚性变换 (推荐)
        umeyama_align: (R, t, s) 全局 Umeyama 对齐 (兼容旧调用)

    Returns:
        dict with confusion_matrix, recall, precision, f1
    """
    import os
    from scipy.spatial import KDTree

    # ── 1. 加载 mesh 并构建 KD-tree ──
    mesh_path = os.path.join(seq_dir, 'coverage_mesh.obj')
    if not os.path.exists(mesh_path):
        print(f"  ⚠ mesh 不存在: {mesh_path}, 跳过 GT mask 评估")
        return None

    print("  加载 coverage_mesh.obj...")
    vertices = []
    with open(mesh_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    vertices = np.array(vertices, dtype=np.float32)
    print(f"    mesh 顶点: {len(vertices)}")
    tree = KDTree(vertices)

    mask_dir = os.path.join(seq_dir, 'generated', 'mask_moving')
    if not os.path.isdir(mask_dir):
        print(f"  ⚠ mask_moving 目录不存在: {mask_dir}")
        return None

    # ── 1.5: VO→GT 坐标系对齐 ──
    if vo_to_gt_align is not None:
        _get_align = lambda fi: vo_to_gt_align.get(fi, (np.eye(3), np.zeros(3)))
        def _ua(p, fi):
            if p is None:
                return None
            R, t = _get_align(fi)
            return R @ p + t
        print(f"  [逐帧对齐] 已启用 VO→GT 坐标系对齐 ({len(vo_to_gt_align)} 帧)")
    elif umeyama_align is not None:
        R_ua, t_ua, s_ua = umeyama_align
        _ua = lambda p, fi: s_ua * R_ua @ p + t_ua if p is not None else None
        print(f"  [Umeyama] 已启用 VO→GT 坐标系对齐 (scale={s_ua:.4f})")
    else:
        _ua = lambda p, fi: p

    # ── 2. 收集运动 track ID ──
    moving_ids = set(t[0]['id'] for t in moving_tracks)

    # ── 3. 逐 track 评估 ──
    tp, fp, fn, tn = 0, 0, 0, 0
    n_skipped = 0
    for track in tracks:
        if isinstance(track, tuple):
            track = track[0]

        tid = track['id']
        pred_moving = tid in moving_ids

        # 取 track 最后一帧
        last_frame = track['frames'][-1]
        frame_idx, u, v = last_frame

        # 反投影到世界坐标
        if frame_idx not in depth_maps or frame_idx >= len(abs_poses):
            n_skipped += 1
            continue
        pw = backproject_to_world(u, v, depth_maps[frame_idx],
                                  abs_poses[frame_idx], K)
        if pw is None:
            n_skipped += 1
            continue

        # VO→GT 坐标系对齐
        pw = _ua(pw, frame_idx)

        # KD-tree 最近顶点
        _, idx = tree.query(pw)

        # GT mask
        mask_path = os.path.join(mask_dir, f'frame_{frame_idx:04d}.npy')
        if not os.path.exists(mask_path):
            n_skipped += 1
            continue
        gt_label = int(np.load(mask_path)[idx])
        gt_moving = gt_label == 1

        if pred_moving and gt_moving:
            tp += 1
        elif pred_moving and not gt_moving:
            fp += 1
        elif not pred_moving and gt_moving:
            fn += 1
        else:
            tn += 1

    n_total = tp + fp + fn + tn
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  GT Mask 运动检测评估                   ║")
    print(f"  ╠══════════════════════════════════════════╣")
    print(f"  ║  评估 track 数: {n_total:>10d}               ║")
    print(f"  ║  TP (检出运动): {tp:>10d}               ║")
    print(f"  ║  TN (正确静止): {tn:>10d}               ║")
    print(f"  ║  FP (误报):     {fp:>10d}               ║")
    print(f"  ║  FN (漏检):     {fn:>10d}               ║")
    print(f"  ║  Recall:        {recall:>10.1%}               ║")
    print(f"  ║  Precision:     {precision:>10.1%}               ║")
    print(f"  ║  F1:            {f1:>10.3f}               ║")
    print(f"  ╚══════════════════════════════════════════╝")

    return {
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn},
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'n_total': n_total,
        'n_skipped': n_skipped,
    }


def evaluate_per_point_vs_gt(chain_data, K, seq_dir, k_factor=DISP_K_FACTOR,
                              umeyama_align=None, vo_to_gt_align=None):
    """逐帧逐点运动检测 vs GT mask_moving 评估 (对齐粒度)。

    与 track 级别分类不同, 这里对每帧的每个匹配点独立判断:
      1. 帧 k→k+1: 对每个匹配点, 分别反投影两帧世界坐标 → 3D 位移
      2. 自适应阈值 (per frame-pair): median + k*std
      3. 位移 > 阈值 → 预测运动点
      4. KD-tree 映射到 mesh 顶点 → 查 GT mask_moving 标签
      5. 汇总混淆矩阵

    Args:
        vo_to_gt_align: dict {frame_idx: (R, t)} 逐帧 VO→GT 刚性变换 (推荐)
        umeyama_align: (R, t, s) VO→GT 坐标系 Umeyama 对齐 (兼容旧调用)

    Returns:
        dict with confusion_matrix, recall, precision, f1
    """
    import os
    from scipy.spatial import KDTree

    all_k0 = chain_data['all_k0']
    all_k1 = chain_data['all_k1']
    depth_maps = chain_data['depth_maps']
    abs_poses = chain_data['abs_poses']

    # ── 1. 加载 mesh KD-tree ──
    mesh_path = os.path.join(seq_dir, 'coverage_mesh.obj')
    if not os.path.exists(mesh_path):
        print(f"  ⚠ mesh 不存在: {mesh_path}, 跳过逐点评估")
        return None

    print("  加载 coverage_mesh.obj ...")
    vertices = []
    with open(mesh_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    vertices = np.array(vertices, dtype=np.float32)
    tree = KDTree(vertices)

    mask_dir = os.path.join(seq_dir, 'generated', 'mask_moving')
    if not os.path.isdir(mask_dir):
        print(f"  ⚠ mask_moving 目录不存在: {mask_dir}")
        return None

    # ── 1.5: VO→GT 坐标系对齐 ──
    if vo_to_gt_align is not None:
        _get_align = lambda fi: vo_to_gt_align.get(fi, (np.eye(3), np.zeros(3)))
        def _ua(p, fi):
            if p is None:
                return None
            R, t = _get_align(fi)
            return R @ p + t
        print(f"  [逐帧对齐] 已启用 VO→GT 坐标系对齐 ({len(vo_to_gt_align)} 帧)")
    elif umeyama_align is not None:
        R_ua, t_ua, s_ua = umeyama_align
        _ua = lambda p, fi: s_ua * R_ua @ p + t_ua if p is not None else None
        print(f"  [Umeyama] 已启用 VO→GT 坐标系对齐 (scale={s_ua:.4f})")
    else:
        _ua = lambda p, fi: p

    # ── 2. 逐帧对评估 ──
    tp, fp, fn, tn = 0, 0, 0, 0
    n_total_pts = 0
    n_skipped = 0
    n_frames_evaluated = 0

    n_pairs = len(all_k0)
    for fi in range(n_pairs):
        k0 = all_k0[fi]
        k1 = all_k1[fi]
        if len(k0) < 4:
            continue

        depth_curr = depth_maps.get(fi)
        depth_next = depth_maps.get(fi + 1)
        if depth_curr is None or depth_next is None:
            continue
        if fi >= len(abs_poses) or fi + 1 >= len(abs_poses):
            continue

        pose_curr = abs_poses[fi]
        pose_next = abs_poses[fi + 1]

        # GT mask for this frame
        mask_path = os.path.join(mask_dir, f'frame_{fi:04d}.npy')
        if not os.path.exists(mask_path):
            continue
        gt_mask = np.load(mask_path)  # (N_verts,)

        # ── 计算每个匹配点的 3D 位移 ──
        displacements = np.full(len(k0), np.nan, dtype=np.float32)
        world_pts = np.zeros((len(k0), 3), dtype=np.float32)

        for j in range(len(k0)):
            u0, v0 = k0[j]
            u1, v1 = k1[j]

            pw0 = backproject_to_world(u0, v0, depth_curr, pose_curr, K)
            pw1 = backproject_to_world(u1, v1, depth_next, pose_next, K)

            if pw0 is None or pw1 is None:
                continue

            disp = float(np.linalg.norm(pw1 - pw0))
            displacements[j] = disp
            world_pts[j] = pw0

        # ── 自适应阈值 ──
        valid_disp = displacements[np.isfinite(displacements)]
        if len(valid_disp) < 10:
            continue

        med = float(np.median(valid_disp))
        std = float(np.std(valid_disp))
        threshold = med + k_factor * std

        # ── 逐点分类 + GT 比较 ──
        n_eval_this_frame = 0
        for j in range(len(k0)):
            disp = displacements[j]
            if not np.isfinite(disp):
                n_skipped += 1
                continue

            # KD-tree → vertex index (先用逐帧对齐到 GT 坐标系)
            pw_aligned = _ua(world_pts[j], fi)
            _, v_idx = tree.query(pw_aligned)

            # GT label
            if v_idx >= len(gt_mask):
                n_skipped += 1
                continue
            gt_label = int(gt_mask[v_idx])
            gt_moving = gt_label == 1

            # Predicted
            pred_moving = disp > threshold

            if pred_moving and gt_moving:
                tp += 1
            elif pred_moving and not gt_moving:
                fp += 1
            elif not pred_moving and gt_moving:
                fn += 1
            else:
                tn += 1

            n_eval_this_frame += 1
            n_total_pts += 1

        n_frames_evaluated += 1

        # 每 50 帧输出进度
        if (fi + 1) % 50 == 0:
            print(f"    逐点评估进度: {fi + 1}/{n_pairs} 帧对")

    if n_total_pts == 0:
        print("  ⚠ 无有效评估点, 跳过")
        return None

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  逐帧逐点 GT Mask 运动检测评估         ║")
    print(f"  ╠══════════════════════════════════════════╣")
    print(f"  ║  评估帧对:     {n_frames_evaluated:>10d}               ║")
    print(f"  ║  评估点总数:   {n_total_pts:>10d}               ║")
    print(f"  ║  TP (检出运动): {tp:>10d}               ║")
    print(f"  ║  TN (正确静止): {tn:>10d}               ║")
    print(f"  ║  FP (误报):     {fp:>10d}               ║")
    print(f"  ║  FN (漏检):     {fn:>10d}               ║")
    print(f"  ║  Recall:        {recall:>10.1%}               ║")
    print(f"  ║  Precision:     {precision:>10.1%}               ║")
    print(f"  ║  F1:            {f1:>10.3f}               ║")
    print(f"  ╚══════════════════════════════════════════╝")

    return {
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn},
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'n_total': n_total_pts,
        'n_frames': n_frames_evaluated,
        'n_skipped': n_skipped,
    }


def run_motion_pipeline(chain_data, seq_name, out_dir, mode):
    """运行运动组织轨迹计算管线.

    Args:
        chain_data: from run_vo_sequence
        seq_name: 序列名
        out_dir: 输出目录
        mode: 'baseline' / 'motionnet'

    Returns:
        summary dict
    """
    # 提取自适应参数
    pp = chain_data.get('pipeline_params', {})
    chain_dist_thresh = pp.get('chain_dist_thresh', 3.0)
    k_factor = pp.get('k_factor', DISP_K_FACTOR)
    K_cam = pp.get('K', K_ORIG)  # 自适应相机内参

    print(f"\n{'='*60}")
    print(f"  [运动轨迹] {seq_name} ({mode})")
    print(f"{'='*60}")

    all_k0 = chain_data['all_k0']
    all_k1 = chain_data['all_k1']
    all_weights = chain_data['all_weights']
    depth_maps = chain_data['depth_maps']
    abs_poses = chain_data['abs_poses']

    # ── 深度尺度恢复 ──
    # 轨迹估计阶段用 ×global_scale 的深度保证相机轨迹尺度与 GT 一致；
    # 组织运动阶段在反投影前 ÷global_scale 恢复原始尺度，保证组织运动位移量级正确。
    # 注: depth_source=='gt' 时深度未乘 global_scale, 无需恢复。
    global_scale = pp.get('global_depth_scale', 1.0)
    depth_src = chain_data.get('depth_source', 'pred')
    if depth_src != 'gt' and global_scale != 1.0:
        for k in depth_maps:
            depth_maps[k] = depth_maps[k] / global_scale
        print(f"  深度尺度恢复: ÷global_scale={global_scale:.4f} (组织运动用原始深度尺度)")

    print(f"  匹配对: {len(all_k0)}, 深度帧: {len(depth_maps)}, 位姿帧: {len(abs_poses)}")

    # Step A: 链式追踪
    print("\n  [A] 链式追踪...")
    tracks = chain_tracks(all_k0, all_k1, dist_thresh=chain_dist_thresh)
    n_long = sum(1 for t in tracks if len(t['frames']) >= 5)
    print(f"  总 tracks: {len(tracks)}, 长度≥5: {n_long}")
    # 限制 track 数量 (采样长 track 优先)
    MAX_TRACKS = 10000
    if len(tracks) > MAX_TRACKS:
        all_tracks = tracks
        tracks_long = [t for t in all_tracks if len(t['frames']) >= 5]
        tracks_short = [t for t in all_tracks if len(t['frames']) < 5]
        n_sample_long = min(len(tracks_long), MAX_TRACKS * 2 // 3)
        n_sample_short = MAX_TRACKS - n_sample_long
        import random
        random.seed(42)
        sampled = random.sample(tracks_long, n_sample_long) + \
                  random.sample(tracks_short, min(n_sample_short, len(tracks_short)))
        print(f"  采样至: {len(sampled)} tracks (长 track: {n_sample_long}, 短 track: {len(sampled)-n_sample_long})")
        tracks = sampled

    # Step B: 运动/静止分类
    print("\n  [B] 静止/运动分类...")
    static_tracks, moving_tracks = classify_tracks(
        tracks, all_weights, depth_maps, abs_poses,
        k_factor=k_factor)
    print(f"  静止: {len(static_tracks)}, 运动: {len(moving_tracks)}")

    # Step C: 统计
    print("\n  [C] 位移统计...")
    summary = compute_motion_summary(moving_tracks, static_tracks,
                                     depth_maps, abs_poses, K_cam)
    print(f"  静止点位移: mean={summary['static_disp_stats']['mean_mm']:.1f}mm "
          f"median={summary['static_disp_stats']['median_mm']:.1f}mm")
    print(f"  运动点位移: mean={summary['moving_disp_stats']['mean_mm']:.1f}mm "
          f"median={summary['moving_disp_stats']['median_mm']:.1f}mm "
          f"max={summary['moving_disp_stats']['max_mm']:.1f}mm")

    # Step D: 保存运动轨迹
    print("\n  [D] 保存结果...")
    save_motion_trajectories(moving_tracks, depth_maps, abs_poses, K_cam,
                             out_dir, f'{seq_name}_{mode}')

    # ── 保存所有 tracks 分类结果供独立评估脚本使用 ──
    tracks_export = []
    for track in tracks:
        t = track if not isinstance(track, tuple) else track[0]
        tracks_export.append({
            'id': t['id'],
            'frames': [[int(fi), float(u), float(v)] for fi, u, v in t['frames']],
        })
    moving_ids = [t[0]['id'] if isinstance(t, tuple) else t['id'] for t in moving_tracks]
    import json
    track_save_path = os.path.join(out_dir, f'{seq_name}_{mode}_tracks.json')
    with open(track_save_path, 'w') as f:
        json.dump({'tracks': tracks_export, 'moving_ids': moving_ids,
                   'n_total': len(tracks_export), 'n_moving': len(moving_ids),
                   'static_disp': summary['static_disp_stats'],
                   'moving_disp': summary['moving_disp_stats']}, f, indent=2)
    print(f"  分类结果已保存: {track_save_path}")

    # Step E: 可视化
    print("\n  [E] 可视化...")
    visualize_top_trajectories(moving_tracks, depth_maps, abs_poses, K_cam,
                                out_dir, f'{seq_name}_{mode}')

    # ── 保存 pipeline 参数供评估脚本用 ──
    summary['pipeline_params'] = {'K': K_cam.tolist(), 'k_factor': k_factor,
                                   'chain_dist_thresh': chain_dist_thresh}

    return summary


# ═══════════════════════════════════════════════════════════
# ATE 评估
# ═══════════════════════════════════════════════════════════


def evaluate_trajectory(vo_traj, gt_traj, seq_name, mode):
    """计算 ATE (Umeyama 对齐)."""
    if len(vo_traj) < 5:
        print(f"  [{seq_name}] [{mode}] 帧数不足 ({len(vo_traj)}), 跳过ATE")
        return None

    vo_mm = vo_traj
    gt_mm = gt_traj

    aligned, errors, R_align, scale = align_trajectory_umeyama(vo_mm, gt_mm)
    ate = compute_ate(aligned, gt_mm)

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  [{mode}] {seq_name}                  ║")
    print(f"  ╠══════════════════════════════════════╣")
    print(f"  ║  ATE RMSE:    {ate['rmse_mm']:>10.2f} mm       ║")
    print(f"  ║  ATE Mean:    {ate['mean_mm']:>10.2f} mm       ║")
    print(f"  ║  ATE Median:  {ate['median_mm']:>10.2f} mm       ║")
    print(f"  ║  ATE Std:     {ate['std_mm']:>10.2f} mm       ║")
    print(f"  ║  Scale:       {scale:>10.4f}         ║")
    print(f"  ║  Frames:      {len(vo_traj):>10d}           ║")
    print(f"  ╚══════════════════════════════════════╝")

    ate['scale'] = float(scale)
    return ate


def main():
    parser = argparse.ArgumentParser(description='V6 + DyEndoVO 集成测试')
    parser.add_argument('--seq', type=str, default=DEFAULT_SEQ,
                        help=f'序列名 (默认: {DEFAULT_SEQ})')
    parser.add_argument('--data_root', type=str, default=DATA_ROOT,
                        help=f'数据集根目录 (默认: {DATA_ROOT})')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='0 配置模式: 直接指定 RGB 帧文件夹 (覆盖 --seq)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='0 配置模式: 输出目录 (默认: input_dir)')
    parser.add_argument('--mode', type=str, default='motionnet',
                        choices=['baseline', 'motionnet', 'both'],
                        help='PnP模式: baseline(硬阈值), motionnet(MotionNet权重), both(两者)')
    parser.add_argument('--max_frames', type=int, default=MAX_FRAMES,
                        help='最大帧对数')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='monodepth2 模型路径')
    parser.add_argument('--depth_source', type=str, default='pred',
                        choices=['pred', 'gt'],
                        help='深度来源: pred(预测深度) 或 gt(GT深度)')
    parser.add_argument('--loftr_source', type=str, default='cache',
                        choices=['online', 'cache'],
                        help='LoFTR来源: online(在线匹配) 或 cache(预计算缓存)')
    parser.add_argument('--motionnet_ckpt', type=str, default=MOTIONNET_CKPT,
                        help='MotionNet checkpoint 路径')
    parser.add_argument('--out_dir', type=str, default='./motion_output',
                        help='运动轨迹输出目录 (默认: ./motion_output)')
    parser.add_argument('--no_gt', action='store_true',
                        help='禁用 GT 校准 (等效 --zero_gt_mode raw)')
    parser.add_argument('--zero_gt_mode', type=str, default=None,
                        choices=['raw', 'depth_prior', 'pnp_scale'],
                        help='零配置模式: raw(默认scale=1), depth_prior(深度先验), pnp_scale(PnP位移反推)')
    parser.add_argument('--target_depth_mm', type=float, default=50.0,
                        help='depth_prior 模式的目标深度中位数 (mm)')
    parser.add_argument('--target_step_mm', type=float, default=0.5,
                        help='pnp_scale 模式的目标帧间位移中位数 (mm)')
    args = parser.parse_args()

    # ── 固定随机种子, 确保管线可复现 ──
    # 核心: cv2.setRNGSeed 消除 solvePnPRansac 的随机性
    import random
    random.seed(42)
    np.random.seed(42)
    cv2.setRNGSeed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_zero_config = args.input_dir is not None

    if is_zero_config:
        # ═══════════════════════════════════════════
        # 0 配置模式: 陌生 RGB 文件夹 → 位姿 + 深度
        # ═══════════════════════════════════════════
        print(f"设备: {device}")
        print(f"模式: {args.mode} (0 配置)")
        print(f"输入: {args.input_dir}")
        print(f"深度模型: {args.model}")

        # 扫描帧
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        frames = sorted([f for f in os.listdir(args.input_dir)
                         if f.lower().endswith(exts)])
        if not frames:
            raise FileNotFoundError(f'未找到图像文件: {args.input_dir}')
        print(f"  找到 {len(frames)} 帧")

        # 自动估算 K
        first_img = cv2.imread(os.path.join(args.input_dir, frames[0]))
        if first_img is None:
            raise FileNotFoundError(f"无法读取图像: {args.input_dir}/{frames[0]}")
        H, W = first_img.shape[:2]
        focal = max(W, H)
        K_auto = np.array([
            [focal, 0, W / 2.0],
            [0, focal, H / 2.0],
            [0, 0, 1]
        ], dtype=np.float64)
        print(f"  自动估算 K: fx={focal:.1f}, fy={focal:.1f}, cx={W/2:.1f}, cy={H/2:.1f} "
              f"(图像 {W}×{H})")

        # 构建 pipeline_params (0配置模式: 无GT, 使用默认参数)
        out_dir = args.output_dir if args.output_dir else args.input_dir
        pipeline_params = {
            'global_depth_scale': 1.0,
            'max_rotation_deg': 30.0,
            'max_translation': 100.0,
            'scale_clip_lo': 0.2,
            'scale_clip_hi': 5.0,
            'chain_dist_thresh': 3.0,
            'k_factor': 0.5,
            'img_w': W, 'img_h': H,
            'fx': focal, 'fy': focal, 'cx': W / 2.0, 'cy': H / 2.0,
            'img_w_orig': W, 'img_h_orig': H,
            'K': K_auto,
            'depth_cache': {},
        }
    else:
        print(f"设备: {device}")
        print(f"序列: {args.seq}")
        print(f"模式: {args.mode}")
        print(f"深度来源: {args.depth_source}")
        print(f"LoFTR来源: {args.loftr_source}")
        pipeline_params = None
        out_dir = os.path.abspath(args.out_dir)

    if args.depth_source == 'pred':
        print(f"深度模型: {args.model}")
    if not is_zero_config:
        print(f"MotionNet: {args.motionnet_ckpt}")

    if args.depth_source == 'gt':
        encoder, depth_decoder, motion_encoder = None, None, None
    else:
        # ── 加载 monodepth2 ──
        print("\n[1/3] 加载深度预测模型...")
        model_manager = ModelManager(model_path=args.model)
        encoder, depth_decoder, _, motion_encoder = model_manager.load_model()

    # ── 加载 MotionNet ──
    motion_net = None
    if args.mode in ('motionnet', 'both'):
        print("[2/3] 加载 MotionNet...")
        ckpt = torch.load(args.motionnet_ckpt, map_location=device, weights_only=False)
        motion_net = MotionNet(pretrained=False, img_h=MOTION_H, img_w=MOTION_W).to(device)
        motion_net.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        print(f"  MotionNet 加载成功 (epoch={ckpt.get('epoch', '?')})")
    else:
        print("[2/3] 跳过 MotionNet (baseline 模式)")

    # ── 运行 VO ──
    print(f"\n[3/3] 运行 VO 管线...")

    results = {}
    motion_summaries = {}
    seq_label = os.path.basename(os.path.abspath(args.input_dir)) if is_zero_config else args.seq

    if args.mode in ('baseline', 'both'):
        print("\n--- Baseline: 标准 EPnP ---")
        vo_baseline, gt_baseline, stats_b, chain_b = run_vo_sequence(
            args.seq, encoder, depth_decoder, device,
            motion_net=None, mode='baseline', max_frames=args.max_frames,
            motion_encoder=motion_encoder, depth_source=args.depth_source,
            loftr_source=args.loftr_source, data_root=args.data_root,
            input_dir=args.input_dir if is_zero_config else None,
            output_dir=out_dir if is_zero_config else None,
            pipeline_params=pipeline_params, no_gt=args.no_gt,
            zero_gt_mode=args.zero_gt_mode)
        print(f"  成功: {stats_b['n_success']}/{stats_b['n_total']}")
        if not is_zero_config and len(gt_baseline) > 0:
            ate_b = evaluate_trajectory(vo_baseline, gt_baseline, seq_label, 'baseline')
            if ate_b:
                results['baseline'] = {'ate': ate_b, 'stats': stats_b}
        else:
            results['baseline'] = {'ate': None, 'stats': stats_b}
        # ── 运动轨迹 ──
        motion_summaries['baseline'] = run_motion_pipeline(
            chain_b, seq_label, out_dir, 'baseline')

    if args.mode in ('motionnet', 'both'):
        print("\n--- MotionNet: 加权 EPnP ---")
        vo_mn, gt_mn, stats_m, chain_m = run_vo_sequence(
            args.seq, encoder, depth_decoder, device,
            motion_net=motion_net, mode='motionnet', max_frames=args.max_frames,
            motion_encoder=motion_encoder, depth_source=args.depth_source,
            loftr_source=args.loftr_source, data_root=args.data_root,
            input_dir=args.input_dir if is_zero_config else None,
            output_dir=out_dir if is_zero_config else None,
            pipeline_params=pipeline_params, no_gt=args.no_gt,
            zero_gt_mode=args.zero_gt_mode)
        print(f"  成功: {stats_m['n_success']}/{stats_m['n_total']}")
        if not is_zero_config and len(gt_mn) > 0:
            ate_m = evaluate_trajectory(vo_mn, gt_mn, seq_label, 'motionnet')
            if ate_m:
                results['motionnet'] = {'ate': ate_m, 'stats': stats_m}
        else:
            results['motionnet'] = {'ate': None, 'stats': stats_m}
        # ── 运动轨迹 ──
        motion_summaries['motionnet'] = run_motion_pipeline(
            chain_m, seq_label, out_dir, 'motionnet')

    # ── 汇总表 ──
    title = "【 0 配置管线输出汇总 】" if is_zero_config else "【 EndoSLAM 黑盒管线评估汇总 】"
    print("\n" + "═" * 65)
    print(f"  {title}")
    print("═" * 65)

    primary_mode = args.mode if args.mode != 'both' else 'baseline'
    ms = motion_summaries.get(primary_mode, {})
    r_entry = results.get(primary_mode, {})
    ate = r_entry.get('ate', {}) if r_entry else {}
    stats_vo = r_entry.get('stats', {}) if r_entry else {}

    depth_ratio = stats_vo.get('depth_ratio')
    ate_rmse = ate.get('rmse_mm') if ate else None
    moving_mean = ms.get('moving_disp_stats', {}).get('mean_mm')
    static_mean = ms.get('static_disp_stats', {}).get('mean_mm')
    sep_ratio = moving_mean / static_mean if static_mean else None
    vo_success = stats_vo.get('n_success', 0)
    vo_total = stats_vo.get('n_total', 0)

    print(f"  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  输出                          │  指标              │  值        │")
    print(f"  ├─────────────────────────────────────────────────────────────────┤")
    print(f"  │  VO 成功率                     │  success           │  {vo_success}/{vo_total}     │")
    if is_zero_config:
        print(f"  │  深度预测                      │  ratio pred/GT     │  N/A (0配置)│")
        print(f"  │  相机绝对位姿                  │  ATE RMSE          │  N/A (0配置)│")
    else:
        print(f"  │  深度预测                      │  ratio pred/GT     │  {depth_ratio:>6.3f}    │" if depth_ratio else
              f"  │  深度预测                      │  ratio pred/GT     │  N/A       │")
        print(f"  │  相机绝对位姿 (ATE)            │  RMSE              │  {ate_rmse:>6.2f} mm │" if ate_rmse else
              f"  │  相机绝对位姿 (ATE)            │  RMSE              │  N/A       │")
    print(f"  │  组织运动位移                  │  运动/静止分离度   │  {sep_ratio:>5.1f}×    │" if sep_ratio else
          f"  │  组织运动位移                  │  运动/静止分离度   │  N/A       │")
    if not is_zero_config:
        print(f"  │  (详细GT评估请运行 evaluate_pipeline.py)                       │")
    print(f"  │  输出目录: {os.path.abspath(out_dir)}")
    print(f"  └─────────────────────────────────────────────────────────────────┘")


if __name__ == '__main__':
    main()
