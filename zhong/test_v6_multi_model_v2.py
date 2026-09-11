#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
5 模型统一 VO 管线测试 —— 复用 test_v6_dyendovo.py 的 run_vo_sequence 管线。

用法:
  python zhong\test_v6_multi_model_v2.py

管线: 与 test_v6_dyendovo.py baseline + pnp_scale 零配置完全一致
  - 5 模型: Monodepth2 / ManyDepth / Lite-Mono / Ours(老) / mdp_v5(新)
  - LoFTR 缓存匹配 (cache) —— 所有模型共用同一组匹配，完全一致
  - 零配置 pnp_scale 全局尺度校准
  - 自适应 3D 位移过滤 + 帧间深度尺度一致性校正
  - 统一 baseline EPnP 模式
"""

import os, sys, json, time
import numpy as np
import torch
import random

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# ── 导入 test_v6_dyendovo 管线函数 ──
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

OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'vo_depth_compare')
os.makedirs(OUT_DIR, exist_ok=True)

# Lite-Mono 模型根目录
_LITEMONO_DIR = r'E:\data1\Lite-Mono-main'

# ── 5 模型配置 ──
MODEL_CONFIGS = [
    {
        'name': 'Monodepth2',
        'path': r'C:\Users\Administrator\tmp\c3vd_md2_multi\models\weights_19',
        'type': 'md2',
    },
    {
        'name': 'ManyDepth',
        'path': r'C:\Users\Administrator\tmp\c3vd_manydepth_multi\models\weights_19',
        'type': 'manydepth',
    },
    {
        'name': 'Lite-Mono',
        'path': r'C:\Users\Administrator\tmp\c3vd_litemono_multi\models\weights_19',
        'type': 'litemono',
    },
    {
        'name': 'Ours',
        'path': r'e:\data1\monodepth2\models\depth',
        'type': 'md2',  # 同 Monodepth2 架构, 但启用 motion_encoder
    },
    {
        'name': 'mdp_v5',
        'path': r'C:\Users\Administrator\tmp\mdp_v5\models\weights_19',
        'type': 'ours_enhanced',  # Lite-Mono 64-bin decoder + MotionEncoder + depth consistency (mdp_v5)
    },
]


# ═══════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════

def load_manydepth_model(model_path, device):
    """加载 ManyDepth 单帧推理模型."""
    from networks import ResnetEncoder, DepthDecoder
    encoder = ResnetEncoder(18, False)
    state_dict = torch.load(os.path.join(model_path, "mono_encoder.pth"),
                           map_location=device)
    for key in ['height', 'width', 'use_stereo']:
        if key in state_dict:
            del state_dict[key]
    encoder.load_state_dict(state_dict)
    encoder.eval().to(device)

    depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(4), num_bins=0)
    depth_state = torch.load(os.path.join(model_path, "mono_depth.pth"),
                             map_location=device)
    depth_decoder.load_state_dict(depth_state, strict=False)
    depth_decoder.eval().to(device)
    return encoder, depth_decoder


def load_litemono_model(model_path, device):
    """加载 Lite-Mono 模型."""
    import importlib.util, importlib.machinery

    _orig_path = sys.path.copy()
    _cached_layers = sys.modules.get('layers')

    litemono_networks = os.path.join(_LITEMONO_DIR, 'networks')
    sys.path.insert(0, _LITEMONO_DIR)
    try:
        loader_layers = importlib.machinery.SourceFileLoader(
            '_litemono_layers', os.path.join(_LITEMONO_DIR, 'layers.py'))
        spec_layers = importlib.util.spec_from_loader('_litemono_layers', loader_layers)
        mod_layers = importlib.util.module_from_spec(spec_layers)
        sys.modules['_litemono_layers'] = mod_layers
        sys.modules['layers'] = mod_layers
        spec_layers.loader.exec_module(mod_layers)

        loader_enc = importlib.machinery.SourceFileLoader(
            '_litemono_enc', os.path.join(litemono_networks, 'depth_encoder.py'))
        spec_enc = importlib.util.spec_from_loader('_litemono_enc', loader_enc)
        mod_enc = importlib.util.module_from_spec(spec_enc)
        sys.modules['_litemono_enc'] = mod_enc
        spec_enc.loader.exec_module(mod_enc)
        LiteMono = mod_enc.LiteMono

        loader_dec = importlib.machinery.SourceFileLoader(
            '_litemono_dec', os.path.join(litemono_networks, 'depth_decoder.py'))
        spec_dec = importlib.util.spec_from_loader('_litemono_dec', loader_dec)
        mod_dec = importlib.util.module_from_spec(spec_dec)
        sys.modules['_litemono_dec'] = mod_dec
        spec_dec.loader.exec_module(mod_dec)
        LitemonoDepthDecoder = mod_dec.DepthDecoder
    finally:
        sys.path = _orig_path
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
    depth_state = torch.load(os.path.join(model_path, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(depth_state)
    depth_decoder.eval().to(device)
    return encoder, depth_decoder


def load_ours_enhanced_model(model_path, device):
    """加载 Ours+Enhanced 模型 (ResNet18 encoder + LiteMonoDepthDecoder + MotionEncoder)."""
    from networks import ResnetEncoder, MotionEncoder

    # ── Encoder (标准 ResNet18) ──
    encoder = ResnetEncoder(18, False)
    state_dict = torch.load(os.path.join(model_path, "encoder.pth"),
                           map_location=device)
    for key in ['height', 'width', 'use_stereo']:
        if key in state_dict:
            del state_dict[key]
    encoder.load_state_dict(state_dict)
    encoder.eval().to(device)

    # ── Depth Decoder (LiteMonoDepthDecoder, 64-bin 分类头) ──
    depth_decoder = LiteMonoDepthDecoder(
        num_ch_enc=np.array([64, 64, 128, 256, 512]),
        scales=range(3), num_output_channels=1, use_skips=True,
        num_bins=64)
    depth_state = torch.load(os.path.join(model_path, "depth.pth"),
                             map_location=device)
    depth_decoder.load_state_dict(depth_state, strict=False)
    depth_decoder.eval().to(device)

    # ── Motion Encoder ──
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
    """统一模型加载接口.
    
    Returns: (encoder, depth_decoder, motion_encoder_or_None)
    """
    model_type = cfg['type']
    model_path = cfg['path']
    name = cfg['name']

    if model_type == 'manydepth':
        print(f'  [{name}] 加载 ManyDepth 模型…')
        enc, dec = load_manydepth_model(model_path, device)
        return enc, dec, None
    elif model_type == 'litemono':
        print(f'  [{name}] 加载 Lite-Mono 模型…')
        enc, dec = load_litemono_model(model_path, device)
        return enc, dec, None
    elif model_type == 'ours_enhanced':
        print(f'  [{name}] 加载 Ours+Enhanced 模型 (Lite-Mono decoder + temporal consistency)…')
        enc, dec, mot = load_ours_enhanced_model(model_path, device)
        return enc, dec, mot
    else:
        # md2 类型 (Monodepth2 / Ours)
        mgr = ModelManager(model_path=model_path, device=device)
        encoder, depth_decoder, _, motion_encoder = mgr.load_model()
        has_motion = motion_encoder is not None
        print(f'  [{name}] 加载 ModelManager 模型 (motion_encoder={"有" if has_motion else "无"})')
        return encoder, depth_decoder, motion_encoder


# ═══════════════════════════════════════════════
# RPE 计算 (对齐 compare_depth_vo.py)
# ═══════════════════════════════════════════════

def rotation_angle_deg(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)) * 180 / np.pi)


def compute_rpe(abs_poses, gt_poses_std):
    """帧间相对位姿误差 RPE-T / RPE-R.
    
    abs_poses:  list of (4,4)  cam→world 位姿
    gt_poses_std: list of (4,4)  GT [R^T, C; 0, 1] 格式
    """
    n = min(len(abs_poses), len(gt_poses_std))
    rpe_trans, rpe_rot = [], []

    for i in range(n - 1):
        # GT: [R_i^T, C_i; 0, 1] → T_cam_world = gt.T
        T_cw_gt_i = np.linalg.inv(gt_poses_std[i].T)
        T_cw_gt_i1 = np.linalg.inv(gt_poses_std[i + 1].T)
        dP_gt = np.linalg.inv(T_cw_gt_i) @ T_cw_gt_i1

        # VO: cam→world → world→cam
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
        'rpe_t_mean': float(np.mean(rpe_trans)),
        'rpe_t_std': float(np.std(rpe_trans)),
        'rpe_t_rmse': float(np.sqrt(np.mean(rpe_trans ** 2))),
        'rpe_r_mean': float(np.mean(rpe_rot)),
        'rpe_r_std': float(np.std(rpe_rot)),
        'rpe_r_rmse': float(np.sqrt(np.mean(rpe_rot ** 2))),
        'n_rpe': len(rpe_trans),
    }


def compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std):
    """计算 ATE + RPE + Scale Error.
    
    Returns: dict of metrics
    """
    # ── ATE (Umeyama 对齐后) ──
    ate_result = tv6.evaluate_trajectory(vo_traj, gt_traj, SEQ_NAME, 'baseline')

    # ── RPE ──
    abs_poses = chain_data['abs_poses']  # list of (4,4) T_world_cam
    rpe = compute_rpe(abs_poses, gt_poses_std)

    # ── Scale Error ──
    scale_error_pct = abs(ate_result.get('scale', 1.0) - 1.0) * 100.0

    return {
        'ate_mean': ate_result.get('mean_mm', float('nan')),
        'ate_std': ate_result.get('std_mm', float('nan')),
        'ate_rmse': ate_result.get('rmse_mm', float('nan')),
        'ate_median': ate_result.get('median_mm', float('nan')),
        'rpe_t_mean': rpe['rpe_t_mean'],
        'rpe_t_std': rpe['rpe_t_std'],
        'rpe_t_rmse': rpe['rpe_t_rmse'],
        'rpe_r_mean': rpe['rpe_r_mean'],
        'rpe_r_std': rpe['rpe_r_std'],
        'rpe_r_rmse': rpe['rpe_r_rmse'],
        'scale_error_pct': scale_error_pct,
        'umeyama_scale': ate_result.get('scale', float('nan')),
        'n_rpe': rpe['n_rpe'],
    }


# ═══════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════

def main():
    # ── 固定随机种子 ──
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"序列: {SEQ_NAME}")
    print(f"管线: baseline + pnp_scale (零配置)")
    print(f"LoFTR: cache (所有模型共用)\n")

    # ── 加载 GT ──
    gt_poses_std = dyendo_load_gt_poses(SEQ_DIR)
    print(f"GT: {len(gt_poses_std)} 帧\n")

    # ── 逐模型测试 ──
    all_results = {}

    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        print(f"\n{'=' * 65}")
        print(f"  [{name}]")
        print(f"  模型路径: {cfg['path']}")
        print(f"{'=' * 65}")

        if not os.path.isdir(cfg['path']):
            print(f"  ⚠ 模型目录不存在, 跳过\n")
            continue

        # ── 加载模型 ──
        encoder, depth_decoder, motion_encoder = load_model(cfg, device)

        # ── 运行 VO (baseline + pnp_scale) ──
        t0 = time.time()
        vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
            SEQ_NAME, encoder, depth_decoder, device,
            motion_net=None, mode='baseline', max_frames=None,
            motion_encoder=motion_encoder, depth_source='pred',
            loftr_source='cache', data_root=DATA_ROOT,
            pipeline_params=None, no_gt=False, zero_gt_mode='pnp_scale')
        elapsed = time.time() - t0

        n_success = stats['n_success']
        n_total = stats['n_total']
        print(f"\n  VO 完成: {n_success}/{n_total} 成功, 耗时 {elapsed:.0f}s")

        # ── 计算指标 ──
        metrics = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
        metrics['n_frames'] = len(vo_traj)
        metrics['n_success'] = n_success
        metrics['n_pairs'] = n_total
        metrics['elapsed_s'] = round(elapsed, 1)

        all_results[name] = metrics

        # 打印单模型结果
        print(f"  ATE:     {metrics['ate_mean']:.2f} ± {metrics['ate_std']:.2f} mm")
        print(f"  RPE-T:   {metrics['rpe_t_mean']:.4f} ± {metrics['rpe_t_std']:.4f} mm")
        print(f"  RPE-R:   {metrics['rpe_r_mean']:.2f} ± {metrics['rpe_r_std']:.2f} °")
        print(f"  Scale Error: {metrics['scale_error_pct']:.2f}%")
        print(f"  Success: {n_success}/{n_total}")

        # ── 释放 GPU 内存 ──
        del encoder, depth_decoder, motion_encoder, vo_traj, gt_traj, chain_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 保存结果 ──
    result_path = os.path.join(OUT_DIR, 'multi_model_results_v2.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存: {result_path}")

    # ── 快速汇总表 ──
    print(f"\n{'=' * 90}")
    print(f"  5 模型对比汇总 (test_v6 pnp_scale 零配置)")
    print(f"{'=' * 90}")
    print(f"  {'Model':<14} {'ATE↓':>8}    {'RPE-T↓':>8}    {'RPE-R↓':>8}    "
          f"{'Scale Err↓':>9}    {'Success':>9}")
    print(f"  {'-' * 80}")
    for name, m in all_results.items():
        print(f"  {name:<14} {m['ate_mean']:>5.2f}±{m['ate_std']:.2f}  "
              f"{m['rpe_t_mean']:>5.3f}±{m['rpe_t_std']:.3f}  "
              f"{m['rpe_r_mean']:>5.2f}±{m['rpe_r_std']:.2f}  "
              f"{m['scale_error_pct']:>8.2f}%  "
              f"{m['n_success']:>4d}/{m['n_pairs']:<4d}")
    print(f"{'=' * 90}")
    print(f"  完整结果: {result_path}")


if __name__ == '__main__':
    main()
