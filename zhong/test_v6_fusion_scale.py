#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方案1: VO 管线级尺度融合验证

核心思路: 用 mdp_v5 预跑得到的全局深度尺度因子 (global_depth_scale)
标定 Ours 的深度/轨迹, 保留 Ours 的相对几何精度 (ATE/RPE) 同时修正绝对尺度 (Scale Error)。

对比 4 组:
  1. Ours baseline        (Ours 自己的 pnp_scale s_ours)
  2. mdp_v5 baseline      (mdp_v5 自己的 pnp_scale s_mdp)
  3. Ours + mdp_v5 尺度   (融合, 核心验证: Ours 深度 × s_mdp)
  4. mdp_v5 + Ours 尺度   (反向融合, 验证尺度传递的对称性)

用法:
  python zhong\\test_v6_fusion_scale.py
"""

import os, sys, json, time
import numpy as np
import torch
import random

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

import test_v6_dyendovo as tv6
from test_v6_multi_model_enhanced import (
    load_model, compute_metrics, MODEL_CONFIGS,
    SEQ_NAME, SEQ_DIR, DATA_ROOT, OUT_DIR)
from dyendovo_dataset import load_gt_poses as dyendo_load_gt_poses


def _recompute_scale_clip(scale):
    """围绕新 scale 重算帧间校正裁剪范围 (与 test_v6_dyendovo.py 逻辑一致)."""
    lo = min(max(0.2, scale * 0.4), 0.9)
    hi = max(min(5.0, max(2.0, scale * 2.5)), 1.1)
    return lo, hi


def estimate_params(seq_dir, encoder, depth_decoder, motion_encoder, device,
                    global_scale_override=None):
    """预跑 estimate_pipeline_params, 可选覆盖 global_depth_scale."""
    params = tv6.estimate_pipeline_params(
        seq_dir, encoder, depth_decoder, motion_encoder, device,
        depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')

    if global_scale_override is not None:
        s = float(global_scale_override)
        params['global_depth_scale'] = s
        params['scale_clip_lo'], params['scale_clip_hi'] = _recompute_scale_clip(s)

    return params


def run_vo(encoder, depth_decoder, motion_encoder, device, pipeline_params, tag):
    """用给定 pipeline_params 跑 VO."""
    gs = pipeline_params['global_depth_scale']
    print(f'  [{tag}] 运行 VO (global_depth_scale={gs:.4f})...')
    t0 = time.time()
    vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
        SEQ_NAME, encoder, depth_decoder, device,
        motion_net=None, mode='baseline', max_frames=None,
        motion_encoder=motion_encoder, depth_source='pred',
        loftr_source='cache', data_root=DATA_ROOT,
        pipeline_params=pipeline_params, no_gt=False, zero_gt_mode='pnp_scale')
    print(f'  [{tag}] 完成: {time.time()-t0:.0f}s')
    return vo_traj, gt_traj, stats, chain_data


def get_cfg(name):
    for c in MODEL_CONFIGS:
        if c['name'] == name:
            return c
    raise ValueError(f'未找到模型配置: {name}')


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    print(f'序列: {SEQ_NAME}')
    print(f'方案1: 用 mdp_v5 的全局尺度标定 Ours 轨迹\n')

    gt_poses_std = dyendo_load_gt_poses(SEQ_DIR)
    print(f'GT: {len(gt_poses_std)} 帧\n')

    ours_cfg = get_cfg('Ours')
    mdp_cfg = get_cfg('mdp_v5')

    # ── 1. 加载两个模型 ──
    print('=' * 65)
    print('[1/3] 加载模型')
    print('=' * 65)
    enc_mdp, dec_mdp, mot_mdp = load_model(mdp_cfg, device)
    enc_ours, dec_ours, mot_ours = load_model(ours_cfg, device)

    # ── 2. 各自预跑, 获取全局尺度因子 ──
    print('\n' + '=' * 65)
    print('[2/3] 预跑: 获取全局尺度因子')
    print('=' * 65)
    print('  [mdp_v5] 预跑 estimate_pipeline_params...')
    t0 = time.time()
    params_mdp = estimate_params(SEQ_DIR, enc_mdp, dec_mdp, mot_mdp, device)
    s_mdp = float(params_mdp['global_depth_scale'])
    print(f'  [mdp_v5] global_depth_scale = {s_mdp:.4f}  ({time.time()-t0:.0f}s)')

    print('  [Ours] 预跑 estimate_pipeline_params...')
    t0 = time.time()
    params_ours = estimate_params(SEQ_DIR, enc_ours, dec_ours, mot_ours, device)
    s_ours = float(params_ours['global_depth_scale'])
    print(f'  [Ours] global_depth_scale = {s_ours:.4f}  ({time.time()-t0:.0f}s)')
    print(f'  尺度比 s_mdp / s_ours = {s_mdp/s_ours:.4f}')

    # ── 3. 四组 VO ──
    results = {}

    print('\n' + '=' * 65)
    print('[3/3] 四组 VO 对比')
    print('=' * 65)

    print('\n  >>> 组1/4: Ours baseline (s_ours) <<<')
    vo_traj, gt_traj, stats, chain_data = run_vo(
        enc_ours, dec_ours, mot_ours, device, params_ours, 'Ours-baseline')
    m = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
    m['n_success'] = stats['n_success']; m['n_pairs'] = stats['n_total']
    results['Ours baseline'] = m

    print('\n  >>> 组2/4: mdp_v5 baseline (s_mdp) <<<')
    vo_traj, gt_traj, stats, chain_data = run_vo(
        enc_mdp, dec_mdp, mot_mdp, device, params_mdp, 'mdp_v5-baseline')
    m = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
    m['n_success'] = stats['n_success']; m['n_pairs'] = stats['n_total']
    results['mdp_v5 baseline'] = m

    print('\n  >>> 组3/4: Ours + mdp_v5 尺度 (融合, 核心验证) <<<')
    params_ours_fused = dict(params_ours)
    params_ours_fused['global_depth_scale'] = s_mdp
    params_ours_fused['scale_clip_lo'], params_ours_fused['scale_clip_hi'] = _recompute_scale_clip(s_mdp)
    vo_traj, gt_traj, stats, chain_data = run_vo(
        enc_ours, dec_ours, mot_ours, device, params_ours_fused, 'Ours+s_mdp')
    m = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
    m['n_success'] = stats['n_success']; m['n_pairs'] = stats['n_total']
    results['Ours + mdp_v5 scale'] = m

    print('\n  >>> 组4/4: mdp_v5 + Ours 尺度 (反向验证) <<<')
    params_mdp_fused = dict(params_mdp)
    params_mdp_fused['global_depth_scale'] = s_ours
    params_mdp_fused['scale_clip_lo'], params_mdp_fused['scale_clip_hi'] = _recompute_scale_clip(s_ours)
    vo_traj, gt_traj, stats, chain_data = run_vo(
        enc_mdp, dec_mdp, mot_mdp, device, params_mdp_fused, 'mdp_v5+s_ours')
    m = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
    m['n_success'] = stats['n_success']; m['n_pairs'] = stats['n_total']
    results['mdp_v5 + Ours scale'] = m

    # ── 4. 汇总 ──
    print('\n' + '=' * 100)
    print('  方案1 融合结果汇总')
    print('=' * 100)
    print(f"  {'组':<22} {'ATE':>8} {'RPE-T':>8} {'RPE-R':>8} "
          f"{'ScaleErr':>10} {'umeyama':>10} {'Success':>9}")
    print('  ' + '-' * 94)
    for name, mm in results.items():
        print(f"  {name:<22} {mm['ate_mean']:>8.2f} {mm['rpe_t_mean']:>8.3f} "
              f"{mm['rpe_r_mean']:>8.2f} {mm['scale_error_pct']:>9.2f}% "
              f"{mm['umeyama_scale']:>10.4f} {mm['n_success']:>8}/{mm['n_pairs']}")

    b = results['Ours baseline']
    f = results['Ours + mdp_v5 scale']
    print('\n' + '=' * 100)
    print('  核心对比: Ours baseline → Ours + mdp_v5 尺度')
    print('=' * 100)
    print(f"  ATE:       {b['ate_mean']:.2f} → {f['ate_mean']:.2f} mm   "
          f"({'改善' if f['ate_mean'] < b['ate_mean'] else '恶化'} {abs(b['ate_mean']-f['ate_mean']):.2f})")
    print(f"  RPE-T:     {b['rpe_t_mean']:.3f} → {f['rpe_t_mean']:.3f} mm")
    print(f"  RPE-R:     {b['rpe_r_mean']:.2f} → {f['rpe_r_mean']:.2f} °")
    print(f"  Scale Err: {b['scale_error_pct']:.2f}% → {f['scale_error_pct']:.2f}%   "
          f"({'改善' if f['scale_error_pct'] < b['scale_error_pct'] else '恶化'} "
          f"{abs(b['scale_error_pct']-f['scale_error_pct']):.2f}%)")

    out_path = os.path.join(OUT_DIR, 'fusion_scale_results.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
