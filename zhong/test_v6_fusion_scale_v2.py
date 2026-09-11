#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方案2: 深度图级尺度融合验证 (隔离 scale_clip 变量)

方案1 失败的根因 (已定位):
  run_vo_sequence 第755-782行的"深度帧间尺度一致性校正" (scale_corr) 会用
  scale_clip_lo/hi 对逐帧深度做非线性缩放。scale_clip 由 global_depth_scale 自动
  计算, 方案1 替换 global_depth_scale 时连带重算了 scale_clip, 破坏了帧间校正行为,
  导致轨迹尺度非线性失控 (Ours+s_mdp 的 Scale Err 从 25.5% 爆炸到 58.9%)。

方案2 核心:
  只替换 global_depth_scale, 但【保持 scale_clip 不变】(沿用各自 baseline 的值),
  从而隔离"scale_clip 变化"这个变量, 验证它是否就是方案1失败的根因。

对比 4 组:
  1. Ours baseline    (s_ours, clip_ours)
  2. mdp_v5 baseline  (s_mdp,  clip_mdp)
  3. Ours  + s_mdp    (global_scale=s_mdp,  clip 保持 clip_ours)  ← 方案2核心
  4. mdp_v5 + s_ours  (global_scale=s_ours, clip 保持 clip_mdp)  ← 反向验证

用法:
  python zhong\\test_v6_fusion_scale_v2.py
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


def estimate_params(seq_dir, encoder, depth_decoder, motion_encoder, device):
    """预跑 estimate_pipeline_params (不覆盖任何参数)."""
    return tv6.estimate_pipeline_params(
        seq_dir, encoder, depth_decoder, motion_encoder, device,
        depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')


def run_vo(encoder, depth_decoder, motion_encoder, device, pipeline_params, tag):
    """用给定 pipeline_params 跑 VO."""
    gs = pipeline_params['global_depth_scale']
    lo = pipeline_params['scale_clip_lo']
    hi = pipeline_params['scale_clip_hi']
    print(f'  [{tag}] 运行 VO (global_scale={gs:.4f}, scale_clip=[{lo:.3f},{hi:.3f}])...')
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


def collect(vo_traj, gt_traj, stats, chain_data, gt_poses_std):
    m = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
    m['n_success'] = stats['n_success']
    m['n_pairs'] = stats['n_total']
    return m


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    print(f'序列: {SEQ_NAME}')
    print(f'方案2: 深度图级尺度融合 (只换 global_depth_scale, 保持 scale_clip 不变)\n')

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

    # ── 2. 各自预跑 ──
    print('\n' + '=' * 65)
    print('[2/3] 预跑: 获取尺度因子与 scale_clip')
    print('=' * 65)
    print('  [mdp_v5] 预跑 estimate_pipeline_params...')
    t0 = time.time()
    params_mdp = estimate_params(SEQ_DIR, enc_mdp, dec_mdp, mot_mdp, device)
    s_mdp = float(params_mdp['global_depth_scale'])
    clip_mdp = (params_mdp['scale_clip_lo'], params_mdp['scale_clip_hi'])
    print(f'  [mdp_v5] s_mdp={s_mdp:.4f}, scale_clip=[{clip_mdp[0]:.3f},{clip_mdp[1]:.3f}]  ({time.time()-t0:.0f}s)')

    print('  [Ours] 预跑 estimate_pipeline_params...')
    t0 = time.time()
    params_ours = estimate_params(SEQ_DIR, enc_ours, dec_ours, mot_ours, device)
    s_ours = float(params_ours['global_depth_scale'])
    clip_ours = (params_ours['scale_clip_lo'], params_ours['scale_clip_hi'])
    print(f'  [Ours] s_ours={s_ours:.4f}, scale_clip=[{clip_ours[0]:.3f},{clip_ours[1]:.3f}]  ({time.time()-t0:.0f}s)')
    print(f'  尺度比 s_mdp/s_ours = {s_mdp/s_ours:.4f}')

    # ── 3. 四组 VO ──
    results = {}
    clips = {}

    print('\n' + '=' * 65)
    print('[3/3] 四组 VO 对比')
    print('=' * 65)

    print('\n  >>> 组1/4: Ours baseline <<<')
    vo_traj, gt_traj, stats, chain_data = run_vo(
        enc_ours, dec_ours, mot_ours, device, params_ours, 'Ours-baseline')
    results['Ours baseline'] = collect(vo_traj, gt_traj, stats, chain_data, gt_poses_std)
    clips['Ours baseline'] = (s_ours, clip_ours)

    print('\n  >>> 组2/4: mdp_v5 baseline <<<')
    vo_traj, gt_traj, stats, chain_data = run_vo(
        enc_mdp, dec_mdp, mot_mdp, device, params_mdp, 'mdp_v5-baseline')
    results['mdp_v5 baseline'] = collect(vo_traj, gt_traj, stats, chain_data, gt_poses_std)
    clips['mdp_v5 baseline'] = (s_mdp, clip_mdp)

    print('\n  >>> 组3/4: Ours + s_mdp (方案2: clip 保持 clip_ours) <<<')
    params_v2_ours = dict(params_ours)
    params_v2_ours['global_depth_scale'] = s_mdp
    # 关键: scale_clip 保持 clip_ours 不变 (不重算)
    params_v2_ours['scale_clip_lo'], params_v2_ours['scale_clip_hi'] = clip_ours
    vo_traj, gt_traj, stats, chain_data = run_vo(
        enc_ours, dec_ours, mot_ours, device, params_v2_ours, 'Ours+s_mdp(clip_ours)')
    results['Ours + s_mdp (clip不变)'] = collect(vo_traj, gt_traj, stats, chain_data, gt_poses_std)
    clips['Ours + s_mdp (clip不变)'] = (s_mdp, clip_ours)

    print('\n  >>> 组4/4: mdp_v5 + s_ours (方案2: clip 保持 clip_mdp) <<<')
    params_v2_mdp = dict(params_mdp)
    params_v2_mdp['global_depth_scale'] = s_ours
    params_v2_mdp['scale_clip_lo'], params_v2_mdp['scale_clip_hi'] = clip_mdp
    vo_traj, gt_traj, stats, chain_data = run_vo(
        enc_mdp, dec_mdp, mot_mdp, device, params_v2_mdp, 'mdp_v5+s_ours(clip_mdp)')
    results['mdp_v5 + s_ours (clip不变)'] = collect(vo_traj, gt_traj, stats, chain_data, gt_poses_std)
    clips['mdp_v5 + s_ours (clip不变)'] = (s_ours, clip_mdp)

    # ── 4. 汇总 ──
    print('\n' + '=' * 108)
    print('  方案2 融合结果汇总')
    print('=' * 108)
    print(f"  {'组':<24} {'ATE':>8} {'RPE-T':>8} {'RPE-R':>8} "
          f"{'ScaleErr':>10} {'umeyama':>10} {'Success':>9}")
    print('  ' + '-' * 102)
    for name, mm in results.items():
        print(f"  {name:<24} {mm['ate_mean']:>8.2f} {mm['rpe_t_mean']:>8.3f} "
              f"{mm['rpe_r_mean']:>8.2f} {mm['scale_error_pct']:>9.2f}% "
              f"{mm['umeyama_scale']:>10.4f} {mm['n_success']:>8}/{mm['n_pairs']}")

    # 核心对比: Ours baseline vs 方案2
    b = results['Ours baseline']
    f = results['Ours + s_mdp (clip不变)']
    print('\n' + '=' * 108)
    print('  核心对比: Ours baseline → Ours + s_mdp (方案2, clip 不变)')
    print('=' * 108)
    print(f"  ATE:       {b['ate_mean']:.2f} → {f['ate_mean']:.2f} mm   "
          f"({'改善' if f['ate_mean'] < b['ate_mean'] else '恶化'} {abs(b['ate_mean']-f['ate_mean']):.2f})")
    print(f"  RPE-T:     {b['rpe_t_mean']:.3f} → {f['rpe_t_mean']:.3f} mm")
    print(f"  RPE-R:     {b['rpe_r_mean']:.2f} → {f['rpe_r_mean']:.2f} °")
    print(f"  Scale Err: {b['scale_error_pct']:.2f}% → {f['scale_error_pct']:.2f}%   "
          f"({'改善' if f['scale_error_pct'] < b['scale_error_pct'] else '恶化'} "
          f"{abs(b['scale_error_pct']-f['scale_error_pct']):.2f}%)")

    # 方案1 vs 方案2 对比 (方案1结果来自 fusion_scale_results.json)
    print('\n' + '=' * 108)
    print('  方案1 (clip重算) vs 方案2 (clip不变) 对比')
    print('=' * 108)
    print(f"  Ours+s_mdp:  方案1 ScaleErr=58.85% (clip[0.698,4.361])  →  方案2 ScaleErr={f['scale_error_pct']:.2f}% (clip[{clip_ours[0]:.3f},{clip_ours[1]:.3f}])")

    out_path = os.path.join(OUT_DIR, 'fusion_scale_v2_results.json')
    out_data = {k: v for k, v in results.items()}
    out_data['_scale_clips'] = {k: {'global_scale': v[0],
                                    'clip': [v[1][0], v[1][1]]}
                                for k, v in clips.items()}
    with open(out_path, 'w') as fp:
        json.dump(out_data, fp, indent=2, default=str)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
