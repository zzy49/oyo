#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V6 方法消融实验 - F1 阈值 (k_factor) 扫描
固定: M1 开启, F2 开启, 标准 EPnP
扫描: F1 的 k_factor ∈ {0.5(当前), 1.0, 1.5, 2.0, 3.0}
参照: w/o F1 (完全不过滤) = 3.99mm / 6.54% (来自 ablation_results.json)

阈值公式: threshold = median + k*std, 且 >= 1mm
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
    load_model, compute_metrics, SEQ_NAME, SEQ_DIR, DATA_ROOT, OUT_DIR)
from dyendovo_dataset import load_gt_poses as dyendo_load_gt_poses

MODEL_PATH = r'e:\data1\monodepth2\models\depth'

K_FACTORS = [0.5, 1.0, 1.5, 2.0, 3.0]


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}, 序列: {SEQ_NAME}\n')

    gt_poses_std = dyendo_load_gt_poses(SEQ_DIR)
    print(f'GT: {len(gt_poses_std)} 帧\n')

    # 加载 Ours 模型 (含 MotionEncoder)
    print('[1/3] 加载 Ours 深度模型...')
    cfg = {'name': 'Ours', 'path': MODEL_PATH, 'type': 'md2'}
    encoder, depth_decoder, motion_encoder = load_model(cfg, device)
    print(f'  motion_encoder: {"有" if motion_encoder is not None else "无"}')

    # estimate_pipeline_params 只跑一次 (M1 开启, 与 k_factor 无关)
    print('\n[2/3] estimate_pipeline_params (M1 开启)...')
    t0 = time.time()
    params = tv6.estimate_pipeline_params(
        SEQ_DIR, encoder, depth_decoder, motion_encoder, device,
        depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')
    s = float(params['global_depth_scale'])
    print(f'  global_depth_scale={s:.4f}  ({time.time()-t0:.0f}s)\n')

    # 读取已有 w/o F1 参照
    prev_path = os.path.join(OUT_DIR, 'ablation_results.json')
    wof1 = None
    if os.path.exists(prev_path):
        with open(prev_path, 'r') as fp:
            wof1 = json.load(fp).get('variants', {}).get('w/o F1')

    results = {}
    for k in K_FACTORS:
        name = f'F1 k={k}'
        print('\n' + '=' * 80)
        print(f'  {name}')
        print('=' * 80)
        t0 = time.time()
        vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
            SEQ_NAME, encoder, depth_decoder, device,
            motion_net=None, mode='baseline', max_frames=None,
            motion_encoder=motion_encoder, depth_source='pred',
            loftr_source='cache', data_root=DATA_ROOT,
            pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale',
            use_displacement_filter=True,
            use_scale_corr=True,
            k_factor=k)
        print(f'  [{name}] VO 完成 ({time.time()-t0:.0f}s), '
              f'成功 {stats["n_success"]}/{stats["n_total"]}')
        m = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
        m['n_success'] = stats['n_success']
        m['n_pairs'] = stats['n_total']
        m['global_depth_scale'] = s
        results[name] = m

    # 汇总
    print('\n' + '=' * 100)
    print('  F1 阈值 (k_factor) 扫描结果 (M1 开, F2 开, 标准 EPnP)')
    print('=' * 100)
    header = (f"  {'变体':<14} {'ATE':>8} {'RPE-T':>8} {'RPE-R':>8} "
              f"{'ScaleErr':>10} {'umeyama':>10} {'Success':>9}")
    print(header)
    print('  ' + '-' * 90)
    for k in K_FACTORS:
        name = f'F1 k={k}'
        m = results[name]
        print(f"  {name:<14} {m['ate_mean']:>8.2f} {m['rpe_t_mean']:>8.3f} "
              f"{m['rpe_r_mean']:>8.2f} {m['scale_error_pct']:>9.2f}% "
              f"{m['umeyama_scale']:>10.4f} {m['n_success']:>8}/{m['n_pairs']}")
    if wof1 and 'error' not in wof1:
        print(f"  {'w/o F1(参照)':<14} {wof1['ate_mean']:>8.2f} {wof1['rpe_t_mean']:>8.3f} "
              f"{wof1['rpe_r_mean']:>8.2f} {wof1['scale_error_pct']:>9.2f}% "
              f"{wof1['umeyama_scale']:>10.4f} {wof1['n_success']:>8}/{wof1['n_pairs']}")

    # 保存
    out_path = os.path.join(OUT_DIR, 'ablation_f1_k_results.json')
    save = {'variants': results}
    if wof1:
        save['w/o F1 (reference)'] = wof1
    with open(out_path, 'w') as fp:
        json.dump(save, fp, indent=2, default=str)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
