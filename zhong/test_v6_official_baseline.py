#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
正式管线验证: Ours 模型 baseline (F1 已默认关闭)
验证 run_vo_sequence 默认 use_displacement_filter=False 生效,
结果应与消融中 w/o F1 一致: ATE≈3.99mm, ScaleErr≈6.54%。

注意: 此处故意不传 use_displacement_filter / k_factor,
     以走 run_vo_sequence 的最新默认值 (F1 关闭)。
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

    # 校准
    print('\n[2/3] estimate_pipeline_params (M1 开启)...')
    t0 = time.time()
    params = tv6.estimate_pipeline_params(
        SEQ_DIR, encoder, depth_decoder, motion_encoder, device,
        depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')
    print(f'  global_depth_scale={float(params["global_depth_scale"]):.4f}  ({time.time()-t0:.0f}s)')

    # 正式 VO (不传 use_displacement_filter, 走默认 False)
    print('\n[3/3] run_vo_sequence (baseline, 默认 F1=关闭)...')
    t0 = time.time()
    vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
        SEQ_NAME, encoder, depth_decoder, device,
        motion_net=None, mode='baseline', max_frames=None,
        motion_encoder=motion_encoder, depth_source='pred',
        loftr_source='cache', data_root=DATA_ROOT,
        pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale')
    print(f'  VO 完成 ({time.time()-t0:.0f}s), 成功 {stats["n_success"]}/{stats["n_total"]}')

    m = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
    m['n_success'] = stats['n_success']
    m['n_pairs'] = stats['n_total']

    print('\n' + '=' * 70)
    print('  Ours baseline (F1 默认关闭) 正式管线结果')
    print('=' * 70)
    print(f"  ATE      : {m['ate_mean']:.2f} mm")
    print(f"  RPE-T    : {m['rpe_t_mean']:.3f} mm")
    print(f"  RPE-R    : {m['rpe_r_mean']:.2f} deg")
    print(f"  ScaleErr : {m['scale_error_pct']:.2f}%")
    print(f"  umeyama  : {m['umeyama_scale']:.4f}")
    print(f"  Success  : {m['n_success']}/{m['n_pairs']}")

    out_path = os.path.join(OUT_DIR, 'official_baseline_f1_off.json')
    with open(out_path, 'w') as fp:
        json.dump(m, fp, indent=2, default=str)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
