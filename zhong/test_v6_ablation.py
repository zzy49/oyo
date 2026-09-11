#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V6 方法消融实验 (推理侧 4 变体)

消融维度 (Ours = M1 + 标准 EPnP, M2 已弃用):
  M1 = MotionEncoder      帧间运动特征注入深度 decoder (两帧推理)
  F1 = 自适应 3D 位移过滤  baseline PnP 前过滤运动点
  F2 = 帧间尺度一致性校正  scale_corr 抑制帧间深度漂移

变体 (全部 mode=baseline, 标准 EPnP):
  Ours (M1+F1+F2): 完整方法
  w/o M1         : 单帧深度 (关闭 MotionEncoder)
  w/o F1         : 关闭自适应 3D 位移过滤
  w/o F2         : 关闭帧间尺度一致性校正

指标: ATE / RPE-T / RPE-R / ScaleErr / umeyama scale

用法:
  python zhong/test_v6_ablation.py
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

VARIANTS = [
    {'name': 'Ours (M1+F1+F2)', 'use_motion_encoder': True,  'use_displacement_filter': True,  'use_scale_corr': True},
    {'name': 'w/o M1',          'use_motion_encoder': False, 'use_displacement_filter': True,  'use_scale_corr': True},
    {'name': 'w/o F1',          'use_motion_encoder': True,  'use_displacement_filter': False, 'use_scale_corr': True},
    {'name': 'w/o F2',          'use_motion_encoder': True,  'use_displacement_filter': True,  'use_scale_corr': False},
]


def eval_variant(cfg, encoder, depth_decoder, motion_encoder, device, gt_poses_std):
    mot = motion_encoder if cfg['use_motion_encoder'] else None

    print(f'\n  [{cfg["name"]}] estimate_pipeline_params (motion_encoder={"有" if mot else "无"})...')
    t0 = time.time()
    params = tv6.estimate_pipeline_params(
        SEQ_DIR, encoder, depth_decoder, mot, device,
        depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')
    s = float(params['global_depth_scale'])
    print(f'  [{cfg["name"]}] global_depth_scale={s:.4f}  ({time.time()-t0:.0f}s)')

    print(f'  [{cfg["name"]}] run_vo_sequence (baseline, F1={"有" if cfg["use_displacement_filter"] else "无"}, '
          f'F2={"有" if cfg["use_scale_corr"] else "无"})...')
    t0 = time.time()
    vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
        SEQ_NAME, encoder, depth_decoder, device,
        motion_net=None, mode='baseline', max_frames=None,
        motion_encoder=mot, depth_source='pred',
        loftr_source='cache', data_root=DATA_ROOT,
        pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale',
        use_displacement_filter=cfg['use_displacement_filter'],
        use_scale_corr=cfg['use_scale_corr'])
    print(f'  [{cfg["name"]}] VO 完成 ({time.time()-t0:.0f}s), '
          f'成功 {stats["n_success"]}/{stats["n_total"]}')

    m = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
    m['n_success'] = stats['n_success']
    m['n_pairs'] = stats['n_total']
    m['global_depth_scale'] = s
    return m


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
    print('[1/2] 加载 Ours 深度模型...')
    cfg = {'name': 'Ours', 'path': MODEL_PATH, 'type': 'md2'}
    encoder, depth_decoder, motion_encoder = load_model(cfg, device)
    print(f'  motion_encoder: {"有" if motion_encoder is not None else "无"}')

    # 评估各变体
    results = {}
    for cfg in VARIANTS:
        print('\n' + '=' * 80)
        print(f'  {cfg["name"]}')
        print('=' * 80)
        try:
            results[cfg['name']] = eval_variant(
                cfg, encoder, depth_decoder, motion_encoder, device, gt_poses_std)
        except Exception as e:
            print(f'  [ERROR] {cfg["name"]} 评估失败: {e}')
            import traceback
            traceback.print_exc()
            results[cfg['name']] = {'error': str(e)}

    # 汇总
    print('\n' + '=' * 100)
    print('  V6 方法消融实验结果 (推理侧 M1/F1/F2)')
    print('=' * 100)
    header = (f"  {'变体':<18} {'ATE':>8} {'RPE-T':>8} {'RPE-R':>8} "
              f"{'ScaleErr':>10} {'umeyama':>10} {'Success':>9}")
    print(header)
    print('  ' + '-' * 92)
    for cfg in VARIANTS:
        name = cfg['name']
        m = results[name]
        if 'error' in m:
            print(f"  {name:<18} {'ERROR':>8} {m['error'][:40]}")
        else:
            print(f"  {name:<18} {m['ate_mean']:>8.2f} {m['rpe_t_mean']:>8.3f} "
                  f"{m['rpe_r_mean']:>8.2f} {m['scale_error_pct']:>9.2f}% "
                  f"{m['umeyama_scale']:>10.4f} {m['n_success']:>8}/{m['n_pairs']}")

    # 保存
    out_path = os.path.join(OUT_DIR, 'ablation_results.json')
    with open(out_path, 'w') as fp:
        json.dump({'variants': results}, fp, indent=2, default=str)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
