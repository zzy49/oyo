#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
以主配置为基准的消融实验：逐个关闭操作，观察效果是否变差。

主配置（Full）包含三个开启的操作：
  1. 运动编码器 (MotionEncoder)：两帧时序深度增强
  2. 零配置尺度恢复 (pnp_scale)：PnP 帧间位移自校准全局尺度
  3. 深度一致性校正 (scale_corr)：帧间深度漂移抑制
（自适应 3D 位移过滤在主配置中已关闭，不属于消融范围）

消融矩阵：
  Full(主配置)          : 运动编码器✓  尺度恢复✓  帧间校正✓
  w/o 运动编码器         : 单帧深度 (ResNet-18 only)
  w/o 零配置尺度恢复      : global_scale 强制 1.0 (保留帧间校正)
  w/o 深度一致性校正      : use_scale_corr=False

序列: c1_transverse1_t1_v2 (117 帧)
模式: baseline (标准 EPnP), use_displacement_filter=False
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
    {'name': 'Full(主配置)',      'use_motion_encoder': True,  'use_scale_restore': True,  'use_scale_corr': True},
    {'name': 'w/o 运动编码器',     'use_motion_encoder': False, 'use_scale_restore': True,  'use_scale_corr': True},
    {'name': 'w/o 零配置尺度恢复',  'use_motion_encoder': True,  'use_scale_restore': False, 'use_scale_corr': True},
    {'name': 'w/o 深度一致性校正',  'use_motion_encoder': True,  'use_scale_restore': True,  'use_scale_corr': False},
]


def eval_variant(cfg, encoder, depth_decoder, motion_encoder, device, gt_poses_std):
    mot = motion_encoder if cfg['use_motion_encoder'] else None

    print(f'\n  [{cfg["name"]}] 估计管线参数 (motion_encoder={"有" if mot else "无"})...')
    t0 = time.time()
    params = tv6.estimate_pipeline_params(
        SEQ_DIR, encoder, depth_decoder, mot, device,
        depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')
    s_orig = float(params['global_depth_scale'])
    if not cfg['use_scale_restore']:
        params['global_depth_scale'] = 1.0
        print(f'  [{cfg["name"]}] 关闭尺度恢复: global_depth_scale {s_orig:.4f} → 1.0')
    else:
        print(f'  [{cfg["name"]}] global_depth_scale={s_orig:.4f}  ({time.time()-t0:.0f}s)')

    print(f'  [{cfg["name"]}] 运行 VO (scale_corr={"有" if cfg["use_scale_corr"] else "无"})...')
    t0 = time.time()
    vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
        SEQ_NAME, encoder, depth_decoder, device,
        motion_net=None, mode='baseline', max_frames=None,
        motion_encoder=mot, depth_source='pred',
        loftr_source='cache', data_root=DATA_ROOT,
        pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale',
        use_displacement_filter=False,
        use_scale_corr=cfg['use_scale_corr'])
    print(f'  [{cfg["name"]}] VO 完成 ({time.time()-t0:.0f}s), '
          f'成功 {stats["n_success"]}/{stats["n_total"]}')

    m = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
    m['n_success'] = stats['n_success']
    m['n_pairs'] = stats['n_total']
    m['global_depth_scale'] = s_orig
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

    print('[1/2] 加载 Ours 深度模型 (含运动编码器)...')
    cfg = {'name': 'Ours', 'path': MODEL_PATH, 'type': 'md2'}
    encoder, depth_decoder, motion_encoder = load_model(cfg, device)
    print(f'  motion_encoder: {"有" if motion_encoder is not None else "无"}\n')

    results = {}
    for vcfg in VARIANTS:
        print('\n' + '=' * 80)
        print(f'  {vcfg["name"]}')
        print('=' * 80)
        try:
            results[vcfg['name']] = eval_variant(
                vcfg, encoder, depth_decoder, motion_encoder, device, gt_poses_std)
        except Exception as e:
            print(f'  [ERROR] {vcfg["name"]} 评估失败: {e}')
            import traceback
            traceback.print_exc()
            results[vcfg['name']] = {'error': str(e)}

    # 汇总表 (均值 ± 标准差)
    print('\n' + '=' * 110)
    print('  消融结果 (主配置为基准, 均值 ± 标准差)')
    print('=' * 110)
    header = (f"  {'变体':<20} {'ATE RMSE':>22} {'ATE Mean':>22} "
              f"{'RPE-T RMSE':>14} {'RPE-R RMSE':>14} {'ScaleErr%':>12} {'Success':>9}")
    print(header)
    print('  ' + '-' * 106)
    for vcfg in VARIANTS:
        name = vcfg['name']
        m = results.get(name)
        if m is None:
            print(f"  {name:<20} {'缺失':>22}")
        elif 'error' in m:
            print(f"  {name:<20} {'ERROR':>22} {m['error'][:30]}")
        else:
            ate = f"{m['ate_rmse']:.2f} ± {m['ate_std']:.2f}"
            ate_m = f"{m['ate_mean']:.2f} ± {m['ate_std']:.2f}"
            rpt = f"{m['rpe_t_rmse']:.3f} ± {m['rpe_t_std']:.3f}"
            rpr = f"{m['rpe_r_rmse']:.2f} ± {m['rpe_r_std']:.2f}"
            print(f"  {name:<20} {ate:>22} {ate_m:>22} {rpt:>14} {rpr:>14} "
                  f"{m['scale_error_pct']:>11.2f}% {m['n_success']:>8}/{m['n_pairs']}")

    # 相对 Full 的退化分析
    full = results.get(VARIANTS[0]['name'], {})
    if 'ate_rmse' in full:
        print('\n  ' + '-' * 70)
        print('  相对主配置 (Full) 的退化量 (正值=关闭后变差=正优化):')
        print('  ' + '-' * 70)
        print(f"  {'变体':<20} {'ΔATE RMSE':>12} {'ΔATE Mean':>12} {'ΔScaleErr':>12}")
        for vcfg in VARIANTS[1:]:
            name = vcfg['name']
            m = results.get(name)
            if m and 'ate_rmse' in m:
                d_rmse = m['ate_rmse'] - full['ate_rmse']
                d_mean = m['ate_mean'] - full['ate_mean']
                d_se = m['scale_error_pct'] - full['scale_error_pct']
                print(f"  {name:<20} {d_rmse:>+12.2f} {d_mean:>+12.2f} {d_se:>+12.2f}")

    out_path = os.path.join(OUT_DIR, 'ablation_main_results.json')
    with open(out_path, 'w') as fp:
        json.dump({'variants': results}, fp, indent=2, default=str)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
