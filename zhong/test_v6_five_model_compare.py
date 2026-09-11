#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五模型 VO 对比 (F1 已默认关闭, 新 Ours 基线)
模型: Monodepth2 / ManyDepth / Lite-Mono / Ours / mdp_v5
序列: c1_transverse1_t1_v2
模式: baseline (标准 EPnP), F1 默认关闭, F2 开启
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
    load_model, compute_metrics, SEQ_NAME, SEQ_DIR, DATA_ROOT, OUT_DIR,
    MODEL_CONFIGS)
from dyendovo_dataset import load_gt_poses as dyendo_load_gt_poses


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

    results = {}
    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        print('\n' + '=' * 70)
        print(f'  [{name}]')
        print('=' * 70)
        if not os.path.isdir(cfg['path']):
            print(f'  模型目录不存在, 跳过: {cfg["path"]}')
            continue

        t0 = time.time()
        encoder, depth_decoder, motion_encoder = load_model(cfg, device)

        # 校准 (motion_encoder 可能为 None)
        print(f'  estimate_pipeline_params (motion_encoder={"有" if motion_encoder is not None else "无"})...')
        params = tv6.estimate_pipeline_params(
            SEQ_DIR, encoder, depth_decoder, motion_encoder, device,
            depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')
        print(f'  global_depth_scale={float(params["global_depth_scale"]):.4f}')

        # VO (F1 默认关闭)
        vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
            SEQ_NAME, encoder, depth_decoder, device,
            motion_net=None, mode='baseline', max_frames=None,
            motion_encoder=motion_encoder, depth_source='pred',
            loftr_source='cache', data_root=DATA_ROOT,
            pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale')

        m = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
        m['n_success'] = stats['n_success']
        m['n_pairs'] = stats['n_total']
        m['elapsed_s'] = round(time.time() - t0, 1)
        results[name] = m
        print(f'  ATE={m["ate_mean"]:.2f}mm, RPE-T={m["rpe_t_mean"]:.3f}mm, '
              f'RPE-R={m["rpe_r_mean"]:.2f}°, ScaleErr={m["scale_error_pct"]:.2f}%, '
              f'Success={m["n_success"]}/{m["n_pairs"]} ({m["elapsed_s"]}s)')

        del encoder, depth_decoder, motion_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 汇总表
    print('\n' + '=' * 100)
    print('  五模型 VO 对比 (F1 关闭, 标准 EPnP, c1_transverse1_t1_v2)')
    print('=' * 100)
    header = (f"  {'Model':<12} {'ATE':>8} {'RPE-T':>8} {'RPE-R':>8} "
              f"{'ScaleErr':>10} {'umeyama':>10} {'Success':>9}")
    print(header)
    print('  ' + '-' * 88)
    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        m = results.get(name)
        if m is None:
            print(f"  {name:<12} {'跳过':>8}")
        else:
            print(f"  {name:<12} {m['ate_mean']:>8.2f} {m['rpe_t_mean']:>8.3f} "
                  f"{m['rpe_r_mean']:>8.2f} {m['scale_error_pct']:>9.2f}% "
                  f"{m['umeyama_scale']:>10.4f} {m['n_success']:>8}/{m['n_pairs']}")

    out_path = os.path.join(OUT_DIR, 'five_model_compare_f1_off.json')
    with open(out_path, 'w') as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
