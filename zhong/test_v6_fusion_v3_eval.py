#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方案3 训练结果 VO 评估: 评估 fusion_v3 checkpoint 的 VO 指标

对比 baseline (来自 fusion_scale_results.json):
  Ours baseline:   ATE 5.11, RPE-T 0.385, RPE-R 0.426, Scale 25.54%
  mdp_v5 baseline: ATE 6.03, RPE-T 0.490, RPE-R 0.696, Scale 13.89%

用法:
  python zhong/test_v6_fusion_v3_eval.py                      # 评估所有已存在 checkpoint
  python zhong/test_v6_fusion_v3_eval.py --cps weights_1      # 只评估指定 checkpoint
"""

import os, sys, json, time, glob, argparse
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

FUSION_MODELS_DIR = r'C:\Users\Administrator\tmp\fusion_v3\models'
BASELINE_JSON = os.path.join(OUT_DIR, 'fusion_scale_results.json')


def eval_checkpoint(ckpt_dir, device, gt_poses_std):
    """加载单个 checkpoint 并跑完整 VO 评估."""
    cfg = {'name': os.path.basename(ckpt_dir), 'path': ckpt_dir, 'type': 'md2'}
    print(f'\n  [{cfg["name"]}] 加载模型...')
    enc, dec, mot = load_model(cfg, device)

    print(f'  [{cfg["name"]}] 预跑 estimate_pipeline_params...')
    t0 = time.time()
    params = tv6.estimate_pipeline_params(
        SEQ_DIR, enc, dec, mot, device,
        depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')
    s = float(params['global_depth_scale'])
    print(f'  [{cfg["name"]}] global_depth_scale = {s:.4f}  ({time.time()-t0:.0f}s)')

    print(f'  [{cfg["name"]}] 运行 VO...')
    t0 = time.time()
    vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
        SEQ_NAME, enc, dec, device,
        motion_net=None, mode='baseline', max_frames=None,
        motion_encoder=mot, depth_source='pred',
        loftr_source='cache', data_root=DATA_ROOT,
        pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale')
    print(f'  [{cfg["name"]}] VO 完成: {time.time()-t0:.0f}s')

    m = compute_metrics(vo_traj, gt_traj, chain_data, gt_poses_std)
    m['n_success'] = stats['n_success']
    m['n_pairs'] = stats['n_total']
    m['global_depth_scale'] = s
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cps', nargs='+', default=None,
                        help='checkpoint 名称列表, 如 weights_0 weights_1')
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}, 序列: {SEQ_NAME}\n')

    gt_poses_std = dyendo_load_gt_poses(SEQ_DIR)
    print(f'GT: {len(gt_poses_std)} 帧\n')

    # 确定要评估的 checkpoint
    if args.cps:
        ckpts = [os.path.join(FUSION_MODELS_DIR, c) for c in args.cps]
    else:
        ckpts = sorted(glob.glob(os.path.join(FUSION_MODELS_DIR, 'weights_*')))
    ckpts = [c for c in ckpts if os.path.isfile(os.path.join(c, 'depth.pth'))]
    if not ckpts:
        print('未找到任何 checkpoint (weights_*), 训练尚未完成第一个 epoch.')
        return
    print(f'待评估 checkpoint: {[os.path.basename(c) for c in ckpts]}')

    results = {}
    for ck in ckpts:
        name = os.path.basename(ck)
        try:
            results[name] = eval_checkpoint(ck, device, gt_poses_std)
        except Exception as e:
            print(f'  [ERROR] {name} 评估失败: {e}')
            results[name] = {'error': str(e)}

    # 读取 baseline
    baseline = {}
    if os.path.isfile(BASELINE_JSON):
        with open(BASELINE_JSON) as fp:
            baseline = json.load(fp)

    # 汇总
    print('\n' + '=' * 100)
    print('  方案3 (fusion_v3) VO 评估汇总')
    print('=' * 100)
    header = (f"  {'模型':<22} {'ATE':>8} {'RPE-T':>8} {'RPE-R':>8} "
              f"{'ScaleErr':>10} {'umeyama':>10} {'Success':>9}")
    print(header)
    print('  ' + '-' * 92)

    def print_row(name, m):
        if 'error' in m:
            print(f"  {name:<22} {'ERROR':>8} {m['error'][:40]}")
            return
        print(f"  {name:<22} {m['ate_mean']:>8.2f} {m['rpe_t_mean']:>8.3f} "
              f"{m['rpe_r_mean']:>8.2f} {m['scale_error_pct']:>9.2f}% "
              f"{m['umeyama_scale']:>10.4f} {m['n_success']:>8}/{m['n_pairs']}")

    for bname in ['Ours baseline', 'mdp_v5 baseline']:
        if bname in baseline:
            print_row(bname, baseline[bname])
    for name, m in results.items():
        print_row(name, m)

    # 与 Ours baseline 核心对比
    if 'Ours baseline' in baseline:
        b = baseline['Ours baseline']
        print('\n' + '=' * 100)
        print('  核心对比: Ours baseline → fusion_v3 (目标: Scale 改善, ATE/RPE 不恶化)')
        print('=' * 100)
        for name, m in results.items():
            if 'error' in m:
                continue
            print(f'\n  [{name}]')
            print(f"    ATE:       {b['ate_mean']:.2f} → {m['ate_mean']:.2f} mm   "
                  f"({'改善' if m['ate_mean'] < b['ate_mean'] else '恶化'} "
                  f"{abs(b['ate_mean']-m['ate_mean']):.2f})")
            print(f"    RPE-T:     {b['rpe_t_mean']:.3f} → {m['rpe_t_mean']:.3f} mm   "
                  f"({'改善' if m['rpe_t_mean'] < b['rpe_t_mean'] else '恶化'} "
                  f"{abs(b['rpe_t_mean']-m['rpe_t_mean']):.3f})")
            print(f"    RPE-R:     {b['rpe_r_mean']:.2f} → {m['rpe_r_mean']:.2f} °   "
                  f"({'改善' if m['rpe_r_mean'] < b['rpe_r_mean'] else '恶化'} "
                  f"{abs(b['rpe_r_mean']-m['rpe_r_mean']):.2f})")
            print(f"    Scale Err: {b['scale_error_pct']:.2f}% → {m['scale_error_pct']:.2f}%   "
                  f"({'改善' if m['scale_error_pct'] < b['scale_error_pct'] else '恶化'} "
                  f"{abs(b['scale_error_pct']-m['scale_error_pct']):.2f}%)")

    out_path = os.path.join(OUT_DIR, 'fusion_v3_eval_results.json')
    with open(out_path, 'w') as fp:
        json.dump({'baseline': baseline, 'fusion_v3': results}, fp, indent=2, default=str)
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()
