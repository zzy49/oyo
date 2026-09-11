#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三基线 + Ours 组织运动对比（论文口径，F1 关闭）。

背景：
  论文 3.4 节表 3 需要 Monodepth2 / ManyDepth / Lite-Mono / Ours 四列的组织运动指标。
  Ours 列已由 eval_pipeline_motion_paper_metric.py 得到（主配置 F1 关闭）。
  三基线重训完成后，用本脚本对每个模型跑 VO 并立即做组织运动评估。

流程（每模型每序列）：
  1. estimate_pipeline_params（pnp_scale 零配置尺度恢复）
  2. run_vo_sequence(mode='baseline', F1 关闭) → 生成 baseline_* 产物到 {seq_dir}
  3. compute_sequence（论文口径：GT 位姿反投影 → gt_disp，VO 位姿反投影 → vo_disp）
     产出 static_rmse_mm / moving_rmse_mm / correlation_r

产物会互相覆盖 → 每模型跑完一个序列立即评估，再跑下一模型/序列。

用法：
  python zhong/run_baselines_motion_5seq.py                      # 三基线全 5 序列
  python zhong/run_baselines_motion_5seq.py --seq c1_transverse1_t1_v2
  python zhong/run_baselines_motion_5seq.py --models Monodepth2,ManyDepth,Lite-Mono,Ours
"""
import os
import sys
import json
import time
import numpy as np
import torch
import random

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

import test_v6_dyendovo as tv6
from test_v6_multi_model_enhanced import load_model, MODEL_CONFIGS
from eval_pipeline_motion_paper_metric import compute_sequence

DATA_ROOT = r'F:\dataset'
SEQ_NAMES = [
    'c1_transverse1_t1_v2',
    'c1_transverse1_t1_v1',
    'c1_transverse1_t2_v1',
    'c1_transverse2_t1_v1',
    'c1_transverse2_t2_v1',
]
if '--seq' in sys.argv:
    SEQ_NAMES = [sys.argv[sys.argv.index('--seq') + 1]]

DEFAULT_MODELS = ['Monodepth2', 'ManyDepth', 'Lite-Mono']
if '--models' in sys.argv:
    DEFAULT_MODELS = sys.argv[sys.argv.index('--models') + 1].split(',')

OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'vo_depth_compare', 'baselines_motion')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    print(f'模型: {DEFAULT_MODELS}')
    print(f'序列: {SEQ_NAMES}\n')

    results = {}
    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        if name not in DEFAULT_MODELS:
            continue
        print(f'\n{"#" * 80}\n#  模型: {name}\n#  路径: {cfg["path"]}\n{"#" * 80}')
        if not os.path.isdir(cfg['path']):
            print(f'  [跳过] 模型目录不存在（训练未完成?）: {cfg["path"]}')
            continue

        encoder, depth_decoder, motion_encoder = load_model(cfg, device)
        results[name] = {}

        for seq in SEQ_NAMES:
            seq_dir = os.path.join(DATA_ROOT, seq)
            print(f'\n  ── [{name}] {seq} ──')
            t0 = time.time()
            try:
                params = tv6.estimate_pipeline_params(
                    seq_dir, encoder, depth_decoder, motion_encoder, device,
                    depth_source='pred', loftr_source='cache', no_gt=False,
                    zero_gt_mode='pnp_scale')
                vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
                    seq, encoder, depth_decoder, device,
                    motion_net=None, mode='baseline', max_frames=None,
                    motion_encoder=motion_encoder, depth_source='pred',
                    loftr_source='cache', data_root=DATA_ROOT,
                    pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale')
                metrics = compute_sequence(seq, use_median_scale=True)
                metrics['vo_success'] = stats['n_success']
                metrics['vo_pairs'] = stats['n_total']
                metrics['global_depth_scale'] = float(params.get('global_depth_scale', 1.0))
                metrics['elapsed_s'] = round(time.time() - t0, 1)
                results[name][seq] = metrics
            except Exception as e:
                print(f'  [FAIL] {name}/{seq}: {e}')
                import traceback
                traceback.print_exc()
                results[name][seq] = None

        del encoder, depth_decoder, motion_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 逐序列保存
    for name, seqs in results.items():
        for seq, m in seqs.items():
            if m is not None:
                with open(os.path.join(OUT_DIR, f'motion_{name}_{seq}.json'), 'w') as f:
                    json.dump(m, f, indent=2)

    # 跨序列汇总表（直接对应论文表 3 的三列指标）
    print('\n' + '=' * 100)
    print('  组织运动对比（论文口径，F1 关闭，跨序列 Mean ± SD）')
    print('=' * 100)
    print(f"  {'Model':<12} {'静态RMSE(mm)':>16} {'运动RMSE(mm)':>16} {'相关系数r':>14}")
    print('  ' + '-' * 66)
    summary = {}
    for name in DEFAULT_MODELS:
        seqs = results.get(name, {})
        valid = {s: m for s, m in seqs.items() if m is not None}
        if not valid:
            print(f"  {name:<12} {'(无有效结果)':>16}")
            continue
        row = {'n_sequences': len(valid), 'sequences': list(valid.keys())}
        for k in ['static_rmse_mm', 'moving_rmse_mm', 'correlation_r']:
            vals = [v[k] for v in valid.values()]
            row[f'{k}_mean'] = float(np.mean(vals))
            row[f'{k}_std'] = float(np.std(vals))
        summary[name] = row
        print(f"  {name:<12} {row['static_rmse_mm_mean']:>7.2f}±{row['static_rmse_mm_std']:<5.2f} "
              f"{row['moving_rmse_mm_mean']:>7.2f}±{row['moving_rmse_mm_std']:<5.2f} "
              f"{row['correlation_r_mean']:>7.3f}±{row['correlation_r_std']:<5.3f}")
    print('=' * 100)

    with open(os.path.join(OUT_DIR, 'motion_summary.json'), 'w') as f:
        json.dump({'models': summary, 'per_seq': results}, f, indent=2, default=str)
    print(f'\n结果已保存: {OUT_DIR}')


if __name__ == '__main__':
    main()
