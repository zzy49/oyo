#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""只跑 Ours 模型 VO, 为 5 序列生成 baseline_* 产物 (供组织运动评估使用)。

背景:
  组织运动评估 (eval_pipeline_motion_vs_gt.py) 依赖每序列的
    - baseline_chain_data.npz / baseline_abs_poses.npy / baseline_depth_maps.npz
  这些产物由 run_vo_sequence(mode='baseline') 生成, 且各模型会互相覆盖。
  因此需在五模型对比之前, 单独用 Ours 跑一遍并立即做组织运动评估。

用法:
  python zhong/run_ours_vo_5seq.py            # 全 5 序列
  python zhong/run_ours_vo_5seq.py --seq c1_transverse1_t1_v2   # 单序列
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

DATA_ROOT = r'F:\dataset'
SEQ_NAMES = [
    'c1_transverse1_t1_v1',
    'c1_transverse1_t2_v1',
    'c1_transverse2_t1_v1',
    'c1_transverse2_t2_v1',
    'c1_transverse1_t1_v2',
]
if '--seq' in sys.argv:
    SEQ_NAMES = [sys.argv[sys.argv.index('--seq') + 1]]

OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'vo_depth_compare', 'ours_vo')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    cfg = next(c for c in MODEL_CONFIGS if c['name'] == 'Ours')
    encoder, depth_decoder, motion_encoder = load_model(cfg, device)

    summary = {}
    for seq in SEQ_NAMES:
        seq_dir = os.path.join(DATA_ROOT, seq)
        print(f'\n{"#" * 80}\n#  [Ours] {seq}\n{"#" * 80}')
        t0 = time.time()
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
        summary[seq] = {
            'n_success': stats['n_success'],
            'n_pairs': stats['n_total'],
            'global_depth_scale': float(params.get('global_depth_scale', 1.0)),
            'elapsed_s': round(time.time() - t0, 1),
        }
        print(f'  [Ours] {seq}: success={stats["n_success"]}/{stats["n_total"]}, '
              f'global_scale={summary[seq]["global_depth_scale"]:.4f}, '
              f'{summary[seq]["elapsed_s"]}s')

    out = os.path.join(OUT_DIR, 'ours_vo_summary.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n汇总已保存: {out}')


if __name__ == '__main__':
    main()
