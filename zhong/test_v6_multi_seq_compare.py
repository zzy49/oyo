#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五模型 VO 多序列对比 (跨序列 mean±std 统计)

序列: c1_transverse1_t1_v1 / c1_transverse1_t2_v1 / c1_transverse2_t1_v1 / c1_transverse2_t2_v1
      (另合并已完成序列 c1_transverse1_t1_v2)
模型: Monodepth2 / ManyDepth / Lite-Mono / Ours / mdp_v5
模式: baseline (标准 EPnP), F1 关闭, F2 开启, zero_gt_mode='pnp_scale'
"""

import os, sys, json, time
import numpy as np
import torch
import random

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

import test_v6_dyendovo as tv6
from test_v6_multi_model_enhanced import load_model, compute_rpe, MODEL_CONFIGS
from dyendovo_dataset import load_gt_poses as dyendo_load_gt_poses

DATA_ROOT = r'F:\dataset'
SEQ_NAMES = [
    'c1_transverse1_t1_v1',
    'c1_transverse1_t2_v1',
    'c1_transverse2_t1_v1',
    'c1_transverse2_t2_v1',
]
OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'vo_depth_compare', 'multi_seq')
os.makedirs(OUT_DIR, exist_ok=True)

# 已完成的单序列结果, 用于合并统计
PREV_SEQ = 'c1_transverse1_t1_v2'
PREV_JSON = os.path.join(PROJECT_DIR, 'zhong', 'vo_depth_compare', 'five_model_compare_f1_off.json')


def compute_metrics_seq(vo_traj, gt_traj, chain_data, gt_poses_std, seq_name):
    """计算单序列指标 (与 compute_metrics 一致, 但显式传入序列名)."""
    ate_result = tv6.evaluate_trajectory(vo_traj, gt_traj, seq_name, 'baseline')
    abs_poses = chain_data['abs_poses']
    rpe = compute_rpe(abs_poses, gt_poses_std)
    scale_error_pct = abs(ate_result.get('scale', 1.0) - 1.0) * 100.0
    return {
        'ate_mean': ate_result.get('mean_mm', float('nan')),
        'ate_std': ate_result.get('std_mm', float('nan')),
        'ate_rmse': ate_result.get('rmse_mm', float('nan')),
        'ate_median': ate_result.get('median_mm', float('nan')),
        'rpe_t_mean': rpe['rpe_t_mean'], 'rpe_t_std': rpe['rpe_t_std'],
        'rpe_t_rmse': rpe['rpe_t_rmse'],
        'rpe_r_mean': rpe['rpe_r_mean'], 'rpe_r_std': rpe['rpe_r_std'],
        'rpe_r_rmse': rpe['rpe_r_rmse'],
        'scale_error_pct': scale_error_pct,
        'umeyama_scale': ate_result.get('scale', float('nan')),
        'n_rpe': rpe['n_rpe'],
    }


def run_seq(seq_name, device):
    seq_dir = os.path.join(DATA_ROOT, seq_name)
    gt_poses_std = dyendo_load_gt_poses(seq_dir)
    print(f'\n{"#" * 90}\n#  序列: {seq_name}  (GT {len(gt_poses_std)} 帧)\n{"#" * 90}')

    results = {}
    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        print('\n' + '=' * 70)
        print(f'  [{seq_name}] [{name}]')
        print('=' * 70)
        if not os.path.isdir(cfg['path']):
            print(f'  模型目录不存在, 跳过: {cfg["path"]}')
            continue
        try:
            t0 = time.time()
            encoder, depth_decoder, motion_encoder = load_model(cfg, device)
            params = tv6.estimate_pipeline_params(
                seq_dir, encoder, depth_decoder, motion_encoder, device,
                depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')
            vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
                seq_name, encoder, depth_decoder, device,
                motion_net=None, mode='baseline', max_frames=None,
                motion_encoder=motion_encoder, depth_source='pred',
                loftr_source='cache', data_root=DATA_ROOT,
                pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale')
            m = compute_metrics_seq(vo_traj, gt_traj, chain_data, gt_poses_std, seq_name)
            m['n_success'] = stats['n_success']
            m['n_pairs'] = stats['n_total']
            m['elapsed_s'] = round(time.time() - t0, 1)
            results[name] = m
            print(f'  [{name}] ATE={m["ate_mean"]:.2f}mm, RPE-T={m["rpe_t_mean"]:.3f}mm, '
                  f'RPE-R={m["rpe_r_mean"]:.2f}°, ScaleErr={m["scale_error_pct"]:.2f}%, '
                  f'Success={m["n_success"]}/{m["n_pairs"]} ({m["elapsed_s"]}s)')
            del encoder, depth_decoder, motion_encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f'  [{name}] 失败: {e}')
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}\n')

    all_results = {}
    for seq in SEQ_NAMES:
        res = run_seq(seq, device)
        all_results[seq] = res
        out_path = os.path.join(OUT_DIR, f'five_model_{seq}.json')
        with open(out_path, 'w') as fp:
            json.dump(res, fp, indent=2, default=str)
        print(f'\n[已保存] {out_path}')

    # 合并已完成的序列
    if os.path.isfile(PREV_JSON):
        with open(PREV_JSON) as fp:
            prev = json.load(fp)
        all_results[PREV_SEQ] = prev
        print(f'[合并] 已有序列 {PREV_SEQ} 结果已读入')

    # ── 跨序列汇总 mean ± std ──
    seq_names = list(all_results.keys())
    model_names = [c['name'] for c in MODEL_CONFIGS]
    keys = ['ate_mean', 'ate_rmse', 'rpe_t_mean', 'rpe_t_rmse',
            'rpe_r_mean', 'rpe_r_rmse', 'scale_error_pct', 'umeyama_scale']

    summary = {}
    print('\n' + '=' * 100)
    print(f'  跨序列汇总 ({len(seq_names)} 序列): {seq_names}')
    print('=' * 100)
    for mn in model_names:
        vals = {k: [] for k in keys}
        total_success = 0
        total_pairs = 0
        for seq in seq_names:
            m = all_results.get(seq, {}).get(mn)
            if m is None:
                continue
            for k in keys:
                v = m.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals[k].append(v)
            total_success += m.get('n_success', 0)
            total_pairs += m.get('n_pairs', 0)

        row = {}
        for k in keys:
            arr = np.array(vals[k], dtype=float)
            row[k + '_mean'] = float(np.mean(arr)) if len(arr) else float('nan')
            row[k + '_std'] = float(np.std(arr)) if len(arr) else float('nan')
            row[k + '_n'] = len(arr)
        row['success'] = f'{total_success}/{total_pairs}'
        summary[mn] = row
        print(f'  {mn:<12} ATE={row["ate_mean_mean"]:.2f}±{row["ate_mean_std"]:.2f} mm | '
              f'RPE-T={row["rpe_t_mean_mean"]:.3f}±{row["rpe_t_mean_std"]:.3f} | '
              f'RPE-R={row["rpe_r_mean_mean"]:.2f}±{row["rpe_r_mean_std"]:.2f}° | '
              f'ScaleErr={row["scale_error_pct_mean"]:.2f}±{row["scale_error_pct_std"]:.2f}%')

    summary_out = os.path.join(OUT_DIR, 'five_model_cross_seq_summary.json')
    with open(summary_out, 'w') as fp:
        json.dump(summary, fp, indent=2, default=str)
    all_out = os.path.join(OUT_DIR, 'five_model_all_seq.json')
    with open(all_out, 'w') as fp:
        json.dump(all_results, fp, indent=2, default=str)
    print(f'\n汇总已保存: {summary_out}')
    print(f'全部结果已保存: {all_out}')


if __name__ == '__main__':
    main()
