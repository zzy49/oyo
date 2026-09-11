#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五模型深度估计精度多序列评估

与 VO 对比实验对齐: 5 序列 × 5 模型, 使用同一套 multi 权重 (MODEL_CONFIGS).
指标 (标准 Eigen 协议, median scaling 对齐): AbsRel↓, SqRel↓, RMSE↓, RMSElog↓,
δ<1.25↑, δ<1.25²↑, δ<1.25³↑.
输出: 每序列结果 + 跨序列 mean±std 汇总.
"""

import os, sys, json, time
import numpy as np
import torch
import random

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

from v6_pipeline.utils import predict_depth
from test_v6_multi_model_enhanced import load_model, MODEL_CONFIGS
from dyendovo_dataset import load_gt_depth

DATA_ROOT = r'F:\dataset'
SEQ_NAMES = [
    'c1_transverse1_t1_v1',
    'c1_transverse1_t2_v1',
    'c1_transverse2_t1_v1',
    'c1_transverse2_t2_v1',
    'c1_transverse1_t1_v2',
]
OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'vo_depth_compare', 'multi_seq')
os.makedirs(OUT_DIR, exist_ok=True)


def compute_depth_metrics(gt, pred):
    """标准 Eigen 深度指标 (median scaling 对齐)."""
    mask = (gt > 0) & (gt < 500) & (pred > 0) & np.isfinite(pred)
    g = gt[mask].astype(np.float64)
    p = pred[mask].astype(np.float64)
    if len(g) < 100:
        return None
    scale = np.median(g) / (np.median(p) + 1e-8)
    p = p * scale
    thresh = np.maximum(g / p, p / g)
    return dict(
        absrel=float(np.mean(np.abs(g - p) / g)),
        sqrel=float(np.mean(((g - p) ** 2) / g)),
        rmse=float(np.sqrt(np.mean((g - p) ** 2))),
        rmse_log=float(np.sqrt(np.mean((np.log(g) - np.log(p)) ** 2))),
        d1=float(np.mean(thresh < 1.25)),
        d2=float(np.mean(thresh < 1.25 ** 2)),
        d3=float(np.mean(thresh < 1.25 ** 3)),
    )


def run_seq(seq_name, device):
    seq_dir = os.path.join(DATA_ROOT, seq_name)
    warp_dir = os.path.join(seq_dir, 'generated', 'rgb_warped')
    frames = sorted([f for f in os.listdir(warp_dir) if f.endswith('.png')])
    print(f'\n{"#" * 90}\n#  序列: {seq_name}  ({len(frames)} 帧)\n{"#" * 90}')

    seq_results = {}
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
            nb = getattr(depth_decoder, 'num_bins', 0)

            frame_metrics = []
            for fi, fname in enumerate(frames):
                img_path = os.path.join(warp_dir, fname)
                try:
                    fidx = int(fname.split('_')[1].split('.')[0])
                except (IndexError, ValueError):
                    # 回退: 用文件序号
                    fidx = fi
                try:
                    gt = load_gt_depth(seq_dir, fidx)
                except Exception:
                    continue  # GT 深度缺失, 跳过该帧

                try:
                    pred = predict_depth(img_path, encoder, depth_decoder, device,
                                         feed_h=256, feed_w=320,
                                         min_depth=1.0, max_depth=500.0, num_bins=nb)
                except Exception as e:
                    print(f'  [WARN] frame {fidx}: {e}')
                    continue

                met = compute_depth_metrics(gt, pred)
                if met:
                    frame_metrics.append(met)

                if (fi + 1) % 100 == 0:
                    print(f'  进度: {fi + 1}/{len(frames)}')

            if frame_metrics:
                keys = ['absrel', 'sqrel', 'rmse', 'rmse_log', 'd1', 'd2', 'd3']
                summary = {k + '_mean': float(np.mean([m[k] for m in frame_metrics]))
                           for k in keys}
                summary['n_frames'] = len(frame_metrics)
                summary['elapsed_s'] = round(time.time() - t0, 1)
                seq_results[name] = summary
                print(f'  [{name}] n={len(frame_metrics)}, AbsRel={summary["absrel_mean"]:.4f}, '
                      f'RMSE={summary["rmse_mean"]:.1f}mm, SqRel={summary["sqrel_mean"]:.4f}, '
                      f'δ1={summary["d1_mean"]:.4f} ({summary["elapsed_s"]}s)')
            else:
                seq_results[name] = {'error': 'no valid frames'}

            del encoder, depth_decoder, motion_encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f'  [{name}] 失败: {e}')
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return seq_results


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    all_results = {}
    for seq in SEQ_NAMES:
        res = run_seq(seq, device)
        all_results[seq] = res
        out_path = os.path.join(OUT_DIR, f'depth_accuracy_{seq}.json')
        with open(out_path, 'w') as fp:
            json.dump(res, fp, indent=2, default=str)
        print(f'\n[已保存] {out_path}')

    # ── 跨序列汇总 mean ± std ──
    seq_names = list(all_results.keys())
    model_names = [c['name'] for c in MODEL_CONFIGS]
    keys = ['absrel_mean', 'sqrel_mean', 'rmse_mean', 'rmse_log_mean',
            'd1_mean', 'd2_mean', 'd3_mean']

    summary = {}
    print('\n' + '=' * 100)
    print(f'  深度指标跨序列汇总 ({len(seq_names)} 序列): {seq_names}')
    print('=' * 100)
    for mn in model_names:
        vals = {k: [] for k in keys}
        total_frames = 0
        for seq in seq_names:
            m = all_results.get(seq, {}).get(mn)
            if m is None or 'error' in m:
                continue
            for k in keys:
                v = m.get(k)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals[k].append(v)
            total_frames += m.get('n_frames', 0)

        row = {}
        for k in keys:
            arr = np.array(vals[k], dtype=float)
            row[k + '_mean'] = float(np.mean(arr)) if len(arr) else float('nan')
            row[k + '_std'] = float(np.std(arr)) if len(arr) else float('nan')
            row[k + '_n'] = len(arr)
        row['total_frames'] = total_frames
        summary[mn] = row
        print(f'  {mn:<12} AbsRel={row["absrel_mean_mean"]:.4f}±{row["absrel_mean_std"]:.4f} | '
              f'RMSE={row["rmse_mean_mean"]:.1f}±{row["rmse_mean_std"]:.1f}mm | '
              f'SqRel={row["sqrel_mean_mean"]:.4f}±{row["sqrel_mean_std"]:.4f} | '
              f'δ1={row["d1_mean_mean"]:.4f}±{row["d1_mean_std"]:.4f}')

    summary_out = os.path.join(OUT_DIR, 'depth_cross_seq_summary.json')
    with open(summary_out, 'w') as fp:
        json.dump(summary, fp, indent=2, default=str)
    all_out = os.path.join(OUT_DIR, 'depth_all_seq.json')
    with open(all_out, 'w') as fp:
        json.dump(all_results, fp, indent=2, default=str)
    print(f'\n汇总已保存: {summary_out}')
    print(f'全部结果已保存: {all_out}')


if __name__ == '__main__':
    main()
