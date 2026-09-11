#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分析 5 模型尺度异常根因.

对比各模型的:
  1. 深度输出分布 (原始 + pnp_scale 校正后)
  2. pnp_scale 计算出的 global_scale (PnP 位移反推)
  3. GT 深度分布
  4. PnP 帧间位移分布 (校正前/后)

结论输出到 zhong/vo_depth_compare/scale_analysis.json
"""
import os, sys, json
import numpy as np
import torch

PROJECT_DIR = r'e:\data1\monodepth2'
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

import test_v6_dyendovo as tv6
from test_v6_multi_model import MODEL_CONFIGS, load_model
from dyendovo_dataset import load_gt_depth

SEQ_DIR = r'F:\dataset\c1_transverse1_t1_v2'
OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'vo_depth_compare')
N_EST = 0  # 0 = 全序列 (与真实运行一致)


def depth_stats(depth_map, name):
    """统计深度图分布."""
    d = depth_map.ravel().astype(np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {}
    lo, hi = np.percentile(d, [1, 99])
    valid = d[(d >= lo) & (d <= hi)]
    return {
        'min': float(d.min()), 'max': float(d.max()),
        'p1': float(lo), 'p99': float(hi),
        'median': float(np.median(valid)), 'mean': float(valid.mean()),
        'std': float(valid.std()),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"序列: {SEQ_DIR}")
    print(f"pnp_scale 目标: target_step_mm={tv6.MAX_TRANSLATION and 0.5}")
    print(f"深度范围: [{tv6.DEPTH_MIN}, {tv6.DEPTH_MAX}] mm\n")

    results = {}

    # ── GT 深度分布 ──
    print(f"{'=' * 70}")
    print(f"  GT 深度 (真值)")
    print(f"{'=' * 70}")
    gt_depths = []
    for fi in range(20):
        try:
            gd = load_gt_depth(SEQ_DIR, fi)
            if gd is not None and np.isfinite(gd).any():
                gt_depths.append(gd)
        except Exception:
            pass
    if gt_depths:
        gt_all = np.concatenate([g.ravel() for g in gt_depths])
        gt_stats = depth_stats(gt_all, 'GT')
        print(f"  GT 深度: {json.dumps(gt_stats, indent=2)}")
        results['GT'] = gt_stats

    # ── 逐模型 ──
    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        print(f"\n{'=' * 70}")
        print(f"  [{name}]")
        print(f"{'=' * 70}")

        if not os.path.isdir(cfg['path']):
            print(f"  ⚠ 模型目录不存在, 跳过")
            continue

        encoder, depth_decoder, motion_encoder = load_model(cfg, device)

        # ── pnp_scale 预跑 (返回 global_scale + depth_cache) ──
        try:
            global_scale, depth_cache = tv6._estimate_scale_zero_gt(
                SEQ_DIR, encoder, depth_decoder, motion_encoder, device,
                n_estimate_frames=N_EST, zero_gt_mode='pnp_scale',
                pnp_max_rot_deg=tv6.MAX_ROTATION_DEG,
                pnp_max_trans=tv6.MAX_TRANSLATION,
                target_depth_mm=50.0, target_step_mm=0.5)
        except Exception as e:
            print(f"  ⚠ pnp_scale 预跑失败: {e}")
            global_scale, depth_cache = None, {}

        print(f"  global_scale = {global_scale}")

        if depth_cache:
            # 原始深度分布
            all_d = np.concatenate([d.ravel() for d in depth_cache.values()])
            raw_stats = depth_stats(all_d, name + '_raw')
            # 校正后深度分布
            scaled_d = all_d * global_scale
            scaled_stats = depth_stats(scaled_d, name + '_scaled')

            print(f"  原始深度: {json.dumps(raw_stats, indent=2)}")
            print(f"  校正后深度 (x{global_scale:.4f}): {json.dumps(scaled_stats, indent=2)}")

            results[name] = {
                'global_scale': global_scale,
                'raw_depth': raw_stats,
                'scaled_depth': scaled_stats,
            }
        else:
            print(f"  ⚠ 无 depth_cache")
            results[name] = {'global_scale': global_scale, 'error': 'no_depth_cache'}

        del encoder, depth_decoder, motion_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 保存 ──
    out_path = os.path.join(OUT_DIR, 'scale_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
