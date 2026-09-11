#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
4 模型 VO 指标对比脚本 —— 读取 test_v6_multi_model.py 的输出 JSON,
生成格式化对比表格。

用法:
  python zhong\compare_multi_model.py

输入: zhong\vo_depth_compare\multi_model_results.json
"""

import os, sys, json
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'vo_depth_compare')
RESULT_PATH = os.path.join(OUT_DIR, 'multi_model_results.json')


def fmt_mm(mean, std, width=8):
    """格式化 mm 指标: mean±std"""
    return f"{mean:>{width-5}.2f}±{std:.2f}"


def fmt_deg(mean, std, width=8):
    """格式化 角度 指标: mean±std"""
    return f"{mean:>{width-5}.2f}±{std:.2f}"


def fmt_pct(val, width=9):
    """格式化百分比"""
    return f"{val:>{width}.2f}%"


def print_header(title):
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")


def main():
    if not os.path.exists(RESULT_PATH):
        print(f"错误: 结果文件不存在: {RESULT_PATH}")
        print(f"请先运行: python zhong\\test_v6_multi_model.py")
        return 1

    with open(RESULT_PATH) as f:
        results = json.load(f)

    models = list(results.keys())

    # ── 验证数据 ──
    for name in models:
        m = results[name]
        print(f"[{name}] ATE={m['ate_mean']:.2f}mm, "
              f"RPE-R={m['rpe_r_mean']:.2f}°, "
              f"ScaleErr={m['scale_error_pct']:.1f}%, "
              f"Success={m['n_success']}/{m['n_pairs']}")

    print_header("4 模型对比 (test_v6_dyendovo.py pipeline, pnp_scale 零配置)")

    # ── 表1: 主指标 ──
    print(f"  {'Model':<14} {'ATE↓(mm)':>12}   {'RPE-T↓(mm)':>12}   "
          f"{'RPE-R↓(°)':>12}   {'ScaleErr↓':>10}   {'Success':>8}   {'Time':>6}")
    print(f"  {'-' * 92}")

    for name in models:
        m = results[name]
        ate_s = fmt_mm(m['ate_mean'], m['ate_std'], 12)
        rpet_s = fmt_mm(m['rpe_t_mean'], m['rpe_t_std'], 12)
        rper_s = fmt_deg(m['rpe_r_mean'], m['rpe_r_std'], 12)
        se_s = fmt_pct(m['scale_error_pct'], 10)
        succ_s = f"{m['n_success']:>3d}/{m['n_pairs']:<3d}"
        time_s = f"{m['elapsed_s']:>5.0f}s"
        print(f"  {name:<14} {ate_s}   {rpet_s}   {rper_s}   {se_s}   {succ_s}   {time_s}")

    # ── 表2: ATE 详细 ──
    print_header("ATE 详细 (Umeyama 对齐)")
    print(f"  {'Model':<14} {'RMSE(mm)':>10}   {'Mean(mm)':>10}   "
          f"{'Std(mm)':>10}   {'Median(mm)':>10}   {'Scale':>10}")
    print(f"  {'-' * 80}")

    for name in models:
        m = results[name]
        print(f"  {name:<14} {m['ate_rmse']:>10.2f}   {m['ate_mean']:>10.2f}   "
              f"{m['ate_std']:>10.2f}   {m['ate_median']:>10.2f}   "
              f"{m['umeyama_scale']:>10.4f}")

    # ── 表3: RPE 详细 ──
    print_header("RPE 详细")
    print(f"  {'Model':<14} {'RPE-T RMSE':>11}   {'RPE-T Mean':>11}   "
          f"{'RPE-T Std':>11}   {'RPE-R RMSE':>11}   {'RPE-R Mean':>11}   "
          f"{'RPE-R Std':>11}")
    print(f"  {'-' * 92}")

    for name in models:
        m = results[name]
        print(f"  {name:<14} {m['rpe_t_rmse']:>11.4f}   {m['rpe_t_mean']:>11.4f}   "
              f"{m['rpe_t_std']:>11.4f}   {m['rpe_r_rmse']:>11.4f}   "
              f"{m['rpe_r_mean']:>11.4f}   {m['rpe_r_std']:>11.4f}")

    # ── 排名分析 ──
    print_header("排名 (↓越小越好)")
    metrics_for_rank = [
        ('ATE↓', 'ate_mean', False),
        ('RPE-T↓', 'rpe_t_mean', False),
        ('RPE-R↓', 'rpe_r_mean', False),
        ('ScaleErr↓', 'scale_error_pct', False),
    ]

    for metric_name, key, reverse in metrics_for_rank:
        ranked = sorted(models, key=lambda n: results[n][key], reverse=reverse)
        ranks = []
        for i, name in enumerate(ranked):
            val = results[name][key]
            if key == 'scale_error_pct':
                rank_str = f"{i+1}. {name} ({val:.1f}%)"
            elif key in ('rpe_t_mean',):
                rank_str = f"{i+1}. {name} ({val:.4f}mm)"
            elif key in ('rpe_r_mean',):
                rank_str = f"{i+1}. {name} ({val:.2f}°)"
            else:
                rank_str = f"{i+1}. {name} ({val:.2f}mm)"
            ranks.append(rank_str)
        print(f"  {metric_name}: {'  →  '.join(ranks)}")

    print(f"\n  结果文件: {RESULT_PATH}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
