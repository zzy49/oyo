#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在 c1_transverse1_t1_v2 上对比三种零配置方案.

配置 A: --zero_gt_mode raw     (global_scale=1.0, scale_corr=禁用)
配置 B: --zero_gt_mode depth_prior (global_scale=深度先验, scale_corr=启用)
配置 C: --zero_gt_mode pnp_scale   (global_scale=PnP位移反推, scale_corr=启用)

输出对比表:
  - ATE RMSE
  - Umeyama scale
  - X/Y/Z 轴 RMS
  - 运行时间
"""

import os, sys, time, subprocess, json
import numpy as np

PROJECT_DIR = r'e:\data1\monodepth2'
SEQ = 'c1_transverse1_t1_v2'
OUT_DIR = os.path.join(PROJECT_DIR, 'zhong', 'zero_gt_compare')
os.makedirs(OUT_DIR, exist_ok=True)

# Umeyama alignment (copy from v6_pipeline/c3vd_loader.py)
def align_trajectory_umeyama(est_positions, gt_positions):
    est = np.array(est_positions, dtype=np.float64)
    gt = np.array(gt_positions, dtype=np.float64)
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]
    est_mean = est.mean(axis=0)
    gt_mean = gt.mean(axis=0)
    est_c, gt_c = est - est_mean, gt - gt_mean
    H = est_c.T @ gt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = np.trace(R @ est_c.T @ gt_c) / np.trace(est_c.T @ est_c)
    if scale < 1e-6:
        scale = 1.0
    est_aligned = (scale * (R @ est.T)).T + gt_mean - (R @ est_mean * scale)
    errors = np.linalg.norm(est_aligned - gt, axis=1)
    return est_aligned, errors, R, scale


def load_gt_traj():
    """加载 GT 位姿轨迹 (c1_transverse1_t1_v2 的 pose.txt)."""
    # Column-major format: [r11,r21,r31,tx, r12,r22,r32,ty, r13,r23,r33,tz, 0,0,0,1]
    # After reshape: translation at row 3, cols 0-2
    DATA_ROOT = r'F:\dataset'
    path = os.path.join(DATA_ROOT, SEQ, 'pose.txt')
    raw = np.loadtxt(path, delimiter=',')
    n_poses = raw.shape[0]
    poses = raw.reshape(n_poses, 4, 4)
    return poses[:, 3, :3]  # (N, 3)


def compute_metrics(vo_traj, gt_traj):
    """计算 ATE RMSE, Umeyama scale, X/Y/Z axis RMS."""
    gt = gt_traj.copy()
    vo = vo_traj.copy()

    # 截断到相同帧数
    n = min(len(vo), len(gt))
    vo, gt = vo[:n], gt[:n]

    aligned, errors, R, scale = align_trajectory_umeyama(vo, gt)
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))

    # Per-axis RMS (aligned - gt)
    diff = aligned - gt
    rms_x = float(np.sqrt(np.mean(diff[:, 0] ** 2)))
    rms_y = float(np.sqrt(np.mean(diff[:, 1] ** 2)))
    rms_z = float(np.sqrt(np.mean(diff[:, 2] ** 2)))

    # 轨迹长度 (mm)
    vo_len = float(np.sum(np.linalg.norm(np.diff(vo, axis=0), axis=1)))
    gt_len = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))

    return {
        'ate_rmse': ate_rmse, 'ate_mean': ate_mean,
        'umeyama_scale': float(scale),
        'rms_x': rms_x, 'rms_y': rms_y, 'rms_z': rms_z,
        'vo_len': vo_len, 'gt_len': gt_len,
        'n_frames': n, 'n_errors': len(errors),
    }


def run_config(label, extra_args):
    """运行 test_v6_dyendovo.py 并返回耗时和输出目录."""
    out_sub = os.path.join(OUT_DIR, f'config_{label}')
    os.makedirs(out_sub, exist_ok=True)

    # 清理旧输出
    for f in ['vo_traj_scaled.npy', 'gt_traj.npy',
              'baseline_abs_poses.npy', 'motionnet_abs_poses.npy']:
        fp = os.path.join(out_sub, f)
        if os.path.exists(fp):
            os.remove(fp)

    cmd = [
        sys.executable, os.path.join(PROJECT_DIR, 'test_v6_dyendovo.py'),
        '--seq', SEQ,
        '--mode', 'baseline',
        '--loftr_source', 'cache',
        '--out_dir', out_sub,
    ] + extra_args

    print(f"\n{'='*70}")
    print(f"  运行配置: {label}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    # 从数据目录复制输出文件到 out_sub
    DATA_ROOT = r'F:\dataset'
    seq_dir = os.path.join(DATA_ROOT, SEQ)
    import shutil
    for fname in ['vo_traj_scaled.npy', 'gt_traj.npy', 'baseline_abs_poses.npy']:
        src = os.path.join(seq_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_sub, fname))

    stdout_path = os.path.join(OUT_DIR, f'config_{label}_stdout.txt')
    stderr_path = os.path.join(OUT_DIR, f'config_{label}_stderr.txt')
    with open(stdout_path, 'w', encoding='utf-8') as f:
        f.write(result.stdout)
    with open(stderr_path, 'w', encoding='utf-8') as f:
        f.write(result.stderr)

    if result.returncode != 0:
        print(f"  ⚠ 进程退出码: {result.returncode}")
        print(f"  STDERR (最后 30 行):")
        for line in result.stderr.strip().split('\n')[-30:]:
            print(f"    {line}")

    return elapsed, out_sub, result


def main():
    # 加载 GT
    print("加载 GT 轨迹...")
    gt_traj = load_gt_traj()
    print(f"  GT: {gt_traj.shape[0]} 帧")

    configs = [
        ('A', ['--zero_gt_mode', 'raw']),
        ('B', ['--zero_gt_mode', 'depth_prior']),
        ('C', ['--zero_gt_mode', 'pnp_scale']),
    ]

    results = []

    for label, extra_args in configs:
        elapsed, out_sub, proc_result = run_config(label, extra_args)

        # 读取 VO 轨迹
        vo_path = os.path.join(out_sub, 'vo_traj_scaled.npy')
        if not os.path.exists(vo_path):
            # 尝试从 baseline_abs_poses.npy 提取
            poses_path = os.path.join(out_sub, 'baseline_abs_poses.npy')
            if os.path.exists(poses_path):
                poses = np.load(poses_path)
                vo_traj = poses[:, :3, 3].copy()
                print(f"  从 baseline_abs_poses.npy 加载 VO 轨迹: {vo_traj.shape}")
            else:
                print(f"  ⚠ 未找到 VO 轨迹文件: {vo_path}")
                results.append({'label': label, 'error': 'no_trajectory', 'time_s': elapsed})
                continue
        else:
            vo_traj = np.load(vo_path)
            print(f"  加载 VO 轨迹: {vo_traj.shape}")

        metrics = compute_metrics(vo_traj, gt_traj)

        # 提取 estimated scale
        est_scale_line = None
        for line in proc_result.stdout.split('\n'):
            if 'global_scale=' in line and '参数' in line:
                est_scale_line = line.strip()
            elif '深度中位数: observed=' in line:
                est_scale_line = line.strip()
            elif 'PnP位移中位数: observed=' in line:
                est_scale_line = line.strip()

        result_entry = {
            'label': f'配置 {label}',
            'mode': {'A': 'raw', 'B': 'depth_prior', 'C': 'pnp_scale'}[label],
            'time_s': elapsed,
            **metrics,
            '_estimated_scale_info': est_scale_line,
        }
        results.append(result_entry)

    # ── 打印对比表 ──
    print("\n" + "="*90)
    print("                    三种零配置方案对比 - c1_transverse1_t1_v2")
    print("="*90)

    header = (f"{'指标':<28} {'配置 A (raw)':>18} {'配置 B (depth_prior)':>18} "
              f"{'配置 C (pnp_scale)':>18}")
    print(header)
    print("-"*90)

    rows = []
    for cur_label in ['配置 A', '配置 B', '配置 C']:
        entry = next((r for r in results if r['label'] == cur_label), None)
        rows.append(entry)

    if not rows or all(r is None for r in rows):
        print("  ⚠ 无有效结果")
        return

    # ATE RMSE
    vals = [f"{r['ate_rmse']:.2f} mm" if r and 'ate_rmse' in r else 'N/A' for r in rows]
    print(f"{'ATE RMSE':<28} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    # ATE Mean
    vals = [f"{r['ate_mean']:.2f} mm" if r and 'ate_mean' in r else 'N/A' for r in rows]
    print(f"{'ATE Mean':<28} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    # Umeyama scale
    vals = [f"{r['umeyama_scale']:.4f}" if r and 'umeyama_scale' in r else 'N/A' for r in rows]
    print(f"{'Umeyama scale':<28} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    # X/Y/Z axis RMS
    for axis in ['rms_x', 'rms_y', 'rms_z']:
        axis_name = {'rms_x': 'X 轴 RMS', 'rms_y': 'Y 轴 RMS', 'rms_z': 'Z 轴 RMS'}[axis]
        vals = [f"{r[axis]:.2f} mm" if r and axis in r else 'N/A' for r in rows]
        print(f"{axis_name:<28} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    # 轨迹长度
    vals = [f"{r['vo_len']:.1f} ({r['gt_len']:.1f}) mm" if r and 'vo_len' in r else 'N/A' for r in rows]
    print(f"{'VO长度 (GT长度)':<28} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    # 运行时间
    vals = [f"{r['time_s']:.1f} s" if r and 'time_s' in r else 'N/A' for r in rows]
    print(f"{'运行时间':<28} {vals[0]:>18} {vals[1]:>18} {vals[2]:>18}")

    print("-"*90)

    # 打印估算信息
    for r in rows:
        if r and '_estimated_scale_info' in r and r['_estimated_scale_info']:
            print(f"  {r['label']} 估算: {r['_estimated_scale_info']}")

    # ── 选出最优 ──
    valid = [r for r in rows if r and 'ate_rmse' in r]
    if valid:
        best = min(valid, key=lambda x: x['ate_rmse'])
        print(f"\n  ★ 推荐方案: {best['label']} (ATE={best['ate_rmse']:.2f}mm)")

        # 保存 JSON
        report_path = os.path.join(OUT_DIR, 'comparison_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"  报告已保存: {report_path}")

    print("="*90)


if __name__ == '__main__':
    main()
