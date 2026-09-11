"""组织运动评估（论文 3.4 口径）：复用 evaluate_pipeline.py 的 GT 位姿反投影口径。

与 eval_pipeline_motion_vs_gt.py 的区别：
  - 本脚本复现 evaluate_pipeline.py 的 Step F 口径（GT 相机位姿反投影 → GT 位移；
    VO 位姿反投影 → VO 位移），即论文表 3 的 0.36 / 3.10 / 0.948 所用口径。
  - 直接用主配置（F1 关闭）的 baseline_* 产物，无需 tracks.json。

口径（与 evaluate_pipeline.py 完全一致）：
  - 深度图 ÷global_scale 恢复原始（物理）尺度
  - gt_disp = 用 GT 位姿反投影 track 首末帧世界坐标差
  - vo_disp = 用 VO 位姿反投影 track 首末帧世界坐标差
  - GT 静止点: gt_disp < 1mm；GT 运动点: gt_disp >= 1mm
  - 静止点 RMSE = ||vo_disp - gt_disp|| 在 GT 静止点上的 RMSE
  - 运动点 RMSE = ||vo_disp - gt_disp|| 在 GT 运动点上的 RMSE
  - 相关系数 = corr(gt_disp, vo_disp)（全部 track）

用法:
  python zhong/eval_pipeline_motion_paper_metric.py --seq c1_transverse1_t1_v2
  python zhong/eval_pipeline_motion_paper_metric.py
"""
import os
import sys
import json
import numpy as np
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

DATA_ROOT = 'F:/dataset'
SEQ_NAMES = [
    'c1_transverse1_t1_v2',
    'c1_transverse1_t1_v1',
    'c1_transverse1_t2_v1',
    'c1_transverse2_t1_v1',
    'c1_transverse2_t2_v1',
]
if '--seq' in sys.argv:
    SEQ_NAMES = [sys.argv[sys.argv.index('--seq') + 1]]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'vo_depth_compare', 'pipeline_motion_paper')

K = np.array([[767.73, 0, 677.74], [0, 767.73, 543.06], [0, 0, 1]], dtype=np.float64)

GT_SCALE = 100.0 / 65535.0  # uint16 -> mm

# 复用 eval_pipeline_motion_vs_gt.py 里的快速 cKDTree 链式追踪与反投影
from eval_pipeline_motion_vs_gt import (
    chain_tracks, backproject_to_world, load_gt_poses_orderF,
)

MAX_TRACKS = 15000
GT_STATIC_THRESH_MM = 1.0


def compute_sequence(seq_name, use_median_scale=False):
    seq_dir = os.path.join(DATA_ROOT, seq_name)
    chain_path = os.path.join(seq_dir, 'baseline_chain_data.npz')
    abs_poses_path = os.path.join(seq_dir, 'baseline_abs_poses.npy')
    depth_path = os.path.join(seq_dir, 'baseline_depth_maps.npz')

    print(f"\n{'='*64}")
    print(f"  [组织运动评估·论文口径] {seq_name}")
    print(f"{'='*64}")

    chain = np.load(chain_path, allow_pickle=True)
    depth_npz = np.load(depth_path, allow_pickle=True)
    abs_poses = np.load(abs_poses_path)          # (N,4,4) world_to_cam
    n_pairs = int(chain['n_pairs'])
    all_k0 = [chain[f'k0_{i:04d}'] for i in range(n_pairs)]
    all_k1 = [chain[f'k1_{i:04d}'] for i in range(n_pairs)]
    chain_dist_thresh = float(chain['chain_dist_thresh']) if 'chain_dist_thresh' in chain else 3.0
    global_scale = float(chain['global_depth_scale']) if 'global_depth_scale' in chain else 1.0
    depth_src = str(chain['depth_source']) if 'depth_source' in chain else 'pred'

    depth_maps = {k: depth_npz[k] for k in depth_npz.files}
    # 与 evaluate_pipeline.py 一致: ÷global_scale 恢复原始物理尺度
    if depth_src != 'gt' and global_scale != 1.0:
        depth_maps = {k: v / global_scale for k, v in depth_maps.items()}
    print(f"  深度尺度恢复: ÷global_scale={global_scale:.4f}")

    # 尺度无关基线：对深度逐帧 median scaling 对齐 GT 深度，恢复到物理毫米尺度。
    # 物理尺度感知方法（Ours）深度已准确，无需此步。
    if use_median_scale:
        gt_depth_dir = os.path.join(seq_dir, 'depth')
        scales = []
        for k, v in depth_maps.items():
            gt_path = os.path.join(gt_depth_dir, f'{int(k):04d}_depth.tiff')
            if not os.path.exists(gt_path):
                continue
            gt_mm = np.array(Image.open(gt_path)).astype(np.float32) * GT_SCALE
            mask = (gt_mm > 0.5) & (v > 0.5)
            if mask.sum() > 100:
                s = float(np.median(gt_mm[mask]) / np.median(v[mask]))
                if np.isfinite(s) and s > 0:
                    depth_maps[k] = v * s
                    scales.append(s)
        if scales:
            print(f"  深度 median scaling 对齐 GT: mean_scale={np.mean(scales):.4f} "
                  f"(n={len(scales)})")

    gt_poses_c2w = load_gt_poses_orderF(seq_dir)  # cam_to_world, mm
    gt_poses_w2c = [np.linalg.inv(T) for T in gt_poses_c2w]
    n_gt = len(gt_poses_w2c)

    print(f"  n_pairs={n_pairs}, VO位姿={abs_poses.shape}, GT位姿={n_gt}")

    # 链式追踪
    tracks = chain_tracks(all_k0, all_k1, dist_thresh=chain_dist_thresh)
    print(f"  总 tracks={len(tracks)}")
    if len(tracks) > MAX_TRACKS:
        import random
        random.seed(42)
        long_t = [t for t in tracks if len(t['frames']) >= 5]
        short_t = [t for t in tracks if len(t['frames']) < 5]
        n_long = min(len(long_t), int(MAX_TRACKS * 2 / 3))
        n_short = MAX_TRACKS - n_long
        sampled = (random.sample(long_t, n_long) if n_long > 0 else []) + \
                  (random.sample(short_t, min(n_short, len(short_t)))
                   if n_short > 0 and short_t else [])
        tracks = sampled
        print(f"  采样至 {len(tracks)} tracks")

    gt_disps = []
    vo_disps = []
    for t in tracks:
        frames = t['frames']
        if len(frames) < 2:
            continue
        f0, f1 = frames[0][0], frames[-1][0]
        if f0 >= n_gt or f1 >= n_gt or f0 == f1:
            continue

        # GT 位移（GT 位姿反投影）
        p0 = backproject_to_world(frames[0][1], frames[0][2],
                                  depth_maps.get(f'{f0:04d}'), gt_poses_w2c[f0])
        p1 = backproject_to_world(frames[-1][1], frames[-1][2],
                                  depth_maps.get(f'{f1:04d}'), gt_poses_w2c[f1])
        if p0 is None or p1 is None:
            continue
        gt_disp = float(np.linalg.norm(p1 - p0))

        # VO 位移（VO 位姿反投影）
        q0 = backproject_to_world(frames[0][1], frames[0][2],
                                  depth_maps.get(f'{f0:04d}'), abs_poses[f0])
        q1 = backproject_to_world(frames[-1][1], frames[-1][2],
                                  depth_maps.get(f'{f1:04d}'), abs_poses[f1])
        if q0 is None or q1 is None:
            continue
        vo_disp = float(np.linalg.norm(q1 - q0))

        gt_disps.append(gt_disp)
        vo_disps.append(vo_disp)

    gt_disps = np.array(gt_disps)
    vo_disps = np.array(vo_disps)
    abs_err = np.abs(vo_disps - gt_disps)

    static_mask = gt_disps < GT_STATIC_THRESH_MM
    moving_mask = ~static_mask

    def rmse(a):
        return float(np.sqrt(np.mean(a ** 2))) if len(a) > 0 else 0.0

    metrics = {
        'n_tracks': int(len(gt_disps)),
        'n_gt_static': int(static_mask.sum()),
        'n_gt_moving': int(moving_mask.sum()),
        'static_rmse_mm': rmse(abs_err[static_mask]),
        'moving_rmse_mm': rmse(abs_err[moving_mask]),
        'correlation_r': float(np.corrcoef(gt_disps, vo_disps)[0, 1])
                         if len(gt_disps) > 2 else 0.0,
        'gt_disp_median_mm': float(np.median(gt_disps)),
        'vo_disp_median_mm': float(np.median(vo_disps)),
    }

    print(f"  有效 track 数: {metrics['n_tracks']}")
    print(f"  GT 静止点 (gt<{GT_STATIC_THRESH_MM}mm): {metrics['n_gt_static']} → 静止点位移 RMSE = {metrics['static_rmse_mm']:.3f} mm")
    print(f"  GT 运动点 (gt>={GT_STATIC_THRESH_MM}mm): {metrics['n_gt_moving']} → 运动点位移 RMSE = {metrics['moving_rmse_mm']:.3f} mm")
    print(f"  位移相关系数 r = {metrics['correlation_r']:.3f}")
    return metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for seq in SEQ_NAMES:
        try:
            results[seq] = compute_sequence(seq)
        except Exception as e:
            print(f"  [FAIL] {seq}: {e}")
            import traceback
            traceback.print_exc()
            results[seq] = None

    valid = {k: v for k, v in results.items() if v is not None}
    for seq, m in valid.items():
        with open(os.path.join(OUT_DIR, f'paper_metric_{seq}.json'), 'w') as f:
            json.dump(m, f, indent=2)

    if valid:
        keys = ['static_rmse_mm', 'moving_rmse_mm', 'correlation_r',
                'n_gt_static', 'n_gt_moving', 'n_tracks']
        summary = {'n_sequences': len(valid), 'sequences': list(valid.keys())}
        for k in keys:
            vals = [v[k] for v in valid.values()]
            summary[f'{k}_mean'] = float(np.mean(vals))
            summary[f'{k}_std'] = float(np.std(vals))
        with open(os.path.join(OUT_DIR, 'paper_metric_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("  论文口径组织运动指标  跨序列汇总 (Mean ± SD)")
        print("=" * 70)
        for k in keys:
            print(f"    {k:<20s}: {summary[f'{k}_mean']:>10.4f} ± {summary[f'{k}_std']:>10.4f}")
        print("=" * 70)
        print(f"  输出目录: {OUT_DIR}")


if __name__ == '__main__':
    main()
