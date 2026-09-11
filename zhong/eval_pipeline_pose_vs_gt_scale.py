"""组织运动绝对位姿差值评估 —— 深度尺度对齐对照实验。

目的：分离"轨迹对齐引入的深度尺度放大(global_scale≈2.2x)"与"位姿/追踪误差"的贡献。

背景：baseline_depth_maps.npz 里保存的预测深度已被 VO 管线的 global_depth_scale
      (Umeyama 轨迹对齐尺度) 放大，导致深度绝对值比 GT 大约 2.2 倍。该放大是为了
      修正相机轨迹尺度，代价是牺牲了深度真值。

对照设计（固定同一批运动 track + 同一分类，仅改变反投影所用深度）：
  [scaled]   baseline_depth_maps.npz 原样（深度已被放大，≈GT 的 2.2 倍）
  [perframe] 逐帧中位数对齐：depth *= median(GT帧) / median(pred帧)
             → 消除尺度偏差，只保留位姿误差 + 深度空间分布误差

另输出每帧 scale_factor 的分布，量化"轨迹对齐引入的深度放大倍数"。

用法:
  python zhong/eval_pipeline_pose_vs_gt_scale.py --seq c1_transverse1_t1_v2 --limit 30
  python zhong/eval_pipeline_pose_vs_gt_scale.py
"""
import os
import sys
import json
import numpy as np
import cv2
from scipy.spatial import cKDTree

LIMIT = None
if '--limit' in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index('--limit') + 1])

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
                       'vo_depth_compare', 'pose_scale')

K = np.array([[767.73, 0, 677.74], [0, 767.73, 543.06], [0, 0, 1]], dtype=np.float64)

CHAIN_DIST_THRESH = 3.0
DISP_K_FACTOR = 0.5
MAX_TRACKS = 15000
SAMPLE_MAX = 5000
MIN_VALID = 0.5  # 深度有效下限 (mm)


def load_gt_poses_orderF(seq_dir):
    """读取 pose.txt, order='F' reshape => cam_to_world (mm)。"""
    with open(os.path.join(seq_dir, 'pose.txt')) as f:
        lines = f.readlines()
    return [np.array([float(x) for x in ln.strip().split(',')]).reshape(4, 4, order='F')
            for ln in lines]


def load_gt_depth(seq_dir, fi):
    """加载 GT 深度 TIFF, 转为 mm。"""
    p = os.path.join(seq_dir, 'depth', f'{fi:04d}_depth.tiff')
    if not os.path.exists(p):
        return None
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return d.astype(np.float64) * (100.0 / 65535.0)


def chain_tracks(all_k0, all_k1, dist_thresh=CHAIN_DIST_THRESH):
    """串联相邻帧 LoFTR 匹配, 形成跨帧 track (cKDTree 加速版)。"""
    n_pairs = len(all_k0)
    tracks = []
    active = {}
    next_id = 0

    for pair_idx in range(n_pairs):
        k0 = all_k0[pair_idx]
        k1 = all_k1[pair_idx]
        if len(k0) == 0:
            active = {}
            continue

        if not active:
            for j in range(len(k0)):
                tid = next_id
                next_id += 1
                tracks.append({'id': tid, 'frames': [
                    (pair_idx, float(k0[j, 0]), float(k0[j, 1])),
                    (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))]})
                active[tid] = (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))
            continue

        active_ids = list(active.keys())
        active_pts = np.array([[active[t][1], active[t][2]] for t in active_ids])
        tree = cKDTree(active_pts)
        dists, nn = tree.query(k0, k=1)

        matched_tids = set()
        new_active = {}
        matched_k0 = np.zeros(len(k0), dtype=bool)

        order = np.argsort(dists)
        for j in order:
            if dists[j] > dist_thresh:
                break
            tid = active_ids[nn[j]]
            if tid in matched_tids:
                continue
            matched_tids.add(tid)
            matched_k0[j] = True
            tracks[tid]['frames'].append(
                (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1])))
            new_active[tid] = (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))

        for j in range(len(k0)):
            if not matched_k0[j]:
                tid = next_id
                next_id += 1
                tracks.append({'id': tid, 'frames': [
                    (pair_idx, float(k0[j, 0]), float(k0[j, 1])),
                    (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))]})
                new_active[tid] = (pair_idx + 1, float(k1[j, 0]), float(k1[j, 1]))

        active = new_active

    return tracks


def backproject_to_world(u, v, depth_map, T_world_cam, scale=1.0):
    """反投影 2D 点到世界 3D 坐标 (mm)，无效返回 None。scale 用于逐帧尺度对齐。"""
    if depth_map is None:
        return None
    h, w = depth_map.shape
    ui = int(np.clip(round(u), 0, w - 1))
    vi = int(np.clip(round(v), 0, h - 1))
    Z = float(depth_map[vi, ui]) * scale
    if Z <= 0.5 or not np.isfinite(Z):
        return None
    p_cam = np.array([(u - K[0, 2]) * Z / K[0, 0],
                      (v - K[1, 2]) * Z / K[1, 1], Z])
    T_cam_world = np.linalg.inv(T_world_cam)
    return T_cam_world[:3, :3] @ p_cam + T_cam_world[:3, 3]


def track_disp_and_med(track, depth_maps, abs_poses):
    """计算 track 逐帧世界坐标 + 相邻帧位移中位数。"""
    p_worlds = []
    for frame_idx, u, v in track['frames']:
        key = f'{frame_idx:04d}'
        if key not in depth_maps or frame_idx >= len(abs_poses):
            p_worlds.append(None)
        else:
            p_worlds.append(backproject_to_world(u, v, depth_maps[key],
                                                 abs_poses[frame_idx]))
    valid = [p for p in p_worlds if p is not None]
    if len(valid) < 2:
        return p_worlds, 0.0
    steps = [np.linalg.norm(valid[i + 1] - valid[i]) for i in range(len(valid) - 1)]
    return p_worlds, float(np.median(steps))


def classify_by_displacement(tracks, depth_maps, abs_poses, k_factor=DISP_K_FACTOR):
    """自适应 3D 位移阈值分类 track 为 (static_tracks, moving_tracks)。"""
    import random
    random.seed(42)
    if len(tracks) > SAMPLE_MAX:
        sample = random.sample(tracks, SAMPLE_MAX)
    else:
        sample = tracks

    meds = []
    for t in sample:
        _, med = track_disp_and_med(t, depth_maps, abs_poses)
        meds.append(med)
    meds = np.array(meds)
    med = float(np.median(meds))
    std = float(np.std(meds))
    threshold = med + k_factor * std
    print(f"  自适应阈值: median={med:.2f}mm std={std:.2f}mm threshold={threshold:.2f}mm")

    static_tracks, moving_tracks = [], []
    for t in tracks:
        _, med_step = track_disp_and_med(t, depth_maps, abs_poses)
        if med_step > threshold:
            moving_tracks.append((t, med_step))
        else:
            static_tracks.append((t, med_step))
    return static_tracks, moving_tracks, threshold


def summarize(err, label):
    err = np.array(err)
    return {
        'n': int(len(err)),
        'mean_mm': float(np.mean(err)),
        'std_mm': float(np.std(err)),
        'median_mm': float(np.median(err)),
        'rmse_mm': float(np.sqrt(np.mean(err ** 2))),
        'p90_mm': float(np.percentile(err, 90)),
        'p95_mm': float(np.percentile(err, 95)),
    }


def evaluate_sequence(seq_name):
    seq_dir = os.path.join(DATA_ROOT, seq_name)
    chain_path = os.path.join(seq_dir, 'baseline_chain_data.npz')
    abs_poses_path = os.path.join(seq_dir, 'baseline_abs_poses.npy')
    depth_path = os.path.join(seq_dir, 'baseline_depth_maps.npz')

    print(f"\n{'='*64}")
    print(f"  [深度尺度对齐对照] {seq_name}")
    print(f"{'='*64}")

    for label, p in [('chain', chain_path), ('VO位姿', abs_poses_path),
                     ('深度图', depth_path)]:
        if not os.path.exists(p):
            print(f"  [FAIL] {label} 不存在: {p}")
            return None

    chain = np.load(chain_path, allow_pickle=True)
    depth_npz = np.load(depth_path, allow_pickle=True)
    abs_poses = np.load(abs_poses_path)
    n_pairs = int(chain['n_pairs'])
    if LIMIT is not None:
        n_pairs = min(n_pairs, LIMIT)
    chain_dist_thresh = float(chain['chain_dist_thresh']) if 'chain_dist_thresh' in chain else CHAIN_DIST_THRESH
    k_factor = float(chain['k_factor']) if 'k_factor' in chain else DISP_K_FACTOR
    gt_poses = load_gt_poses_orderF(seq_dir)          # cam_to_world
    gt_poses_w2c = [np.linalg.inv(T) for T in gt_poses]
    n_gt = len(gt_poses)
    print(f"  n_pairs={n_pairs} VO位姿={abs_poses.shape} GT位姿={n_gt} chain_thresh={chain_dist_thresh}")

    all_k0 = [chain[f'k0_{i:04d}'] for i in range(n_pairs)]
    all_k1 = [chain[f'k1_{i:04d}'] for i in range(n_pairs)]

    # 深度图一次性加载到内存 dict (scaled 版, 已被 global_scale 放大)
    depth_maps = {k: depth_npz[k] for k in depth_npz.files if k.startswith('0') or k.startswith('1')}
    print(f"  预测深度帧已加载: {len(depth_maps)}")

    def vo_to_gt(i):
        T = gt_poses[i] @ abs_poses[i]
        return T[:3, :3], T[:3, 3]

    gt_depth_cache = {}
    frame_scale = {}  # key -> scale_i = median(GT帧) / median(pred帧)

    def get_gt_depth(fi):
        if fi not in gt_depth_cache:
            gt_depth_cache[fi] = load_gt_depth(seq_dir, fi)
        return gt_depth_cache[fi]

    def get_scale(key, fi):
        if key in frame_scale:
            return frame_scale[key]
        pred = depth_maps.get(key)
        if pred is None:
            frame_scale[key] = None
            return None
        gt = get_gt_depth(fi)
        if gt is None:
            frame_scale[key] = None
            return None
        pm = float(np.median(pred[pred > MIN_VALID]))
        gm = float(np.median(gt[gt > MIN_VALID]))
        if pm <= 0 or gm <= 0 or not np.isfinite(pm) or not np.isfinite(gm):
            frame_scale[key] = None
            return None
        s = gm / pm
        frame_scale[key] = s
        return s

    # Step A: 链式追踪
    print("  [A] 链式追踪 ...")
    tracks = chain_tracks(all_k0, all_k1, dist_thresh=chain_dist_thresh)
    print(f"  总 tracks={len(tracks)}")
    if len(tracks) > MAX_TRACKS:
        long_t = [t for t in tracks if len(t['frames']) >= 5]
        short_t = [t for t in tracks if len(t['frames']) < 5]
        import random
        random.seed(42)
        n_long = min(len(long_t), int(MAX_TRACKS * 2 / 3))
        n_short = MAX_TRACKS - n_long
        sampled = (random.sample(long_t, n_long) if n_long > 0 else []) + \
                  (random.sample(short_t, min(n_short, len(short_t)))
                   if n_short > 0 and short_t else [])
        print(f"  采样至 {len(sampled)} tracks")
        tracks = sampled

    # Step B: 运动/静止分类 (用 scaled 深度)
    print("  [B] 运动/静止分类 (scaled 深度) ...")
    static_tracks, moving_tracks, threshold = classify_by_displacement(
        tracks, depth_maps, abs_poses, k_factor=k_factor)
    print(f"  静止={len(static_tracks)} 运动={len(moving_tracks)}")

    # Step C: 逐帧绝对位姿误差 (scaled vs perframe)
    err_scaled = []
    err_perframe = []

    for t, med_step in moving_tracks:
        if len(t['frames']) < 2:
            continue
        for frame_idx, u, v in t['frames']:
            if frame_idx >= n_gt:
                continue
            pred_key = f'{frame_idx:04d}'
            pred_depth = depth_maps.get(pred_key)
            if pred_depth is None:
                continue
            gt_depth = get_gt_depth(frame_idx)
            if gt_depth is None:
                continue
            s = get_scale(pred_key, frame_idx)
            if s is None:
                continue

            p_vo_scaled = backproject_to_world(u, v, pred_depth, abs_poses[frame_idx], scale=1.0)
            p_vo_perframe = backproject_to_world(u, v, pred_depth, abs_poses[frame_idx], scale=s)
            p_gt = backproject_to_world(u, v, gt_depth, gt_poses_w2c[frame_idx])
            if p_vo_scaled is None or p_vo_perframe is None or p_gt is None:
                continue
            R_i, t_i = vo_to_gt(frame_idx)
            err_scaled.append(float(np.linalg.norm(R_i @ p_vo_scaled + t_i - p_gt)))
            err_perframe.append(float(np.linalg.norm(R_i @ p_vo_perframe + t_i - p_gt)))

    if not err_scaled:
        print("  [FAIL] 无有效评估帧")
        return None

    scales = np.array([v for v in frame_scale.values() if v is not None])
    m_scaled = summarize(err_scaled, 'scaled')
    m_perframe = summarize(err_perframe, 'perframe')

    print(f"\n  ┌────────────────────────────────────────────────────────────┐")
    print(f"  │  组织运动绝对位姿差值  深度尺度对齐对照                    │")
    print(f"  ├────────────────────────────────────────────────────────────┤")
    print(f"  │  评估帧数:                {len(err_scaled):>6d}                          │")
    print(f"  │  每帧 scale_factor (GT/pred):  median={np.median(scales):.4f} mean={np.mean(scales):.4f}  │")
    print(f"  ├────────────────────────────────────────────────────────────┤")
    print(f"  │  [scaled]   深度已放大   Mean±Std = {m_scaled['mean_mm']:>7.2f} ± {m_scaled['std_mm']:>5.2f} mm  │")
    print(f"  │              RMSE = {m_scaled['rmse_mm']:>8.3f}  Median = {m_scaled['median_mm']:>8.3f} mm        │")
    print(f"  │  [perframe] 逐帧对齐     Mean±Std = {m_perframe['mean_mm']:>7.2f} ± {m_perframe['std_mm']:>5.2f} mm  │")
    print(f"  │              RMSE = {m_perframe['rmse_mm']:>8.3f}  Median = {m_perframe['median_mm']:>8.3f} mm        │")
    print(f"  └────────────────────────────────────────────────────────────┘")

    return {
        'seq': seq_name,
        'n_frames_eval': int(len(err_scaled)),
        'n_moving_tracks': len(moving_tracks),
        'threshold_mm': float(threshold),
        'scale_factor_median': float(np.median(scales)),
        'scale_factor_mean': float(np.mean(scales)),
        'scale_factor_std': float(np.std(scales)),
        'scaled': m_scaled,
        'perframe': m_perframe,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for seq in SEQ_NAMES:
        try:
            m = evaluate_sequence(seq)
            results[seq] = m
        except Exception as e:
            print(f"  [FAIL] {seq}: {e}")
            import traceback
            traceback.print_exc()
            results[seq] = None

    for seq, m in results.items():
        if m is not None:
            with open(os.path.join(OUT_DIR, f'pose_scale_{seq}.json'), 'w') as f:
                json.dump(m, f, indent=2)

    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        keys_scaled = ['mean_mm', 'std_mm', 'median_mm', 'rmse_mm', 'p90_mm', 'p95_mm']
        summary = {'n_sequences': len(valid), 'sequences': list(valid.keys())}
        for mode in ['scaled', 'perframe']:
            for k in keys_scaled:
                vals = np.array([v[mode][k] for v in valid.values()], dtype=float)
                summary[f'{mode}_{k}_mean'] = float(np.mean(vals))
                summary[f'{mode}_{k}_sd'] = float(np.std(vals))
        scale_vals = np.array([v['scale_factor_median'] for v in valid.values()], dtype=float)
        summary['scale_factor_median_mean'] = float(np.mean(scale_vals))
        summary['scale_factor_median_sd'] = float(np.std(scale_vals))

        with open(os.path.join(OUT_DIR, 'pose_scale_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 72)
        print("  深度尺度对齐对照  跨序列 Mean ± SD 汇总")
        print("=" * 72)
        print(f"  每帧 scale_factor (GT/pred) 跨序列: "
              f"{summary['scale_factor_median_mean']:.4f} ± {summary['scale_factor_median_sd']:.4f}")
        for mode in ['scaled', 'perframe']:
            mean = summary[f'{mode}_mean_mm_mean']
            std = summary[f'{mode}_std_mm_mean']
            rmse = summary[f'{mode}_rmse_mm_mean']
            med = summary[f'{mode}_median_mm_mean']
            print(f"  [{mode:8s}] Mean±Std = {mean:>7.2f} ± {std:>5.2f} mm   "
                  f"RMSE = {rmse:>7.2f}   Median = {med:>7.2f}")
        print("=" * 72)
        print(f"  输出目录: {OUT_DIR}")


if __name__ == '__main__':
    main()
