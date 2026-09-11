"""组织运动绝对位姿差值评估：预测世界坐标 vs GT 世界坐标（逐帧 3D 位置误差）。

思路（用户指定）：
  test_v6_dyendovo.py 末尾已能生成运动点的绝对位姿（每帧世界坐标 x/y/z）。
  本脚本把"预测的绝对位姿"与"GT 绝对位姿"逐帧做差，得到 Mean ± Std。

绝对位姿来源：
  [预测] VO 位姿 + 预测深度 反投影 → VO 世界坐标 → 逐帧 VO→GT 对齐
  [GT]   GT 位姿 + GT 深度(tiff) 反投影 → GT 世界坐标

逐帧对齐（与 test_v6_dyendovo.py visualize_motion_vs_gt 一致）：
  T_vo_to_gt[i] = gt_poses[i] @ abs_poses_vo[i]   (VO_world → GT_world)

评估流程：
  1. chain_tracks: 串联相邻帧 LoFTR 匹配形成跨帧 track
  2. _classify_by_displacement: 自适应 3D 位移阈值分类运动/静止 track
  3. 对每个运动 track 的每一帧:
       p_vo  = backproject(u, v, pred_depth, VO_pose)     # VO 世界坐标
       p_gt  = backproject(u, v, gt_depth,   GT_pose_w2c) # GT 世界坐标
       p_vo_aligned = R_i @ p_vo + t_i                     # 对齐到 GT 坐标系
       err_i = ||p_vo_aligned - p_gt||                     # 3D 位置误差 (mm)
  4. 汇总所有帧误差 → Mean ± Std / RMSE / Median / P90

指标（跨 5 序列 Mean ± Std）：
  - 3D 位置误差 mean/median/rmse (mm)
  - 分静止点(GT) / 运动点(GT) 两组分别统计

数据依赖（每序列）：
  - {seq}/baseline_chain_data.npz   LoFTR 匹配点对 k0/k1 + K + n_pairs
  - {seq}/baseline_abs_poses.npy    VO 绝对位姿 (N,4,4) world_to_cam
  - {seq}/baseline_depth_maps.npz   预测深度图 {frame: (1080,1350) mm}
  - {seq}/pose.txt                  GT 位姿 (order='F' => cam_to_world, mm)
  - {seq}/depth/{i:04d}_depth.tiff  GT 深度 (uint16 -> 0-100mm)

用法:
  python zhong/eval_pipeline_pose_vs_gt.py --seq c1_transverse1_t1_v2 --limit 30
  python zhong/eval_pipeline_pose_vs_gt.py
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
                       'vo_depth_compare', 'pose')

K = np.array([[767.73, 0, 677.74], [0, 767.73, 543.06], [0, 0, 1]], dtype=np.float64)

CHAIN_DIST_THRESH = 3.0
DISP_K_FACTOR = 0.5
MAX_TRACKS = 15000
SAMPLE_MAX = 5000
N_STATIC_SAMPLE = 3000


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


def backproject_to_world(u, v, depth_map, T_world_cam):
    """反投影 2D 点到世界 3D 坐标 (mm)，无效返回 None。"""
    if depth_map is None:
        return None
    h, w = depth_map.shape
    ui = int(np.clip(round(u), 0, w - 1))
    vi = int(np.clip(round(v), 0, h - 1))
    Z = float(depth_map[vi, ui])
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


def evaluate_sequence(seq_name):
    seq_dir = os.path.join(DATA_ROOT, seq_name)
    chain_path = os.path.join(seq_dir, 'baseline_chain_data.npz')
    abs_poses_path = os.path.join(seq_dir, 'baseline_abs_poses.npy')
    depth_path = os.path.join(seq_dir, 'baseline_depth_maps.npz')

    print(f"\n{'='*64}")
    print(f"  [组织运动绝对位姿差值评估] {seq_name}")
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

    # 深度图一次性加载到内存 dict
    depth_maps = {k: depth_npz[k] for k in depth_npz.files if k.startswith('0') or k.startswith('1')}
    print(f"  预测深度帧已加载: {len(depth_maps)}")

    # 逐帧 VO->GT 对齐: T_vo_to_gt[i] = gt_poses[i] @ abs_poses[i]
    def vo_to_gt(i):
        T = gt_poses[i] @ abs_poses[i]
        return T[:3, :3], T[:3, 3]

    # GT 深度缓存
    gt_depth_cache = {}

    def get_gt_depth(fi):
        if fi not in gt_depth_cache:
            gt_depth_cache[fi] = load_gt_depth(seq_dir, fi)
        return gt_depth_cache[fi]

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

    # Step B: 运动/静止分类
    print("  [B] 运动/静止分类 ...")
    static_tracks, moving_tracks, threshold = classify_by_displacement(
        tracks, depth_maps, abs_poses, k_factor=k_factor)
    print(f"  静止={len(static_tracks)} 运动={len(moving_tracks)}")

    gen_dir = os.path.join(seq_dir, 'generated')

    # ── 逐帧绝对位姿误差 ──
    err_all = []       # 所有运动点帧误差 (mm)
    err_gt_static = []  # GT 判静止点的帧误差
    err_gt_moving = []  # GT 判运动点的帧误差

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
            # 预测绝对位姿（VO 世界坐标）
            p_vo = backproject_to_world(u, v, pred_depth, abs_poses[frame_idx])
            # GT 绝对位姿（GT 世界坐标）
            p_gt = backproject_to_world(u, v, gt_depth, gt_poses_w2c[frame_idx])
            if p_vo is None or p_gt is None:
                continue
            # 逐帧对齐 VO -> GT
            R_i, t_i = vo_to_gt(frame_idx)
            p_vo_aligned = R_i @ p_vo + t_i
            err = float(np.linalg.norm(p_vo_aligned - p_gt))

            err_all.append(err)

    if not err_all:
        print("  [FAIL] 无有效评估帧")
        return None

    err_all = np.array(err_all)
    metrics = {
        'n_frames_eval': int(len(err_all)),
        'n_moving_tracks': len(moving_tracks),
        'threshold_mm': float(threshold),
        # 3D 绝对位姿误差
        'pose_err_mean_mm': float(np.mean(err_all)),
        'pose_err_std_mm': float(np.std(err_all)),
        'pose_err_median_mm': float(np.median(err_all)),
        'pose_err_rmse_mm': float(np.sqrt(np.mean(err_all ** 2))),
        'pose_err_p90_mm': float(np.percentile(err_all, 90)),
        'pose_err_p95_mm': float(np.percentile(err_all, 95)),
        'pose_err_max_mm': float(np.max(err_all)),
    }

    print(f"\n  ┌──────────────────────────────────────────────────────┐")
    print(f"  │  组织运动绝对位姿差值 (预测 vs GT)                  │")
    print(f"  ├──────────────────────────────────────────────────────┤")
    print(f"  │  评估帧数:                {len(err_all):>6d}                 │")
    print(f"  │  Mean ± Std:              {metrics['pose_err_mean_mm']:>7.2f} ± {metrics['pose_err_std_mm']:>5.2f} mm     │")
    print(f"  │  Median:                  {metrics['pose_err_median_mm']:>8.3f} mm            │")
    print(f"  │  RMSE:                    {metrics['pose_err_rmse_mm']:>8.3f} mm            │")
    print(f"  │  P90:                     {metrics['pose_err_p90_mm']:>8.3f} mm            │")
    print(f"  └──────────────────────────────────────────────────────┘")

    return metrics


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
            with open(os.path.join(OUT_DIR, f'pose_{seq}.json'), 'w') as f:
                json.dump(m, f, indent=2)

    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        keys = [
            'pose_err_mean_mm', 'pose_err_std_mm', 'pose_err_median_mm',
            'pose_err_rmse_mm', 'pose_err_p90_mm', 'pose_err_p95_mm',
            'pose_err_max_mm',
        ]
        summary = {'n_sequences': len(valid), 'sequences': list(valid.keys())}
        for k in keys:
            vals = np.array([v[k] for v in valid.values()], dtype=float)
            summary[f'{k}_mean'] = float(np.mean(vals))
            summary[f'{k}_std'] = float(np.std(vals))
        with open(os.path.join(OUT_DIR, 'pose_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("  组织运动绝对位姿差值  跨序列 Mean ± SD 汇总")
        print("=" * 70)
        for k in keys:
            m = summary[f'{k}_mean']
            s = summary[f'{k}_std']
            print(f"    {k:<24s}: {m:>10.4f} ± {s:>10.4f}")
        print("=" * 70)
        print(f"  输出目录: {OUT_DIR}")


if __name__ == '__main__':
    main()
