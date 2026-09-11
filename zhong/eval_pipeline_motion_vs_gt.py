"""组织运动端到端评估：管线运动点累计位移 vs GT 顶点位移（直接差值）。

思路（用户指定）：
  不再用逐帧对的间接信号（VO 位姿反投影 / 残差光流），而是直接取管线
  端到端走完后输出的组织运动结果 —— 链式追踪 track 的累计位移 —— 与
  GT motion_gt 顶点位移做差值，评估两者误差。

端到端流程（复现 test_v6_dyendovo.py 的运动轨迹管线）：
  1. chain_tracks: 串联相邻帧 LoFTR 匹配形成跨帧 track
  2. _classify_by_displacement: 自适应 3D 位移阈值分类运动/静止 track
  3. 对每个运动 track: pred = ||p_last - p_first|| (VO 世界坐标累计位移 mm)
  4. 逐帧 VO->GT 对齐 + cKDTree 匹配 GT mesh 顶点
  5. GT 位移 = ||motion_gt[last] - motion_gt[first]|| * 1000 (同窗口帧间差分 mm)
  6. 直接差值: RMSE / MAE / Pearson r / bias / 分离度

口径说明（关键改进）：
  - pred: track 首帧→末帧位移（VO 位姿 + 预测深度反投影）
  - gt:  GT 顶点在同一时间窗口 [first, last] 的位移（motion_gt 帧间差分）
  - 两者都是"同一时间窗口的累计位移"，口径对齐。

数据依赖（每序列）：
  - {seq}/baseline_chain_data.npz   LoFTR 匹配点对 k0/k1 + K + n_pairs
  - {seq}/baseline_abs_poses.npy    VO 绝对位姿 (N,4,4) world_to_cam
  - {seq}/baseline_depth_maps.npz   预测深度图 {frame: (1080,1350) mm}
  - {seq}/pose.txt                  GT 位姿 (order='F' => cam_to_world, mm)
  - {seq}/generated/motion_gt/frame_{i:04d}.npy    顶点累计位移 (米)
  - {seq}/generated/vertex_static/frame_{i:04d}.npy 顶点世界坐标 (mm)
  - {seq}/generated/mask_static/frame_{i:04d}.npy   静止=1/运动=0

用法:
  python zhong/eval_pipeline_motion_vs_gt.py --limit 30            # 单序列快速验证
  python zhong/eval_pipeline_motion_vs_gt.py --seq c1_transverse1_t1_v2
  python zhong/eval_pipeline_motion_vs_gt.py                       # 全量 5 序列
"""
import os
import sys
import json
import numpy as np
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
                       'vo_depth_compare', 'pipeline_motion')

K = np.array([[767.73, 0, 677.74], [0, 767.73, 543.06], [0, 0, 1]], dtype=np.float64)

CHAIN_DIST_THRESH = 3.0    # 跨帧匹配距离阈值 (像素)
DISP_K_FACTOR = 0.5        # 自适应阈值: median + k * std
MAX_TRACKS = 15000         # track 数上限（优先长 track）
SAMPLE_MAX = 5000          # 阈值估计采样数
N_STATIC_SAMPLE = 3000     # 分离度基线: 静止 track 采样数


def load_gt_poses_orderF(seq_dir):
    """读取 pose.txt, order='F' reshape => cam_to_world (mm)。"""
    with open(os.path.join(seq_dir, 'pose.txt')) as f:
        lines = f.readlines()
    return [np.array([float(x) for x in ln.strip().split(',')]).reshape(4, 4, order='F')
            for ln in lines]


def chain_tracks(all_k0, all_k1, dist_thresh=CHAIN_DIST_THRESH):
    """串联相邻帧 LoFTR 匹配, 形成跨帧 track (cKDTree 加速版)。

    Returns:
        tracks: list of dict {'id', 'frames': [(frame_idx, u, v), ...]}
    """
    n_pairs = len(all_k0)
    tracks = []
    active = {}   # tid -> (frame, u, v)
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

        # 贪心: 按距离升序, 每个 active track 至多延续一个 k0
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

        # 未匹配的 k0 点建新 track
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
    """计算 track 的逐帧世界坐标 + 累计位移 + 相邻帧位移中位数。

    Returns:
        p_worlds: list of (3,) or None
        total_disp: float (首末帧位移 mm)
        med_step: float (相邻帧位移中位数 mm)
    """
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
        return p_worlds, 0.0, 0.0
    total_disp = float(np.linalg.norm(valid[-1] - valid[0]))
    steps = [np.linalg.norm(valid[i + 1] - valid[i]) for i in range(len(valid) - 1)]
    return p_worlds, total_disp, float(np.median(steps))


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
        _, _, med = track_disp_and_med(t, depth_maps, abs_poses)
        meds.append(med)
    meds = np.array(meds)
    med = float(np.median(meds))
    std = float(np.std(meds))
    threshold = med + k_factor * std
    print(f"  自适应阈值: median={med:.2f}mm std={std:.2f}mm threshold={threshold:.2f}mm")

    static_tracks, moving_tracks = [], []
    for t in tracks:
        _, total, med_step = track_disp_and_med(t, depth_maps, abs_poses)
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
    print(f"  [端到端组织运动差值评估] {seq_name}")
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
    # 链式追踪阈值（沿用管线保存的自适应参数）
    chain_dist_thresh = float(chain['chain_dist_thresh']) if 'chain_dist_thresh' in chain else CHAIN_DIST_THRESH
    k_factor = float(chain['k_factor']) if 'k_factor' in chain else DISP_K_FACTOR
    gt_poses = load_gt_poses_orderF(seq_dir)
    n_gt = len(gt_poses)
    print(f"  n_pairs={n_pairs} VO位姿={abs_poses.shape} GT位姿={n_gt} chain_thresh={chain_dist_thresh}")

    all_k0 = [chain[f'k0_{i:04d}'] for i in range(n_pairs)]
    all_k1 = [chain[f'k1_{i:04d}'] for i in range(n_pairs)]

    # 深度图一次性加载到内存 dict（避免 npz 反复解压）
    depth_maps = {k: depth_npz[k] for k in depth_npz.files if k.startswith('0') or k.startswith('1')}
    print(f"  深度帧已加载: {len(depth_maps)}")

    # ── 深度尺度恢复 ──
    # chain_data.npz 里的深度是 ×global_scale 的轨迹尺度深度;
    # 组织运动反投影需原始深度尺度, 故在此 ÷global_scale 恢复。
    # depth_source=='gt' 时深度未乘 global_scale, 无需恢复。
    global_scale = float(chain['global_depth_scale']) if 'global_depth_scale' in chain else 1.0
    depth_src = str(chain['depth_source']) if 'depth_source' in chain else 'pred'
    if depth_src != 'gt' and global_scale != 1.0:
        for k in depth_maps:
            depth_maps[k] = depth_maps[k] / global_scale
        print(f"  深度尺度恢复: ÷global_scale={global_scale:.4f} (组织运动用原始深度尺度)")

    # 逐帧 VO->GT 对齐: T_vo_to_gt[i] = gt_poses[i] @ abs_poses[i]
    def vo_to_gt(i):
        T = gt_poses[i] @ abs_poses[i]
        return T[:3, :3], T[:3, 3]

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
    motion_dir = os.path.join(gen_dir, 'motion_gt')
    vertex_dir = os.path.join(gen_dir, 'vertex_static')
    mask_dir = os.path.join(gen_dir, 'mask_static')

    # ── 收集运动 track 的 pred/gt 差值 ──
    # 按 f_first 分组处理，控制顶点云内存峰值
    by_first = {}
    for t, med_step in moving_tracks:
        if len(t['frames']) < 2:
            continue
        f_first = t['frames'][0][0]
        f_last = t['frames'][-1][0]
        if f_first >= n_gt or f_last >= n_gt:
            continue
        if f_first == f_last:
            continue
        by_first.setdefault(f_first, []).append((t, med_step))

    pred_disp_all = []
    gt_disp_all = []
    gt_moving_all = []
    pred_moving_all = []  # 运动 track 的 pred（用于分离度）

    n_eval = 0
    for f_first, items in sorted(by_first.items()):
        vertex_path = os.path.join(vertex_dir, f'frame_{f_first:04d}.npy')
        motion_first_path = os.path.join(motion_dir, f'frame_{f_first:04d}.npy')
        if not (os.path.exists(vertex_path) and os.path.exists(motion_first_path)):
            continue
        vertex_first = np.load(vertex_path)
        motion_first = np.load(motion_first_path)
        R_align, t_align = vo_to_gt(f_first)
        tree = cKDTree(vertex_first)

        # 该组需要加载的 f_last 帧的 motion/mask（按需缓存）
        cache = {}
        for t, med_step in items:
            f_last = t['frames'][-1][0]
            u0, v0 = t['frames'][0][1], t['frames'][0][2]
            u1, v1 = t['frames'][-1][1], t['frames'][-1][2]
            # pred: VO 世界坐标累计位移
            p0 = backproject_to_world(u0, v0, depth_maps.get(f'{f_first:04d}'),
                                      abs_poses[f_first])
            p1 = backproject_to_world(u1, v1, depth_maps.get(f'{f_last:04d}'),
                                      abs_poses[f_last])
            if p0 is None or p1 is None:
                continue
            pred = float(np.linalg.norm(p1 - p0))

            # GT: 对齐首帧 VO 点到 GT 坐标, 匹配顶点
            p0_gt = R_align @ p0 + t_align
            _, idx = tree.query(p0_gt)
            if f_last not in cache:
                mp = os.path.join(motion_dir, f'frame_{f_last:04d}.npy')
                sp = os.path.join(mask_dir, f'frame_{f_last:04d}.npy')
                if not (os.path.exists(mp) and os.path.exists(sp)):
                    cache[f_last] = None
                else:
                    cache[f_last] = (np.load(mp), np.load(sp))
            if cache[f_last] is None:
                continue
            motion_last, mask_last = cache[f_last]
            gt_vec = (motion_last[idx] - motion_first[idx]) * 1000.0  # 米->mm
            gt = float(np.linalg.norm(gt_vec))
            gt_moving = bool(mask_last[idx] <= 0.5)

            pred_disp_all.append(pred)
            gt_disp_all.append(gt)
            gt_moving_all.append(gt_moving)
            pred_moving_all.append(pred)
            n_eval += 1

    # ── 静止 track 的 pred 位移（分离度基线，采样）──
    import random
    random.seed(42)
    static_sample = (random.sample(static_tracks, min(N_STATIC_SAMPLE, len(static_tracks)))
                     if static_tracks else [])
    pred_static_all = []
    for t, med_step in static_sample:
        if len(t['frames']) < 2:
            continue
        f_first = t['frames'][0][0]
        f_last = t['frames'][-1][0]
        if f_first >= n_gt or f_last >= n_gt or f_first == f_last:
            continue
        p0 = backproject_to_world(t['frames'][0][1], t['frames'][0][2],
                                  depth_maps.get(f'{f_first:04d}'), abs_poses[f_first])
        p1 = backproject_to_world(t['frames'][-1][1], t['frames'][-1][2],
                                  depth_maps.get(f'{f_last:04d}'), abs_poses[f_last])
        if p0 is None or p1 is None:
            continue
        pred_static_all.append(float(np.linalg.norm(p1 - p0)))

    if n_eval == 0:
        print("  [FAIL] 无有效评估 track")
        return None

    pred = np.array(pred_disp_all)
    gt = np.array(gt_disp_all)
    gt_moving = np.array(gt_moving_all)
    pred_moving = np.array(pred_moving_all)
    pred_static = np.array(pred_static_all) if pred_static_all else np.array([])

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) > 0 else 0.0

    def _mae(a, b):
        return float(np.mean(np.abs(a - b))) if len(a) > 0 else 0.0

    def _pearson(a, b):
        a = np.asarray(a); b = np.asarray(b)
        if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    bias = float(np.mean(pred - gt))
    # 运动 track 中 GT 也判为运动的占比
    gt_true_moving_ratio = float(np.mean(gt_moving)) if len(gt_moving) > 0 else 0.0

    metrics = {
        'n_moving_tracks_eval': int(n_eval),
        'n_static_tracks_sampled': int(len(pred_static)),
        'threshold_mm': float(threshold),
        # 直接差值指标（核心）
        'rmse_mm': _rmse(pred, gt),
        'mae_mm': _mae(pred, gt),
        'bias_mm': bias,                      # mean(pred - gt)
        'bias_abs_mm': float(np.mean(np.abs(pred - gt))),
        # 相对误差
        'rel_mae': _mae(pred, gt) / (float(np.mean(gt)) + 1e-8),
        # 相关性
        'pearson_r': _pearson(pred, gt),
        'pearson_r_moving_gt': _pearson(pred[gt_moving], gt[gt_moving])
                               if gt_moving.sum() > 0 else 0.0,
        # 分布
        'pred_median_mm': float(np.median(pred)),
        'gt_median_mm': float(np.median(gt)),
        'pred_mean_mm': float(np.mean(pred)),
        'gt_mean_mm': float(np.mean(gt)),
        'pred_std_mm': float(np.std(pred)),
        'gt_std_mm': float(np.std(gt)),
        # 运动/静止分离度
        'moving_disp_median_mm': float(np.median(pred_moving)),
        'static_disp_median_mm': float(np.median(pred_static)) if len(pred_static) > 0 else 0.0,
        'separation_mm': float(np.median(pred_moving) - np.median(pred_static))
                         if len(pred_static) > 0 else 0.0,
        # 运动点 GT 真运动占比（我们判的运动点里, GT 也判运动的比例）
        'gt_true_moving_ratio': gt_true_moving_ratio,
    }

    print(f"\n  ┌──────────────────────────────────────────────────────┐")
    print(f"  │  端到端组织运动 vs GT 差值                          │")
    print(f"  ├──────────────────────────────────────────────────────┤")
    print(f"  │  运动 track 评估数:        {n_eval:>6d}                 │")
    print(f"  │  RMSE (pred vs GT):       {metrics['rmse_mm']:>8.3f} mm            │")
    print(f"  │  MAE  (pred vs GT):       {metrics['mae_mm']:>8.3f} mm            │")
    print(f"  │  Bias (pred - GT):        {metrics['bias_mm']:>+8.3f} mm            │")
    print(f"  │  Rel MAE:                 {metrics['rel_mae']:>8.3f}                │")
    print(f"  │  Pearson r:               {metrics['pearson_r']:>8.3f}                │")
    print(f"  │  pred 中位数:             {metrics['pred_median_mm']:>8.3f} mm            │")
    print(f"  │  GT   中位数:             {metrics['gt_median_mm']:>8.3f} mm            │")
    print(f"  │  分离度 (动-静):          {metrics['separation_mm']:>8.3f} mm            │")
    print(f"  │  GT 真运动占比:           {metrics['gt_true_moving_ratio']:>8.1%}                │")
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
            with open(os.path.join(OUT_DIR, f'pipeline_motion_{seq}.json'), 'w') as f:
                json.dump(m, f, indent=2)

    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        keys = [
            'rmse_mm', 'mae_mm', 'bias_mm', 'bias_abs_mm', 'rel_mae',
            'pearson_r', 'pearson_r_moving_gt',
            'pred_median_mm', 'gt_median_mm', 'pred_mean_mm', 'gt_mean_mm',
            'pred_std_mm', 'gt_std_mm',
            'moving_disp_median_mm', 'static_disp_median_mm', 'separation_mm',
            'gt_true_moving_ratio',
        ]
        summary = {'n_sequences': len(valid), 'sequences': list(valid.keys())}
        for k in keys:
            vals = np.array([v[k] for v in valid.values()], dtype=float)
            summary[f'{k}_mean'] = float(np.mean(vals))
            summary[f'{k}_std'] = float(np.std(vals))
        with open(os.path.join(OUT_DIR, 'pipeline_motion_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 70)
        print("  端到端组织运动 vs GT 差值  跨序列 Mean ± SD 汇总")
        print("=" * 70)
        for k in keys:
            m = summary[f'{k}_mean']
            s = summary[f'{k}_std']
            print(f"    {k:<24s}: {m:>10.4f} ± {s:>10.4f}")
        print("=" * 70)
        print(f"  输出目录: {OUT_DIR}")


if __name__ == '__main__':
    main()
