"""组织运动分析评估脚本：预测运动位移 vs GT motion_gt 顶点位移。

评估内容（对应《中国图象图形学报》组织运动分析指标）：
  1. 运动位移 RMSE vs GT（运动点 pred_disp vs motion_gt 位移，mm）
  2. Pearson r（预测位移 vs GT 位移相关性）
  3. 运动检测 Recall / Precision / F1（自适应阈值 vs mask_static 标签）
  4. 运动/静止分离度（运动点 vs 静止点 预测位移中位数之差）

数据依赖（每序列）：
  - {seq}/baseline_abs_poses.npy   VO 绝对位姿 (N,4,4) world_to_cam
  - {seq}/baseline_depth_maps.npz   预测深度图 {frame: (1080,1350) mm}
  - {seq}/baseline_chain_data.npz   匹配点对 k0/k1 + K + n_pairs
  - {seq}/pose.txt                  GT 位姿 cam_to_world
  - {seq}/generated/motion_gt/frame_{i:04d}.npy    顶点运动位移 (米)
  - {seq}/generated/vertex_static/frame_{i:04d}.npy 顶点世界坐标 (mm)
  - {seq}/generated/mask_static/frame_{i:04d}.npy   静止=1/运动=0

用法:
  python zhong/eval_motion_vs_gt.py
"""
import os
import sys
import json
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dyendovo_dataset import load_gt_poses

DATA_ROOT = 'F:/dataset'
SEQ_NAMES = [
    'c1_transverse1_t1_v2',
    'c1_transverse1_t1_v1',
    'c1_transverse1_t2_v1',
    'c1_transverse2_t1_v1',
    'c1_transverse2_t2_v1',
]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'vo_depth_compare', 'motion')

# 自适应阈值：位移 > median + k_factor * std 判为运动
K_FACTOR = 0.5


def compute_per_frame_vo_to_gt(gt_poses, abs_poses_vo):
    """逐帧计算 VO 世界坐标 -> GT 世界坐标 的刚性变换。

    T_vo_to_gt[i] = gt_poses[i] @ abs_poses_vo[i]
    其中 gt_poses[i] 是 cam_to_world_gt, abs_poses_vo[i] 是 world_to_cam_vo。
    组合后: VO_world -> VO_cam -> GT_cam -> GT_world。
    """
    n = min(len(gt_poses), len(abs_poses_vo))
    align = {}
    for i in range(n):
        T = gt_poses[i] @ abs_poses_vo[i]
        align[i] = (T[:3, :3].copy(), T[:3, 3].copy())
    return align


def backproject_batch(uv, depth_map, T_world_cam, K):
    """批量反投影 2D 点到世界 3D 坐标。

    Args:
        uv: (N,2) 像素坐标 (原始分辨率 1350x1080)
        depth_map: (H,W) 深度图 (mm)
        T_world_cam: (4,4) world_to_cam
        K: (3,3) 内参
    Returns:
        (N,3) 世界坐标 (mm)，无效点为 nan
    """
    h, w = depth_map.shape
    u = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, w - 1)
    v = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, h - 1)
    Z = depth_map[v, u].astype(np.float64)
    valid = (Z > 0.5) & np.isfinite(Z)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    p_cam = np.zeros((len(uv), 3), dtype=np.float64)
    p_cam[:, 0] = (uv[:, 0] - cx) * Z / fx
    p_cam[:, 1] = (uv[:, 1] - cy) * Z / fy
    p_cam[:, 2] = Z

    T_cam_world = np.linalg.inv(T_world_cam)
    p_world = (T_cam_world[:3, :3] @ p_cam.T).T + T_cam_world[:3, 3]
    p_world[~valid] = np.nan
    return p_world


def evaluate_sequence(seq_name):
    """对单个序列做组织运动 vs GT 评估。"""
    seq_dir = os.path.join(DATA_ROOT, seq_name)
    abs_poses_path = os.path.join(seq_dir, 'baseline_abs_poses.npy')
    depth_path = os.path.join(seq_dir, 'baseline_depth_maps.npz')
    chain_path = os.path.join(seq_dir, 'baseline_chain_data.npz')

    print(f"\n{'='*60}")
    print(f"  [组织运动评估] {seq_name}")
    print(f"{'='*60}")

    for label, p in [('VO位姿', abs_poses_path), ('深度图', depth_path),
                     ('匹配数据', chain_path)]:
        if not os.path.exists(p):
            print(f"  ⚠ {label} 不存在: {p}")
            return None
    print(f"  加载 VO 产物 ...")

    abs_poses = np.load(abs_poses_path)          # (N,4,4) world_to_cam
    depth_data = np.load(depth_path, allow_pickle=True)   # lazy
    chain = np.load(chain_path, allow_pickle=True)
    n_pairs = int(chain['n_pairs'])
    K = chain['K']

    gt_poses = load_gt_poses(seq_dir)            # list cam_to_world
    n_gt = len(gt_poses)

    print(f"  VO位姿帧数={len(abs_poses)}, GT位姿帧数={n_gt}, 匹配对={n_pairs}")

    # 逐帧 VO->GT 对齐
    vo_to_gt = compute_per_frame_vo_to_gt(gt_poses, abs_poses)

    gen_dir = os.path.join(seq_dir, 'generated')
    motion_dir = os.path.join(gen_dir, 'motion_gt')
    vertex_dir = os.path.join(gen_dir, 'vertex_static')
    mask_dir = os.path.join(gen_dir, 'mask_static')

    # 汇总容器
    all_pred_disp = []    # 预测运动位移 (mm, VO 世界坐标模长)
    all_gt_disp = []      # GT 运动位移 (mm, 米->毫米)
    all_gt_moving = []    # GT 运动标签 (bool)
    all_fi = []           # 帧对索引（用于逐帧对阈值）

    n_processed = 0

    for fi in range(n_pairs):
        k0_key = f'k0_{fi:04d}'
        k1_key = f'k1_{fi:04d}'
        if k0_key not in chain or k1_key not in chain:
            continue
        # 深度帧键
        d0_key = f'{fi:04d}'
        d1_key = f'{fi+1:04d}'
        if d0_key not in depth_data or d1_key not in depth_data:
            continue
        if fi >= len(abs_poses) or fi + 1 >= len(abs_poses):
            continue
        if fi not in vo_to_gt:
            continue

        # GT 文件
        motion_path = os.path.join(motion_dir, f'frame_{fi:04d}.npy')
        vertex_path = os.path.join(vertex_dir, f'frame_{fi:04d}.npy')
        mask_path = os.path.join(mask_dir, f'frame_{fi:04d}.npy')
        if not (os.path.exists(motion_path) and os.path.exists(vertex_path)
                and os.path.exists(mask_path)):
            continue

        k0 = chain[k0_key]
        k1 = chain[k1_key]
        depth_curr = depth_data[d0_key]
        depth_next = depth_data[d1_key]
        pose_curr = abs_poses[fi]
        pose_next = abs_poses[fi + 1]

        # ── 预测运动位移（VO 世界坐标） ──
        pw0_vo = backproject_batch(k0, depth_curr, pose_curr, K)   # (N,3)
        pw1_vo = backproject_batch(k1, depth_next, pose_next, K)
        valid = ~(np.isnan(pw0_vo).any(axis=1) | np.isnan(pw1_vo).any(axis=1))
        if valid.sum() < 10:
            continue

        pred_disp = np.linalg.norm(pw1_vo - pw0_vo, axis=1)

        # ── 对齐到 GT 世界坐标 + cKDTree 匹配顶点 ──
        motion_gt = np.load(motion_path)       # (V,3) 米
        vertex_static = np.load(vertex_path)   # (V,3) mm
        mask_static = np.load(mask_path)       # (V,) 1=静止 0=运动

        R_align, t_align = vo_to_gt[fi]
        pw0_gt = (R_align @ pw0_vo.T).T + t_align

        tree = cKDTree(vertex_static)
        _, idxs = tree.query(pw0_gt[valid])

        gt_disp = np.linalg.norm(motion_gt[idxs], axis=1) * 1000.0   # 米->毫米
        gt_moving = mask_static[idxs] <= 0.5

        all_pred_disp.append(pred_disp[valid])
        all_gt_disp.append(gt_disp)
        all_gt_moving.append(gt_moving)
        all_fi.append(fi)
        n_processed += 1

        if (fi + 1) % 100 == 0:
            print(f"    进度: {fi+1}/{n_pairs} 帧对")

    if not all_pred_disp:
        print("  ⚠ 无有效评估数据")
        return None

    pred_disp = np.concatenate(all_pred_disp)
    gt_disp = np.concatenate(all_gt_disp)
    gt_moving = np.concatenate(all_gt_moving)

    n_total = len(pred_disp)
    n_moving = int(gt_moving.sum())
    n_static = n_total - n_moving
    print(f"  评估点总数={n_total}, GT运动点={n_moving}, GT静止点={n_static}")

    # ═══════════════════════════════════════════
    # 1. 运动位移 RMSE / MAE（运动点）
    # ═══════════════════════════════════════════
    moving_mask = gt_moving
    static_mask = ~gt_moving

    pred_m = pred_disp[moving_mask]
    gt_m = gt_disp[moving_mask]
    pred_s = pred_disp[static_mask]

    # 中位数尺度归一化（消除 VO/GT 尺度差异，稳健）
    if len(pred_m) > 0 and np.median(pred_m) > 1e-6:
        scale = np.median(gt_m) / (np.median(pred_m) + 1e-8)
    else:
        scale = 1.0

    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) > 0 else 0.0

    def _mae(a, b):
        return float(np.mean(np.abs(a - b))) if len(a) > 0 else 0.0

    metrics = {
        'n_total': n_total,
        'n_moving': n_moving,
        'n_static': n_static,
        'scale_median': float(scale),
        # 运动点：预测位移 vs GT 位移
        'motion_rmse_mm': _rmse(pred_m * scale, gt_m),
        'motion_mae_mm': _mae(pred_m * scale, gt_m),
        'motion_rmse_raw_mm': _rmse(pred_m, gt_m),
        # 静止点：预测位移基线（应≈0，反映位姿误差）
        'static_disp_median_mm': float(np.median(pred_s)) if len(pred_s) > 0 else 0.0,
        'static_disp_mean_mm': float(np.mean(pred_s)) if len(pred_s) > 0 else 0.0,
        # 运动/静止分离度
        'moving_disp_median_mm': float(np.median(pred_m)) if len(pred_m) > 0 else 0.0,
        'separation_mm': float(np.median(pred_m) - np.median(pred_s))
                         if (len(pred_m) > 0 and len(pred_s) > 0) else 0.0,
        # GT 位移统计
        'gt_moving_median_mm': float(np.median(gt_m)) if len(gt_m) > 0 else 0.0,
        'gt_moving_mean_mm': float(np.mean(gt_m)) if len(gt_m) > 0 else 0.0,
    }

    # ═══════════════════════════════════════════
    # 2. Pearson r（预测 vs GT）
    # ═══════════════════════════════════════════
    def _pearson(a, b):
        if len(a) < 3:
            return 0.0
        a = np.asarray(a); b = np.asarray(b)
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    metrics['pearson_r_all'] = _pearson(pred_disp, gt_disp)
    metrics['pearson_r_moving'] = _pearson(pred_m, gt_m)

    # ═══════════════════════════════════════════
    # 3. 运动检测 Recall / Precision / F1（逐帧对自适应阈值）
    # ═══════════════════════════════════════════
    tp = fp = fn = tn = 0
    offset = 0
    for i, disp_arr in enumerate(all_pred_disp):
        n = len(disp_arr)
        sl = slice(offset, offset + n)
        disp = pred_disp[sl]
        lbl = gt_moving[sl]
        if len(disp) < 10:
            offset += n
            continue
        med = float(np.median(disp))
        std = float(np.std(disp))
        threshold = med + K_FACTOR * std
        pred_moving = disp > threshold
        tp += int((pred_moving & lbl).sum())
        fp += int((pred_moving & ~lbl).sum())
        fn += int((~pred_moving & lbl).sum())
        tn += int((~pred_moving & ~lbl).sum())
        offset += n

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics.update({
        'detection_tp': tp, 'detection_fp': fp, 'detection_fn': fn, 'detection_tn': tn,
        'detection_recall': float(recall),
        'detection_precision': float(precision),
        'detection_f1': float(f1),
    })

    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  组织运动 vs GT (motion_gt) 评估          ║")
    print(f"  ╠══════════════════════════════════════════╣")
    print(f"  ║  运动点 RMSE:   {metrics['motion_rmse_mm']:>8.3f} mm           ║")
    print(f"  ║  运动点 MAE:    {metrics['motion_mae_mm']:>8.3f} mm           ║")
    print(f"  ║  Pearson r(全): {metrics['pearson_r_all']:>8.3f}               ║")
    print(f"  ║  Pearson r(动): {metrics['pearson_r_moving']:>8.3f}               ║")
    print(f"  ║  Recall:        {metrics['detection_recall']:>8.1%}               ║")
    print(f"  ║  Precision:     {metrics['detection_precision']:>8.1%}               ║")
    print(f"  ║  F1:            {metrics['detection_f1']:>8.3f}               ║")
    print(f"  ║  分离度(中位):  {metrics['separation_mm']:>8.3f} mm           ║")
    print(f"  ╚══════════════════════════════════════════╝")

    return metrics


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for seq in SEQ_NAMES:
        try:
            m = evaluate_sequence(seq)
            results[seq] = m
        except Exception as e:
            print(f"  ✗ {seq} 评估失败: {e}")
            import traceback
            traceback.print_exc()
            results[seq] = None

    # ── 保存单序列结果 ──
    for seq, m in results.items():
        if m is not None:
            out_path = os.path.join(OUT_DIR, f'motion_{seq}.json')
            with open(out_path, 'w') as f:
                json.dump(m, f, indent=2)
            print(f"  已保存: {out_path}")

    # ── 跨序列 mean±std 汇总 ──
    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        keys = [
            'motion_rmse_mm', 'motion_mae_mm', 'motion_rmse_raw_mm',
            'pearson_r_all', 'pearson_r_moving',
            'detection_recall', 'detection_precision', 'detection_f1',
            'static_disp_median_mm', 'static_disp_mean_mm',
            'moving_disp_median_mm', 'separation_mm',
            'gt_moving_median_mm', 'gt_moving_mean_mm',
        ]
        summary = {'n_sequences': len(valid), 'sequences': list(valid.keys())}
        for k in keys:
            vals = np.array([v[k] for v in valid.values()], dtype=float)
            summary[f'{k}_mean'] = float(np.mean(vals))
            summary[f'{k}_std'] = float(np.std(vals))
        out_path = os.path.join(OUT_DIR, 'motion_multi_seq_summary.json')
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  已保存汇总: {out_path}")

        print("\n" + "="*70)
        print("  组织运动分析 跨序列 Mean ± SD 汇总")
        print("="*70)
        for k in keys:
            m = summary[f'{k}_mean']
            s = summary[f'{k}_std']
            print(f"    {k:<24s}: {m:>10.4f} ± {s:>10.4f}")
        print("="*70)


if __name__ == '__main__':
    main()
