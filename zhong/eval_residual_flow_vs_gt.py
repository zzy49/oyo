"""组织运动分析：残差光流评估脚本。

残差光流 = 总光流(稀疏匹配点 k1-k0) - 相机光流(位姿+深度重投影)

这是对上一版"VO 位姿反投影位移"评估的替代方案：不再用 VO 位姿直接反投影
3D 位移（VO 位姿误差 ATE 6-8mm 淹没了组织运动信号），而是先在图像平面
做残差光流（扣除相机诱导光流），再将残差光流反投影为 3D 位移（mm）。

两套对照配置：
  [GT 补偿版 - 理想上界]  相机光流 = GT位姿(order='F' cam_to_world) + GT深度
  [VO 补偿版 - 我们方法]  相机光流 = VO位姿(world_to_cam) + 预测深度

正确 T_rel 方向（已由探针6 实证验证，resid < total）：
  GT: T_rel = inv(gt[i+1]) @ gt[i]   （cam_i -> cam_j）
  VO: T_rel = vo[i+1] @ inv(vo[i])   （cam_i -> cam_j）

评估指标（跨帧对汇总，Mean ± SD 跨 5 序列）：
  1. 残差光流 3D 位移 vs GT 位移 RMSE/MAE（moving 点，vs 帧间差分）
  2. Pearson r（残差 3D 位移 vs GT 累计位移 / 帧间差分）
  3. 运动检测 Recall / Precision / F1（逐帧对自适应阈值 vs mask_static）
  4. 运动/静止分离度（moving vs static 残差 3D 位移中位数差，mm）

数据依赖（每序列）：
  - {seq}/baseline_chain_data.npz   匹配点对 k0/k1 (LoFTR, 原始分辨率)
  - {seq}/baseline_abs_poses.npy    VO 绝对位姿 (N,4,4) world_to_cam
  - {seq}/baseline_depth_maps.npz   预测深度图 {frame: (1080,1350) mm}
  - {seq}/pose.txt                  GT 位姿 (order='F' => cam_to_world, mm)
  - {seq}/depth/{i:04d}_depth.tiff  GT 深度 (uint16 -> mm)
  - {seq}/generated/motion_gt/frame_{i:04d}.npy   顶点运动位移 (米)
  - {seq}/generated/vertex_static/frame_{i:04d}.npy 顶点世界坐标 (mm)
  - {seq}/generated/mask_static/frame_{i:04d}.npy   静止=1/运动=0

用法:
  python zhong/eval_residual_flow_vs_gt.py
"""
import os
import sys
import json
import numpy as np
import cv2
from scipy.spatial import cKDTree

# 快速验证：python zhong/eval_residual_flow_vs_gt.py --limit 20 [--seq 序列名]
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
                       'vo_depth_compare', 'residual_flow')

# 自适应检测阈值：位移 > median + k_factor * std 判为运动
K_FACTOR = 0.5

K = np.array([[767.73, 0, 677.74], [0, 767.73, 543.06], [0, 0, 1]], dtype=np.float64)
FX, FY, CX, CY = K[0, 0], K[1, 1], K[0, 2], K[1, 2]


def load_gt_poses_orderF(seq_dir):
    """读取 pose.txt，order='F' reshape => cam_to_world (mm)。"""
    pose_path = os.path.join(seq_dir, 'pose.txt')
    with open(pose_path) as f:
        lines = f.readlines()
    return [np.array([float(x) for x in ln.strip().split(',')]).reshape(4, 4, order='F')
            for ln in lines]


def load_gt_depth(seq_dir, fi):
    """加载 GT 深度 TIFF，转为 mm。"""
    p = os.path.join(seq_dir, 'depth', f'{fi:04d}_depth.tiff')
    if not os.path.exists(p):
        return None
    d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return d.astype(np.float64) * (100.0 / 65535.0)


def camera_flow_at(k0, depth, T_rel):
    """k0 点（frame i 像素）的相机诱导光流（像素位移）。

    P = 反投影(k0, depth)  ->  P' = T_rel @ P  ->  flow = project(P') - k0
    T_rel: cam_i -> cam_j 的 4x4 变换。
    """
    ui = np.clip(np.round(k0[:, 0]).astype(np.int64), 0, 1349)
    vi = np.clip(np.round(k0[:, 1]).astype(np.int64), 0, 1079)
    Z = depth[vi, ui]
    valid = Z > 0.5
    X = (k0[:, 0] - CX) * Z / FX
    Y = (k0[:, 1] - CY) * Z / FY
    P = np.stack([X, Y, Z], axis=-1)
    R = T_rel[:3, :3]
    t = T_rel[:3, 3]
    Pn = (R @ P.T + t.reshape(3, 1)).T
    Zn = Pn[:, 2]
    ok = Zn > 1e-6
    uv = np.zeros((len(Pn), 2), dtype=np.float64)
    uv[ok, 0] = FX * Pn[ok, 0] / Zn[ok] + CX
    uv[ok, 1] = FY * Pn[ok, 1] / Zn[ok] + CY
    flow = uv - k0
    flow[~(valid & ok)] = np.nan
    return flow


def align_to_gt(k0, d_gt, T_cw, gen_dir, fi):
    """把 k0 点反投影到 GT 世界坐标，cKDTree 匹配 mesh 顶点。

    Returns:
        moving: (N,) bool  运动标签 (mask_static <= 0.5)
        gt_cum: (N,) mm    GT 累计位移 ||motion_i||
        gt_frm: (N,) mm    GT 帧间差分 ||motion_{i+1} - motion_i||
        valid: (N,) bool   深度有效 + 匹配成功
    """
    ui = np.clip(np.round(k0[:, 0]).astype(np.int64), 0, 1349)
    vi = np.clip(np.round(k0[:, 1]).astype(np.int64), 0, 1079)
    Z = d_gt[vi, ui]
    valid = Z > 0.5
    X = (k0[:, 0] - CX) * Z / FX
    Y = (k0[:, 1] - CY) * Z / FY
    Pc = np.stack([X, Y, Z], axis=-1)
    Pw = (T_cw[:3, :3] @ Pc.T).T + T_cw[:3, 3]

    vertex = np.load(os.path.join(gen_dir, 'vertex_static', f'frame_{fi:04d}.npy'))
    motion_i = np.load(os.path.join(gen_dir, 'motion_gt', f'frame_{fi:04d}.npy'))
    motion_j = np.load(os.path.join(gen_dir, 'motion_gt', f'frame_{fi+1:04d}.npy'))
    mask = np.load(os.path.join(gen_dir, 'mask_static', f'frame_{fi:04d}.npy'))

    tree = cKDTree(vertex)
    _, idx = tree.query(Pw[valid])
    # 全长度数组（无效点填 False/0，后续统一用 valid 索引）
    moving_full = np.zeros(len(k0), dtype=bool)
    gt_cum_full = np.zeros(len(k0), dtype=np.float64)
    gt_frm_full = np.zeros(len(k0), dtype=np.float64)
    moving_full[valid] = mask[idx] <= 0.5
    gt_cum_full[valid] = np.linalg.norm(motion_i[idx], axis=1) * 1000.0
    gt_frm_full[valid] = np.linalg.norm(motion_j[idx] - motion_i[idx], axis=1) * 1000.0
    return moving_full, gt_cum_full, gt_frm_full, valid


def evaluate_sequence(seq_name):
    seq_dir = os.path.join(DATA_ROOT, seq_name)
    chain_path = os.path.join(seq_dir, 'baseline_chain_data.npz')
    abs_poses_path = os.path.join(seq_dir, 'baseline_abs_poses.npy')
    depth_path = os.path.join(seq_dir, 'baseline_depth_maps.npz')

    print(f"\n{'='*70}")
    print(f"  [残差光流评估] {seq_name}")
    print(f"{'='*70}")

    for label, p in [('匹配数据', chain_path), ('VO位姿', abs_poses_path),
                     ('预测深度', depth_path)]:
        if not os.path.exists(p):
            print(f"  [WARN] {label} 不存在: {p}")
            return None

    chain = np.load(chain_path, allow_pickle=True)
    vo = np.load(abs_poses_path)                 # (N,4,4) world_to_cam
    dpz = np.load(depth_path, allow_pickle=True)  # 预测深度 (1080,1350) mm
    gt = load_gt_poses_orderF(seq_dir)            # list cam_to_world (mm)
    n_pairs = int(chain['n_pairs'])

    gen_dir = os.path.join(seq_dir, 'generated')
    print(f"  VO位姿帧数={len(vo)}, GT位姿帧数={len(gt)}, 匹配对={n_pairs}")

    # 汇总容器（按 配置 x 帧对）
    frames_gt = []   # 每帧对的 (resid_3d, gt_cum, gt_frm, moving, valid)
    frames_vo = []

    n_processed = 0
    fi_end = min(n_pairs, LIMIT) if LIMIT else n_pairs
    for fi in range(fi_end):
        k0_key = f'k0_{fi:04d}'
        k1_key = f'k1_{fi:04d}'
        d_key = f'{fi:04d}'
        if k0_key not in chain or k1_key not in chain:
            continue
        if d_key not in dpz:
            continue
        if fi + 1 >= len(gt) or fi + 1 >= len(vo):
            continue

        # GT 文件
        motion_path = os.path.join(gen_dir, 'motion_gt', f'frame_{fi:04d}.npy')
        vertex_path = os.path.join(gen_dir, 'vertex_static', f'frame_{fi:04d}.npy')
        mask_path = os.path.join(gen_dir, 'mask_static', f'frame_{fi:04d}.npy')
        if not (os.path.exists(motion_path) and os.path.exists(vertex_path)
                and os.path.exists(mask_path)):
            continue

        d_gt = load_gt_depth(seq_dir, fi)
        if d_gt is None:
            continue

        k0 = chain[k0_key]
        k1 = chain[k1_key]
        d_pred = dpz[d_key].astype(np.float64)

        total_flow = k1 - k0  # (N,2) 真实匹配位移

        # ── GT 补偿版相机光流（理想上界）──
        T_rel_gt = np.linalg.inv(gt[fi + 1]) @ gt[fi]   # cam_i -> cam_j
        cam_gt = camera_flow_at(k0, d_gt, T_rel_gt)
        resid_gt = total_flow - cam_gt                  # (N,2) 像素残差

        # ── VO 补偿版相机光流（我们方法）──
        T_rel_vo = vo[fi + 1] @ np.linalg.inv(vo[fi])   # cam_i -> cam_j
        cam_vo = camera_flow_at(k0, d_pred, T_rel_vo)
        resid_vo = total_flow - cam_vo

        # ── GT 对齐 ──
        moving, gt_cum, gt_frm, valid = align_to_gt(k0, d_gt, gt[fi], gen_dir, fi)
        valid &= np.isfinite(resid_gt).all(axis=1) & np.isfinite(resid_vo).all(axis=1)

        # 残差光流幅值 -> 3D 位移 (mm)（用各自深度反投影）
        Zg = d_gt[np.clip(np.round(k0[:, 1]).astype(np.int64), 0, 1079),
                  np.clip(np.round(k0[:, 0]).astype(np.int64), 0, 1349)]
        resid_gt_3d = np.linalg.norm(resid_gt, axis=1) * Zg / FX
        resid_vo_3d = np.linalg.norm(resid_vo, axis=1) * Zg / FX

        if valid.sum() < 10:
            continue

        frames_gt.append((resid_gt_3d[valid], gt_cum[valid], gt_frm[valid], moving[valid]))
        frames_vo.append((resid_vo_3d[valid], gt_cum[valid], gt_frm[valid], moving[valid]))
        n_processed += 1

        if (fi + 1) % 100 == 0:
            print(f"    进度: {fi+1}/{fi_end} 帧对")

    if n_processed == 0:
        print("  [WARN] 无有效评估数据")
        return None

    print(f"  处理帧对数={n_processed}")

    def compute_metrics(frames):
        resid_all = np.concatenate([f[0] for f in frames])
        gt_cum_all = np.concatenate([f[1] for f in frames])
        gt_frm_all = np.concatenate([f[2] for f in frames])
        moving_all = np.concatenate([f[3] for f in frames]).astype(bool)

        n_total = len(resid_all)
        n_moving = int(moving_all.sum())
        n_static = n_total - n_moving

        def _rmse(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) > 0 else 0.0

        def _mae(a, b):
            return float(np.mean(np.abs(a - b))) if len(a) > 0 else 0.0

        def _pearson(a, b):
            if len(a) < 3:
                return 0.0
            a = np.asarray(a); b = np.asarray(b)
            if np.std(a) < 1e-9 or np.std(b) < 1e-9:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        rm = resid_all[moving_all]
        rs = resid_all[~moving_all]
        gm = gt_frm_all[moving_all]

        # 中位数尺度归一化（残差光流幅值 vs GT 帧间位移）
        if len(rm) > 0 and np.median(rm) > 1e-6:
            scale = np.median(gm) / (np.median(rm) + 1e-8)
        else:
            scale = 1.0

        metrics = {
            'n_total': n_total,
            'n_moving': n_moving,
            'n_static': n_static,
            'scale_median': float(scale),
            # moving 点残差 vs GT 帧间位移
            'motion_rmse_mm': _rmse(rm * scale, gm),
            'motion_mae_mm': _mae(rm * scale, gm),
            'motion_rmse_raw_mm': _rmse(rm, gm),
            # 分离度
            'moving_resid_median_mm': float(np.median(rm)) if len(rm) > 0 else 0.0,
            'static_resid_median_mm': float(np.median(rs)) if len(rs) > 0 else 0.0,
            'separation_mm': float(np.median(rm) - np.median(rs))
                             if (len(rm) > 0 and len(rs) > 0) else 0.0,
            # Pearson r
            'pearson_r_cum': _pearson(resid_all, gt_cum_all),
            'pearson_r_frm': _pearson(resid_all, gt_frm_all),
            'pearson_r_frm_moving': _pearson(rm, gm),
            # GT 统计
            'gt_frm_moving_median_mm': float(np.median(gm)) if len(gm) > 0 else 0.0,
            'gt_cum_moving_median_mm': float(np.median(gt_cum_all[moving_all]))
                                       if len(gt_cum_all[moving_all]) > 0 else 0.0,
        }

        # 运动检测（逐帧对自适应阈值）
        tp = fp = fn = tn = 0
        for (resid, _, _, moving) in frames:
            if len(resid) < 10:
                continue
            med = float(np.median(resid))
            std = float(np.std(resid))
            thr = med + K_FACTOR * std
            pred_moving = resid > thr
            tp += int((pred_moving & moving).sum())
            fp += int((pred_moving & ~moving).sum())
            fn += int((~pred_moving & moving).sum())
            tn += int((~pred_moving & ~moving).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics.update({
            'detection_recall': float(recall),
            'detection_precision': float(precision),
            'detection_f1': float(f1),
        })
        return metrics

    metrics_gt = compute_metrics(frames_gt)
    metrics_vo = compute_metrics(frames_vo)

    print(f"\n  ┌──────────────────────────────────────────────┐")
    print(f"  │  GT 补偿版（理想上界）                       │")
    print(f"  ├──────────────────────────────────────────────┤")
    print(f"  │  分离度:        {metrics_gt['separation_mm']:>8.4f} mm        │")
    print(f"  │  Pearson r(帧): {metrics_gt['pearson_r_frm']:>8.4f}             │")
    print(f"  │  Pearson r(累): {metrics_gt['pearson_r_cum']:>8.4f}             │")
    print(f"  │  Recall:        {metrics_gt['detection_recall']:>8.1%}             │")
    print(f"  │  Precision:     {metrics_gt['detection_precision']:>8.1%}             │")
    print(f"  │  F1:            {metrics_gt['detection_f1']:>8.4f}             │")
    print(f"  │  RMSE(帧):      {metrics_gt['motion_rmse_mm']:>8.4f} mm        │")
    print(f"  └──────────────────────────────────────────────┘")
    print(f"  ┌──────────────────────────────────────────────┐")
    print(f"  │  VO 补偿版（我们方法）                       │")
    print(f"  ├──────────────────────────────────────────────┤")
    print(f"  │  分离度:        {metrics_vo['separation_mm']:>8.4f} mm        │")
    print(f"  │  Pearson r(帧): {metrics_vo['pearson_r_frm']:>8.4f}             │")
    print(f"  │  Pearson r(累): {metrics_vo['pearson_r_cum']:>8.4f}             │")
    print(f"  │  Recall:        {metrics_vo['detection_recall']:>8.1%}             │")
    print(f"  │  Precision:     {metrics_vo['detection_precision']:>8.1%}             │")
    print(f"  │  F1:            {metrics_vo['detection_f1']:>8.4f}             │")
    print(f"  │  RMSE(帧):      {metrics_vo['motion_rmse_mm']:>8.4f} mm        │")
    print(f"  └──────────────────────────────────────────────┘")

    return {'gt': metrics_gt, 'vo': metrics_vo}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for seq in SEQ_NAMES:
        try:
            m = evaluate_sequence(seq)
            results[seq] = m
        except Exception as e:
            print(f"  [FAIL] {seq} 评估失败: {e}")
            import traceback
            traceback.print_exc()
            results[seq] = None

    # 保存单序列结果
    for seq, m in results.items():
        if m is not None:
            with open(os.path.join(OUT_DIR, f'residual_flow_{seq}.json'), 'w') as f:
                json.dump(m, f, indent=2)
            print(f"  已保存: residual_flow_{seq}.json")

    # 跨序列 mean±std 汇总
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        print("  无有效结果，跳过汇总")
        return

    keys = [
        'separation_mm', 'pearson_r_frm', 'pearson_r_cum', 'pearson_r_frm_moving',
        'detection_recall', 'detection_precision', 'detection_f1',
        'motion_rmse_mm', 'motion_mae_mm', 'motion_rmse_raw_mm',
        'moving_resid_median_mm', 'static_resid_median_mm',
        'gt_frm_moving_median_mm', 'gt_cum_moving_median_mm',
    ]

    summary = {'n_sequences': len(valid), 'sequences': list(valid.keys())}
    for cfg in ['gt', 'vo']:
        for k in keys:
            vals = np.array([v[cfg][k] for v in valid.values()], dtype=float)
            summary[f'{cfg}_{k}_mean'] = float(np.mean(vals))
            summary[f'{cfg}_{k}_std'] = float(np.std(vals))

    with open(os.path.join(OUT_DIR, 'residual_flow_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  已保存汇总: residual_flow_summary.json")

    print("\n" + "=" * 78)
    print("  残差光流评估 跨序列 Mean ± SD 汇总")
    print("=" * 78)
    for cfg, label in [('gt', 'GT补偿版(理想上界)'), ('vo', 'VO补偿版(我们方法)')]:
        print(f"\n  【{label}】")
        for k in keys:
            m = summary[f'{cfg}_{k}_mean']
            s = summary[f'{cfg}_{k}_std']
            print(f"    {k:<26s}: {m:>10.4f} ± {s:>10.4f}")
    print("=" * 78)


if __name__ == '__main__':
    main()
