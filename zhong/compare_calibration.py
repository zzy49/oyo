"""
对比两种校准配置的 4 项指标: ATE RMSE, 静止点 RMSE, 运动点 RMSE, 相关性 r
"""
import json, numpy as np, os
from scipy.stats import pearsonr

SEQ_DIR = r'F:\dataset\c1_transverse1_t1_v2'
A_DIR = r'e:\data1\monodepth2\zhong\config_a'
B_DIR = r'e:\data1\monodepth2\zhong\config_b'


def load_vo_traj(config_dir):
    """从 config_dir 的 abs_poses.npy 加载 VO 轨迹."""
    path = os.path.join(config_dir, 'baseline_abs_poses.npy')
    poses = np.load(path)
    return poses[:, :3, 3]  # (N, 3) translations


def load_gt_traj():
    """加载 GT 位姿轨迹 (F:\dataset)."""
    path = os.path.join(SEQ_DIR, 'pose.txt')
    # CSV 格式 4×4 列主序: [r11,r21,r31,tx, r12,r22,r32,ty, r13,r23,r33,tz, 0,0,0,1]
    raw = np.loadtxt(path, delimiter=',')
    n_poses = raw.shape[0]
    poses = raw.reshape(n_poses, 4, 4)
    # translation 在最后一行 (列主序→C reshape后 t 在 row3 col0-2)
    return poses[:, 3, :3]  # (N, 3) translations


def umeyama(X, Y):
    n = X.shape[0]
    mu_x, mu_y = X.mean(0), Y.mean(0)
    X_c, Y_c = X - mu_x, Y - mu_y
    sigma_x = np.sum(X_c ** 2) / n
    S = (Y_c.T @ X_c) / n
    U, s_vec, Vt = np.linalg.svd(S)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = U @ Vt
    s = np.trace(np.diag(s_vec)) / sigma_x if sigma_x > 1e-10 else 1.0
    t = mu_y - s * R @ mu_x
    return s * (X @ R.T) + t, (R, t, s)


def compute_ate(vo, gt):
    aligned, (R, t, s) = umeyama(vo, gt)
    err = np.linalg.norm(aligned - gt, axis=1)
    return {'rmse': float(np.sqrt(np.mean(err**2))), 'mean': float(np.mean(err)),
            'median': float(np.median(err)), 'std': float(np.std(err)), 'scale': float(s)}


def load_tracks(config_dir):
    import glob
    files = glob.glob(os.path.join(config_dir, '*_baseline_tracks.json'))
    with open(files[0]) as f:
        return json.load(f)


def compute_motion_metrics(tracks_data):
    """从 tracks 数据提取静止/运动点统计."""
    tr = tracks_data['tracks']
    moving_ids = set(tracks_data['moving_ids'])
    static_disp = []
    moving_disp = []
    for t in tr:
        tid = t['id']
        frames = t['frames']
        if len(frames) < 2:
            continue
        # 首帧到尾帧位移 (图像坐标位移, proxy for 3D)
        f0, fn = frames[0], frames[-1]
        du = float(fn[1]) - float(f0[1])
        dv = float(fn[2]) - float(f0[2])
        disp_2d = np.sqrt(du**2 + dv**2)
        if tid in moving_ids:
            moving_disp.append(disp_2d)
        else:
            static_disp.append(disp_2d)
    static_disp = np.array(static_disp)
    moving_disp = np.array(moving_disp)
    all_disp = np.concatenate([static_disp, moving_disp])
    labels = np.concatenate([np.zeros(len(static_disp)), np.ones(len(moving_disp))])

    r, pval = pearsonr(all_disp, labels) if len(np.unique(labels)) >= 2 else (0, 1)

    return {
        'static_rmse_px': float(np.sqrt(np.mean(static_disp**2))),
        'static_median_px': float(np.median(static_disp)),
        'moving_rmse_px': float(np.sqrt(np.mean(moving_disp**2))),
        'moving_median_px': float(np.median(moving_disp)),
        'correlation_r': float(r),
        'n_static': len(static_disp),
        'n_moving': len(moving_disp),
    }


def main():
    print("=" * 65)
    print("  【校准对比: A (GT校准) vs B (--no_gt) on c1_transverse1_t1_v2】")
    print("=" * 65)

    gt = load_gt_traj()
    vo_a = load_vo_traj(A_DIR)
    vo_b = load_vo_traj(B_DIR)

    n = min(len(gt), len(vo_a), len(vo_b))
    gt, vo_a, vo_b = gt[:n], vo_a[:n], vo_b[:n]

    ate_a = compute_ate(vo_a, gt)
    ate_b = compute_ate(vo_b, gt)

    tracks_a = load_tracks(A_DIR)
    tracks_b = load_tracks(B_DIR)

    mA = compute_motion_metrics(tracks_a)
    mB = compute_motion_metrics(tracks_b)

    print()
    print(f"  {'指标':<28} {'配置 A (GT校准)':>16} {'配置 B (--no_gt)':>16}")
    print("  " + "-" * 60)
    print(f"  {'ATE RMSE (mm)':<28} {ate_a['rmse']:>16.2f} {ate_b['rmse']:>16.2f}")
    print(f"  {'ATE Mean (mm)':<28} {ate_a['mean']:>16.2f} {ate_b['mean']:>16.2f}")
    print(f"  {'ATE Std (mm)':<28} {ate_a['std']:>16.2f} {ate_b['std']:>16.2f}")
    print(f"  {'Umeyama Scale':<28} {ate_a['scale']:>16.4f} {ate_b['scale']:>16.4f}")
    print()
    print(f"  {'--- 运动检测 (图像坐标) ---':<28}")
    print(f"  {'静止点 n':<28} {mA['n_static']:>16d} {mB['n_static']:>16d}")
    print(f"  {'运动点 n':<28} {mA['n_moving']:>16d} {mB['n_moving']:>16d}")
    print(f"  {'静止点 RMSE (px)':<28} {mA['static_rmse_px']:>16.2f} {mB['static_rmse_px']:>16.2f}")
    print(f"  {'运动点 RMSE (px)':<28} {mA['moving_rmse_px']:>16.2f} {mB['moving_rmse_px']:>16.2f}")
    print(f"  {'相关性 r':<28} {mA['correlation_r']:>16.4f} {mB['correlation_r']:>16.4f}")

    print()
    print("=" * 65)
    print("  【幅度对比】")
    d_ate = (ate_b['rmse'] - ate_a['rmse']) / ate_a['rmse'] * 100
    print(f"  ATE:      无GT校准 → {d_ate:+.1f}%")
    d_static = (mB['static_rmse_px'] - mA['static_rmse_px']) / mA['static_rmse_px'] * 100
    print(f"  静止 RMSE: 无GT校准 → {d_static:+.1f}%")
    d_moving = (mB['moving_rmse_px'] - mA['moving_rmse_px']) / mA['moving_rmse_px'] * 100
    print(f"  运动 RMSE: 无GT校准 → {d_moving:+.1f}%")
    d_r = mB['correlation_r'] - mA['correlation_r']
    print(f"  相关性 r:  无GT校准 → {d_r:+.4f}")
    print("=" * 65)


if __name__ == '__main__':
    main()
