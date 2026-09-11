import numpy as np

def align_traj(est, gt):
    est, gt = np.array(est, np.float64), np.array(gt, np.float64)
    n = min(len(est), len(gt)); est, gt = est[:n], gt[:n]
    em, gm = est.mean(0), gt.mean(0)
    ec, gc = est - em, gt - gm
    U, _, Vt = np.linalg.svd(ec.T @ gc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[-1] *= -1; R = Vt.T @ U.T
    s = np.trace(R @ ec.T @ gc) / np.trace(ec.T @ ec)
    if s < 1e-6: s = 1.0
    aligned = (s * (R @ est.T)).T + gm - (R @ em * s)
    return aligned, np.linalg.norm(aligned - gt, axis=1), R, s

raw = np.loadtxt(r'F:\dataset\c1_transverse1_t1_v2\pose.txt', delimiter=',')
gt = raw.reshape(raw.shape[0], 4, 4)[:, 3, :3]

vc = np.load(r'e:\data1\monodepth2\zhong\zero_gt_compare\config_C\vo_traj_scaled.npy')
vg = np.load(r'e:\data1\monodepth2\zhong\zero_gt_compare\config_gt\vo_traj_scaled.npy')

results = {}
for name, vo in [('GT校准', vg), ('配置C', vc)]:
    a, e, R, s = align_traj(vo, gt)
    d = a - gt[:len(a)]
    rms = np.sqrt(np.mean(e**2))
    rxyz = [np.sqrt(np.mean(d[:,i]**2)) for i in range(3)]
    vl = np.sum(np.linalg.norm(np.diff(vo, axis=0), axis=1))
    gl = np.sum(np.linalg.norm(np.diff(gt[:len(vo)], axis=0), axis=1))
    results[name] = {'ate_rmse': rms, 'umeyama_s': s,
                     'rms_x': rxyz[0], 'rms_y': rxyz[1], 'rms_z': rxyz[2],
                     'vo_len': vl, 'gt_len': gl}

hdr = "{:<28} {:>12} {:>15} {:>10}".format('指标', 'GT校准', '配置C', '差异')
print(hdr)
print('-' * 68)

def row(label, vg_val, vc_val, fmt='{:.2f} mm'):
    if isinstance(vg_val, str) and isinstance(vc_val, str):
        return "{:<28} {:>12} {:>15} {:>10}".format(label, vg_val, vc_val, '-')
    diff = vc_val - vg_val
    ds = fmt.format(diff) if not isinstance(diff, str) else diff
    vgs = fmt.format(vg_val) if not isinstance(vg_val, str) else vg_val
    vcs = fmt.format(vc_val) if not isinstance(vc_val, str) else vc_val
    return "{:<28} {:>12} {:>15} {:>10}".format(label, vgs, vcs, ds)

print(row('ATE RMSE', results['GT校准']['ate_rmse'], results['配置C']['ate_rmse']))

# Per-axis RMS
for a, aname in [('rms_x','X轴 RMS'), ('rms_y','Y轴 RMS'), ('rms_z','Z轴 RMS')]:
    print(row(aname, results['GT校准'][a], results['配置C'][a]))

# Umeyama scale
print(row('Umeyama scale', results['GT校准']['umeyama_s'], results['配置C']['umeyama_s'], '{:.4f}'))

# VO length ratio
vo_ratio_gt = results['GT校准']['vo_len'] / results['GT校准']['gt_len'] * 100
vo_ratio_c = results['配置C']['vo_len'] / results['配置C']['gt_len'] * 100
print(row('VO/GT 长度比', f'{vo_ratio_gt:.1f}%', f'{vo_ratio_c:.1f}%', '{}'))

ate_rmse_gt = results['GT校准']['ate_rmse']
ate_rmse_c = results['配置C']['ate_rmse']
delta = ate_rmse_c - ate_rmse_gt
print(f"\n  ATE RMSE 差异: 配置C - GT校准 = {delta:+.2f}mm ({(delta/ate_rmse_gt)*100:+.1f}%)")
print(f"  结论: {'配置C 与 GT校准 精度接近' if abs(delta) < 0.5 else '配置C 精度明显不如 GT校准'}")
