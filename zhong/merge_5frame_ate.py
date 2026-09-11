import numpy as np, sys
sys.path.insert(0, r'e:\data1\monodepth2\baselines')
from compute_5frame_ate import umeyama, apply_sim3


def window_ate_list(gt_poses, pred_poses, window=5):
    n = gt_poses.shape[0]
    ate = []
    for i in range(0, n - window + 1):
        g = gt_poses[i:i + window, :3, 3]
        p = pred_poses[i:i + window, :3, 3]
        s, R, t = umeyama(p, g)
        aligned = apply_sim3(p, s, R, t)
        err = np.linalg.norm(aligned - g, axis=1)
        ate.append(np.sqrt(np.mean(err ** 2)))
    return np.array(ate)


def accum(rel):
    traj = [np.eye(4)]
    c2w = np.eye(4)
    for T in rel:
        c2w = c2w @ T
        traj.append(c2w.copy())
    return np.array(traj)


def method_mean_std(rel_paths, gt_paths):
    all_ate = []
    for rp, gp in zip(rel_paths, gt_paths):
        rel = np.load(rp)['data']
        gt = np.load(gp)['data']
        pred = accum(rel)
        n = min(pred.shape[0], gt.shape[0])
        all_ate.append(window_ate_list(gt[:n], pred[:n]))
    ate = np.concatenate(all_ate)
    return ate.mean(), ate.std()


base = r'e:\data1\monodepth2\baselines'
fast = base + r'\Endo_FASt3r\splits\endovis'
afsl = base + r'\AF-SfMLearner-main\splits\endovis'

scared = {
    'Ours': ([fast + r'\pred_pose_ours_sq1.npz', fast + r'\pred_pose_ours_sq2.npz'],
             [fast + r'\gt_poses_sq1.npz', fast + r'\gt_poses_sq2.npz']),
    'BodySLAM': ([fast + r'\pred_pose_bodyslam_sq1.npz', fast + r'\pred_pose_bodyslam_sq2.npz'],
                 [fast + r'\gt_poses_sq1.npz', fast + r'\gt_poses_sq2.npz']),
    'Endo-FASt3r': ([fast + r'\pred_pose_sq1.npz', fast + r'\pred_pose_sq2.npz'],
                    [fast + r'\gt_poses_sq1.npz', fast + r'\gt_poses_sq2.npz']),
    'AF-SfMLearner': ([afsl + r'\pred_pose_afsl_sq1.npz', afsl + r'\pred_pose_afsl_sq2.npz'],
                      [afsl + r'\gt_poses_sq1.npz', afsl + r'\gt_poses_sq2.npz']),
}

print('=== SCARED 合并 5 帧窗口 ATE (mm) ===')
for m, (rp, gp) in scared.items():
    mn, sd = method_mean_std(rp, gp)
    print(f'{m}: {mn*1000:.2f} +/- {sd*1000:.2f}')

esl = r'e:\data1\monodepth2\EndoSLAM\UnityCam'
scenes = ['Colon', 'Small Intestine', 'Stomach']
endoslam = {}
for m, fn in [('Ours', 'pred_pose_ours'), ('BodySLAM', 'pred_pose_bodyslam'),
              ('Endo-FASt3r', 'pred_pose_endofast3r'), ('AF-SfMLearner', 'pred_pose_afsl')]:
    rp = [f'{esl}\\{s}\\{fn}_{s}.npz' for s in scenes]
    gp = [f'{esl}\\{s}\\gt_poses_{s}.npz' for s in scenes]
    endoslam[m] = (rp, gp)

print('=== EndoSLAM 合并 5 帧窗口 ATE (mm) ===')
for m, (rp, gp) in endoslam.items():
    mn, sd = method_mean_std(rp, gp)
    print(f'{m}: {mn*1000:.2f} +/- {sd*1000:.2f}')
