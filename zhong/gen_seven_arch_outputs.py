# -*- coding: utf-8 -*-
"""
从 rgb_warped 完整跑 V6 正式流程 (M1 开 + F1 关 + F2 开),
生成架构图所需的 7 张图片产物。
输出目录: zhong/arch_fig/outputs
"""
import os, sys, json, time, shutil
import numpy as np
import torch
import random
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = r'e:\data1\monodepth2'
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

import test_v6_dyendovo as tv6
from test_v6_multi_model_enhanced import (
    load_model, SEQ_NAME, SEQ_DIR, DATA_ROOT)
from dyendovo_dataset import load_gt_poses as dyendo_load_gt_poses

MODEL_PATH = r'e:\data1\monodepth2\models\depth'
RGB_DIR = r'F:\dataset\c1_transverse1_t1_v2\generated\rgb_warped'
OUT = r'e:\data1\monodepth2\zhong\arch_fig\outputs'
os.makedirs(OUT, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def save_depth_viz(depth_mm, path, vmax=None):
    d = np.asarray(depth_mm, dtype=np.float32)
    d = np.clip(d, 1.0, 500.0)
    if vmax is None:
        valid = d[d > 1.0]
        vmax = float(np.percentile(valid, 95)) if len(valid) > 0 else 100.0
    norm = np.clip((d - 1.0) / max(vmax - 1.0, 1e-6), 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(path, vis)
    return vmax


def camera_positions(abs_poses):
    pos = []
    for T in abs_poses:
        T = np.asarray(T, dtype=np.float64)
        T_cw = np.linalg.inv(T)
        pos.append(T_cw[:3, 3])
    return np.array(pos)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    gt_poses_std = dyendo_load_gt_poses(SEQ_DIR)
    print(f'GT: {len(gt_poses_std)} 帧')

    # [1] 加载 Ours 模型
    print('\n[1/4] 加载 Ours 深度模型...')
    cfg = {'name': 'Ours', 'path': MODEL_PATH, 'type': 'md2'}
    encoder, depth_decoder, motion_encoder = load_model(cfg, device)
    print(f'  motion_encoder: {"有" if motion_encoder is not None else "无"}')

    # [2] 校准 (pnp_scale)
    print('\n[2/4] estimate_pipeline_params (M1 开, pnp_scale)...')
    t0 = time.time()
    params = tv6.estimate_pipeline_params(
        SEQ_DIR, encoder, depth_decoder, motion_encoder, device,
        depth_source='pred', loftr_source='cache', no_gt=False, zero_gt_mode='pnp_scale')
    print(f'  global_depth_scale={float(params["global_depth_scale"]):.4f} ({time.time()-t0:.0f}s)')

    # [3] 正式 VO (F1 默认关闭)
    print('\n[3/4] run_vo_sequence (M1开 + F1关 + F2开)...')
    t0 = time.time()
    vo_traj, gt_traj, stats, chain_data = tv6.run_vo_sequence(
        SEQ_NAME, encoder, depth_decoder, device,
        motion_net=None, mode='baseline', max_frames=None,
        motion_encoder=motion_encoder, depth_source='pred',
        loftr_source='cache', data_root=DATA_ROOT,
        pipeline_params=params, no_gt=False, zero_gt_mode='pnp_scale')
    print(f'  VO 完成 ({time.time()-t0:.0f}s), 成功 {stats["n_success"]}/{stats["n_total"]}')

    all_k0 = chain_data['all_k0']
    all_k1 = chain_data['all_k1']
    all_weights = chain_data['all_weights']
    depth_maps = chain_data['depth_maps']
    abs_poses = chain_data['abs_poses']
    K = np.asarray(params.get('K', np.eye(3)))
    print(f'  匹配对: {len(all_k0)}, 深度帧: {len(depth_maps)}, 位姿帧: {len(abs_poses)}')

    # [4] 生成 7 张图
    print('\n[4/4] 生成 7 张架构图产物...')

    # 图3: 深度图 (帧0)
    vmax3 = save_depth_viz(depth_maps[0], os.path.join(OUT, '03_depth_map.png'))
    print(f'  [3/7] 深度图 (帧0, vmax={vmax3:.0f}mm)')

    # 图5: 输出深度图 (帧30)
    k30 = 30 if 30 in depth_maps else max(depth_maps.keys())
    vmax5 = save_depth_viz(depth_maps[k30], os.path.join(OUT, '05_output_depth.png'))
    print(f'  [5/7] 输出深度图 (帧{k30}, vmax={vmax5:.0f}mm)')

    # 相机位置
    cam = camera_positions(abs_poses)

    # 图4: 位姿轨迹 (3D)
    fig = plt.figure(figsize=(6, 5), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(cam[:, 0], cam[:, 1], cam[:, 2], '-o', color='#9B59B6', markersize=2.5, linewidth=1.3)
    ax.scatter(*cam[0], c='#66BB6A', s=70, zorder=5, label='start')
    ax.scatter(*cam[-1], c='#F5A623', s=70, zorder=5, label='end')
    ax.set_xlabel('X (mm)', fontsize=8)
    ax.set_ylabel('Y (mm)', fontsize=8)
    ax.set_zlabel('Z (mm)', fontsize=8)
    ax.set_title('Camera Pose Trajectory (3D)', fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '04_pose_trajectory.png'), bbox_inches='tight')
    plt.close()
    print('  [4/7] 位姿轨迹 (3D)')

    # 图6: 相机 XY 俯视轨迹
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    ax.plot(cam[:, 0], cam[:, 1], '-o', color='#9B59B6', markersize=2.5, linewidth=1.3)
    ax.scatter(cam[0, 0], cam[0, 1], c='#66BB6A', s=60, zorder=5, label='start')
    ax.scatter(cam[-1, 0], cam[-1, 1], c='#F5A623', s=60, zorder=5, label='end')
    ax.set_xlabel('X (mm)', fontsize=8)
    ax.set_ylabel('Y (mm)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '06_camera_trajectory.png'), bbox_inches='tight')
    plt.close()
    print('  [6/7] 相机 XY 俯视轨迹')

    # 图2: 匹配点 (帧0 vs 帧1)
    img0 = cv2.imread(os.path.join(RGB_DIR, 'frame_0000.png'))
    if img0 is not None and len(all_k0) > 0:
        k0 = all_k0[0]
        k1 = all_k1[0]
        rng = np.random.RandomState(0)
        idx = rng.choice(len(k0), min(300, len(k0)), replace=False)
        vis = img0.copy()
        for i in idx:
            p0 = tuple(k0[i].astype(int))
            p1 = tuple(k1[i].astype(int))
            cv2.line(vis, p0, p1, (0, 255, 0), 1)
            cv2.circle(vis, p0, 2, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(OUT, '02_match_points.png'), vis)
        print(f'  [2/7] 匹配点 ({len(idx)} 对)')
    else:
        print('  [2/7] 匹配点: 跳过 (无图像/无匹配)')

    # 图7: 运动组织轨迹
    chain_dist_thresh = float(params.get('chain_dist_thresh', 3.0))
    k_factor = float(params.get('k_factor', 0.5))
    tracks = tv6.chain_tracks(all_k0, all_k1, dist_thresh=chain_dist_thresh)
    static_tracks, moving_tracks = tv6.classify_tracks(
        tracks, all_weights, depth_maps, abs_poses, k_factor=k_factor)
    print(f'  tracks={len(tracks)}, static={len(static_tracks)}, moving={len(moving_tracks)}')
    if moving_tracks:
        tv6.visualize_top_trajectories(moving_tracks, depth_maps, abs_poses, K, OUT, SEQ_NAME)
        src = os.path.join(OUT, f'{SEQ_NAME}_motion_3d.png')
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, '07_tissue_motion_trajectory.png'))
            print('  [7/7] 运动组织轨迹')
    else:
        print('  [7/7] 运动组织轨迹: 无运动点')

    # 图1: 概率图 P(static) — MotionNet 推理
    try:
        from dyendovo_network import MotionNet
        MOTION_H, MOTION_W = 384, 512
        net = MotionNet(pretrained=False, img_h=MOTION_H, img_w=MOTION_W).to(device)
        ckpt = torch.load(r'e:\data1\monodepth2\models\dyendovo\best_model.pth',
                          map_location=device, weights_only=False)
        net.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        net.eval()

        def load_rgb(p):
            im = cv2.imread(p)
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        it = cv2.resize(load_rgb(os.path.join(RGB_DIR, 'frame_0000.png')), (MOTION_W, MOTION_H))
        ip1 = cv2.resize(load_rgb(os.path.join(RGB_DIR, 'frame_0001.png')), (MOTION_W, MOTION_H))
        pair = np.concatenate([it, ip1], axis=-1)
        pair_t = torch.from_numpy(pair).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            pmap = net(pair_t)
        pmap = pmap[0]
        if pmap.dim() == 3:
            pmap = pmap[0] if pmap.shape[0] == 1 else pmap.mean(0)
        pmap_np = pmap.cpu().numpy()
        pmap_np = 1.0 / (1.0 + np.exp(-pmap_np))
        pmap_viz = np.clip(pmap_np * 255.0, 0, 255).astype(np.uint8)
        pmap_color = cv2.applyColorMap(pmap_viz, cv2.COLORMAP_JET)
        if img0 is not None:
            pmap_color = cv2.resize(pmap_color, (img0.shape[1], img0.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(OUT, '01_probability_map.png'), pmap_color)
        print('  [1/7] 概率图 P(static)')
    except Exception as e:
        print(f'  [1/7] 概率图失败: {e}')

    print('\n全部产物输出目录:', OUT)
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        print(f'  {f}  {os.path.getsize(p)//1024}KB')


if __name__ == '__main__':
    main()
