#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断: Ours 位移被放大的根源 (深度误差 vs 位姿误差).

对若干 track, 逐帧反投影并做消融:
  A = Ours深度 + Ours位姿  (完整 Ours 反投影)
  B = Ours深度 + GT位姿    (隔离: 只换位姿, 保留 Ours 深度)
  C = GT深度   + Ours位姿  (隔离: 只换深度, 保留 Ours 位姿)
  D = GT深度   + GT位姿    (完整 GT 反投影)

对比 A/B/C/D 的首末净位移, 判断放大主要来自深度还是位姿。

输入:
  zhong/ours_rerun/c1_transverse1_t1_v2_baseline_motion_trajectories.json
  F:/dataset/c1_transverse1_t1_v2/baseline_depth_maps.npz  (Ours 预测深度)
  F:/dataset/c1_transverse1_t1_v2/baseline_abs_poses.npy   (Ours VO 位姿)
  F:/dataset/c1_transverse1_t1_v2/depth/{fi:04d}_depth.tiff (GT 深度)
  F:/dataset/c1_transverse1_t1_v2/pose.txt                  (GT 位姿)
"""

import os
import sys
import json
from collections import defaultdict

import numpy as np
from PIL import Image

PROJECT_DIR = r'e:\data1\monodepth2'
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'zhong'))

from eval_pipeline_motion_vs_gt import backproject_to_world, load_gt_poses_orderF

BASE = r'e:\data1\monodepth2\zhong'
SEQ_DIR = r'F:\dataset\c1_transverse1_t1_v2'
SEQ = 'c1_transverse1_t1_v2'
GT_SCALE = 100.0 / 65535.0

TRAJ_JSON = os.path.join(BASE, 'ours_rerun',
                         f'{SEQ}_baseline_motion_trajectories.json')
DEPTH_NPZ = os.path.join(SEQ_DIR, 'baseline_depth_maps.npz')
POSES_NPY = os.path.join(SEQ_DIR, 'baseline_abs_poses.npy')

# 要诊断的 track: 真运动(2661, 20540) + 伪运动(3078, 17981, 50083)
TARGET_IDS = [3078, 17981, 2661, 20540, 50083]


def disp_of(plist):
    valid = [p for p in plist if p is not None]
    if len(valid) < 2:
        return 0.0
    return float(np.linalg.norm(valid[-1] - valid[0]))


def main():
    with open(TRAJ_JSON) as f:
        trajs = json.load(f)
    by_id = {t['track_id']: t for t in trajs}

    # Ours 深度 + Ours 位姿
    depth_npz = np.load(DEPTH_NPZ, allow_pickle=True)
    depth_ours = {k: depth_npz[k] for k in depth_npz.files}
    # 深度尺度恢复 ÷global_scale (与 run_motion_pipeline 一致)
    # 注意: baseline_depth_maps.npz 是 ×global_scale 的轨迹尺度深度
    global_scale = 2.2390
    for k in depth_ours:
        depth_ours[k] = depth_ours[k] / global_scale
    poses_ours = np.load(POSES_NPY)   # world_to_cam (117,4,4)

    # GT 位姿 (cam_to_world -> world_to_cam) + GT 深度
    gt_c2w = load_gt_poses_orderF(SEQ_DIR)
    poses_gt = [np.linalg.inv(T) for T in gt_c2w]

    print(f'{"track":>7} | {"Ours深度+Ours位姿":>16} | {"Ours深度+GT位姿":>15} '
          f'| {"GT深度+Ours位姿":>15} | {"GT深度+GT位姿":>14} | 帧数 | 结论')
    print('-' * 110)

    for tid in TARGET_IDS:
        t = by_id[tid]
        A, B, C, D = [], [], [], []
        for fr in t['frames']:
            fi = int(fr['frame'])
            u, v = float(fr['u']), float(fr['v'])
            key = f'{fi:04d}'
            d_ours = depth_ours.get(key)
            p_ours = poses_ours[fi] if fi < len(poses_ours) else None
            p_gt = poses_gt[fi] if fi < len(poses_gt) else None
            d_gt = None
            if fi < len(poses_gt):
                d_gt = np.array(Image.open(os.path.join(
                    SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))).astype(np.float32) * GT_SCALE

            A.append(backproject_to_world(u, v, d_ours, p_ours) if (d_ours is not None and p_ours is not None) else None)
            B.append(backproject_to_world(u, v, d_ours, p_gt) if (d_ours is not None and p_gt is not None) else None)
            C.append(backproject_to_world(u, v, d_gt, p_ours) if (d_gt is not None and p_ours is not None) else None)
            D.append(backproject_to_world(u, v, d_gt, p_gt) if (d_gt is not None and p_gt is not None) else None)

        dA, dB, dC, dD = disp_of(A), disp_of(B), disp_of(C), disp_of(D)
        # 结论: 换位姿后 A->B 下降多少 = 位姿误差贡献; 换深度后 A->C 下降多少 = 深度误差贡献
        if dA > 2.0:
            pose_contrib = dA - dB
            depth_contrib = dA - dC
            if pose_contrib > depth_contrib:
                concl = '位姿误差主导'
            elif depth_contrib > pose_contrib:
                concl = '深度误差主导'
            else:
                concl = '两者相当'
        else:
            concl = '位移小, 正常'
        print(f'{tid:>7} | {dA:>16.2f} | {dB:>15.2f} | {dC:>15.2f} | '
              f'{dD:>14.2f} | {t["n_frames"]:>4} | {concl}')


if __name__ == '__main__':
    main()
