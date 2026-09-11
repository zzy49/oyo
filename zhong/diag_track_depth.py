#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断 Top 伪运动 track 的 GT/Ours 深度逐帧变化模式."""
import os, sys, json
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
TRAJ_JSON = os.path.join(BASE, 'ours_rerun', f'{SEQ}_baseline_motion_trajectories.json')
DEPTH_NPZ = os.path.join(SEQ_DIR, 'baseline_depth_maps.npz')
CHAIN_NPZ = os.path.join(SEQ_DIR, 'baseline_chain_data.npz')

TARGETS = [1904, 1821, 12849]

trajs = json.load(open(TRAJ_JSON))
t2t = {t['track_id']: t for t in trajs}

chain = np.load(CHAIN_NPZ, allow_pickle=True)
gs = float(chain['global_depth_scale'])
npz = np.load(DEPTH_NPZ, allow_pickle=True)
ours = {k: npz[k] / gs for k in npz.files}

gt_c2w = load_gt_poses_orderF(SEQ_DIR)
poses_gt = [np.linalg.inv(T) for T in gt_c2w]

for tid in TARGETS:
    t = t2t.get(tid)
    if not t:
        print(f'track {tid} not found')
        continue
    frames = t['frames']
    print(f'\n===== track {tid}: {len(frames)} frames, total_disp={t["total_displacement_mm"]:.2f}mm =====')
    # 首3帧 + 末3帧 + 中间几个
    idxs = list(range(0, min(3, len(frames)))) + \
           list(range(max(3, len(frames)//2-1), min(len(frames)//2+2, len(frames)))) + \
           list(range(max(0, len(frames)-3), len(frames)))
    idxs = sorted(set(idxs))
    print(f'  {"fi":>4} {"u":>7} {"v":>7} {"GT_Z":>8} {"Ours_Z":>8} {"GT_world":>16} {"Ours_world":>16}')
    for i in idxs:
        fr = frames[i]
        fi = int(fr['frame'])
        u, v = float(fr['u']), float(fr['v'])
        d_gt = np.array(Image.open(os.path.join(SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))).astype(np.float32) * GT_SCALE
        d_ours = ours.get(f'{fi:04d}')
        T = poses_gt[fi]
        p_gt = backproject_to_world(u, v, d_gt, T)
        p_ours = backproject_to_world(u, v, d_ours, T) if d_ours is not None else None
        z_gt = float(d_gt[int(np.clip(round(v),0,d_gt.shape[0]-1)), int(np.clip(round(u),0,d_gt.shape[1]-1))])
        z_ours = float(d_ours[int(np.clip(round(v),0,d_ours.shape[0]-1)), int(np.clip(round(u),0,d_ours.shape[1]-1))]) if d_ours is not None else -1
        s_gt = f'({p_gt[0]:.1f},{p_gt[1]:.1f},{p_gt[2]:.1f})' if p_gt is not None else '-'
        s_ours = f'({p_ours[0]:.1f},{p_ours[1]:.1f},{p_ours[2]:.1f})' if p_ours is not None else '-'
        print(f'  {fi:>4} {u:>7.1f} {v:>7.1f} {z_gt:>8.1f} {z_ours:>8.1f} {s_gt:>16} {s_ours:>16}')
