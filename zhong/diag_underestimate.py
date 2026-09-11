#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断: 真实大运动点为何被 Ours 低估 (逐帧深度对比)."""
import sys, json, os
import numpy as np
from PIL import Image

sys.path.insert(0, r'e:\data1\monodepth2')
sys.path.insert(0, r'e:\data1\monodepth2\zhong')

SEQ_DIR = r'F:\dataset\c1_transverse1_t1_v2'
GT_SCALE = 100.0 / 65535.0

TRAJ = r'e:\data1\monodepth2\zhong\ours_rerun\c1_transverse1_t1_v2_baseline_motion_trajectories.json'

TARGET = [12827, 7051, 6742]


def main():
    trajs = json.load(open(TRAJ))
    by = {t['track_id']: t for t in trajs}
    dn = np.load(os.path.join(SEQ_DIR, 'baseline_depth_maps.npz'), allow_pickle=True)
    do = {k: dn[k] / 2.2390 for k in dn.files}

    for tid in TARGET:
        t = by[tid]
        print(f'\n=== track {tid} (n_frames={t["n_frames"]}) ===')
        print(f'  {"frame":>5} | {"u":>7} | {"v":>7} | {"OursZ":>7} | {"GTZ":>7} | {"dOurs":>6} | {"dGT":>6}')
        prev_o = prev_g = None
        for fr in t['frames']:
            fi = int(fr['frame'])
            u, v = float(fr['u']), float(fr['v'])
            ui, vi = int(round(u)), int(round(v))
            oz = do.get(f'{fi:04d}')
            oz = oz[vi, ui] if oz is not None else -1
            if fi < 117:
                gz = np.array(Image.open(os.path.join(
                    SEQ_DIR, 'depth', f'{fi:04d}_depth.tiff'))).astype(np.float32)[vi, ui] * GT_SCALE
            else:
                gz = -1
            dO = f'{oz-prev_o:+.1f}' if prev_o is not None else '  -'
            dG = f'{gz-prev_g:+.1f}' if prev_g is not None else '  -'
            print(f'  {fi:>5} | {u:7.1f} | {v:7.1f} | {oz:7.1f} | {gz:7.1f} | {dO:>6} | {dG:>6}')
            prev_o, prev_g = oz, gz


if __name__ == '__main__':
    main()
