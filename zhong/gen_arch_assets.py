# -*- coding: utf-8 -*-
"""生成 V6 总体架构图所需的图像产物 + 缩略图素材"""
import os, sys, cv2
import numpy as np

sys.path.insert(0, r'e:\data1\monodepth2')

OUT = r'e:\data1\monodepth2\zhong\arch_fig\assets'
os.makedirs(OUT, exist_ok=True)

RGB_DIR = r'F:\dataset\c1_transverse1_t1_v2\generated\rgb_warped'
LOFTR = r'F:\dataset\c1_transverse1_t1_v2\loftr_cache\matches_0000.npz'
DEPTH_VIZ = r'e:\data1\monodepth2\compare_output\baseline\depth_viz'
TRAJ_3D = r'e:\data1\monodepth2\compare_output\baseline\output_trajectory_3d.png'
MOTION_3D = r'e:\data1\monodepth2\compare_output\baseline\output_motion_3d.png'
ABS_POSES = r'e:\data1\monodepth2\compare_output\baseline\abs_poses.npy'

# ═══ 1. 匹配点可视化 (IMG_M) ═══
img0 = cv2.imread(os.path.join(RGB_DIR, 'frame_0000.png'))
img1 = cv2.imread(os.path.join(RGB_DIR, 'frame_0001.png'))
m = np.load(LOFTR)
pts0, pts1 = m['pts0'], m['pts1']
rng = np.random.RandomState(0)
idx = rng.choice(len(pts0), min(300, len(pts0)), replace=False)
vis = img0.copy()
for i in idx:
    p0 = tuple(pts0[i].astype(int))
    p1 = tuple(pts1[i].astype(int))
    cv2.line(vis, p0, p1, (0, 255, 0), 1)
    cv2.circle(vis, p0, 2, (0, 0, 255), -1)
cv2.imwrite(os.path.join(OUT, 'match_points.png'), vis)
print('[1/4] match_points.png saved')

# ═══ 2. MotionNet P(static) 概率图 (IMG_S) ═══
import torch
from dyendovo_network import MotionNet

MOTION_H, MOTION_W = 384, 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
pmap_np = 1.0 / (1.0 + np.exp(-pmap_np))  # sigmoid -> [0,1] 概率
pmap_np = pmap_np * 255.0
pmap_viz = np.clip(pmap_np, 0, 255).astype(np.uint8)
pmap_color = cv2.applyColorMap(pmap_viz, cv2.COLORMAP_JET)
cv2.imwrite(os.path.join(OUT, 'prob_map.png'), pmap_color)
print(f'[2/4] prob_map.png saved  shape={pmap_np.shape} min={pmap_np.min():.2f} max={pmap_np.max():.2f}')

# ═══ 3. 相机 XY 俯视轨迹图 (O2) ═══
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

poses = np.load(ABS_POSES)  # (N, 4, 4) 或 (N, 3)
if poses.ndim == 3:
    t = poses[:, :3, 3]
else:
    t = poses[:, :3]
fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
ax.plot(t[:, 0], t[:, 1], '-o', color='#9B59B6', markersize=2, linewidth=1.2)
ax.scatter(t[0, 0], t[0, 1], c='#66BB6A', s=60, zorder=5, label='start')
ax.set_xlabel('X (mm)', fontsize=7)
ax.set_ylabel('Y (mm)', fontsize=7)
ax.tick_params(labelsize=6)
ax.set_aspect('equal', adjustable='datalim')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'cam_traj_xy.png'), bbox_inches='tight')
plt.close()
print('[3/4] cam_traj_xy.png saved')

# ═══ 4. 8 张缩略图 (统一 240x180, 供 draw.io 嵌入) ═══
THUMB = (240, 180)
def thumb(src, dst):
    im = cv2.imread(src)
    if im is None:
        print(f'  !! missing: {src}')
        return
    im = cv2.resize(im, THUMB, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, dst), im)

thumb(os.path.join(RGB_DIR, 'frame_0000.png'), 'img_in.png')          # IMG_IN 输入RGB
thumb(os.path.join(DEPTH_VIZ, '0000_depth.png'), 'img_d.png')          # IMG_D  深度图
thumb(os.path.join(OUT, 'prob_map.png'), 'img_s.png')                  # IMG_S  概率图
thumb(os.path.join(OUT, 'match_points.png'), 'img_m.png')              # IMG_M  匹配点
thumb(TRAJ_3D, 'img_t.png')                                            # IMG_T  位姿轨迹
thumb(os.path.join(DEPTH_VIZ, '0030_depth.png'), 'o1.png')             # O1     输出深度图
thumb(os.path.join(OUT, 'cam_traj_xy.png'), 'o2.png')                  # O2     相机轨迹
thumb(MOTION_3D, 'o3.png')                                             # O3     运动组织轨迹
print('[4/4] 8 thumbnails saved ->', OUT)
