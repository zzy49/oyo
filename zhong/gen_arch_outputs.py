# -*- coding: utf-8 -*-
"""整理生成 7 张 V6 管线图片产物 (输入: rgb_warped)"""
import os, cv2, shutil

BASE = r'e:\data1\monodepth2'
OUT = os.path.join(BASE, r'zhong\arch_fig\outputs')
os.makedirs(OUT, exist_ok=True)

ASSETS = os.path.join(BASE, r'zhong\arch_fig\assets')
CMP = os.path.join(BASE, r'compare_output\baseline')

# 1. 概率图 P(static) — MotionNet 输出, 放大到原图尺寸 1350x1080
prob = cv2.imread(os.path.join(ASSETS, 'prob_map.png'))
prob_big = cv2.resize(prob, (1350, 1080), interpolation=cv2.INTER_LINEAR)
cv2.imwrite(os.path.join(OUT, '01_probability_map.png'), prob_big)
print('[1/7] 01_probability_map.png  (384x512 -> 1350x1080)')

# 2. 匹配点 (LoFTR 帧间对应点)
shutil.copy(os.path.join(ASSETS, 'match_points.png'),
            os.path.join(OUT, '02_match_points.png'))
print('[2/7] 02_match_points.png  (1350x1080)')

# 3. 深度图 (单帧深度预测可视化)
shutil.copy(os.path.join(CMP, 'depth_viz', '0000_depth.png'),
            os.path.join(OUT, '03_depth_map.png'))
print('[3/7] 03_depth_map.png  (1350x1080)')

# 4. 位姿轨迹 (VO 相机位姿 3D 轨迹)
shutil.copy(os.path.join(CMP, 'output_trajectory_3d.png'),
            os.path.join(OUT, '04_pose_trajectory.png'))
print('[4/7] 04_pose_trajectory.png')

# 5. 输出的深度图 (最终输出深度, 帧30)
shutil.copy(os.path.join(CMP, 'depth_viz', '0030_depth.png'),
            os.path.join(OUT, '05_output_depth.png'))
print('[5/7] 05_output_depth.png  (1350x1080)')

# 6. 相机轨迹 (XY 俯视)
shutil.copy(os.path.join(ASSETS, 'cam_traj_xy.png'),
            os.path.join(OUT, '06_camera_trajectory.png'))
print('[6/7] 06_camera_trajectory.png')

# 7. 运动组织轨迹 (运动点世界坐标 3D)
shutil.copy(os.path.join(CMP, 'output_motion_3d.png'),
            os.path.join(OUT, '07_tissue_motion_trajectory.png'))
print('[7/7] 07_tissue_motion_trajectory.png')

print('\n全部产物输出目录:', OUT)
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f'  {f}  {os.path.getsize(p)//1024}KB')
