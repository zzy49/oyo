"""提取并预处理 SCARED 数据
从 scard/*.zip 中提取 Left_Image.png + left_depth_map.tiff，
转换为 trainer 可用的格式: frame_000000.png + frame_000000_depth.npy
同时提取 rgb.mp4 帧作为连续序列。
"""

import os
import sys
import zipfile
import subprocess
import numpy as np
from PIL import Image

# TIFF 深度图读取: PIL 不支持 float TIFF, 使用 tifffile 或 cv2
try:
    import tifffile as tiff
except ImportError:
    tiff = None
    import cv2

SCARD_DIR = r'E:\data1\monodepth2\scard'
OUT_DIR = r'E:\data1\monodepth2\scared_extracted'
TEMP_DIR = r'F:\scared_temp'  # 临时解压到 F: 盘

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# 检查 ffmpeg
has_ffmpeg = False
try:
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    has_ffmpeg = True
except:
    print("WARNING: ffmpeg 未找到, 将跳过 mp4 帧提取")

zips = sorted([f for f in os.listdir(SCARD_DIR) if f.endswith('.zip')])
print(f"找到 {len(zips)} 个 zip 文件")

total_keyframes = 0
total_frames = 0

for zip_name in zips:
    zip_path = os.path.join(SCARD_DIR, zip_name)
    dataset_name = zip_name.replace('.zip', '')
    print(f"\n处理: {zip_name} ({dataset_name})")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # 找到所有 keyframe 目录
        keyframes = set()
        for entry in zf.namelist():
            parts = entry.split('/')
            if len(parts) >= 2 and parts[1].startswith('keyframe_'):
                keyframes.add(f'{parts[0]}/{parts[1]}')
        keyframes = sorted(keyframes)
        print(f"  找到 {len(keyframes)} 个 keyframes")

        for kf in keyframes:
            kf_short = kf.split('/')[1]  # keyframe_1
            out_subdir = os.path.join(OUT_DIR, dataset_name, kf_short)
            os.makedirs(out_subdir, exist_ok=True)

            # 提取 Left_Image.png
            left_img_path = f'{kf}/Left_Image.png'
            if left_img_path in zf.namelist():
                out_img = os.path.join(out_subdir, 'frame_000000.png')
                if not os.path.exists(out_img):
                    zf.extract(left_img_path, TEMP_DIR)
                    temp_img = os.path.join(TEMP_DIR, left_img_path)
                    if os.path.exists(temp_img):
                        img = Image.open(temp_img)
                        img.save(out_img)
                        total_frames += 1
                        print(f"    提取: {dataset_name}/{kf_short}/frame_000000.png")

            # 提取并转换 left_depth_map.tiff → .npy
            depth_path = f'{kf}/left_depth_map.tiff'
            if depth_path in zf.namelist():
                out_depth = os.path.join(out_subdir, 'frame_000000_depth.npy')
                if not os.path.exists(out_depth):
                    zf.extract(depth_path, TEMP_DIR)
                    temp_depth = os.path.join(TEMP_DIR, depth_path)
                    if os.path.exists(temp_depth):
                        try:
                            if tiff is not None:
                                depth_arr = tiff.imread(temp_depth).astype(np.float32)
                            else:
                                depth_arr = cv2.imread(temp_depth, cv2.IMREAD_UNCHANGED).astype(np.float32)
                            if depth_arr.ndim == 3:
                                depth_arr = depth_arr[:, :, 0]  # 取第一个通道
                            # 过滤异常值
                            depth_arr = np.nan_to_num(depth_arr, nan=0.0, posinf=0.0, neginf=0.0)
                            np.save(out_depth, depth_arr)
                            valid = depth_arr[depth_arr > 0]
                            if len(valid) > 0:
                                print(f"    深度: {dataset_name}/{kf_short}/frame_000000_depth.npy "
                                      f"({valid.min():.1f}~{valid.max():.1f} mm)")
                            else:
                                print(f"    深度: {dataset_name}/{kf_short}/frame_000000_depth.npy (无有效值)")
                        except Exception as e:
                            print(f"    深度转换失败: {e}")

            # 提取 rgb.mp4 帧
            mp4_path = f'{kf}/data/rgb.mp4'
            if has_ffmpeg and mp4_path in zf.namelist():
                # 计算已有帧数
                existing_frames = len([f for f in os.listdir(out_subdir)
                                       if f.startswith('frame_') and f.endswith('.png')
                                       and f != 'frame_000000.png'])
                if existing_frames == 0:
                    zf.extract(mp4_path, TEMP_DIR)
                    temp_mp4 = os.path.join(TEMP_DIR, mp4_path)
                    if os.path.exists(temp_mp4):
                        try:
                            # 提取帧 (每5帧取1帧以减少数据量)
                            out_pattern = os.path.join(out_subdir, 'frame_%06d.png')
                            result = subprocess.run([
                                'ffmpeg', '-i', temp_mp4,
                                '-vf', 'select=not(mod(n\,5))',
                                '-vsync', 'vfr', '-q:v', '2',
                                '-start_number', '1',
                                out_pattern
                            ], capture_output=True, text=True, timeout=300)
                            # 重命名帧文件
                            png_files = sorted([f for f in os.listdir(out_subdir)
                                               if f.startswith('frame_') and f.endswith('.png')
                                               and f != 'frame_000000.png'])
                            n_frames = len(png_files)
                            total_frames += n_frames
                            print(f"    mp4 帧: {n_frames} 帧")
                        except Exception as e:
                            print(f"    mp4 提取失败: {e}")

            total_keyframes += 1

    # 清理临时文件
    import shutil
    temp_dataset = os.path.join(TEMP_DIR, dataset_name)
    if os.path.exists(temp_dataset):
        shutil.rmtree(temp_dataset, ignore_errors=True)

# 清理临时目录
import shutil
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

print(f"\n{'='*60}")
print(f"完成! 总计 {total_keyframes} keyframes, {total_frames} 帧")
print(f"输出目录: {OUT_DIR}")
