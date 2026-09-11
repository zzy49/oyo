"""生成统一的 C3VD 格式多源数据集
创建 F:\multi_unified\ 目录，包含所有三个数据源的符号链接，
使基线 C3VD dataset 类无需修改即可使用。

目录结构:
  F:\multi_unified\
    endo_{scene_name}\rgb\0000.png → EndoSLAM 原始图片
    scared_{dataset}_{keyframe}\rgb\0000.png + depth\ → SCARED
    c1_transverse1...\rgb\0000.png + depth\ → F:\zhuan (symlink)
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image

UNIFIED_DIR = r'F:\multi_unified'
os.makedirs(UNIFIED_DIR, exist_ok=True)

# ── 数据源配置 ──
ENDOSLAM_DIR = r'E:\data1\monodepth2\EndoSLAM'
ENDOSLAM_TRAIN = os.path.join(ENDOSLAM_DIR, 'train_files.txt')
ENDOSLAM_VAL = os.path.join(ENDOSLAM_DIR, 'val_files.txt')

SCARED_DIR = r'E:\data1\monodepth2\scared_extracted'

ZHuan = r'F:\zhuan'
REAL_SEQS = [
    'c1_transverse1_t1_v1', 'c1_transverse1_t1_v2', 'c1_transverse1_t2_v1',
    'c1_transverse2_t1_v1', 'c1_transverse2_t2_v1',
    'c2_cecum_t3_v1', 'c2_sigmoid_t2_v1',
    'c2_transverse1_t2_v1', 'c2_transverse1_t3_v1', 'c2_transverse1_t3_v2',
    'c2_transverse2_t2_v1', 'c2_transverse2_t2_v2',
    'c2_transverse2_t3_v1', 'c2_transverse2_t3_v2',
]

SPLIT_DIR = r'e:\data1\monodepth2\splits\multi_c3vd'
os.makedirs(SPLIT_DIR, exist_ok=True)

import random
random.seed(42)

train_lines = []
val_lines = []

def safe_seq_name(name):
    """将名称转为安全的 C3VD 序列名"""
    return name.replace('/', '_').replace('\\', '_').replace(' ', '_')

# ═══════════════════════════════════════════════════════
# 1. EndoSLAM
# ═══════════════════════════════════════════════════════
print("=== EndoSLAM ===")
endo_count = 0
with open(ENDOSLAM_TRAIN, 'r') as f:
    endo_train_scenes = [l.strip() for l in f if l.strip()]
with open(ENDOSLAM_VAL, 'r') as f:
    endo_val_scenes = [l.strip() for l in f if l.strip()]

for scene in endo_train_scenes + endo_val_scenes:
    is_train = scene in set(endo_train_scenes)
    seq_name = f'endo_{safe_seq_name(scene)}'
    rgb_src = os.path.join(ENDOSLAM_DIR, scene, 'rgb')
    if not os.path.isdir(rgb_src):
        continue

    seq_dir = os.path.join(UNIFIED_DIR, seq_name)
    rgb_dst = os.path.join(seq_dir, 'rgb')
    os.makedirs(rgb_dst, exist_ok=True)

    # 扫描源图片
    src_images = sorted([f for f in os.listdir(rgb_src)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    for i, fname in enumerate(src_images):
        src_path = os.path.join(rgb_src, fname)
        ext = os.path.splitext(fname)[1]
        dst_name = f'{i:04d}{ext}'
        dst_path = os.path.join(rgb_dst, dst_name)
        if not os.path.exists(dst_path):
            try:
                os.symlink(src_path, dst_path)
            except OSError:
                # Windows 可能不支持 symlink，用硬拷贝
                try:
                    img = Image.open(src_path)
                    img.save(dst_path)
                except:
                    continue

    # 添加到 split
    n_frames = len(src_images)
    for idx in range(n_frames):
        line = f'{seq_name} {idx} l'
        if is_train:
            train_lines.append(line)
        else:
            val_lines.append(line)
    endo_count += n_frames

print(f"  EndoSLAM: {endo_count} 帧")

# ═══════════════════════════════════════════════════════
# 2. SCARED (scared_extracted)
# ═══════════════════════════════════════════════════════
print("=== SCARED ===")
scared_count = 0
for dname in sorted(os.listdir(SCARED_DIR)):
    dpath = os.path.join(SCARED_DIR, dname)
    if not os.path.isdir(dpath):
        continue
    for kf in sorted(os.listdir(dpath)):
        if not kf.startswith('keyframe_'):
            continue
        kf_path = os.path.join(dpath, kf)
        frames = sorted([f for f in os.listdir(kf_path)
                        if f.startswith('frame_') and f.endswith('.png')])
        if not frames:
            continue

        seq_name = f'scared_{dname}_{kf}'
        seq_dir = os.path.join(UNIFIED_DIR, seq_name)
        rgb_dst = os.path.join(seq_dir, 'rgb')
        depth_dst = os.path.join(seq_dir, 'depth')
        os.makedirs(rgb_dst, exist_ok=True)

        has_depth = False
        for fname in frames:
            src_img = os.path.join(kf_path, fname)
            # frame_000000.png → 0000.png
            frame_idx = int(os.path.splitext(fname)[0].split('_')[1])
            dst_img = os.path.join(rgb_dst, f'{frame_idx:04d}.png')
            if not os.path.exists(dst_img):
                try:
                    img = Image.open(src_img)
                    img.save(dst_img)
                except:
                    continue

            # 深度: frame_000000_depth.npy → 0000_depth.tiff
            depth_src = os.path.join(kf_path, f'frame_{frame_idx:06d}_depth.npy')
            if os.path.exists(depth_src):
                os.makedirs(depth_dst, exist_ok=True)
                dst_depth = os.path.join(depth_dst, f'{frame_idx:04d}_depth.tiff')
                if not os.path.exists(dst_depth):
                    try:
                        depth_arr = np.load(depth_src)
                        depth_arr = np.nan_to_num(depth_arr, nan=0.0).astype(np.float32)
                        Image.fromarray(depth_arr).save(dst_depth)
                        has_depth = True
                    except Exception as e:
                        print(f"    深度转换失败 {seq_name} frame {frame_idx}: {e}")

        # 添加到 split (9:1)
        for fname in frames:
            frame_idx = int(os.path.splitext(fname)[0].split('_')[1])
            line = f'{seq_name} {frame_idx} l'
            if random.random() < 0.9:
                train_lines.append(line)
            else:
                val_lines.append(line)
        scared_count += len(frames)

print(f"  SCARED: {scared_count} 帧")

# ═══════════════════════════════════════════════════════
# 3. Real Colon (F:\zhuan)
# ═══════════════════════════════════════════════════════
print("=== Real Colon (F:\\zhuan) ===")
real_count = 0
for seq in REAL_SEQS:
    src_rgb = os.path.join(ZHuan, seq, 'rgb')
    src_depth = os.path.join(ZHuan, seq, 'depth')
    if not os.path.isdir(src_rgb):
        print(f"  SKIP: {src_rgb} 不存在")
        continue

    seq_dir = os.path.join(UNIFIED_DIR, seq)
    rgb_dst = os.path.join(seq_dir, 'rgb')
    depth_dst = os.path.join(seq_dir, 'depth')
    os.makedirs(rgb_dst, exist_ok=True)

    # RGB 符号链接
    src_images = sorted([f for f in os.listdir(src_rgb)
                        if f.lower().endswith(('.png', '.jpg'))])
    for fname in src_images:
        src_path = os.path.join(src_rgb, fname)
        dst_path = os.path.join(rgb_dst, fname)
        if not os.path.exists(dst_path):
            try:
                os.symlink(src_path, dst_path)
            except OSError:
                try:
                    img = Image.open(src_path)
                    img.save(dst_path)
                except:
                    continue

    # 深度 symlink
    if os.path.isdir(src_depth):
        os.makedirs(depth_dst, exist_ok=True)
        for fname in os.listdir(src_depth):
            src_path = os.path.join(src_depth, fname)
            dst_path = os.path.join(depth_dst, fname)
            if not os.path.exists(dst_path):
                try:
                    os.symlink(src_path, dst_path)
                except OSError:
                    shutil.copy2(src_path, dst_path)

    # 添加到 split (9:1)
    for i, fname in enumerate(src_images):
        frame_idx = int(os.path.splitext(fname)[0])
        line = f'{seq} {frame_idx} l'
        if i % 10 < 9:
            train_lines.append(line)
        else:
            val_lines.append(line)
    real_count += len(src_images)

print(f"  Real Colon: {real_count} 帧")

# ═══════════════════════════════════════════════════════
# 写入 split 文件
# ═══════════════════════════════════════════════════════
with open(os.path.join(SPLIT_DIR, 'train_files.txt'), 'w') as f:
    f.write('\n'.join(train_lines) + '\n')
with open(os.path.join(SPLIT_DIR, 'val_files.txt'), 'w') as f:
    f.write('\n'.join(val_lines) + '\n')

print(f"\n{'='*60}")
print(f"Train: {len(train_lines)}, Val: {len(val_lines)}")
print(f"统一目录: {UNIFIED_DIR}")
print(f"Split 目录: {SPLIT_DIR}")
print(f"\n用法:")
print(f"  --data_path {UNIFIED_DIR}")
print(f"  --split multi_c3vd")
