"""生成多源 C3VD split 文件（带前缀标识数据源）
格式: @endo@scene_name frame_idx side
      @scared@dataset/keyframe frame_idx side  
      seq_name frame_idx side  (Real Colon from F:\zhuan)

输出: splits/multi_c3vd/
"""

import os
import random

SPLIT_DIR = r'e:\data1\monodepth2\splits\multi_c3vd'
ENDOSLAM_TRAIN = r'E:\data1\monodepth2\EndoSLAM\train_files.txt'
ENDOSLAM_VAL = r'E:\data1\monodepth2\EndoSLAM\val_files.txt'
ENDOSLAM_ROOT = r'E:\data1\monodepth2\EndoSLAM'
SCARED_ROOT = r'E:\data1\monodepth2\scared_extracted'
REAL_ROOT = r'F:\zhuan'

REAL_SEQS = [
    'c1_transverse1_t1_v1', 'c1_transverse1_t1_v2', 'c1_transverse1_t2_v1',
    'c1_transverse2_t1_v1', 'c1_transverse2_t2_v1',
    'c2_cecum_t3_v1', 'c2_sigmoid_t2_v1',
    'c2_transverse1_t2_v1', 'c2_transverse1_t3_v1', 'c2_transverse1_t3_v2',
    'c2_transverse2_t2_v1', 'c2_transverse2_t2_v2',
    'c2_transverse2_t3_v1', 'c2_transverse2_t3_v2',
]

random.seed(42)
os.makedirs(SPLIT_DIR, exist_ok=True)

train_lines = []
val_lines = []

# ═══════════════════════════════════════════════════════
# 1. EndoSLAM — 展开每个场景的所有帧
# ═══════════════════════════════════════════════════════
# EndoSLAM train_files: 每行 scene_name [optional_frame_idx]
# 对于无帧索引的场景，加载所有帧

with open(ENDOSLAM_TRAIN, 'r') as f:
    endo_train = [l.strip() for l in f if l.strip()]
with open(ENDOSLAM_VAL, 'r') as f:
    endo_val = [l.strip() for l in f if l.strip()]

endo_train_set = set(endo_train)

endo_count = [0, 0]

# 预扫描所有 EndoSLAM 场景的帧数（缓存避免重复扫描）
scene_frames = {}
for line in endo_train + endo_val:
    line = line.strip()
    parts = line.split()
    scene = line
    if len(parts) >= 2:
        try:
            int(parts[-1])
            scene = ' '.join(parts[:-1])
        except ValueError:
            pass
    if scene in scene_frames:
        continue
    rgb_dir = os.path.join(ENDOSLAM_ROOT, scene, 'Frames_jpg')
    if not os.path.isdir(rgb_dir):
        rgb_dir = os.path.join(ENDOSLAM_ROOT, scene, 'Frames')
    if os.path.isdir(rgb_dir):
        frames = sorted([f for f in os.listdir(rgb_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        scene_frames[scene] = frames
    else:
        scene_frames[scene] = []

for line in endo_train + endo_val:
    line = line.strip()
    # 尝试解析：最后一段如果是纯数字则是 frame_index
    parts = line.split()
    frame_idx = None
    scene = line
    if len(parts) >= 2:
        try:
            frame_idx = int(parts[-1])
            scene = ' '.join(parts[:-1])
        except ValueError:
            pass
    is_train = line in endo_train_set

    frames = scene_frames.get(scene, [])
    if frame_idx is not None:
        # 指定了帧索引
        if frame_idx < len(frames):
            entry = f'@endo@{scene} {frame_idx} l'
            if is_train:
                train_lines.append(entry)
                endo_count[0] += 1
            else:
                val_lines.append(entry)
                endo_count[1] += 1
    else:
        # 场景级条目：展开所有帧
        for fi in range(len(frames)):
            entry = f'@endo@{scene} {fi} l'
            if is_train:
                train_lines.append(entry)
                endo_count[0] += 1
            else:
                val_lines.append(entry)
                endo_count[1] += 1

print(f"EndoSLAM: train={endo_count[0]}, val={endo_count[1]}")

# ═══════════════════════════════════════════════════════
# 2. SCARED — scared_extracted 中的帧
# ═══════════════════════════════════════════════════════
scared_count = [0, 0]
for dname in sorted(os.listdir(SCARED_ROOT)):
    dpath = os.path.join(SCARED_ROOT, dname)
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
        subdir = f'{dname}/{kf}'
        for fname in frames:
            frame_idx = int(os.path.splitext(fname)[0].split('_')[1])
            entry = f'@scared@{subdir} {frame_idx} l'
            if random.random() < 0.9:
                train_lines.append(entry)
                scared_count[0] += 1
            else:
                val_lines.append(entry)
                scared_count[1] += 1

print(f"SCARED: train={scared_count[0]}, val={scared_count[1]}")

# ═══════════════════════════════════════════════════════
# 3. Real Colon (F:\zhuan)
# ═══════════════════════════════════════════════════════
real_count = [0, 0]
for seq in REAL_SEQS:
    rgb_dir = os.path.join(REAL_ROOT, seq, 'rgb')
    if not os.path.isdir(rgb_dir):
        print(f"  SKIP: {rgb_dir}")
        continue
    frames = sorted([f for f in os.listdir(rgb_dir)
                    if f.lower().endswith(('.png', '.jpg'))])
    for i, fname in enumerate(frames):
        frame_idx = int(os.path.splitext(fname)[0])
        entry = f'{seq} {frame_idx} l'
        if i % 10 < 9:
            train_lines.append(entry)
            real_count[0] += 1
        else:
            val_lines.append(entry)
            real_count[1] += 1

print(f"Real Colon: train={real_count[0]}, val={real_count[1]}")

# ═══════════════════════════════════════════════════════
# 写入
# ═══════════════════════════════════════════════════════
with open(os.path.join(SPLIT_DIR, 'train_files.txt'), 'w') as f:
    f.write('\n'.join(train_lines) + '\n')
with open(os.path.join(SPLIT_DIR, 'val_files.txt'), 'w') as f:
    f.write('\n'.join(val_lines) + '\n')

total_train = endo_count[0] + scared_count[0] + real_count[0]
total_val = endo_count[1] + scared_count[1] + real_count[1]
print(f"\n{'='*60}")
print(f"Train: {total_train} (EndoSLAM={endo_count[0]}, SCARED={scared_count[0]}, Real={real_count[0]})")
print(f"Val:   {total_val} (EndoSLAM={endo_count[1]}, SCARED={scared_count[1]}, Real={real_count[1]})")
print(f"Split 目录: {SPLIT_DIR}")
