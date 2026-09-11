"""生成 splits/multi/train_files.txt 和 val_files.txt
三个数据源: EndoSLAM + SCARED + Real Colon (F:\zhuan 子集)
"""

import os
import random

SPLIT_DIR = r'e:\data1\monodepth2\splits\multi'
ENDOSLAM_TRAIN = r'E:\data1\monodepth2\EndoSLAM\train_files.txt'
ENDOSLAM_VAL = r'E:\data1\monodepth2\EndoSLAM\val_files.txt'
REAL_SEQS = [
    'c1_transverse1_t1_v1', 'c1_transverse1_t1_v2', 'c1_transverse1_t2_v1',
    'c1_transverse2_t1_v1', 'c1_transverse2_t2_v1',
    'c2_cecum_t3_v1', 'c2_sigmoid_t2_v1',
    'c2_transverse1_t2_v1', 'c2_transverse1_t3_v1', 'c2_transverse1_t3_v2',
    'c2_transverse2_t2_v1', 'c2_transverse2_t2_v2',
    'c2_transverse2_t3_v1', 'c2_transverse2_t3_v2',
]
ZHuan = r'F:\zhuan'
SCARED_DIR = r'E:\data1\monodepth2\scared_extracted'  # 预处理后 SCARED 数据

random.seed(42)

os.makedirs(SPLIT_DIR, exist_ok=True)

lines = []

# ── 1. EndoSLAM ──
with open(ENDOSLAM_TRAIN, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            lines.append(('train', line))  # 无前缀 = EndoSLAM

with open(ENDOSLAM_VAL, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            lines.append(('val', line))

# ── 2. Real Colon (F:\zhuan) ──
for seq in REAL_SEQS:
    rgb_dir = os.path.join(ZHuan, seq, 'rgb')
    if not os.path.isdir(rgb_dir):
        print(f"  WARNING: {rgb_dir} 不存在, 跳过")
        continue
    frames = sorted([
        f for f in os.listdir(rgb_dir)
        if f.lower().endswith(('.png', '.jpg'))
    ])
    # 挑帧 (间隔1帧作为训练，9:1 split)
    for i, fname in enumerate(frames):
        frame_idx = int(os.path.splitext(fname)[0])
        if i % 10 < 9:  # 90% train
            lines.append(('train', f'real {seq} {frame_idx}'))
        else:
            lines.append(('val', f'real {seq} {frame_idx}'))

# ── 3. SCARED ──
# 扫描预处理后的 SCARED: scared_extracted/dataset_X/keyframe_Y/frame_XXXXXX.png
scared_train = []
scared_val = []
for dname in sorted(os.listdir(SCARED_DIR)):
    dpath = os.path.join(SCARED_DIR, dname)
    if not os.path.isdir(dpath):
        continue
    kfs = sorted([k for k in os.listdir(dpath) if k.startswith('keyframe_')])
    if not kfs:
        continue
    for kf in kfs:
        kf_path = os.path.join(dpath, kf)
        # 找所有帧
        frames = sorted([f for f in os.listdir(kf_path)
                        if f.startswith('frame_') and f.endswith('.png')])
        if not frames:
            continue
        subdir = f'{dname}/{kf}'
        for fname in frames:
            # frame_000000.png → frame_idx=0, frame_000001.png → frame_idx=1
            frame_idx = int(os.path.splitext(fname)[0].split('_')[1])
            if random.random() < 0.9:
                scared_train.append(f'scared {subdir} {frame_idx}')
            else:
                scared_val.append(f'scared {subdir} {frame_idx}')

for entry in scared_train:
    lines.append(('train', entry))
for entry in scared_val:
    lines.append(('val', entry))

# ── 写入 ──
train_lines = [l for t, l in lines if t == 'train']
val_lines = [l for t, l in lines if t == 'val']

# 统计
endo_train = sum(1 for l in train_lines if not l.startswith('scared') and not l.startswith('real'))
scared_train_n = sum(1 for l in train_lines if l.startswith('scared'))
real_train_n = sum(1 for l in train_lines if l.startswith('real'))

endo_val = sum(1 for l in val_lines if not l.startswith('scared') and not l.startswith('real'))
scared_val_n = sum(1 for l in val_lines if l.startswith('scared'))
real_val_n = sum(1 for l in val_lines if l.startswith('real'))

print(f"=== splits/multi ===")
print(f"  Train: {len(train_lines)} (EndoSLAM={endo_train}, SCARED={scared_train_n}, Real={real_train_n})")
print(f"  Val:   {len(val_lines)} (EndoSLAM={endo_val}, SCARED={scared_val_n}, Real={real_val_n})")

with open(os.path.join(SPLIT_DIR, 'train_files.txt'), 'w') as f:
    f.write('\n'.join(train_lines) + '\n')
with open(os.path.join(SPLIT_DIR, 'val_files.txt'), 'w') as f:
    f.write('\n'.join(val_lines) + '\n')

print(f"\n已写入: {SPLIT_DIR}/train_files.txt")
print(f"已写入: {SPLIT_DIR}/val_files.txt")
