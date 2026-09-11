"""Generate C3VD train/val split files for F:\\zhuan.
Excludes _model/_mold directories (mesh data, not images).
Output format: folder frame_index l  (one per line)
"""
import os

DATA_ROOT = r"F:\zhuan"
OUT_DIR = r"E:\data1\monodepth2\splits\c3vd_full"

os.makedirs(OUT_DIR, exist_ok=True)

seqs = []
for d in sorted(os.listdir(DATA_ROOT)):
    dpath = os.path.join(DATA_ROOT, d)
    if not os.path.isdir(dpath):
        continue
    if '_model' in d or '_mold' in d:
        continue
    rgb_dir = os.path.join(dpath, 'rgb')
    if not os.path.isdir(rgb_dir):
        continue
    pngs = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    if not pngs:
        continue
    seqs.append((d, len(pngs)))

print(f"Found {len(seqs)} valid sequences, total frames: {sum(s[1] for s in seqs)}")

# Sort by frame count, use ~10% for validation (pick every 10th)
seqs.sort(key=lambda x: x[1])

train_lines = []
val_lines = []

for i, (seq, n_frames) in enumerate(seqs):
    rgb_dir = os.path.join(DATA_ROOT, seq, 'rgb')
    pngs = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png')])
    # frame index = stem of filename (e.g. 0000.png -> 0)
    frame_indices = [int(os.path.splitext(f)[0]) for f in pngs]
    frame_indices.sort()

    is_val = (i % 10 == 0)  # every 10th seq is val

    for fi in frame_indices:
        # C3VD has no left/right stereo, use 'l' as convention
        line = f"{seq} {fi} l\n"
        if is_val:
            val_lines.append(line)
        else:
            train_lines.append(line)

    status = "VAL" if is_val else "TRAIN"
    print(f"  [{status}] {seq}: {n_frames} frames")

with open(os.path.join(OUT_DIR, 'train_files.txt'), 'w') as f:
    f.writelines(train_lines)
with open(os.path.join(OUT_DIR, 'val_files.txt'), 'w') as f:
    f.writelines(val_lines)

print(f"\nTrain: {len(train_lines)} frames from {len(seqs) - len(seqs)//10} seqs")
print(f"Val:   {len(val_lines)} frames from {len(seqs)//10} seqs")
print(f"Saved to: {OUT_DIR}")
