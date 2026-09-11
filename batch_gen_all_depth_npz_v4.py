"""
一键生成所有数据集的 motion_train_data (V4 13通道版本)
===========================================================
0. (跳过 - motion_gt 已全部就绪)
1. 运行 gen_motion_training_data_depth_v4.py（生成 per-window npz）
2. 运行 gen_motion_train_npz_v4.py（合并为 patches.npy + labels.npy）

输出: {data_dir}/generated/patches.npy + labels.npy (mmap 兼容)
"""
import os, sys, subprocess, time

ROOT = os.path.dirname(__file__)

# ── 所有需要的数据集 (14 序列) ──
DATASETS = [
    # C1 训练序列 (4)
    (r'F:\dataset\c1_transverse1_t1_v1', 'c1_transverse1_t1_v1'),
    (r'F:\dataset\c1_transverse1_t2_v1', 'c1_transverse1_t2_v1'),
    (r'F:\dataset\c1_transverse2_t1_v1', 'c1_transverse2_t1_v1'),
    (r'F:\dataset\c1_transverse2_t2_v1', 'c1_transverse2_t2_v1'),
    # C2 训练序列 (8)
    (r'F:\dataset\c2_cecum_t3_v1', 'c2_cecum_t3_v1'),
    (r'F:\dataset\c2_sigmoid_t2_v1', 'c2_sigmoid_t2_v1'),
    (r'F:\dataset\c2_transverse1_t2_v1', 'c2_transverse1_t2_v1'),
    (r'F:\dataset\c2_transverse1_t3_v1', 'c2_transverse1_t3_v1'),
    (r'F:\dataset\c2_transverse1_t3_v2', 'c2_transverse1_t3_v2'),
    (r'F:\dataset\c2_transverse2_t2_v1', 'c2_transverse2_t2_v1'),
    (r'F:\dataset\c2_transverse2_t2_v2', 'c2_transverse2_t2_v2'),
    (r'F:\dataset\c2_transverse2_t3_v1', 'c2_transverse2_t3_v1'),
    # C2 验证序列 (固定)
    (r'F:\dataset\c2_transverse2_t3_v2', 'c2_transverse2_t3_v2'),
    # C1 测试序列
    (r'F:\dataset\c1_transverse1_t1_v2', 'c1_transverse1_t1_v2'),
]

OUTPUT_BASE = r'E:\data1\monodepth2\motion_cls_data_depth_v4'
GEN_SCRIPT = os.path.join(ROOT, 'gen_motion_training_data_depth_v4.py')
MERGE_SCRIPT = os.path.join(ROOT, 'gen_motion_train_npz_v4.py')


def count_frames(data_dir):
    """统计 rgb_warped 帧数."""
    d = os.path.join(data_dir, 'generated', 'rgb_warped')
    if os.path.exists(d):
        return len([f for f in os.listdir(d) if f.startswith('frame_') and f.endswith('.png') and f.count('_') == 1])
    d = os.path.join(data_dir, 'rgb')
    return len([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg'))])


def is_v4_valid(data_dir):
    """Check if V4-diff data exists (patches.npy with 11 channels)."""
    try:
        import numpy as np
        patches_path = os.path.join(data_dir, 'generated', 'patches.npy')
        labels_path = os.path.join(data_dir, 'generated', 'labels.npy')
        if not os.path.exists(patches_path) or not os.path.exists(labels_path):
            return False
        data = np.load(patches_path, mmap_mode='r')
        return data.shape[-1] == 11
    except Exception:
        return False


def main():
    for ds_path, ds_name in DATASETS:
        generated_dir = os.path.join(ds_path, 'generated')

        # Skip if valid V4 patches.npy already exists
        if is_v4_valid(ds_path):
            patches_path = os.path.join(generated_dir, 'patches.npy')
            sz_gb = os.path.getsize(patches_path) / 1024**3
            print(f'\n[跳过] {ds_name}: V4 patches.npy 已存在 ({sz_gb:.1f} GB)')
            continue

        n_frames = count_frames(ds_path)
        npz_out_dir = os.path.join(OUTPUT_BASE, ds_name)

        print(f'\n{"="*60}')
        print(f'[生成] {ds_name}: {n_frames} frames')
        print(f'{"="*60}')

        # Step 0: (跳过 - motion_gt 已全部就绪)

        # Step 1: 生成 per-window npz
        print(f'\n  Step 1: gen_motion_training_data_depth_v4.py (5-frame windows)')
        t0 = time.time()
        ret = subprocess.run([
            sys.executable, GEN_SCRIPT,
            '--data_dir', ds_path,
            '--output_dir', npz_out_dir,
            '--start_frame', '0',
            '--end_frame', str(n_frames - 1),
            '--prefix', ds_name,
            '--max_samples', '2000',
        ])

        elapsed = time.time() - t0
        if ret.returncode != 0:
            err_msg = 'N/A'
            if hasattr(ret, 'stderr') and ret.stderr:
                err_msg = ret.stderr[-500:]
            print(f'  ERROR in gen: {err_msg}')
            continue
        print(f'  Step 1 完成: {elapsed:.0f}s')

        # Step 2: 合并为 patches.npy + labels.npy
        print(f'\n  Step 2: gen_motion_train_npz_v4.py')
        t0 = time.time()
        ret = subprocess.run([
            sys.executable, MERGE_SCRIPT,
            '--data_dir', ds_path,
            '--npz_dir', npz_out_dir,
        ])
        elapsed = time.time() - t0
        if ret.returncode != 0:
            err_msg = 'N/A'
            if hasattr(ret, 'stderr') and ret.stderr:
                err_msg = ret.stderr[-500:]
            print(f'  ERROR in merge: {err_msg}')
            continue
        print(f'  Step 2 完成: {elapsed:.0f}s')

    print('\n\n全部完成！')


if __name__ == '__main__':
    main()
