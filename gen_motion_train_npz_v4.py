"""
合并 V4-diff per-window npz → patches.npy + labels.npy (11通道差分版本)
=============================================================
输出格式:
  patches: (N, 16, 16, 11) float32, HWC
    通道顺序: [depth_t, Δdepth_01, Δdepth_12, Δdepth_23, Δdepth_34,
               Δflow_x_01-12, Δflow_x_12-23, Δflow_x_23-34,
               Δflow_y_01-12, Δflow_y_12-23, Δflow_y_23-34]
  labels:  (N, 16, 16) float32, per-pixel (由 per-point 标签扩展)
  patch_size: int64
  depth_stats: (2,) float32 [mean, std]
  flow_stats: (4,) float32 [mean_x, std_x, mean_y, std_y]

用法:
  python gen_motion_train_npz_v4.py --data_dir F:\dataset\c2_sigmoid_t2_v1 ^
      --npz_dir e:\data1\monodepth2\motion_cls_data_depth_v4\c2_sigmoid_t2_v1

输出到: {data_dir}\generated\patches.npy + labels.npy (11通道)
"""
import os, sys, argparse, glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def consolidate(npz_dir, output_dir):
    """合并所有 per-window npz → patches.npy + labels.npy (13通道, mmap 兼容)."""
    files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
    if not files:
        print(f'  ERROR: 在 {npz_dir} 中未找到 npz 文件')
        return None

    print(f'  合并 {len(files)} 个 npz...')

    # 从第一个文件读取归一化统计量
    first = np.load(files[0])
    depth_stats = first.get('depth_stats', np.array([50.0, 30.0], dtype=np.float32))
    flow_stats = first.get('flow_stats', np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    patch_size = int(first.get('patch_size', 16))
    print(f'  归一化统计: depth mean={depth_stats[0]:.1f} std={depth_stats[1]:.1f}mm, '
          f'flow_x mean={flow_stats[0]:.2f} std={flow_stats[1]:.2f}px, '
          f'flow_y mean={flow_stats[2]:.2f} std={flow_stats[3]:.2f}px')

    # V4-diff 11通道: depth_t + 4Δdepth + 3Δflow_x + 3Δflow_y
    CHANNEL_KEYS = [
        'depth_0', 'depth_1', 'depth_2', 'depth_3', 'depth_4',       # 5 深度差分通道
        'flow_x_0', 'flow_x_1', 'flow_x_2',                          # 3 残差流 x 差分
        'flow_y_0', 'flow_y_1', 'flow_y_2',                          # 3 残差流 y 差分
    ]

    all_patches = []
    all_labels = []

    for fpath in files:
        data = np.load(fpath)
        n = len(data['labels'])
        if n == 0:
            continue

        # 验证所需 key 存在
        missing_keys = [k for k in CHANNEL_KEYS if k not in data]
        if missing_keys:
            print(f'  WARNING: {os.path.basename(fpath)} 缺少 key: {missing_keys}, 跳过')
            continue

        # Stack 11 channels: (N, 16, 16, 11) HWC
        channels = [data[k] for k in CHANNEL_KEYS]
        patch = np.stack(channels, axis=-1).astype(np.float32)  # (N, 16, 16, 11)
        all_patches.append(patch)

        # Expand per-point labels to per-pixel: (N,) → (N, 16, 16)
        lbl = data['labels'].astype(np.float32)  # (N,)
        lbl_per_pixel = np.broadcast_to(lbl[:, None, None], (n, 16, 16)).copy()
        all_labels.append(lbl_per_pixel)

    patches = np.concatenate(all_patches, axis=0)    # (N_total, 16, 16, 11)
    labels = np.concatenate(all_labels, axis=0)       # (N_total, 16, 16)

    n_total = len(labels)
    n_moving = int(labels.max(axis=(1, 2)).sum())  # any pixel moving → patch is moving
    print(f'  总计: {n_total:,} 样本, moving={n_moving:,} '
          f'({100*n_moving/max(n_total,1):.1f}%)')

    # 保存为 .npy (未压缩，支持 mmap)
    os.makedirs(output_dir, exist_ok=True)
    patches_path = os.path.join(output_dir, 'patches.npy')
    labels_path = os.path.join(output_dir, 'labels.npy')

    np.save(patches_path, patches)
    np.save(labels_path, labels)

    size_p = os.path.getsize(patches_path) / 1e6
    size_l = os.path.getsize(labels_path) / 1e6
    print(f'  已保存: {patches_path} ({size_p:.1f} MB)')
    print(f'  已保存: {labels_path} ({size_l:.1f} MB)')
    return n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='数据集根目录 (输出到 {data_dir}/generated/patches.npy + labels.npy)')
    parser.add_argument('--npz_dir', required=True,
                        help='per-window npz 目录')
    args = parser.parse_args()

    output_dir = os.path.join(args.data_dir, 'generated')
    consolidate(args.npz_dir, output_dir)


if __name__ == '__main__':
    main()
