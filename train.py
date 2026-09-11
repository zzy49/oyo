"""
train.py - 单源自监督深度估计训练 (Monodepth2 标准入口)

基于 Monodepth2 架构, 扩展支持:
- 64-bin 分类+残差深度头 (1–500mm log-uniform bins)
- 尺度锚定损失 + 方差下界损失
- 断点续训 (num_epochs 为绝对终点值)

用法:
  # 从头训练 C3VDv2
  python train.py --data_path F:/dataset/cecum_t1_a --log_dir ./logs/c3vd --epochs 50

  # 从 epoch 18 续训 5 轮 (必须设 epochs=23)
  python train.py --data_path ... --model_path ./models/depth/weights_18 --epochs 23
"""

import os
import sys
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer import Trainer


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="单源自监督深度估计训练")

    # 数据
    parser.add_argument('--data_path', type=str, required=True,
                        help='训练数据根目录')
    parser.add_argument('--val_path', type=str, default=None,
                        help='验证数据目录 (None 则从 train 中划分)')

    # 模型
    parser.add_argument('--model_path', type=str, default=None,
                        help='预训练权重目录 (续训起点)')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='日志与 checkpoint 目录')

    # 训练
    parser.add_argument('--epochs', type=int, default=30,
                        help='绝对终点 epoch 编号 (续训时需设为 S+T)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=320)

    # 损失
    parser.add_argument('--disable_photometric', action='store_true')
    parser.add_argument('--disable_scale_anchor', action='store_true')
    parser.add_argument('--w_photo', type=float, default=1.0)
    parser.add_argument('--w_smooth', type=float, default=1.0)
    parser.add_argument('--w_variance', type=float, default=0.02)
    parser.add_argument('--w_anchor', type=float, default=0.1)

    # 其他
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_checkpoint', type=str, default=None)

    args = parser.parse_args()

    # ── 固定参数 ──
    args.min_depth = 1.0
    args.max_depth = 500.0
    args.num_bins = 64
    args.num_layers = 18
    args.pretrained = True
    args.use_posecnn = True       # 单源训练使用 PoseCNN
    args.use_motion_encoder = False
    args.learning_rate = args.lr
    args.early_stop_patience = 10

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 60)
    print("单源自监督深度估计训练 (Monodepth2)")
    print(f"  数据: {args.data_path}")
    print(f"  深度: {args.min_depth}–{args.max_depth} mm, {args.num_bins} bins")
    print(f"  Epochs: {args.epochs} (绝对终点)")
    print("=" * 60)

    # ── 数据集 ──
    from train_multi_source import (
        MultiSourceDataset, _load_c3vd_intrinsics,
    )

    # ── 自动检测数据格式 ──
    import cv2

    # EndoSLAM 格式: seq_dir/generated/rgb_warped/*.png
    endoslam_rgb = os.path.join(args.data_path, 'generated', 'rgb_warped')
    if os.path.isdir(endoslam_rgb):
        is_endoslam = True
        rgb_dir = endoslam_rgb
        depth_dir = args.data_path
        from dyendovo_dataset import load_gt_poses, K_MAT
        try:
            gt_poses = load_gt_poses(args.data_path)
        except Exception:
            gt_poses = None
        K = K_MAT.copy()
        has_gt = False  # EndoSLAM训练序列通常无GT深度
    else:
        # C3VD 格式: seq_dir/*_color.png
        is_endoslam = False
        rgb_dir = args.data_path
        depth_dir = args.data_path
        K = _load_c3vd_intrinsics(args.data_path)
        gt_poses = None
        pose_path = os.path.join(args.data_path, 'pose.txt')
        if os.path.isfile(pose_path):
            poses_raw = np.loadtxt(pose_path, delimiter=',').reshape(-1, 4, 4)
            gt_poses = [poses_raw[i] for i in range(len(poses_raw))]
        # 检测 GT 深度
        frames_all = sorted([
            f for f in os.listdir(rgb_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        has_gt = any(
            os.path.isfile(os.path.join(rgb_dir, f.replace('.png', '_depth.tiff')))
            or os.path.isfile(os.path.join(rgb_dir, f.replace('_color.png', '_depth.tiff')))
            for f in frames_all[:3]
        )

    # 扫描帧
    color_frames = sorted([
        f for f in os.listdir(rgb_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    # C3VD: 过滤非 _color 文件
    if not is_endoslam:
        filtered = [f for f in color_frames if '_color' in f or not any(
            tag in f.lower() for tag in ['_depth', '_normals', '_flow', '_occlusion']
        )]
        if len(filtered) > 0:
            color_frames = filtered

    print(f"  格式: {'EndoSLAM' if is_endoslam else 'C3VD'}, 帧数: {len(color_frames)}")

    full_ds = MultiSourceDataset(
        data_dir=args.data_path, frames=color_frames, K=K,
        gt_poses=gt_poses, depth_dir=depth_dir,
        has_gt=has_gt, is_c3vd=not is_endoslam,
        height=args.height, width=args.width,
        rgb_subdir=os.path.join('generated', 'rgb_warped') if is_endoslam else None,
    )
    print(f"  总样本: {len(full_ds)}")

    # 划分训练/验证
    n_val = max(1, int(len(full_ds) * 0.05))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # ── 训练 ──
    trainer = Trainer(args)
    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        trainer.load_checkpoint(args.resume_checkpoint)

    trainer.run(train_loader, val_loader)


if __name__ == '__main__':
    main()
