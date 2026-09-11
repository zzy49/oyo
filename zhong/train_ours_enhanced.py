r"""
Ours 增强版训练: Lite-Mono 解码器 + 时序一致性损失 + 多源混合训练

关键特性:
  - LiteMonoDepthDecoder: 3层 bilinear 上采样, Truncated Normal 初始化
  - 时序一致性损失 (w_temporal=0.1): 惩罚相邻帧深度预测的不一致性
  - 边缘感知平滑损失: RGB 梯度加权 (原版已实现)
  - 多源混合: EndoSLAM (自监督) + SCARED (GT深度) + Real Colon (GT深度)

用法:
  python zhong/train_ours_enhanced.py

模型输出: C:/Users/Administrator/tmp/ours_litemono_head/weights_X/
"""

from __future__ import absolute_import, division, print_function

import os, sys
import torch

# ── 训练加速 ──
torch.backends.cudnn.benchmark = True  # 自动寻找最优 cuDNN 卷积算法

# 路径设置
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sys_123 = os.path.join(_project_dir, '123')
if _sys_123 not in sys.path:
    sys.path.insert(0, _sys_123)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from options import MonodepthOptions
# 需要在 sys.path 中先加入 zhong 目录
sys.path.insert(0, os.path.join(_project_dir, 'zhong'))
from trainer_litemono_head import TrainerLitemonoHead

if __name__ == "__main__":
    options = MonodepthOptions()

    # 分类头默认值 (对齐 Ours 的 64-bin 配置, 深度监督更稳定)
    options.parser.set_defaults(
        num_bins=64,
        freeze_residual=False,
        dataset='multi_source',
        split='multi',
        scales=[0, 1, 2],  # Lite-Mono decoder 仅3层输出
    )

    # 数据源配比
    options.parser.add_argument("--endo_ratio",
                                type=float, default=0.2,
                                help="EndoSLAM sampling weight")
    options.parser.add_argument("--scared_ratio",
                                type=float, default=0.4,
                                help="SCARED sampling weight")
    options.parser.add_argument("--real_ratio",
                                type=float, default=0.4,
                                help="Real colon sampling weight")
    options.parser.add_argument("--zhuan_ratio",
                                type=float, default=0.2,
                                help="Zhuan (extra colon) sampling weight")

    # 时序一致性损失权重 (新增)
    options.parser.add_argument("--w_temporal",
                                type=float, default=0.1,
                                help="temporal consistency loss weight")

    # 混合精度训练 (新增)
    options.parser.add_argument("--use_amp",
                                action="store_true",
                                help="enable mixed precision (AMP) training")

    # 梯度裁剪 (防止 AMP NaN)
    options.parser.add_argument("--grad_clip",
                                type=float, default=1.0,
                                help="gradient clipping max_norm (0=disabled)")

    opts = options.parse()

    # ── 参数覆盖 ──
    if not opts.model_name:
        opts.model_name = "ours_litemono_head"

    if not opts.log_dir:
        opts.log_dir = os.path.join(os.path.expanduser("~"), "tmp")

    # ── 训练加速默认值 ──
    if opts.num_workers == 12:  # 未手动指定, 用加速默认值
        opts.num_workers = 4

    # 深度范围: 1.0-500.0mm (对齐多源训练)
    if opts.min_depth == 0.1:
        opts.min_depth = 1.0
    if opts.max_depth == 100.0:
        opts.max_depth = 500.0

    print("=" * 65)
    print("  Ours 增强版训练 (Lite-Mono 解码器 + 时序一致性)")
    print(f"  模型: {opts.model_name}")
    print(f"  输出: {opts.log_dir}/{opts.model_name}")
    print(f"  数据: multi_source (EndoSLAM+SCARED+Real)")
    print(f"  Decoder: LiteMonoDepthDecoder (3层, bilinear)")
    print(f"  Loss: photo + edge_smooth + classify + temporal(w={opts.w_temporal})")
    print(f"  Epochs: {opts.num_epochs}, Batch: {opts.batch_size}, Workers: {opts.num_workers}")
    print(f"  AMP: {getattr(opts, 'use_amp', False)}, GradClip: {getattr(opts, 'grad_clip', 0)}, Benchmark: True")
    print(f"  Scales: {opts.scales} (Lite-Mono 仅3层, 需 --scales 0 1 2)")
    print("=" * 65)

    trainer = TrainerLitemonoHead(opts)
    trainer.train()
