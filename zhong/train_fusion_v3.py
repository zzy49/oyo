"""
方案3: 训练级融合 —— 保留 Ours 老模型的 DepthDecoder(5层) + 打开深度/时序一致性

背景:
  方案1(管线级尺度传递)与方案2(深度图级融合)已证伪:
    umeyama × global_depth_scale ≈ 常数, 后处理校准会自动抵消深度整体缩放。
  因此唯一可行路径是通过训练让 Ours 的 DepthDecoder 学到 mdp_v5 的绝对尺度优势。

mdp_v5 尺度优势三要素 (定位自 123/trainer.py + trainer_litemono_head.py):
  ① use_motion_encoder      MotionEncoder 帧间运动特征注入 encoder skip (Ours 已有)
  ② use_depth_consistency   两帧独立深度预测, 共享权重, 输入顺序相反
  ③ w_temporal=0.1          时序一致性 L1 loss = |D_0 - warp(D_t, T_{0->t})|

Ours 老模型 (models/depth) 结构:
  ResNet18 + DepthDecoder(原版5层, scales=[0,1,2,3], 64bin 分类+残差头)
  + PoseCNN + MotionEncoder (有 motion_encoder.pth, 无 pose_encoder.pth)
  → 已具备 ①, 缺 ②③。

本脚本:
  TrainerFusionV3(TrainerLitemonoHead) 在 __init__ 中直接调用 Trainer.__init__
  (跳过 TrainerLitemonoHead 的解码器替换步骤), 从而保留原版 DepthDecoder(5层);
  同时继承其 compute_temporal_consistency_loss + compute_losses (时序一致性损失)。

用法:
  python zhong/train_fusion_v3.py --load_weights_folder models/depth \
      --use_motion_encoder --use_depth_consistency --pose_model_type posecnn \
      --num_epochs 5 --model_name fusion_v3
"""

from __future__ import absolute_import, division, print_function

import os, sys
import torch

torch.backends.cudnn.benchmark = True

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sys_123 = os.path.join(_project_dir, '123')
if _sys_123 not in sys.path:
    sys.path.insert(0, _sys_123)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from options import MonodepthOptions
sys.path.insert(0, os.path.join(_project_dir, 'zhong'))
from trainer_litemono_head import TrainerLitemonoHead, Trainer


class TrainerFusionV3(TrainerLitemonoHead):
    """方案3: 保留原版 DepthDecoder(5层) + 时序一致性损失.

    与 TrainerLitemonoHead 的唯一区别:
      TrainerLitemonoHead.__init__ 会把 DepthDecoder 替换为 LiteMonoDepthDecoder(3层);
      本类跳过替换, 直接使用 Trainer.__init__ 创建的原版 DepthDecoder(5层)。
    """

    def __init__(self, options):
        # 直接调用 123/trainer.py 的 Trainer.__init__ (保留原版 DepthDecoder)
        Trainer.__init__(self, options)

        # 时序一致性损失权重
        self.w_temporal = getattr(self.opt, 'w_temporal', 0.1)

        n_dec = sum(p.numel() for p in self.models['depth'].parameters()) / 1000.0
        print(f"  [FusionV3] 保留原版 DepthDecoder(5层, {n_dec:.1f}K params)")
        print(f"  [FusionV3] temporal consistency weight: {self.w_temporal}")
        print(f"  [FusionV3] use_depth_consistency: "
              f"{getattr(self.opt, 'use_depth_consistency', False)}")


if __name__ == "__main__":
    options = MonodepthOptions()

    # 对齐 Ours 老模型: 64-bin 分类+残差头, 5层解码器 4 尺度
    options.parser.set_defaults(
        num_bins=64,
        freeze_residual=False,
        dataset='multi_source',
        split='multi',
        scales=[0, 1, 2, 3],
        pose_model_type='posecnn',   # Ours 只有 pose.pth (PoseCNN)
    )

    # 数据源配比 (对齐 mdp_v5 / train_ours_enhanced.py)
    options.parser.add_argument("--endo_ratio", type=float, default=0.2,
                                help="EndoSLAM sampling weight")
    options.parser.add_argument("--scared_ratio", type=float, default=0.4,
                                help="SCARED sampling weight")
    options.parser.add_argument("--real_ratio", type=float, default=0.4,
                                help="Real colon sampling weight")
    options.parser.add_argument("--zhuan_ratio", type=float, default=0.2,
                                help="Zhuan (extra colon) sampling weight")

    # 时序一致性损失权重
    options.parser.add_argument("--w_temporal", type=float, default=0.1,
                                help="temporal consistency loss weight")

    # 混合精度 + 梯度裁剪
    options.parser.add_argument("--use_amp", action="store_true",
                                help="enable mixed precision (AMP) training")
    options.parser.add_argument("--grad_clip", type=float, default=1.0,
                                help="gradient clipping max_norm (0=disabled)")

    opts = options.parse()

    if not opts.model_name:
        opts.model_name = "fusion_v3"

    if not opts.log_dir:
        opts.log_dir = os.path.join(os.path.expanduser("~"), "tmp")

    # 关键: 显式把 motion_encoder 加入加载列表 (默认不含它, 否则会随机初始化)
    if opts.load_weights_folder is not None:
        opts.models_to_load = ["encoder", "depth", "pose", "motion_encoder"]

    # 深度范围: 1.0-500.0mm (对齐多源训练)
    if opts.min_depth == 0.1:
        opts.min_depth = 1.0
    if opts.max_depth == 100.0:
        opts.max_depth = 500.0

    # 未手动指定 num_workers 时用加速默认值
    if opts.num_workers == 12:
        opts.num_workers = 4

    print("=" * 65)
    print("  方案3: 训练级融合 (Ours DepthDecoder 5层 + 深度/时序一致性)")
    print(f"  模型: {opts.model_name}")
    print(f"  输出: {opts.log_dir}/{opts.model_name}")
    print(f"  续训起点: {opts.load_weights_folder}")
    print(f"  数据: multi_source (EndoSLAM+SCARED+Real+Zhuan)")
    print(f"  Decoder: 原版 DepthDecoder(5层, scales={opts.scales})")
    print(f"  Loss: photo + edge_smooth + classify + temporal(w={opts.w_temporal})")
    print(f"  use_motion_encoder: {opts.use_motion_encoder}")
    print(f"  use_depth_consistency: {opts.use_depth_consistency}")
    print(f"  Epochs: {opts.num_epochs}, Batch: {opts.batch_size}, "
          f"Workers: {opts.num_workers}")
    print("=" * 65)

    trainer = TrainerFusionV3(opts)
    trainer.train()
