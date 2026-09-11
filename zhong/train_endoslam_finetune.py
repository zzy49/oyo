"""EndoSLAM 微调: 在现有模型(models/depth)基础上, 用 EndoSLAM UnityCam 深度 GT 监督微调。

背景:
  现有模型 models/depth = ResNet18 + DepthDecoder(原版5层, scales=[0,1,2,3],
  64-bin 分类+残差头) + PoseCNN + MotionEncoder。
  (depth.pth 有 18 个 decoder 模块, 对应原版 5 层 DepthDecoder, 非 LiteMono 3 层)

  目标: 在 EndoSLAM 训练集上微调, 使深度与位姿适配 EndoSLAM 域,
        以便与 BodySLAM 等基线在 EndoSLAM 上公平对比 (不要求超越, 差距小即可)。

  数据: EndoSLAMDepthDataset (datasets/endoslam_depth_dataset.py)
        RGB=Frames_jpg/image_XXXX.jpg, 深度=Pixelwise Depths/aov_image_XXXX.png
        深度编码 R 通道值单位厘米 → 毫米 = value * 10 (范围 1~450mm)。

用法:
  python zhong/train_endoslam_finetune.py \
      --load_weights_folder models/depth \
      --use_motion_encoder --use_depth_consistency \
      --use_depth_gt --depth_gt_weight 0.3 \
      --num_epochs 5 --model_name endoslam_finetune
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
# 复用 TrainerFusionV3: 直接调用 Trainer.__init__, 保留原版 5 层 DepthDecoder,
# 并继承 TrainerLitemonoHead 的时序一致性损失。
from train_fusion_v3 import TrainerFusionV3


if __name__ == "__main__":
    options = MonodepthOptions()

    # 对齐现有模型 models/depth: 原版 5 层 DepthDecoder + 64-bin + PoseCNN
    options.parser.set_defaults(
        num_bins=64,
        freeze_residual=False,
        dataset='endoslam_depth',
        split='endoslam_full',
        scales=[0, 1, 2, 3],
        pose_model_type='posecnn',   # models/depth 只有 pose.pth (PoseCNN), 无 pose_encoder.pth
        height=320,
        width=320,
        learning_rate=1e-5,          # 微调小学习率
        num_epochs=5,
        min_depth=1.0,
        max_depth=500.0,
    )

    # 时序一致性 / 混合精度 / 梯度裁剪 (与 train_fusion_v3.py 一致)
    options.parser.add_argument("--w_temporal", type=float, default=0.1,
                                help="temporal consistency loss weight")
    options.parser.add_argument("--use_amp", action="store_true",
                                help="enable mixed precision (AMP) training")
    options.parser.add_argument("--grad_clip", type=float, default=1.0,
                                help="gradient clipping max_norm (0=disabled)")

    opts = options.parse()

    if not opts.model_name or opts.model_name == "mdp":
        opts.model_name = "endoslam_finetune"
    if not opts.log_dir:
        opts.log_dir = os.path.join(os.path.expanduser("~"), "tmp")

    # 数据路径默认值: EndoSLAM 根目录 (含 UnityCam/Colon 等)
    if opts.data_path.endswith("kitti_data"):
        opts.data_path = os.path.join(_project_dir, "EndoSLAM")

    # 关键: 显式把 motion_encoder 加入加载列表 (默认不含它, 否则随机初始化)
    if opts.load_weights_folder is not None:
        opts.models_to_load = ["encoder", "depth", "pose", "motion_encoder"]

    # 深度范围 1.0-500.0mm (对齐现有模型, EndoSLAM GT 10~450mm)
    if opts.min_depth == 0.1:
        opts.min_depth = 1.0
    if opts.max_depth == 100.0:
        opts.max_depth = 500.0

    # 未手动指定 num_workers 时用加速默认值
    if opts.num_workers == 12:
        opts.num_workers = 4

    print("=" * 65)
    print("  EndoSLAM 微调 (现有模型 + EndoSLAM 深度 GT 监督)")
    print(f"  模型: {opts.model_name}")
    print(f"  输出: {opts.log_dir}/{opts.model_name}")
    print(f"  续训起点: {opts.load_weights_folder}")
    print(f"  数据: {opts.dataset} / {opts.split} @ {opts.data_path}")
    print(f"  分辨率: {opts.height}x{opts.width}, Scales: {opts.scales}")
    print(f"  Decoder: 原版 DepthDecoder(5层) + PoseCNN + MotionEncoder")
    print(f"  Depth GT: use_depth_gt={opts.use_depth_gt}, weight={opts.depth_gt_weight}")
    print(f"  use_motion_encoder: {opts.use_motion_encoder}")
    print(f"  use_depth_consistency: {opts.use_depth_consistency}")
    print(f"  Epochs: {opts.num_epochs}, Batch: {opts.batch_size}, "
          f"LR: {opts.learning_rate}, Workers: {opts.num_workers}")
    print("=" * 65)

    trainer = TrainerFusionV3(opts)
    trainer.train()
