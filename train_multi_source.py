"""
多源混合训练: EndoSLAM + SCARED + F:\\dataset (excl. c1_descending_t4_v4 + xinshuju)

训练方式: monodepth2自监督光度损失
模型: ResNet18 encoder + DepthDecoder (分类+残差混合头)
输出: ~/tmp/multi_source_e1/weights_X/

关键参数:
  --dataset multi_source    (触发多源混合模式)
  --split multi             (从 splits/multi/ 读取train/val文件)
  --height 192 --width 640
  --batch_size 12
  --num_epochs 20
  --min_depth 1.0 --max_depth 500.0   (对齐 C3VD 模型, 64 bin 对数分布)
  --endo_ratio 0.2  --scared_ratio 0.4  --real_ratio 0.4
  --classify_weight 2.0  --residual_weight 1.0
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import torch

# 确保项目在 path 中
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from trainer import Trainer
from options import MonodepthOptions

if __name__ == "__main__":
    options = MonodepthOptions()

    # 覆盖分类头默认值 (已在 options.py 中注册)
    options.parser.set_defaults(num_bins=64,
                                freeze_residual=False,
                                dataset='multi_source',
                                split='multi')

    # 数据源配比: 降低无GT的EndoSLAM占比, 提升有深度监督的数据占比至80%
    options.parser.add_argument("--endo_ratio",
                                type=float,
                                help="EndoSLAM sampling weight (default 0.2)",
                                default=0.2)

    options.parser.add_argument("--scared_ratio",
                                type=float,
                                help="SCARED sampling weight (default 0.4)",
                                default=0.4)

    options.parser.add_argument("--real_ratio",
                                type=float,
                                help="Real colon (F:\\dataset) sampling weight (default 0.4)",
                                default=0.4)

    opts = options.parse()

    # 默认参数覆盖
    if not opts.model_name:
        opts.model_name = "multi_source_e1"

    if not opts.log_dir:
        opts.log_dir = os.path.join(os.path.expanduser("~"), "tmp")

    # bin 范围: 对齐 C3VD 验证过的 1.0-500.0mm (对数分布有利于近场内窥镜场景)
    if opts.min_depth == 0.1:      # 未通过命令行显式指定时覆盖
        opts.min_depth = 1.0
    if opts.max_depth == 100.0:    # 未通过命令行显式指定时覆盖
        opts.max_depth = 500.0
