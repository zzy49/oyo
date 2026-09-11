"""ManyDepth 基线 + GT深度监督 训练启动脚本"""
import sys, os

# Add ManyDepth parent to path (manydepth package is at e:\data1\manydepth-master\manydepth)
sys.path.insert(0, r'e:\data1\manydepth-master')
os.chdir(r'e:\data1\manydepth-master')

from manydepth.trainer import Trainer
from manydepth.options import MonodepthOptions

options = MonodepthOptions()
opts = options.parser.parse_args([
    '--data_path', 'F:/zhuan',
    '--dataset', 'c3vd',
    '--split', 'multi_c3vd_gt',
    '--model_name', 'c3vd_manydepth_multi',
    '--log_dir', 'C:/Users/Administrator/tmp',
    '--num_layers', '18',
    '--height', '192',
    '--width', '640',
    '--batch_size', '4',
    '--learning_rate', '1e-4',
    '--num_epochs', '20',
    '--scheduler_step_size', '15',
    '--min_depth', '2.0',
    '--max_depth', '100.0',
    '--depth_supervision_weight', '1.0',
    '--disparity_smoothness', '1e-3',
    '--frame_ids', '0', '-1', '1',
    '--scales', '0', '1', '2', '3',
    '--png',
    '--weights_init', 'pretrained',
    '--num_workers', '4',
    '--save_frequency', '1',
    '--log_frequency', '250',
    '--disable_motion_masking',
    '--no_matching_augmentation',
])

if __name__ == '__main__':
    print("=" * 60)
    print("  ManyDepth 基线训练 (多源: EndoSLAM+SCARED+Real)")
    print(f"  数据: {opts.data_path}")
    print(f"  模型: {opts.model_name}")
    print(f"  深度监督权重: {opts.depth_supervision_weight}")
    print(f"  Epochs: {opts.num_epochs}")
    print("=" * 60)
    trainer = Trainer(opts)
    trainer.train()
