"""Lite-Mono 基线 + GT深度监督 训练启动脚本"""
import sys, os

sys.path.insert(0, r'E:\data1\Lite-Mono-main')
from trainer import Trainer
from options import LiteMonoOptions

options = LiteMonoOptions()
opts = options.parser.parse_args([
    '--data_path', 'F:/zhuan',
    '--dataset', 'c3vd',
    '--split', 'multi_c3vd_gt',
    '--model_name', 'c3vd_litemono_multi',
    '--log_dir', 'C:/Users/Administrator/tmp',
    '--num_layers', '18',
    '--height', '192',
    '--width', '640',
    '--batch_size', '8',
    '--lr', '0.0001', '5e-6', '31', '0.0001', '1e-5', '31',
    '--num_epochs', '20',
    '--scheduler_step_size', '15',
    '--min_depth', '2.0',
    '--max_depth', '100.0',
    '--depth_supervision_weight', '1.0',
    '--disparity_smoothness', '1e-3',
    '--frame_ids', '0', '-1', '1',
    '--scales', '0', '1', '2',
    '--png',
    '--weights_init', 'pretrained',
    '--pose_model_input', 'pairs',
    '--pose_model_type', 'separate_resnet',
    '--num_workers', '4',
    '--save_frequency', '1',
    '--log_frequency', '250',
])

if __name__ == '__main__':
    print("=" * 60)
    print("  Lite-Mono 基线训练 (多源: EndoSLAM+SCARED+Real)")
    print(f"  数据: {opts.data_path}")
    print(f"  模型: {opts.model_name}")
    print(f"  LR: {opts.lr}")
    print(f"  深度监督权重: {opts.depth_supervision_weight}")
    print(f"  Epochs: {opts.num_epochs}")
    print("=" * 60)
    trainer = Trainer(opts)
    trainer.train()
