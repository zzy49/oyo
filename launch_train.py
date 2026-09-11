"""Quick test launcher for monodepth2 training on EndoSLAM."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from options import MonodepthOptions
from trainer import Trainer

# Build options directly (skip argparse to avoid double-parse)
opts_mgr = MonodepthOptions()
opts = opts_mgr.parse()

# Override for endoslam test
opts.dataset = 'endoslam'
opts.split = 'endoslam'
opts.height = 192
opts.width = 640
opts.batch_size = 2
opts.num_epochs = 2
opts.num_layers = 18
opts.learning_rate = 1e-4
opts.model_name = 'test_endoslam'
opts.log_dir = './logs/test_depth'
opts.frame_ids = [0, -1, 1]
opts.scales = [0, 1, 2, 3]
opts.min_depth = 1.0
opts.max_depth = 500.0
opts.num_bins = 64
opts.num_workers = 0
opts.no_ssim = True
opts.disable_automasking = True
opts.use_stereo = False
opts.weights_init = 'pretrained'
opts.pose_model_type = 'posecnn'
opts.pose_model_input = 'pairs'
opts.img_ext = '.png'
opts.png = True
opts.scheduler_step_size = 15
opts.data_path = r'F:\dataset'
opts.avgrgb_census = False
opts.disable_motion_masking = True
print(f"Dataset: {opts.dataset}, Split: {opts.split}")
print(f"Data path: {opts.data_path}")
print(f"Model: {opts.model_name}")

trainer = Trainer(opts)
trainer.train()
