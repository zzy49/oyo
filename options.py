# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "endoslam", "cameras", "mixed", "c3vd", "multi", "multi_c3vd", "c3vd_full", "endoslam_full"],
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "endoscopic", "scared", "endoslam", "cameras", "mixed", "c3vd", "endoslam_depth", "multi_source"])
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--img_ext",
                                 type=str,
                                 help="image extension to use (.png or .jpg)",
                                 default='.jpg')
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")

        # SEMI-SUPERVISED options
        self.parser.add_argument("--use_depth_gt",
                                 help="if set, uses ground truth depth as auxiliary supervision",
                                 action="store_true")
        self.parser.add_argument("--depth_gt_weight",
                                 type=float,
                                 help="weight for ground truth depth supervision loss",
                                 default=0.1)
        self.parser.add_argument("--depth_gt_scale",
                                 type=float,
                                 help="scale factor applied to GT depth before loss (e.g. 100 for 0.1/100 range)",
                                 default=1.0)
        self.parser.add_argument("--photometric_weight",
                                 type=float,
                                 help="weight for photometric loss",
                                 default=1.0)
        self.parser.add_argument("--use_depth_consistency",
                                 help="if set, uses depth consistency loss between frames",
                                 action="store_true")
        self.parser.add_argument("--depth_consistency_weight",
                                 type=float,
                                 help="weight for depth consistency loss",
                                 default=0.1)
        self.parser.add_argument("--use_gt_pose",
                                 help="if set, uses GT camera poses for photometric warping (C3VD only)",
                                 action="store_true")
        self.parser.add_argument("--gt_pose_photometric_weight",
                                 type=float,
                                 help="weight for photometric loss with GT poses (replaces standard photometric)",
                                 default=0.3)
        self.parser.add_argument("--gt_depth_consistency_weight",
                                 type=float,
                                 help="weight for multi-view depth consistency loss with GT poses",
                                 default=0.1)
        self.parser.add_argument("--use_log_consistency",
                                 help="use log-space L1 for depth consistency loss (scale-invariant, prevents model from being rewarded for predicting small depths)",
                                 action="store_true",
                                 default=True)
        self.parser.add_argument("--scale_anchor_weight",
                                 type=float,
                                 help="weight for explicit scale anchor loss: ((pred_mean/gt_mean)-1)^2. Anchors depth scale to GT. 0=disabled. Recommended 0.05-0.1.",
                                 default=0.0)
        self.parser.add_argument("--gt_pose_supervision_weight",
                                 type=float,
                                 help="weight for GT pose supervision loss (trains PoseCNN to match GT poses, C3VD only). 0=disabled.",
                                 default=0.0)
        self.parser.add_argument("--pose_loss_type",
                                 type=str,
                                 help="pose supervision loss type: 'l1_matrix' (L1 on full 4x4) or 'log_translation' (log-scale + direction + rotation)",
                                 default="l1_matrix",
                                 choices=["l1_matrix", "log_translation"])
        self.parser.add_argument("--freeze_encoder_depth",
                                 help="freeze encoder and depth decoder (Stage 2: train PoseCNN only)",
                                 action="store_true")
        self.parser.add_argument("--use_motion_encoder",
                                 help="use MotionEncoder to inject frame-pair motion features into depth decoder",
                                 action="store_true")
        self.parser.add_argument("--freeze_motion_depth",
                                 help="freeze depth decoder when training motion encoder (only train motion+fuse)",
                                 action="store_true")
        self.parser.add_argument("--freeze_pose",
                                 help="freeze PoseCNN/PoseNet (for motion encoder training, keep pretrained pose)",
                                 action="store_true")
        self.parser.add_argument("--pnf_depth_weight",
                                 type=float,
                                 help="weight for PnP-friendly sparse depth consistency loss (0=disabled). Uses GT poses to verify geometric consistency of sparse feature points across views, without touching PoseNet.",
                                 default=0.0)
        self.parser.add_argument("--pnf_num_samples",
                                 type=int,
                                 help="number of sparse sample points per frame pair for PnP-friendly depth loss",
                                 default=1024)
        self.parser.add_argument("--use_pose_smooth",
                                 help="if set, uses pose smoothness loss",
                                 action="store_true")
        self.parser.add_argument("--pose_smooth_weight",
                                 type=float,
                                 help="weight for pose smoothness loss",
                                 default=0.2)
        self.parser.add_argument("--use_depth_regularization",
                                 help="if set, uses depth regularization to prevent collapse to max_depth",
                                 action="store_true")
        self.parser.add_argument("--depth_reg_weight",
                                 type=float,
                                 help="weight for depth regularization loss",
                                 default=0.05)
        self.parser.add_argument("--use_anti_collapse",
                                 help="if set, uses anti-collapse loss to prevent trivial solutions",
                                 action="store_true")
        self.parser.add_argument("--use_color_jitter",
                                 help="if set, uses color jittering augmentation",
                                 action="store_true")
        self.parser.add_argument("--anti_collapse_weight",
                                 type=float,
                                 help="weight for anti-collapse loss",
                                 default=0.01)
        # Stage2 v3: 深度结构改进
        self.parser.add_argument("--use_depth_gradient",
                                 help="if set, uses depth gradient matching loss",
                                 action="store_true")
        self.parser.add_argument("--depth_gradient_weight",
                                 type=float,
                                 help="weight for depth gradient matching loss",
                                 default=0.5)
        self.parser.add_argument("--use_silog",
                                 help="if set, uses SILog loss instead of log-L1 for GT depth",
                                 action="store_true")
        self.parser.add_argument("--use_multiscale_gt",
                                 help="if set, applies GT depth supervision at multiple scales",
                                 action="store_true")
        self.parser.add_argument("--use_normal_loss",
                                 help="if set, uses surface normal consistency loss",
                                 action="store_true")
        self.parser.add_argument("--normal_loss_weight",
                                 type=float,
                                 help="weight for surface normal consistency loss",
                                 default=0.2)
        # 分类深度头
        self.parser.add_argument("--num_bins",
                                 type=int,
                                 help="number of depth bins for classification head (0=regression)",
                                 default=0)
        self.parser.add_argument("--classify_weight",
                                 type=float,
                                 help="weight for bin classification (cross-entropy) loss",
                                 default=2.0)
        self.parser.add_argument("--residual_weight",
                                 type=float,
                                 help="weight for residual regression loss",
                                 default=1.0)
        self.parser.add_argument("--freeze_residual",
                                 help="if set, freezes residual head during training",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])
        self.parser.add_argument("--reset_epoch",
                                 help="if set, start training from epoch 0 even when loading weights",
                                 action="store_true")
        self.parser.add_argument("--start_epoch",
                                 type=int,
                                 help="manually override starting epoch. Use -1 for auto-detect from folder name.",
                                 default=-1)

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
