# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


def normalize_image(x):
    """Normalize tensor to [0,1] for TensorBoard display."""
    if len(x.shape) >= 3:
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales,
            num_bins=self.opt.num_bins)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # 预计算 bin centers (分类头)
        if self.opt.num_bins > 0:
            self.bin_centers = make_bin_centers(
                self.opt.min_depth, self.opt.max_depth, self.opt.num_bins)
        else:
            self.bin_centers = None

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        # K-归一化位姿 (方案B): 统一标准相机, PoseNet 在一致坐标空间学习
        # 三个数据源的物理 K (fx 210~518) 全部映射到这个中间值标准相机
        self.K_CANON = torch.tensor([
            [320.0, 0.0,   320.0],
            [0.0,   96.0,  96.0],
            [0.0,   0.0,   1.0]
        ], dtype=torch.float32)
        self.K_CANON_INV = torch.inverse(self.K_CANON)

        # Motion Encoder: 帧间运动特征注入深度decoder (时序深度模型)
        # 必须在 optimizer 创建之前初始化, 确保参数被优化器追踪
        self.use_motion_encoder = getattr(self.opt, 'use_motion_encoder', False)
        if self.use_motion_encoder:
            self.models["motion_encoder"] = networks.MotionEncoder()
            self.models["motion_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["motion_encoder"].parameters())
            print("  [motion] MotionEncoder added (%.1fK params)" %
                  (sum(p.numel() for p in self.models["motion_encoder"].parameters()) / 1000))

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        # 冻结残差头 (阶段1训练)
        if self.opt.num_bins > 0 and getattr(self.opt, 'freeze_residual', False):
            for name, param in self.models["depth"].named_parameters():
                if 'resconv' in name:
                    param.requires_grad = False
                    print(f"  [freeze] {name}")

        # 冻结 encoder + depth (Stage 2: 专门训 PoseCNN)
        if getattr(self.opt, 'freeze_encoder_depth', False):
            for name, param in self.models["encoder"].named_parameters():
                param.requires_grad = False
            for name, param in self.models["depth"].named_parameters():
                param.requires_grad = False
            print("  [freeze] encoder + depth (only training PoseCNN)")

        # Motion Encoder 冻结逻辑 (motion_encoder 已在 optimizer 之前创建)
        if self.use_motion_encoder:
            # 冻结深度decoder (可选: 只训练motion_encoder + fusion)
            if getattr(self.opt, 'freeze_motion_depth', False):
                for name, param in self.models["depth"].named_parameters():
                    param.requires_grad = False
                print("  [freeze] depth decoder (only training motion_encoder)")

        # 冻结 PoseCNN/PoseNet (motion encoder训练时保持预训练位姿网络)
        if getattr(self.opt, 'freeze_pose', False) and self.use_pose_net:
            if "pose_encoder" in self.models:
                for name, param in self.models["pose_encoder"].named_parameters():
                    param.requires_grad = False
            if "pose" in self.models:
                for name, param in self.models["pose"].named_parameters():
                    param.requires_grad = False
            print("  [freeze] PoseCNN/PoseNet (keeping pretrained pose)")

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "endoscopic": datasets.EndoscopicDataset,
                         "endoslam": datasets.EndoscopicDataset,
                         "cameras": datasets.EndoscopicDataset,
                         "mixed": datasets.EndoscopicDataset,
                         "scared": datasets.SCAREDDataset,
                         "c3vd": datasets.C3VDDataset,
                         "multi_source": "__multi__"}  # 特殊标记: 多源混合
        self.dataset = datasets_dict[self.opt.dataset]

        # endoscopic/scared: 从 data_path 下读取文件列表
        # endoslam/mixed/c3vd: 从 splits/ 下读取
        # multi_source: 特殊处理（见下方）
        if self.opt.dataset == 'multi_source':
            # 多源混合训练: EndoSLAM + SCARED
            # 从 splits/multi/ 读取端到端文件列表
            fpath = os.path.join(os.path.dirname(__file__), "splits", "multi", "{}_files.txt")
            train_filenames_path = fpath.format("train")
            val_filenames_path = fpath.format("val")
            train_filenames = readlines(train_filenames_path)
            val_filenames = readlines(val_filenames_path)
            img_ext = getattr(self.opt, 'img_ext', '.jpg')
            print(f"DEBUG: Multi-source mode")
            print(f"DEBUG: train_filenames = {len(train_filenames)} lines")
        elif self.opt.dataset in ['endoscopic', 'scared']:
            train_filenames_path = os.path.join(self.opt.data_path, "train_files.txt")
            val_filenames_path = os.path.join(self.opt.data_path, "val_files.txt")
            train_filenames = readlines(train_filenames_path)
            val_filenames = readlines(val_filenames_path)
            img_ext = getattr(self.opt, 'img_ext', '.jpg')
        else:
            fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
            train_filenames_path = fpath.format("train")
            val_filenames_path = fpath.format("val")
            train_filenames = readlines(train_filenames_path)
            val_filenames = readlines(val_filenames_path)
            img_ext = getattr(self.opt, 'img_ext', '.jpg')
        
        # 确保 data_path 是绝对路�?
        self.opt.data_path = os.path.abspath(self.opt.data_path)
        
        print(f"DEBUG: img_ext = {img_ext}")
        print(f"DEBUG: data_path = {self.opt.data_path}")

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        if self.opt.dataset == 'multi_source':
            # 分拆 filenames: endoslam / scared / real
            endo_files = []
            scared_files = []
            real_files = []
            for line in train_filenames:
                line = line.strip()
                if not line: continue
                if line.startswith('scared'):
                    scared_files.append(line)
                elif line.startswith('real'):
                    real_files.append(line)
                else:
                    endo_files.append(line)
            print(f"  EndoSLAM: {len(endo_files)}, SCARED: {len(scared_files)}, Real: {len(real_files)}")

            # 多源数据路径
            multi_cfg = getattr(self.opt, 'multi_data_paths', {})
            endo_path = multi_cfg.get('endoslam', r'E:\data1\monodepth2\EndoSLAM')
            scared_path = multi_cfg.get('scared', r'E:\data1\monodepth2\scared_extracted')
            real_path = multi_cfg.get('real', r'F:\dataset')

            # 构建子数据集
            sub_ds_list = []  # (dataset, weight)

            endo_ds = datasets.EndoscopicDataset(
                endo_path, endo_files, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext='.png')
            scared_ds = datasets.SCAREDDataset(
                scared_path, scared_files, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext='.png')
            real_ds = datasets.RealColonDataset(
                real_path, real_files, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext='.png')

            # 混合权重 (可通过命令行覆盖)
            endo_weight = getattr(self.opt, 'endo_ratio', 0.4)
            scared_weight = getattr(self.opt, 'scared_ratio', 0.2)
            real_weight = getattr(self.opt, 'real_ratio', 0.4)
            total_w = endo_weight + scared_weight + real_weight
            train_dataset = datasets.MultiSourceDataset(
                [(endo_ds, endo_weight / total_w),
                 (scared_ds, scared_weight / total_w),
                 (real_ds, real_weight / total_w)])

            # Val: 同理
            endo_val_files = [l.strip() for l in val_filenames
                              if l.strip() and not l.strip().startswith('scared')
                              and not l.strip().startswith('real')]
            scared_val_files = [l.strip() for l in val_filenames
                                if l.strip() and l.strip().startswith('scared')]
            real_val_files = [l.strip() for l in val_filenames
                              if l.strip() and l.strip().startswith('real')]
            val_endo_ds = datasets.EndoscopicDataset(
                endo_path, endo_val_files, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext='.png')
            val_scared_ds = datasets.SCAREDDataset(
                scared_path, scared_val_files, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext='.png')
            val_real_ds = datasets.RealColonDataset(
                real_path, real_val_files, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext='.png')
            val_dataset = datasets.MultiSourceDataset(
                [(val_endo_ds, 1.0), (val_scared_ds, 1.0), (val_real_ds, 1.0)])
        else:
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext,
                depth_gt_scale=getattr(self.opt, 'depth_gt_scale', 1.0),
                pnf_num_samples=getattr(self.opt, 'pnf_num_samples', 0))
            val_dataset = self.dataset(
                self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext,
                depth_gt_scale=getattr(self.opt, 'depth_gt_scale', 1.0),
                pnf_num_samples=getattr(self.opt, 'pnf_num_samples', 0))
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        # 尝试从加载的权重文件夹推断起始epoch
        self.epoch = 0
        if getattr(self.opt, 'start_epoch', -1) >= 0:
            self.epoch = self.opt.start_epoch
            print(f"Resuming training from epoch {self.epoch} (manual override)")
        elif self.opt.load_weights_folder and not self.opt.reset_epoch:
            folder_name = os.path.basename(self.opt.load_weights_folder)
            if folder_name.startswith("weights_"):
                try:
                    start_epoch = int(folder_name.split("_")[1])
                    self.epoch = start_epoch
                    print(f"Resuming training from epoch {self.epoch}")
                except ValueError:
                    pass
        elif self.opt.reset_epoch:
            print("Starting training from epoch 0 (reset_epoch enabled)")
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.epoch, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
                # post-epoch callback: 如评估隔离集
                if hasattr(self.opt, 'post_epoch_fn'):
                    self.opt.post_epoch_fn(self)

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if ("depth_gt", 0, 0) in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                # 训练初期打印各项loss量级，用于校准loss权重
                if self.step < self.opt.log_frequency * 10:
                    loss_items = sorted(
                        [(k, v.cpu().item() if hasattr(v, 'cpu') else v)
                         for k, v in losses.items() if k.startswith("loss/")],
                        key=lambda x: x[1], reverse=True)
                    loss_str = " | ".join(
                        ["{}: {:.6f}".format(k.replace("loss/", ""), v)
                         for k, v in loss_items])
                    print(f"[LOSS BREAKDOWN step={self.step}] {loss_str}")

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])

            # MotionEncoder: 注入帧间运动特征到encoder skip连接
            motion_pair_id = 0  # 0=自配对(无运动信号), 1/-1=有效帧对
            if self.use_motion_encoder:
                # 获取相邻帧 (优先 +1, 否则回退到 -1, 最后自配对)
                if ("color_aug", 1, 0) in inputs:
                    next_frame = inputs[("color_aug", 1, 0)]
                    motion_pair_id = 1
                elif ("color_aug", -1, 0) in inputs:
                    next_frame = inputs[("color_aug", -1, 0)]
                    motion_pair_id = -1
                else:
                    next_frame = inputs[("color_aug", 0, 0)]  # 自配对 (零运动信号)

                # Frame 0→t: MotionEnc(frame_0, frame_t) → 注入encoder skip
                frame_pair_0t = torch.cat(
                    [inputs[("color_aug", 0, 0)], next_frame], dim=1)
                motion_feats_0t = self.models["motion_encoder"](frame_pair_0t)

                # 融合运动特征到encoder skip连接 (level 0~3)
                for level in range(4):
                    features[level] = self.models["motion_encoder"].fuse_skip(
                        features[level], motion_feats_0t[level], level)

            outputs = self.models["depth"](features)

            # ── 强时序深度一致性: 独立预测 frame_t 的深度 ──
            # depth_t = Decoder(Enc(frame_t) + MotionEnc(frame_t, frame_0))
            # 两帧独立推理, 共享权重, 输入顺序相反
            if self.use_motion_encoder and motion_pair_id != 0 \
                    and self.opt.use_depth_consistency:
                features_t = self.models["encoder"](next_frame)
                frame_pair_t0 = torch.cat(
                    [next_frame, inputs[("color_aug", 0, 0)]], dim=1)
                motion_feats_t0 = self.models["motion_encoder"](frame_pair_t0)
                for level in range(4):
                    features_t[level] = self.models["motion_encoder"].fuse_skip(
                        features_t[level], motion_feats_t0[level], level)
                outputs_t = self.models["depth"](features_t)
                # DepthDecoder 输出 bins/residual (分类头) 或 disp (回归头)
                # 需要手动计算实际深度值
                if self.opt.num_bins > 0:
                    bin_logits_t = outputs_t[("bins", 0)]
                    residual_t = outputs_t.get(("residual", 0), None)
                    depth_t = depth_from_bins(bin_logits_t, self.bin_centers, residual_t)
                else:
                    disp_t = outputs_t[("disp", 0)]
                    _, depth_t = disp_to_depth(disp_t, self.opt.min_depth, self.opt.max_depth)
                outputs[("depth", motion_pair_id, 0)] = depth_t

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def _normalize_images_for_pose(self, images, K_phys_batch):
        """将物理相机图像 warp 到标准相机 (K_CANON) 空间。

        方案B核心: 不同内参的图像映射到统一虚拟相机,
        PoseNet 在统一坐标下预测位姿, 天然解耦内参差异。

        Args:
            images:     [B, C, H, W] 物理相机图像
            K_phys_batch: [B, 3, 3] 每样本的物理 K 矩阵 (已缩放到 H×W 分辨率)
        Returns:
            [B, C, H, W] 标准相机空间图像
        """
        B, C, H, W = images.shape
        device = images.device
        K_can = self.K_CANON.to(device)

        # 标准相机像素坐标 → 归一化坐标
        y_idx = torch.arange(H, dtype=torch.float32, device=device).view(1, H, 1)
        x_idx = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, W)
        x_can_norm = (x_idx - K_can[0, 2]) / K_can[0, 0]  # [1, 1, W]
        y_can_norm = (y_idx - K_can[1, 2]) / K_can[1, 1]  # [1, H, 1]

        # 归一化坐标 → 物理相机像素坐标 (broadcast to [B, H, W])
        fx_phys = K_phys_batch[:, 0, 0].view(B, 1, 1)  # [B, 1, 1]
        fy_phys = K_phys_batch[:, 1, 1].view(B, 1, 1)
        cx_phys = K_phys_batch[:, 0, 2].view(B, 1, 1)
        cy_phys = K_phys_batch[:, 1, 2].view(B, 1, 1)

        u_phys = (fx_phys * x_can_norm + cx_phys).expand(B, H, W)  # [B, H, W]
        v_phys = (fy_phys * y_can_norm + cy_phys).expand(B, H, W)  # [B, H, W]

        # grid_sample 归一化到 [-1, 1]
        grid_x = 2.0 * u_phys / W - 1.0
        grid_y = 2.0 * v_phys / H - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]

        return F.grid_sample(images, grid, mode='bilinear',
                             padding_mode='border', align_corners=False)

    def _convert_pose_canonical_to_physical(self, axisangle, translation, K_phys_batch):
        """将 PoseNet 输出从标准相机空间转换到物理相机空间。

        旋转 R 不变 (3D 旋转与相机内参无关)。
        平移 t 需缩放: t_phys_i = t_can_i * f_can_i / f_phys_i  (i ∈ {x, y})
        z 分量不变 (深度方向不受 fx/fy 影响)。

        Args:
            axisangle:   [B, 1, 3] 或 [B, 1, 1, 3]
            translation: [B, 1, 3] 或 [B, 1, 1, 3]
            K_phys_batch: [B, 3, 3]
        Returns:
            (axisangle, translation_phys)
        """
        B = axisangle.shape[0]
        device = axisangle.device
        K_can = self.K_CANON.to(device)

        # 标准化到 [B, 1, 3] (兼容 PoseCNN 的 [B,1,1,3] 输出)
        if translation.dim() == 4:
            translation = translation.squeeze(2)
            axisangle = axisangle.squeeze(2)

        fx_phys = K_phys_batch[:, 0, 0].view(B, 1, 1).clamp(min=50.0)  # [B, 1, 1]
        fy_phys = K_phys_batch[:, 1, 1].view(B, 1, 1).clamp(min=50.0)

        sx = K_can[0, 0] / fx_phys  # [B, 1, 1]
        sy = K_can[1, 1] / fy_phys

        translation_phys = translation.clone()
        translation_phys[:, :, 0:1] = translation[:, :, 0:1] * sx
        translation_phys[:, :, 1:2] = translation[:, :, 1:2] * sy
        # z 分量保留不变

        return axisangle, translation_phys

    def predict_poses(self, inputs, features):
        """Predict poses between input frames (K-归一化坐标方案B)。

        关键改动:
        1. 将输入图像 warp 到标准相机 (K_CANON) 再喂给 PoseNet
        2. PoseNet 预测标准相机下的位姿 (T_canonical)
        3. 用 K_phys 将 T_canonical 转换为物理位姿 T_physical
        4. T_physical 用于后续投影 (与现有管道兼容)
        """
        outputs = {}

        # 获取物理 K 矩阵 (scale 0)
        K_phys = inputs[("K", 0)]  # [B, 3, 3]

        if self.num_pose_frames == 2:
            # 逐对预测位姿 (pairs 模式，默认)
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {}
                for f_i in self.opt.frame_ids:
                    if f_i != "s":
                        # 方案B: 归一化图像 → PoseNet 在统一坐标空间下学习
                        pose_feats[f_i] = self._normalize_images_for_pose(
                            inputs["color_aug", f_i, 0], K_phys)

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](
                            torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)

                    # 方案B: 标准相机位姿 → 物理相机位姿
                    axisangle_phys, translation_phys = \
                        self._convert_pose_canonical_to_physical(
                            axisangle, translation, K_phys)

                    outputs[("axisangle", 0, f_i)] = axisangle_phys
                    outputs[("translation", 0, f_i)] = translation_phys
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle_phys[:, 0:1], translation_phys[:, 0:1],
                        invert=(f_i < 0))

        else:
            # all 模式: 所有帧一起预测
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [self._normalize_images_for_pose(
                        inputs[("color_aug", i, 0)], K_phys)
                     for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    axisangle_phys, translation_phys = \
                        self._convert_pose_canonical_to_physical(
                            axisangle[:, i:i+1], translation[:, i:i+1], K_phys)
                    outputs[("axisangle", 0, f_i)] = axisangle_phys
                    outputs[("translation", 0, f_i)] = translation_phys
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle_phys[:, 0], translation_phys[:, 0])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if ("depth_gt", 0, 0) in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.几何变换
        """
        for scale in self.opt.scales:
            if self.opt.num_bins > 0:
                # 分类头: bins + residual → depth → disp (用于光流warp)
                bin_logits = outputs[("bins", scale)]
                residual = outputs.get(("residual", scale), None)
                depth = depth_from_bins(bin_logits, self.bin_centers, residual)
                # 反向: depth → normalized disp (用于平滑损失和下游)
                min_disp = 1.0 / self.opt.max_depth
                max_disp = 1.0 / self.opt.min_depth
                disp = (1.0 / (depth + 1e-7) - min_disp) / (max_disp - min_disp)
                # 保存深度输出
                outputs[("disp", scale)] = disp
                outputs[("depth", 0, scale)] = depth
                if scale == 0:
                    outputs[("depth", 0, 0)] = depth
            else:
                disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                elif self.opt.use_gt_pose and ("gt_cam_T_cam", 0, frame_id) in inputs:
                    T = inputs[("gt_cam_T_cam", 0, frame_id)]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0:1], translation[:, 0:1] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K_4x4", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.2 * ssim_loss + 0.8 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch,计算loss
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)
            
            # 应用高光mask（如果存在）
            if ("specular_mask", 0, source_scale) in inputs:
                specular_mask = inputs[("specular_mask", 0, source_scale)]
                # 扩展mask维度以匹配reprojection_losses
                specular_mask = specular_mask.unsqueeze(1).expand_as(reprojection_losses)
                reprojection_losses = reprojection_losses * (1 - specular_mask.float())
                # 归一化
                valid_pixels = (1 - specular_mask.float()).sum() + 1e-7
                reprojection_losses = reprojection_losses * reprojection_losses.numel() / valid_pixels
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            # 应用光度权重 (GT位姿模式使用专用权重)
            pw = self.opt.gt_pose_photometric_weight if self.opt.use_gt_pose else self.opt.photometric_weight
            loss += pw * to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales

        # ── 分类头损失 (num_bins > 0) ──
        if self.opt.num_bins > 0 and ("depth_gt", 0, 0) in inputs:
            depth_gt = inputs[("depth_gt", 0, 0)].to(self.device)
            gt_scale = getattr(self.opt, 'depth_gt_scale', 1.0)
            valid_mask = (depth_gt > 1.0 * gt_scale) & (depth_gt < 500.0 * gt_scale)

            if valid_mask.sum() > 0:
                bin_centers_t = torch.from_numpy(self.bin_centers).to(self.device).to(depth_gt.dtype)

                for scale in self.opt.scales:
                    bin_logits = outputs[("bins", scale)]  # [B, N, H, W]
                    residual = outputs.get(("residual", scale), None)

                    # 将GT对齐到当前scale的分辨率
                    h_s, w_s = bin_logits.shape[2], bin_logits.shape[3]
                    gt_s = F.interpolate(depth_gt, [h_s, w_s], mode="nearest")
                    valid_s = (gt_s > 1.0 * gt_scale) & (gt_s < 500.0 * gt_scale)

                    if valid_s.sum() < 100:
                        continue

                    # 分类损失: GT → nearest bin index → cross-entropy
                    gt_s_flat = gt_s[valid_s]  # (N_valid,)
                    # Find nearest bin for each valid pixel
                    dists = torch.abs(gt_s_flat.unsqueeze(1) - bin_centers_t.unsqueeze(0))
                    gt_bin_idx = torch.argmin(dists, dim=1)  # (N_valid,)

                    bin_logits_flat = bin_logits.permute(0, 2, 3, 1)[valid_s.squeeze(1)]  # (N_valid, N)
                    classify_loss = F.cross_entropy(bin_logits_flat, gt_bin_idx, reduction='mean')

                    # 权重递减
                    scale_w = self.opt.classify_weight / (2 ** scale)
                    total_loss += scale_w * classify_loss
                    losses[f"loss/classify_{scale}"] = classify_loss

                    # 残差损失 (仅GT监督)
                    if residual is not None:
                        # 计算粗深度 (without residual)
                        coarse = depth_from_bins(bin_logits, self.bin_centers, residual=None)
                        # 逐像素局部 bin 宽度 (detach: 残差target不应反向影响分类)
                        with torch.no_grad():
                            probs = torch.softmax(bin_logits, dim=1)
                            bw = bin_centers_t[1:] - bin_centers_t[:-1]  # (N-1,)
                            adj_avg = (probs[:, :-1] + probs[:, 1:]) / 2.0
                            local_width = (adj_avg * bw[None, :, None, None]).sum(dim=1, keepdim=True)
                        residual_target = (gt_s - coarse) / (local_width / 2.0)
                        residual_target = torch.clamp(residual_target, -1.0, 1.0)
                        res_loss = F.l1_loss(
                            residual[valid_s], residual_target[valid_s], reduction='mean')
                        scale_w_res = self.opt.residual_weight / (2 ** scale)
                        total_loss += scale_w_res * res_loss
                        losses[f"loss/residual_{scale}"] = res_loss

        # Depth consistency loss（帧间深度一致性）
        if self.opt.use_depth_consistency and len(self.opt.frame_ids) > 1:
            depth_consistency_loss = self.compute_depth_consistency_loss(inputs, outputs)
            dcw = self.opt.gt_depth_consistency_weight if self.opt.use_gt_pose else self.opt.depth_consistency_weight
            total_loss += dcw * depth_consistency_loss
            losses["loss/depth_consistency"] = depth_consistency_loss

        # 显式尺度锚定: 强制深度均值与GT一致 (与log-space consistency互补)
        if self.opt.scale_anchor_weight > 0:
            scale_anchor_loss = self.compute_scale_anchor_loss(inputs, outputs)
            total_loss += self.opt.scale_anchor_weight * scale_anchor_loss
            losses["loss/scale_anchor"] = scale_anchor_loss

        # PnP-friendly 稀疏特征点深度一致性 (GT位姿几何验证，不动PoseNet)
        if self.opt.pnf_depth_weight > 0 and len(self.opt.frame_ids) > 1:
            pnf_loss = self.compute_pnf_depth_loss(inputs, outputs)
            total_loss += self.opt.pnf_depth_weight * pnf_loss
            losses["loss/pnf_depth"] = pnf_loss

        # 半监督：深度真值辅助监督项（仅当 --use_depth_gt 且有真值时生效）
        if self.opt.use_depth_gt and ("depth_gt", 0, 0) in inputs:
            depth_pred = outputs[("depth", 0, 0)]         # [B, 1, H, W]
            depth_gt   = inputs[("depth_gt", 0, 0)].to(self.device)  # [B, 1, H, W]

            # 有效像素掩码：根据 GT 缩放因子调整范围
            gt_scale = getattr(self.opt, 'depth_gt_scale', 1.0)
            valid_mask = (depth_gt > 1.0 * gt_scale) & (depth_gt < 500.0 * gt_scale)

            if valid_mask.sum() > 0:
                # === GT 深度主损失 ===
                if getattr(self.opt, 'use_silog', False):
                    # SILog Loss: scale-invariant，强制结构一致
                    depth_gt_loss = self.compute_silog_loss(depth_pred, depth_gt, valid_mask)
                else:
                    # 对数尺度 L1 损失（log 域更平滑）
                    log_diff = torch.abs(
                        torch.log(depth_pred[valid_mask] + 1e-7) -
                        torch.log(depth_gt[valid_mask]   + 1e-7)
                    )
                    depth_gt_loss = log_diff.mean()
                total_loss += self.opt.depth_gt_weight * depth_gt_loss
                losses["loss/depth_gt"] = depth_gt_loss
                
                # === 深度梯度匹配损失 ===
                if getattr(self.opt, 'use_depth_gradient', False):
                    grad_loss = self.compute_depth_gradient_loss(depth_pred, depth_gt, valid_mask)
                    total_loss += self.opt.depth_gradient_weight * grad_loss
                    losses["loss/depth_gradient"] = grad_loss
                
                # === 表面法线一致性损失 ===
                if getattr(self.opt, 'use_normal_loss', False):
                    normal_loss = self.compute_normal_consistency_loss(depth_pred, depth_gt, valid_mask)
                    total_loss += self.opt.normal_loss_weight * normal_loss
                    losses["loss/normal"] = normal_loss
                
                # === 多尺度GT监督 ===
                if getattr(self.opt, 'use_multiscale_gt', False):
                    for s in self.opt.scales[1:]:  # scale 1, 2, 3
                        depth_pred_s = outputs[("depth", 0, s)]
                        # 将GT下采样到对应尺度
                        h_s = self.opt.height // (2 ** s)
                        w_s = self.opt.width // (2 ** s)
                        depth_gt_s = F.interpolate(depth_gt, [h_s, w_s], mode="nearest")
                        depth_pred_s_up = F.interpolate(depth_pred_s, [h_s, w_s], mode="bilinear", align_corners=False)
                        valid_s = (depth_gt_s > 0.01 * gt_scale) & (depth_gt_s < 1.0 * gt_scale)
                        if valid_s.sum() > 0:
                            if getattr(self.opt, 'use_silog', False):
                                ms_loss = self.compute_silog_loss(depth_pred_s_up, depth_gt_s, valid_s)
                            else:
                                ms_loss = torch.abs(
                                    torch.log(depth_pred_s_up[valid_s] + 1e-7) -
                                    torch.log(depth_gt_s[valid_s] + 1e-7)
                                ).mean()
                            # 多尺度权重递减
                            scale_weight = self.opt.depth_gt_weight / (2 ** s)
                            total_loss += scale_weight * ms_loss
                            losses[f"loss/depth_gt_s{s}"] = ms_loss
                
                # 打印调试信息（每1000次迭代）
                if self.step % 1000 == 0:
                    valid_ratio = valid_mask.float().mean().item() * 100
                    debug_str = f"[Stage2] gt_loss={depth_gt_loss.item():.6f}, valid={valid_ratio:.1f}%"
                    debug_str += f", pred_mean={depth_pred[valid_mask].mean().item():.4f}"
                    debug_str += f", gt_mean={depth_gt[valid_mask].mean().item():.4f}"
                    if getattr(self.opt, 'use_depth_gradient', False) and 'loss/depth_gradient' in losses:
                        debug_str += f", grad={losses['loss/depth_gradient'].item():.6f}"
                    if getattr(self.opt, 'use_normal_loss', False) and 'loss/normal' in losses:
                        debug_str += f", normal={losses['loss/normal'].item():.6f}"
                    print(debug_str)
        
        # Depth regularization：防止深度塌缩到 min_depth 或 max_depth
        # ===== Depth Regularization（已禁用）=====
        # 用户要求只保留：photometric loss + disparity smoothness
        if False and self.opt.use_depth_regularization:
            pass  # 禁用depth regularization
        
        # ===== Anti-collapse loss（已禁用：半监督GT深度约束替代）=====
        if False and self.opt.use_anti_collapse:
            pass
        
        # ===== Pose smoothness（已禁用）=====
        # 用户要求只保留：photometric loss + disparity smoothness
        if False and self.opt.use_pose_smooth and self.use_pose_net:
            pose_smooth_loss = self.compute_pose_smoothness_loss(inputs, outputs)
            if pose_smooth_loss is not None:
                total_loss += self.opt.pose_smooth_weight * pose_smooth_loss
                losses["loss/pose_smooth"] = pose_smooth_loss

        # GT位姿监督损失: 训练PoseCNN匹配GT位姿
        if getattr(self.opt, 'gt_pose_supervision_weight', 0) > 0 and self.use_pose_net:
            gt_pose_loss = self.compute_gt_pose_supervision_loss(inputs, outputs)
            total_loss += self.opt.gt_pose_supervision_weight * gt_pose_loss
            losses["loss/gt_pose"] = gt_pose_loss

        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)].detach()
        depth_gt = inputs[("depth_gt", 0, 0)]

        # 使用 depth_valid_mask 过滤无效GT
        if ("depth_valid_mask", 0, 0) in inputs:
            valid = inputs[("depth_valid_mask", 0, 0)].bool()
        else:
            valid = depth_gt > 0

        # KITTI 模式: mask + garg/eigen crop
        if "mask" in inputs:
            mask = inputs["mask"] * valid
            depth_pred = torch.clamp(F.interpolate(
                depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
            depth_pred = depth_pred.detach()
            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]
            depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
            depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
        else:
            # 通用模式: 对有效像素计算误差
            mask = valid
            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]

        if depth_gt.numel() < 10:
            return  # 有效像素太少，跳过

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = {}
        for k, v in self.opt.__dict__.copy().items():
            if not callable(v):
                to_save[k] = v

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def compute_pose_smoothness_loss(self, inputs, outputs):
        """计算位姿平滑损失
        
        让相邻帧位姿变化更平滑：||t_t - t_{t+1}||_2 + β * ||R_t - R_{t+1}||_2
        """
        pose_smooth_loss = 0
        beta = 0.1  # 旋转权重
        
        for frame_id in self.opt.frame_ids[1:]:
            if frame_id == "s":
                continue
            
            # 获取相邻帧的位姿
            if ("axisangle", 0, frame_id) in outputs and ("translation", 0, frame_id) in outputs:
                axisangle = outputs[("axisangle", 0, frame_id)]
                translation = outputs[("translation", 0, frame_id)]
                
                # 计算平移平滑（L2范数）
                trans_smooth = torch.norm(translation, p=2, dim=1).mean()
                
                # 计算旋转平滑（L2范数）
                rot_smooth = torch.norm(axisangle, p=2, dim=1).mean()
                
                pose_smooth_loss += trans_smooth + beta * rot_smooth
        
        return pose_smooth_loss / max(len(self.opt.frame_ids) - 1, 1) if pose_smooth_loss != 0 else None

    def compute_depth_consistency_loss(self, inputs, outputs):
        """强时序深度一致性损失 (GT位姿) — 尺度不变版本
        
        将 depth_0 通过 GT 位姿 warp 到相邻帧, 与独立预测的 depth_t 做逐像素 L1。
        使用 **log-space L1** 消除对尺度压缩的隐式奖励:
          |log(Z_warped) - log(depth_t)| = |log(Z_warped / depth_t)|
        
        旧版本 (绝对 L1):
          模型预测深度整体×0.5 → Z_warped和depth_t都缩小一半
          → |Z_warped - depth_t|也缩小一半 → loss下降 → 模型被鼓励欠估计
          → 这是 Umeyama scale=2.06 的根源
        
        新版本 (log L1):
          log(Z_warped/2) - log(depth_t/2) = log(Z_warped) - log(depth_t)
          尺度压缩被完全抵消 → loss不变 → 模型不再获得"偷懒"奖励
          → 模型被强制学习真实的运动感知, 而非缩放深度来讨好loss
        
        公式: loss = |log(Z_warped) - log(depth_t)|₁  (仅有效投影区域)
        """
        total_loss = 0.0
        n_valid = 0

        for frame_id in self.opt.frame_ids[1:]:
            if frame_id == "s":
                continue

            # 必须有独立预测的 depth_t (由 process_batch 中 MotionEncoder 反向推理产生)
            depth_t_key = ("depth", frame_id, 0)
            if depth_t_key not in outputs:
                continue

            pose_key = ("gt_cam_T_cam", 0, frame_id)
            if pose_key not in inputs:
                continue

            depth_0 = outputs[("depth", 0, 0)]   # (B, 1, H, W)
            depth_t = outputs[depth_t_key]        # (B, 1, H, W)
            T = inputs[pose_key]                  # (B, 4, 4) 帧0→帧t
            inv_K = inputs[("inv_K", 0)]
            K_4x4 = inputs[("K_4x4", 0)]

            # 1. 反投影 depth_0 → 3D 点云 (帧0相机系)
            cam_points_0 = self.backproject_depth[0](depth_0, inv_K)  # (B, 4, H*W)

            # 2. GT 位姿变换到帧t相机系
            cam_points_t = torch.matmul(T, cam_points_0)  # (B, 4, H*W)

            # 3. 提取变换后的 Z → 即 depth_0 在帧t视角下的"应有深度"
            Z_warped = cam_points_t[:, 2, :].view_as(depth_0)  # (B, 1, H, W)

            # 4. 投影有效性检查: 3D点是否落在帧t图像范围内
            pix_coords = self.project_3d[0](
                cam_points_0, K_4x4, T)  # (B, H, W, 2), 归一化 [-1, 1]
            in_bounds = ((pix_coords >= -1.0) & (pix_coords <= 1.0)).all(
                dim=-1).float().unsqueeze(1)  # (B, 1, H, W)

            # 5. 有效像素 mask: Z>0 且 投影在图像内 且 depth_t 有效
            valid = (Z_warped > 1.0) & (depth_t > 1.0) & (in_bounds > 0.5)

            n_pixels = valid.sum()
            if n_pixels < 100:
                continue

            # 6. 逐像素 L1: 尺度不变 (log-space) 或 绝对尺度
            if getattr(self.opt, 'use_log_consistency', False):
                # log-space: 尺度压缩被抵消, 模型必须学习真实运动
                eps = 1e-6
                l1 = torch.abs(torch.log(Z_warped + eps) - torch.log(depth_t + eps))
            else:
                # 原始绝对 L1 (向后兼容)
                l1 = torch.abs(Z_warped - depth_t)

            frame_loss = (l1 * valid).sum() / (n_pixels + 1e-8)

            total_loss += frame_loss
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=self.device)
        return total_loss / n_valid

    def compute_scale_anchor_loss(self, inputs, outputs):
        """显式尺度锚定损失: 强制深度预测的均值与 GT 一致
        
        单独 log-space depth_consistency 消除了"偷懒"激励,
        但不能主动把尺度拉回正确量级。scale anchor 填补这个缺口:
          scale_loss = ((pred_mean / gt_mean) - 1)²
        
        只在有 GT 深度的帧上计算 (≤0.05-0.1 权重, 弱约束防漂移)。
        log-space L1 让模型不敢压缩尺度, scale anchor 告诉它正确尺度在哪。
        两者互补, 预期最终深度不再需要 Umeyama scale 2.06。
        """
        depth_gt_key = ("depth_gt", 0, 0)
        if depth_gt_key not in inputs:
            return torch.tensor(0.0, device=self.device)

        depth_pred = outputs[("depth", 0, 0)]      # (B, 1, H, W)
        depth_gt = inputs[depth_gt_key].to(self.device)  # (B, 1, H, W)

        gt_scale = getattr(self.opt, 'depth_gt_scale', 1.0)
        valid_mask = (depth_gt > 1.0 * gt_scale) & (depth_gt < 500.0 * gt_scale)

        batch_losses = []
        for b in range(depth_pred.shape[0]):
            v = valid_mask[b]
            if v.sum() < 100:
                continue
            pred_mean = depth_pred[b][v].mean()
            gt_mean = depth_gt[b][v].mean()
            ratio = pred_mean / (gt_mean + 1e-8)
            batch_losses.append((ratio - 1.0) ** 2)

        if len(batch_losses) == 0:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(batch_losses).mean()

    def compute_gt_pose_supervision_loss(self, inputs, outputs):
        """GT位姿监督损失: 训练PoseCNN预测与GT一致的位姿
        
        支持两种loss类型:
          - l1_matrix: L1 on full 4x4 (默认, 平移+旋转联合监督)
          - log_translation: log(|t|) L1 + direction L1 + rotation MSE
            对数空间压缩尺度差距, 适合模型输出与GT差很多倍的情况
        """
        total_loss = 0.0
        n_valid = 0

        for frame_id in self.opt.frame_ids[1:]:
            if frame_id == "s":
                continue

            pose_key = ("gt_cam_T_cam", 0, frame_id)
            pred_key = ("cam_T_cam", 0, frame_id)

            if pose_key not in inputs or pred_key not in outputs:
                continue

            T_gt = inputs[pose_key]   # (B, 4, 4)
            T_pred = outputs[pred_key]  # (B, 4, 4)

            if getattr(self.opt, 'pose_loss_type', 'l1_matrix') == 'log_translation':
                # ── Log尺度平移 loss: log(|t_pred|) vs log(|t_gt|) ──
                t_pred = T_pred[:, :3, 3]  # (B, 3)
                t_gt = T_gt[:, :3, 3]      # (B, 3)

                # 尺度: log L1 — 55倍差距→log(55)≈4, 梯度均衡
                t_pred_norm = torch.norm(t_pred, p=2, dim=1) + 1e-7
                t_gt_norm = torch.norm(t_gt, p=2, dim=1) + 1e-7
                scale_loss = F.l1_loss(torch.log(t_pred_norm), torch.log(t_gt_norm))

                # 方向: L1 on normalized direction
                t_pred_dir = t_pred / t_pred_norm.unsqueeze(1)
                t_gt_dir = t_gt / t_gt_norm.unsqueeze(1)
                dir_loss = F.l1_loss(t_pred_dir, t_gt_dir)

                # 旋转: MSE on rotation matrix
                R_pred = T_pred[:, :3, :3]
                R_gt = T_gt[:, :3, :3]
                rot_loss = F.mse_loss(R_pred, R_gt)

                frame_loss = 2.0 * scale_loss + 1.0 * dir_loss + 0.5 * rot_loss
            else:
                # L1 on full 4x4: 旋转+平移联合监督
                frame_loss = F.l1_loss(T_pred, T_gt, reduction='mean')

            total_loss += frame_loss
            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=self.device)
        return total_loss / n_valid

    def compute_pnf_depth_loss(self, inputs, outputs):
        """PnP-friendly 特征点全3D坐标一致性损失

        用LK跟踪的静止特征点进行全(X,Y,Z)坐标监督:
        1. 获取预计算的LK跟踪点 (u0,v0,u1,v1)
        2. 采样depth_0在(u0,v0), depth_t在(u1,v1)
        3. 反投影到3D: P_0, P_t
        4. GT位姿变换: P_0_to_t = T @ P_0
        5. loss = ||P_t - P_0_to_t||  (L1, 全3D)

        关键改进 vs 旧版:
        - 监督全(X,Y,Z)而非仅Z: 惩罚XY错误和深度错误的组合效应
        - 使用LK跟踪特征点而非随机采样: 与PnP实际输入一致
        - 静止点筛选: 过滤运动点，保证几何一致性可计算
        """
        N = self.opt.pnf_num_samples
        total_loss = 0.0
        n_valid_frames = 0

        for frame_id in self.opt.frame_ids[1:]:
            if frame_id == "s":
                continue

            track_key = ("lk_tracks", 0, frame_id)
            pose_key = ("gt_cam_T_cam", 0, frame_id)
            depth_t_key = ("depth", frame_id, 0)

            if track_key not in inputs or pose_key not in inputs or depth_t_key not in outputs:
                continue

            tracks = inputs[track_key]       # (B, N, 4) — (u0,v0,u1,v1)
            T = inputs[pose_key]             # (B, 4, 4) — GT相对位姿
            depth_0 = outputs[("depth", 0, 0)]  # (B, 1, H, W)
            depth_t = outputs[depth_t_key]       # (B, 1, H, W)

            B, N, _ = tracks.shape
            H, W = depth_0.shape[2], depth_0.shape[3]
            K_3x3 = inputs[("K_4x4", 0)][:, :3, :3]  # (B, 3, 3)

            # 提取像素坐标
            u0 = tracks[:, :, 0]  # (B, N)
            v0 = tracks[:, :, 1]  # (B, N)
            u1 = tracks[:, :, 2]  # (B, N)
            v1 = tracks[:, :, 3]  # (B, N)

            # 有效性mask: 有效点 > 0, 填充点 = -1
            valid_mask = (u0 >= 0) & (v0 >= 0) & (u1 >= 0) & (v1 >= 0)  # (B, N)

            # grid_sample: 归一化坐标 [-1, 1]
            u0_norm = u0 / (W - 1) * 2.0 - 1.0
            v0_norm = v0 / (H - 1) * 2.0 - 1.0
            grid_0 = torch.stack([u0_norm, v0_norm], dim=-1).unsqueeze(2)  # (B, N, 1, 2)
            depth_0_samples = torch.nn.functional.grid_sample(
                depth_0, grid_0, mode='bilinear',
                padding_mode='border', align_corners=True
            ).squeeze(3).squeeze(1)  # (B, N)

            u1_norm = u1 / (W - 1) * 2.0 - 1.0
            v1_norm = v1 / (H - 1) * 2.0 - 1.0
            grid_1 = torch.stack([u1_norm, v1_norm], dim=-1).unsqueeze(2)
            depth_t_samples = torch.nn.functional.grid_sample(
                depth_t, grid_1, mode='bilinear',
                padding_mode='border', align_corners=True
            ).squeeze(3).squeeze(1)  # (B, N)

            # 反投影到3D (帧0坐标系)
            fx = K_3x3[:, 0, 0].unsqueeze(1)  # (B, 1)
            fy = K_3x3[:, 1, 1].unsqueeze(1)
            cx = K_3x3[:, 0, 2].unsqueeze(1)
            cy = K_3x3[:, 1, 2].unsqueeze(1)

            X_0 = (u0 - cx) / fx * depth_0_samples  # (B, N)
            Y_0 = (v0 - cy) / fy * depth_0_samples
            Z_0 = depth_0_samples

            # 反投影到3D (帧t坐标系)
            X_t = (u1 - cx) / fx * depth_t_samples  # (B, N)
            Y_t = (v1 - cy) / fy * depth_t_samples
            Z_t = depth_t_samples

            # 构建齐次坐标
            ones = torch.ones(B, N, device=self.device)
            P_0 = torch.stack([X_0, Y_0, Z_0, ones], dim=1)  # (B, 4, N)
            P_t = torch.stack([X_t, Y_t, Z_t], dim=1)        # (B, 3, N)

            # GT位姿变换: P_0_to_t = T @ P_0
            P_0_to_t_homo = torch.bmm(T, P_0)      # (B, 4, N)
            P_0_to_t = P_0_to_t_homo[:, :3, :]     # (B, 3, N)

            # ★ 全3D坐标一致性: L1 in XYZ
            diff_3d = torch.abs(P_t - P_0_to_t)  # (B, 3, N)
            point_l1 = diff_3d.mean(dim=1)        # (B, N) — 对XYZ取平均

            # 有效性筛选: 深度>1mm 且 坐标有效
            valid_geom = (depth_0_samples > 1.0) & (depth_t_samples > 1.0) & valid_mask
            valid_f = valid_geom.float()
            n_valid = valid_f.sum()

            if n_valid < 10:
                continue

            frame_loss = (point_l1 * valid_f).sum() / (n_valid + 1e-8)
            total_loss += frame_loss
            n_valid_frames += 1

        if n_valid_frames == 0:
            return torch.tensor(0.0, device=self.device)
        return total_loss / n_valid_frames

    def compute_silog_loss(self, depth_pred, depth_gt, valid_mask):
        """Scale-Invariant Logarithmic Loss (Eigen et al. 2014)
        
        SILog = mean(d^2) - lambda * mean(d)^2
        where d = log(pred) - log(gt)
        
        This loss is invariant to global scale shifts and focuses on structural accuracy.
        """
        log_pred = torch.log(depth_pred[valid_mask] + 1e-7)
        log_gt = torch.log(depth_gt[valid_mask] + 1e-7)
        log_diff = log_pred - log_gt
        
        # SILog: variance of log differences (lambda=0.85 标准值)
        silog = torch.mean(log_diff ** 2) - 0.85 * (torch.mean(log_diff) ** 2)
        return torch.sqrt(silog + 1e-8)  # 取sqrt使梯度更稳定

    def compute_depth_gradient_loss(self, depth_pred, depth_gt, valid_mask):
        """Depth Gradient Matching Loss
        
        强制预测深度的空间梯度与GT深度的空间梯度一致。
        这直接改善深度图的局部空间结构。
        在log域计算梯度，使其对尺度不敏感。
        """
        # 在log域计算梯度（尺度无关）
        log_pred = torch.log(depth_pred + 1e-7)
        log_gt = torch.log(depth_gt + 1e-7)
        
        # 水平梯度
        grad_pred_x = log_pred[:, :, :, :-1] - log_pred[:, :, :, 1:]
        grad_gt_x = log_gt[:, :, :, :-1] - log_gt[:, :, :, 1:]
        
        # 垂直梯度
        grad_pred_y = log_pred[:, :, :-1, :] - log_pred[:, :, 1:, :]
        grad_gt_y = log_gt[:, :, :-1, :] - log_gt[:, :, 1:, :]
        
        # 有效区域的梯度mask（两个相邻像素都有效才计算）
        valid_x = valid_mask[:, :, :, :-1] & valid_mask[:, :, :, 1:]
        valid_y = valid_mask[:, :, :-1, :] & valid_mask[:, :, 1:, :]
        
        # L1梯度差异
        grad_loss_x = torch.abs(grad_pred_x - grad_gt_x)
        grad_loss_y = torch.abs(grad_pred_y - grad_gt_y)
        
        loss_x = grad_loss_x[valid_x].mean() if valid_x.sum() > 0 else torch.tensor(0.0, device=depth_pred.device)
        loss_y = grad_loss_y[valid_y].mean() if valid_y.sum() > 0 else torch.tensor(0.0, device=depth_pred.device)
        
        return loss_x + loss_y

    def compute_normal_consistency_loss(self, depth_pred, depth_gt, valid_mask):
        """Surface Normal Consistency Loss
        
        从深度图计算表面法线，然后强制预测与GT的法线一致。
        这比单纯的深度值匹配更能捕获局部几何结构。
        """
        # 从深度图计算表面法线（用有限差分近似）
        def depth_to_normals(depth):
            # dx, dy梯度
            dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]  # [B,1,H,W-1]
            dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]  # [B,1,H-1,W]
            # 裁剪到相同尺寸
            dx = dx[:, :, :-1, :]  # [B,1,H-1,W-1]
            dy = dy[:, :, :, :-1]  # [B,1,H-1,W-1]
            # 法线 = (-dx, -dy, 1) 归一化
            ones = torch.ones_like(dx)
            normals = torch.cat([-dx, -dy, ones], dim=1)  # [B,3,H-1,W-1]
            normals = F.normalize(normals, dim=1, eps=1e-6)
            return normals
        
        normal_pred = depth_to_normals(depth_pred)
        normal_gt = depth_to_normals(depth_gt)
        
        # 有效区域的mask（3x3邻域都有效）
        valid_normal = valid_mask[:, :, :-1, :-1] & valid_mask[:, :, 1:, :-1] & \
                       valid_mask[:, :, :-1, 1:] & valid_mask[:, :, 1:, 1:]
        valid_normal = valid_normal.expand_as(normal_pred)  # [B,3,H-1,W-1]
        
        # 余弦相似度损失: 1 - cos(angle)
        cos_sim = (normal_pred * normal_gt).sum(dim=1, keepdim=True)  # [B,1,H-1,W-1]
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        
        # 只计算有效区域
        valid_cos = valid_mask[:, :, :-1, :-1] & valid_mask[:, :, 1:, :-1] & \
                    valid_mask[:, :, :-1, 1:]
        
        if valid_cos.sum() > 0:
            normal_loss = (1.0 - cos_sim[valid_cos]).mean()
        else:
            normal_loss = torch.tensor(0.0, device=depth_pred.device)
        
        return normal_loss

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            try:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            except (ValueError, RuntimeError) as e:
                print(f"Cannot load Adam state (mismatch, ok): {e}")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
