"""
增强版 Trainer: Lite-Mono 风格深度解码器 + 时序一致性损失

关键改动 vs 原版 123/trainer.py:
  1. DepthDecoder → LiteMonoDepthDecoder (3层, bilinear上采样, Truncated Normal初始化)
  2. 添加时序一致性损失 L_temporal:
     L_temporal = 1/N Σ|D_0(p) - D_t(warp(p, T_{0→t}))|
     利用 PoseCNN 预测的位姿, 将相邻帧深度 warp 到当前帧, 计算 L1 差异
  3. 边缘感知平滑损失 (原版 get_smooth_loss 已实现 e^{-|∇I|} 加权, 保持不变)

用法:
  python zhong/train_ours_enhanced.py
"""

from __future__ import absolute_import, division, print_function

import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# 确保能导入 123/ 下的 trainer 和父目录下的 networks
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sys_123 = os.path.join(_project_dir, '123')
# 父类 123/trainer.py 需要 import networks (位于项目根目录)
if _sys_123 not in sys.path:
    sys.path.insert(0, _sys_123)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# 显式导入 123/trainer.py
import importlib
_trainer_123 = importlib.util.spec_from_file_location("_trainer_123", os.path.join(_sys_123, "trainer.py"))
_trainer_123 = importlib.util.module_from_spec(_trainer_123)
import sys as _sys_mod
_sys_mod.modules["_trainer_123"] = _trainer_123
_trainer_123.__spec__.loader.exec_module(_trainer_123)
Trainer = _trainer_123.Trainer

# 保留原 import 以便其他模块使用
_IMPORTED_123_TRAINER = True
from networks.litemono_decoder import LiteMonoDepthDecoder
from layers import BackprojectDepth, Project3D, transformation_from_parameters


class TrainerLitemonoHead(Trainer):
    """增强版 Trainer: Lite-Mono 解码器 + 时序一致性损失.

    继承自原版 Trainer, 覆盖深度解码器创建和损失计算.
    """

    def __init__(self, options):
        """初始化增强版 trainer.

        先调用父类构造 (创建 encoder, pose_net, motion_encoder, 数据集等),
        然后将 DepthDecoder 替换为 LiteMonoDepthDecoder.
        """
        # ── 阶段1: 调用父类 __init__ (会用原版 DepthDecoder) ──
        super().__init__(options)

        # ── 阶段2: 替换深度解码器为 Lite-Mono 版本 ──
        old_depth = self.models["depth"]

        # 从 parameters_to_train 中移除旧解码器参数
        old_param_ids = {id(p) for p in old_depth.parameters()}
        self.parameters_to_train = [
            p for p in self.parameters_to_train if id(p) not in old_param_ids
        ]

        # 创建 Lite-Mono 风格深度解码器
        # scales: 默认 [0,1,2,3], Lite-Mono decoder 只支持 3 层 → 需通过 --scales 0 1 2 设置
        num_ch_enc = self.models["encoder"].num_ch_enc  # [64, 64, 128, 256, 512]
        self.models["depth"] = LiteMonoDepthDecoder(
            num_ch_enc,
            scales=self.opt.scales,
            num_output_channels=1,
            use_skips=True,
            num_bins=self.opt.num_bins,
        )
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # ── 断点续训: 加载 LiteMonoDepthDecoder 权重 ──
        if self.opt.load_weights_folder is not None:
            depth_path = os.path.join(self.opt.load_weights_folder, "depth.pth")
            if os.path.isfile(depth_path):
                try:
                    depth_ckpt = torch.load(depth_path)
                    depth_state = self.models["depth"].state_dict()
                    depth_ckpt = {k: v for k, v in depth_ckpt.items() if k in depth_state}
                    depth_state.update(depth_ckpt)
                    self.models["depth"].load_state_dict(depth_state, strict=False)
                    print(f"  [LiteMono] Loaded depth weights from checkpoint")
                except Exception as e:
                    print(f"  [LiteMono] Could not load depth weights (fresh init): {e}")

        # 重新创建优化器 (参数列表已变更)
        self.model_optimizer = optim.Adam(
            self.parameters_to_train, self.opt.learning_rate
        )

        # ── 断点续训: 加载 Adam 状态 ──
        if self.opt.load_weights_folder is not None:
            adam_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
            if os.path.isfile(adam_path):
                try:
                    adam_dict = torch.load(adam_path)
                    self.model_optimizer.load_state_dict(adam_dict)
                    print(f"  [LiteMono] Loaded Adam state from checkpoint")
                except Exception as e:
                    print(f"  [LiteMono] Adam state mismatch (fresh init, ok): {e}")

        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1
        )

        # 时序一致性权重
        self.w_temporal = getattr(self.opt, 'w_temporal', 0.1)

        print(f"  [LiteMono] DepthDecoder replaced: {sum(p.numel() for p in self.models['depth'].parameters())/1000:.1f}K params")
        print(f"  [LiteMono] Temporal consistency weight: {self.w_temporal}")

    def _find_motion_pair(self, outputs):
        """从 outputs 中自动检测帧间深度 pair.

        查找形如 ("depth", fid, 0) 的 key, 其中 fid != 0.
        这是 process_batch 中深度一致性分支写入的相邻帧深度.

        Returns:
            (frame_id, depth_key, cam_T_key) 或 (None, None, None)
        """
        for key in outputs:
            if isinstance(key, tuple) and len(key) == 3:
                if key[0] == "depth" and key[2] == 0 and key[1] != 0:
                    fid = key[1]
                    t_key = ("cam_T_cam", 0, fid)
                    if t_key in outputs:
                        return fid, key, t_key
        return None, None, None

    def compute_temporal_consistency_loss(self, inputs, outputs):
        """计算时序一致性损失.

        L_temporal = |D_0 - warp(D_t, T_{0→t})|_1

        将相邻帧深度 warp 到当前帧, 惩罚深度预测的时序不一致性.
        仅在有效区域计算 (深度在合理范围内).

        Returns:
            temporal_loss: 标量 tensor, 无效时返回 0.0
        """
        motion_pair_id, depth_key, t_key = self._find_motion_pair(outputs)
        if motion_pair_id is None:
            return torch.tensor(0.0, device=self.device)

        # ── 获取深度和位姿 ──
        depth_0 = outputs[("depth", 0, 0)]  # 当前帧深度 [B, 1, H, W]
        depth_t = outputs[depth_key]         # 相邻帧深度 [B, 1, H, W]
        T_0_to_t = outputs[t_key]            # T_{0→t} [B, 4, 4]

        # T_{t→0} = inv(T_{0→t})
        T_t_to_0 = torch.inverse(T_0_to_t.float()).to(depth_0.dtype)

        # ── 有效区域掩码 ──
        valid_0 = (depth_0 > self.opt.min_depth) & (depth_0 < self.opt.max_depth)
        valid_t = (depth_t > self.opt.min_depth) & (depth_t < self.opt.max_depth)
        valid = (valid_0 & valid_t).float()

        if valid.sum() < 100:
            return torch.tensor(0.0, device=self.device)

        # ── Warp depth_t → frame_0 视角 ──
        # 使用 scale=0 的 BackprojectDepth / Project3D
        H, W = depth_0.shape[2], depth_0.shape[3]
        inv_K = inputs[("inv_K", 0)]  # [B, 4, 4]
        K_4x4 = inputs[("K_4x4", 0)]  # [B, 4, 4]

        # 确保 BackprojectDepth/Project3D 尺寸匹配
        bp = self.backproject_depth.get(0)
        proj = self.project_3d.get(0)

        if bp is None or bp.height != H or bp.width != W:
            # 动态创建 (适配不同分辨率)
            B = depth_0.shape[0]
            bp = BackprojectDepth(B, H, W).to(self.device)
            proj = Project3D(B, H, W).to(self.device)

        cam_points_t = bp(depth_t, inv_K)
        pix_coords_t = proj(cam_points_t, K_4x4, T_t_to_0)

        # Grid sample: 将 depth_t 按 pix_coords_t warp 到 frame_0 坐标
        depth_t_warped = F.grid_sample(
            depth_t, pix_coords_t,
            mode='bilinear', padding_mode='zeros', align_corners=False,
        )

        # ── L1 损失 ──
        temporal_diff = torch.abs(depth_0 - depth_t_warped)
        temporal_loss = (temporal_diff * valid).sum() / (valid.sum() + 1e-7)

        return temporal_loss

    def compute_losses(self, inputs, outputs):
        """覆盖父类 compute_losses, 追加时序一致性损失.

        先调用父类计算标准损失 (光度 + 平滑 + 分类 + ...),
        再追加时序一致性损失.
        """
        losses = super().compute_losses(inputs, outputs)
        total_loss = losses["loss"]

        # ── 时序一致性损失 ──
        if self.w_temporal > 0:
            temporal_loss = self.compute_temporal_consistency_loss(inputs, outputs)
            if temporal_loss.item() > 0:
                total_loss = total_loss + self.w_temporal * temporal_loss
                losses["loss/temporal"] = temporal_loss

        losses["loss"] = total_loss
        return losses
