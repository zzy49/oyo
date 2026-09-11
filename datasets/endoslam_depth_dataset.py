"""EndoSLAMDepthDataset: EndoSLAM UnityCam 合成数据, 带像素级深度 GT 监督。

数据格式:
    {data_path}/UnityCam/{Colon|Small Intestine|Stomach}/
        Frames_jpg/image_{:04d}.jpg          # RGB 320x320
        Pixelwise Depths/aov_image_{:04d}.png # 深度(厘米, 8bit RGBA, R通道)

深度编码: R 通道值单位为「厘米」, 转毫米 = value * 10 (范围 1~450mm),
与分类头 64-bin 深度范围 [1, 500] mm 对齐。

filenames 格式: "UnityCam/Colon" (整序列) 或 "UnityCam/Colon frame_idx" (单帧)。
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class EndoSLAMDepthDataset(Dataset):
    """EndoSLAM UnityCam dataset with depth supervision.

    Constructor:
        EndoSLAMDepthDataset(data_path, filenames, height, width, frame_ids,
                             num_scales, is_train=True, img_ext='.jpg', **kwargs)
    """

    # UnityCam 内参 (320x320)
    K_UNITY = np.array([[156.0418, 0.0, 178.5604],
                        [0.0, 155.7529, 181.8043],
                        [0.0, 0.0, 1.0]], dtype=np.float32)
    NATIVE_W, NATIVE_H = 320, 320
    DEPTH_SCALE = 10.0  # 厘米 -> 毫米

    def __init__(self, data_path, filenames, height, width,
                 frame_ids, num_scales, is_train=True, img_ext='.jpg', **kwargs):
        super().__init__()
        self.data_path = data_path
        self.height = height
        self.width = width
        self.frame_ids = frame_ids
        self.num_scales = num_scales
        self.is_train = is_train
        self.img_ext = img_ext

        # 解析 filenames -> (scene, frame_idx)
        # 注意: 场景名可能含空格 (如 "UnityCam/Small Intestine"), 因此从右侧分割。
        self.samples = []
        for line in filenames:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(None, 1)  # [scene, frame_idx]
            scene = parts[0]
            frame_idx = None
            if len(parts) == 2:
                try:
                    frame_idx = int(parts[1])
                except ValueError:
                    frame_idx = None

            if frame_idx is not None:
                self.samples.append((scene, frame_idx))
            else:
                # 整序列: 扫描 Frames_jpg 目录展开所有帧
                rgb_dir = os.path.join(data_path, scene, 'Frames_jpg')
                if os.path.isdir(rgb_dir):
                    for fname in sorted(os.listdir(rgb_dir)):
                        if fname.endswith(self.img_ext):
                            try:
                                num = int(os.path.splitext(fname)[0].split('_')[-1])
                                self.samples.append((scene, num))
                            except (ValueError, IndexError):
                                pass

        self.K = self.K_UNITY.copy()

        self.resize_rgb = transforms.Resize((self.height, self.width),
                                            interpolation=Image.LANCZOS)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def _get_rgb_path(self, scene, fid):
        return os.path.join(self.data_path, scene, 'Frames_jpg',
                            f'image_{fid:04d}{self.img_ext}')

    def _get_depth_path(self, scene, fid):
        return os.path.join(self.data_path, scene, 'Pixelwise Depths',
                            f'aov_image_{fid:04d}.png')

    def _resize_for_scale(self, img, scale):
        if scale == 0:
            return img
        h = self.height // (2 ** scale)
        w = self.width // (2 ** scale)
        return img.resize((w, h), Image.LANCZOS)

    def _color_aug(self, img):
        from torchvision.transforms import functional as TF
        factor = random.uniform(0.8, 1.2)
        img = TF.adjust_brightness(img, factor)
        factor = random.uniform(0.8, 1.2)
        img = TF.adjust_contrast(img, factor)
        factor = random.uniform(0.8, 1.2)
        img = TF.adjust_saturation(img, factor)
        factor = random.uniform(-0.1, 0.1)
        img = TF.adjust_hue(img, factor)
        return img

    def __getitem__(self, idx):
        inputs = {}
        scene, frame_idx = self.samples[idx]

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and np.random.random() > 0.5

        for frame_id in self.frame_ids:
            fid = frame_idx + frame_id
            if fid < 0:
                fid = 0

            img_path = self._get_rgb_path(scene, fid)
            if not os.path.isfile(img_path):
                img = Image.new('RGB', (self.width, self.height))
            else:
                img = pil_loader(img_path)
                img = self.resize_rgb(img)
                if do_flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

            for scale in range(self.num_scales):
                scaled = self._resize_for_scale(img, scale)
                inputs[("color", frame_id, scale)] = self.to_tensor(scaled)
                aug = scaled
                if do_color_aug:
                    aug = self._color_aug(scaled)
                inputs[("color_aug", frame_id, scale)] = self.to_tensor(aug)

        # 内参 (按训练尺寸缩放)
        for scale in range(self.num_scales):
            scale_factor = 2 ** scale
            K = self.K.copy()
            K[0, :] *= self.width / self.NATIVE_W
            K[1, :] *= self.height / self.NATIVE_H
            K[0, :] /= scale_factor
            K[1, :] /= scale_factor
            inputs[("K", scale)] = torch.from_numpy(K.copy())
            inputs[("inv_K", scale)] = torch.from_numpy(np.linalg.inv(K))
            K_4x4 = np.eye(4, dtype=np.float32)
            K_4x4[:3, :3] = K
            inputs[("K_4x4", scale)] = torch.from_numpy(K_4x4)
            inv_K_4x4 = np.eye(4, dtype=np.float32)
            inv_K_4x4[:3, :3] = np.linalg.inv(K)
            inputs[("inv_K_4x4", scale)] = torch.from_numpy(inv_K_4x4)

        # 深度 GT (厘米 -> 毫米)
        depth_path = self._get_depth_path(scene, frame_idx)
        if os.path.isfile(depth_path):
            try:
                with Image.open(depth_path) as dimg:
                    d_arr = np.array(dimg)[:, :, 0].astype(np.float32)  # R 通道
                depth_mm = d_arr * self.DEPTH_SCALE
                if do_flip:
                    depth_mm = np.fliplr(depth_mm)
                # resize 到训练尺寸 (最近邻, 保持深度值)
                d_img = Image.fromarray(depth_mm.astype(np.float32))
                d_img = d_img.resize((self.width, self.height), Image.NEAREST)
                depth_mm = np.array(d_img, dtype=np.float32)

                valid_mask = ((depth_mm > 1.0) & (depth_mm < 500.0)).astype(np.float32)
                depth_mm = np.clip(depth_mm, 1.0, 500.0)

                inputs[("depth_gt", 0)] = torch.from_numpy(depth_mm).unsqueeze(0)
                inputs[("depth_valid_mask", 0)] = torch.from_numpy(valid_mask).unsqueeze(0)
            except Exception:
                pass

        return inputs
