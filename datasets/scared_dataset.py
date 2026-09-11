"""SCAREDDataset: SCARED laparoscopic dataset."""

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


class SCAREDDataset(Dataset):
    """SCARED structured-light laparoscopic dataset.
    
    Constructor:
        SCAREDDataset(data_path, filenames, height, width, frame_ids, num_scales,
                      is_train=True, img_ext='.png')
    """
    
    def __init__(self, data_path, filenames, height, width,
                 frame_ids, num_scales, is_train=True, img_ext='.png', **kwargs):
        super().__init__()
        self.data_path = data_path
        self.height = height
        self.width = width
        self.frame_ids = frame_ids
        self.num_scales = num_scales
        self.is_train = is_train
        self.img_ext = img_ext
        
        # Parse filenames
        self.samples = []
        for line in filenames:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 格式: "scared dataset_X/keyframe_Y frame_idx" 或 "scared dataset_X/keyframe_Y"
            if len(parts) >= 3:
                # "scared" prefix + folder + frame_idx
                try:
                    self.samples.append((parts[1], int(parts[2])))
                except ValueError:
                    self.samples.append((parts[1], None))
            elif len(parts) >= 2:
                # 不含 frame_idx
                try:
                    self.samples.append((parts[1], int(parts[-1])))
                except ValueError:
                    self.samples.append((parts[1], None))
            else:
                self.samples.append((parts[0], None))
        
        # SCARED intrinsics (typical 1280x1024 endoscope)
        self.K = np.array([[800.0, 0, 640.0, 0],
                           [0, 800.0, 512.0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.resize = transforms.Resize((self.height, self.width),
                                         interpolation=Image.LANCZOS)
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        inputs = {}
        subdir, frame_idx = self.samples[idx]
        
        if frame_idx is None:
            frame_idx = 0
        
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and np.random.random() > 0.5
        
        for frame_id in self.frame_ids:
            fid = frame_idx + frame_id
            if fid < 0:
                fid = 0
            
            img_path = os.path.join(self.data_path, subdir,
                                     f'frame_{fid:06d}{self.img_ext}')
            if not os.path.isfile(img_path):
                img = Image.new('RGB', (self.width, self.height))
            else:
                img = pil_loader(img_path)
                img = self.resize(img)
                if do_flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            for scale in range(self.num_scales):
                scaled = self._resize_for_scale(img, scale)
                inputs[("color", frame_id, scale)] = self.to_tensor(scaled)
                aug = scaled
                if do_color_aug:
                    aug = self._color_aug(scaled)
                inputs[("color_aug", frame_id, scale)] = self.to_tensor(aug)
        
        for scale in range(self.num_scales):
            K = self.K.copy()
            scale_factor = 2 ** scale
            K[0, :] *= self.width / 1280.0
            K[1, :] *= self.height / 1024.0
            K[0, :] /= scale_factor
            K[1, :] /= scale_factor
            K_3x3 = K[:3, :3]
            inputs[("K", scale)] = torch.from_numpy(K_3x3.copy())
            inputs[("inv_K", scale)] = torch.from_numpy(np.linalg.inv(K_3x3))
            K_4x4 = np.eye(4, dtype=np.float32)
            K_4x4[:3, :3] = K_3x3
            inputs[("K_4x4", scale)] = torch.from_numpy(K_4x4)
            inv_K_4x4 = np.eye(4, dtype=np.float32)
            inv_K_4x4[:3, :3] = np.linalg.inv(K_3x3)
            inputs[("inv_K_4x4", scale)] = torch.from_numpy(inv_K_4x4)
        
        return inputs
    
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
