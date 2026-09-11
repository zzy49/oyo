"""RealColonDataset: Real colonoscopy data from F:/dataset."""

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


def load_depth_tiff(path, h, w):
    """Load 16-bit TIFF depth in mm."""
    img = Image.open(path)
    img = img.resize((w, h), Image.NEAREST)
    depth = np.array(img, dtype=np.float32)
    return torch.from_numpy(depth).unsqueeze(0)


class RealColonDataset(Dataset):
    """Real colonoscopy data (F:\\dataset).
    
    Constructor:
        RealColonDataset(data_path, filenames, height, width, frame_ids, num_scales,
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
        self.has_gt = False  # Will be set True if depth files found
        
        # Parse filenames: "real seq_name frame_idx"
        self.samples = []
        for line in filenames:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                seq_name = parts[1]
                frame_idx = int(parts[2])
                self.samples.append((seq_name, frame_idx))
            elif len(parts) == 2:
                self.samples.append((parts[1], int(0)))
        
        # Check for GT depth availability
        self._check_gt_availability()
        
        self.K = np.array([[767.73, 0, 677.74, 0],
                           [0, 767.73, 543.06, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        self.resize = transforms.Resize((self.height, self.width),
                                         interpolation=Image.LANCZOS)
        self.to_tensor = transforms.ToTensor()
    
    def _check_gt_availability(self):
        """Check if first sample has depth data."""
        if not self.samples:
            return
        seq_name, frame_idx = self.samples[0]
        depth_dir = os.path.join(self.data_path, seq_name, 'depth')
        if os.path.isdir(depth_dir):
            files = os.listdir(depth_dir)
            if any(f.endswith(('.tiff', '.tif')) for f in files):
                self.has_gt = True
    
    def __len__(self):
        return len(self.samples)
    
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
        seq_name, frame_idx = self.samples[idx]
        seq_dir = os.path.join(self.data_path, seq_name)
        
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and np.random.random() > 0.5
        
        for frame_id in self.frame_ids:
            fid = frame_idx + frame_id
            if fid < 0:
                fid = 0
            
            # Try multiple image locations
            img_found = False
            for subd in ['frames', 'rgb', 'images', '']:
                for ext in [self.img_ext, '.jpg', '.png']:
                    for fmt in [f'frame_{fid:06d}', f'{fid:06d}', f'{fid:04d}']:
                        path = os.path.join(seq_dir, subd, f'{fmt}{ext}') if subd else os.path.join(seq_dir, f'{fmt}{ext}')
                        if os.path.isfile(path):
                            img = pil_loader(path)
                            img_found = True
                            break
                    if img_found:
                        break
                if img_found:
                    break
            
            if not img_found:
                img = Image.new('RGB', (self.width, self.height))
            else:
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
            K[0, :] *= self.width / 1350.0
            K[1, :] *= self.height / 1080.0
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
        
        # GT depth if available
        if self.has_gt:
            for subd in ['depth', '']:
                depth_dir = os.path.join(seq_dir, subd) if subd else seq_dir
                if os.path.isdir(depth_dir):
                    for ext in ['.tiff', '.tif']:
                        for fmt in [f'frame_{frame_idx:06d}_depth', f'{frame_idx:06d}_depth',
                                    f'frame_{frame_idx:04d}_depth', f'{frame_idx:04d}_depth']:
                            path = os.path.join(depth_dir, f'{fmt}{ext}')
                            if os.path.isfile(path):
                                try:
                                    depth_gt = load_depth_tiff(path, self.height, self.width)
                                    # C3VDv2 格式: uint16 0-65535 线性映射到 0-100 mm
                                    depth_gt = depth_gt * (100.0 / 65535.0)
                                    valid_mask = ((depth_gt > 1.0) & (depth_gt < 100.0)).float()
                                    depth_gt = torch.clamp(depth_gt, 1.0, 100.0)
                                    inputs[("depth_gt", 0)] = depth_gt
                                    inputs[("depth_valid_mask", 0)] = valid_mask
                                except Exception:
                                    pass
                                break
                    break
        
        return inputs
