"""C3VDDataset: C3VD synthetic colonoscopy dataset."""

import os
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
    """Load 16-bit TIFF depth in mm, resize to (h,w)."""
    img = Image.open(path)
    img = img.resize((w, h), Image.NEAREST)
    depth = np.array(img, dtype=np.float32)
    return torch.from_numpy(depth).unsqueeze(0)


class C3VDDataset(Dataset):
    """C3VD synthetic colonoscopy dataset.
    
    Image naming: frame_XXXXXX_color.png
    Depth naming: frame_XXXXXX_depth.tiff (16-bit, mm)
    
    Constructor:
        C3VDDataset(data_path, filenames, height, width, frame_ids, num_scales,
                    is_train=True, img_ext='.jpg', depth_gt_scale=1.0)
    """
    
    def __init__(self, data_path, filenames, height, width,
                 frame_ids, num_scales, is_train=True, img_ext='.jpg',
                 depth_gt_scale=1.0, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.height = height
        self.width = width
        self.frame_ids = frame_ids
        self.num_scales = num_scales
        self.is_train = is_train
        self.img_ext = img_ext
        self.depth_gt_scale = depth_gt_scale
        
        # Parse filenames — C3VD uses simple frame names
        self.frames = []
        for line in filenames:
            line = line.strip()
            if not line:
                continue
            # Format: "frame_name" or "frame_name side"
            parts = line.split()
            self.frames.append(parts[0])
        
        # C3VD intrinsics (from C3VD metadata)
        # Default: 1350x1080 with standard endoscope FOV ~120deg
        # These should match the actual data
        self.K = np.array([[600.0, 0, 512.0, 0],
                           [0, 600.0, 384.0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        # Try to load actual K
        k_path = os.path.join(data_path, 'intrinsics.txt')
        if os.path.isfile(k_path):
            try:
                k_vals = np.loadtxt(k_path)
                if k_vals.shape == (3, 3):
                    self.K[:3, :3] = k_vals
            except Exception:
                pass
        
        self.resize = transforms.Resize((self.height, self.width),
                                         interpolation=Image.LANCZOS)
        self.to_tensor = transforms.ToTensor()
    
    def _get_rgb_path(self, frame_name):
        """Find RGB image path. Supports multiple naming conventions."""
        candidates = [
            os.path.join(self.data_path, 'rgb', f'{frame_name}_color.png'),
            os.path.join(self.data_path, 'rgb', f'{frame_name}.png'),
            os.path.join(self.data_path, 'images', f'{frame_name}_color.png'),
            os.path.join(self.data_path, 'images', f'{frame_name}.png'),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None
    
    def _get_depth_path(self, frame_name):
        """Find depth TIFF path."""
        candidates = [
            os.path.join(self.data_path, 'depth', f'{frame_name}_depth.tiff'),
            os.path.join(self.data_path, 'depth', f'{frame_name}.tiff'),
            os.path.join(self.data_path, 'depth', f'{frame_name}_depth.tif'),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        inputs = {}
        frame_name = self.frames[idx]
        
        # Find frame number from name
        try:
            frame_num = int(frame_name.replace('frame_', '').replace('_color', ''))
        except ValueError:
            frame_num = idx
        
        do_flip = self.is_train and np.random.random() > 0.5
        
        for frame_id in self.frame_ids:
            src_num = frame_num + frame_id
            src_name = f'frame_{src_num:06d}'
            
            img_path = self._get_rgb_path(src_name)
            if img_path is None:
                img = Image.new('RGB', (self.width, self.height))
            else:
                img = pil_loader(img_path)
                img = self.resize(img)
                if do_flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            inputs[("color", frame_id)] = self.to_tensor(img)
        
        # Intrinsics
        K = self.K.copy()
        # No scaling needed for C3VD (fixed resolution)
        inputs[("K", 0)] = torch.from_numpy(K)
        inputs[("inv_K", 0)] = torch.from_numpy(np.linalg.inv(K))
        
        # GT depth (C3VD provides ground truth)
        depth_path = self._get_depth_path(frame_name)
        if depth_path and os.path.isfile(depth_path):
            try:
                depth_gt = load_depth_tiff(depth_path, self.height, self.width)
                if self.depth_gt_scale != 1.0:
                    depth_gt = depth_gt * self.depth_gt_scale
                valid_mask = ((depth_gt > 1.0) & (depth_gt < 500.0)).float()
                depth_gt = torch.clamp(depth_gt, 1.0, 500.0)
                inputs[("depth_gt", 0)] = depth_gt
                inputs[("depth_valid_mask", 0)] = valid_mask
            except Exception:
                pass
        
        return inputs
