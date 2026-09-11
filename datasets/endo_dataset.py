"""EndoscopicDataset: EndoSLAM / Cameras / Mixed format."""

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


def parse_line(line):
    """Parse a line from train_files.txt.
    
    Format: "Camera/Seq [SubSeq frame_idx lr]"
    Returns: (camera, sequence, subseq, frame_idx, side)
    If no frame info, subseq/frame_idx/side = None.
    """
    line = line.strip()
    if not line:
        return None, None, None, None, None
    
    parts = line.split(' ')
    path_parts = parts[0].split('/')
    camera = path_parts[0]
    sequence = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ''
    
    if len(parts) >= 2:
        subseq = '/'.join(path_parts[1:-1]) if len(path_parts) > 2 else path_parts[1]
        frame_idx = int(parts[1])
        side = parts[2] if len(parts) >= 3 else 'l'
    else:
        subseq = sequence  # no frame, whole sequence
        frame_idx = None
        side = 'l'
    
    return camera, sequence, subseq, frame_idx, side


def find_rgb_files(data_path, subdir, frame_idx, img_ext='.png'):
    """Find RGB file for a given frame index."""
    rgb_dir = os.path.join(data_path, 'generated', subdir)
    if not os.path.isdir(rgb_dir):
        return None
    
    if frame_idx is not None:
        fname = f'frame_{frame_idx:06d}{img_ext}'
        path = os.path.join(rgb_dir, fname)
        if os.path.isfile(path):
            return path
    else:
        # Return the directory for sequence-level entries
        files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(img_ext)])
        return [os.path.join(rgb_dir, f) for f in files]
    
    return None


def load_intrinsics(data_path):
    """Load 4x4 intrinsic matrix from dataset."""
    # Default EndoSLAM intrinsic (from dyendovo_dataset K_MAT)
    K = np.array([[767.73, 0, 677.74, 0],
                  [0, 767.73, 543.06, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    
    # Try to load from K.txt if present
    k_path = os.path.join(data_path, 'generated', 'K.txt')
    if os.path.isfile(k_path):
        try:
            k_vals = np.loadtxt(k_path)
            if k_vals.shape == (3, 3):
                K = np.eye(4, dtype=np.float32)
                K[:3, :3] = k_vals
            elif k_vals.shape == (4, 4):
                K = k_vals.astype(np.float32)
        except Exception:
            pass
    
    return K


def load_gt_poses(data_path):
    """Load GT poses from generated/pose.txt (EndoSLAM format: R|t per line)."""
    pose_path = os.path.join(data_path, 'generated', 'pose.txt')
    if not os.path.isfile(pose_path):
        return None
    
    poses = []
    with open(pose_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split(',')))
            T = np.array(vals).reshape(4, 4)
            # EndoSLAM format: [R|0; t|1] -> convert to standard [R|t; 0|1]
            # Original format has translation in first 3 rows, last column
            poses.append(T)
    
    return poses


class EndoscopicDataset(Dataset):
    """EndoSLAM / Cameras dataset for monodepth2 training.
    
    Constructor:
        EndoscopicDataset(data_path, filenames, height, width, frame_ids, num_scales,
                         is_train=True, img_ext='.png')
    
    __getitem__ returns:
        {
            ("color", frame_id): Tensor[B,3,H,W],
            ("K", 0): Tensor[4,4],
            ("inv_K", 0): Tensor[4,4],
            ("gt_pose", -1) or ("gt_pose", 1): Tensor[4,4],  # if available
        }
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
        
        # Color augmentation params
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        
        # Parse filenames into (subdir, frame_idx) pairs
        self.samples = []
        for line in filenames:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(' ')
            
            # 尝试最后一部分是否为 frame index (数字)
            try:
                maybe_frame = int(parts[-1])
            except (ValueError, IndexError):
                maybe_frame = None
            
            if len(parts) >= 2 and maybe_frame is not None:
                # Per-frame: "Cam/Seq/SubSeq frame_idx"
                # 注意: parts[-1] 已被确认为整数, 前面的部分为路径
                path_str = ' '.join(parts[:-1])
                path_parts = path_str.split('/')
                subdir = '/'.join(path_parts[1:]) if len(path_parts) > 1 else path_parts[0]
                self.samples.append((subdir, maybe_frame))
            else:
                # Whole sequence: "Cam/Seq" — expand to all frames
                # 或者路径中包含空格 (如 "Small Intestine") — 重新合并
                path_str = ' '.join(parts)
                path_parts = path_str.split('/')
                seq_dir = os.path.join(data_path, *path_parts)
                rgb_dir = None
                for d in ['generated/rgb_warped', 'generated/rgb', 'rgb']:
                    candidate = os.path.join(seq_dir, d)
                    if os.path.isdir(candidate):
                        rgb_dir = candidate
                        break
                if rgb_dir:
                    frames = sorted([f for f in os.listdir(rgb_dir) if f.endswith(img_ext)])
                    for fname in frames:
                        try:
                            # frame_000000.png -> 0
                            num = int(os.path.splitext(fname)[0].split('_')[-1])
                            subdir = '/'.join(path_parts[1:]) if len(path_parts) > 1 else path_parts[0]
                            self.samples.append((subdir, num))
                        except (ValueError, IndexError):
                            subdir = '/'.join(path_parts[1:]) if len(path_parts) > 1 else path_parts[0]
                            self.samples.append((subdir, fname))
        
        # Load intrinsics and GT poses
        self.K = load_intrinsics(data_path)
        self.gt_poses = load_gt_poses(data_path)
        
        self.resize = transforms.Resize((self.height, self.width),
                                         interpolation=Image.LANCZOS)
        self.to_tensor = transforms.ToTensor()
    
    def _resize_for_scale(self, img, scale):
        """Resize PIL Image for multi-scale pyramid."""
        if scale == 0:
            return img
        h = self.height // (2 ** scale)
        w = self.width // (2 ** scale)
        return img.resize((w, h), Image.LANCZOS)
    
    def _color_aug(self, img):
        """Apply color augmentation (brightness, contrast, saturation, hue)."""
        from torchvision.transforms import functional as TF
        
        # Brightness
        factor = random.uniform(*self.brightness)
        img = TF.adjust_brightness(img, factor)
        
        # Contrast
        factor = random.uniform(*self.contrast)
        img = TF.adjust_contrast(img, factor)
        
        # Saturation
        factor = random.uniform(*self.saturation)
        img = TF.adjust_saturation(img, factor)
        
        # Hue
        factor = random.uniform(*self.hue)
        img = TF.adjust_hue(img, factor)
        
        return img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        inputs = {}
        do_color_aug = self.is_train and random.random() > 0.5

        subdir, frame_idx = self.samples[index]
        seq_dir = os.path.join(self.data_path, subdir)
        rgb_subdirs = ['generated/rgb_warped', 'generated/rgb', 'rgb']
        
        do_flip = self.is_train and np.random.random() > 0.5
        
        for frame_id in self.frame_ids:
            fid = frame_idx + frame_id
            if fid < 0:
                fid = 0
            
            img_path = None
            name_patterns = [
                f'frame_{fid:06d}', f'frame_{fid:04d}', f'frame_{fid:d}',
                f'{fid:06d}', f'{fid:04d}', f'{fid:d}',
            ]
            for subd in rgb_subdirs:
                for name in name_patterns:
                    candidate = os.path.join(seq_dir, subd, f'{name}{self.img_ext}')
                    if os.path.isfile(candidate):
                        img_path = candidate
                        break
                if img_path:
                    break
            
            if img_path is None:
                img = Image.new('RGB', (self.width, self.height))
            else:
                img = pil_loader(img_path)
                img = self.resize(img)
                if do_flip:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Multi-scale + augmentation
            for scale in range(self.num_scales):
                scaled = self._resize_for_scale(img, scale)
                inputs[("color", frame_id, scale)] = self.to_tensor(scaled)
                
                aug = scaled
                if do_color_aug:
                    aug = self._color_aug(scaled)
                inputs[("color_aug", frame_id, scale)] = self.to_tensor(aug)
        
        # Adjust intrinsics for each scale
        for scale in range(self.num_scales):
            scale_factor = 2 ** scale
            K = self.K[:3, :3].copy()  # Extract 3x3 from 4x4
            K[0, :] *= self.width / 1920.0
            K[1, :] *= self.height / 1080.0
            K[0, :] /= scale_factor
            K[1, :] /= scale_factor
            
            # 3x3 intrinsics
            inputs[("K", scale)] = torch.from_numpy(K.copy()).float()
            inputs[("inv_K", scale)] = torch.from_numpy(np.linalg.inv(K)).float()
            
            # 4x4 intrinsics (for projective geometry ops)
            K_4x4 = np.eye(4, dtype=np.float32)
            K_4x4[:3, :3] = K
            inputs[("K_4x4", scale)] = torch.from_numpy(K_4x4)
            inv_K_4x4 = np.eye(4, dtype=np.float32)
            inv_K_4x4[:3, :3] = np.linalg.inv(K)
            inputs[("inv_K_4x4", scale)] = torch.from_numpy(inv_K_4x4)
        
        # GT poses if available
        if self.gt_poses is not None:
            for frame_id in self.frame_ids:
                if frame_id == 0:
                    continue
                src_idx = frame_idx + frame_id
                if (0 <= frame_idx < len(self.gt_poses) and
                    0 <= src_idx < len(self.gt_poses)):
                    T_w2c_0 = self.gt_poses[frame_idx]
                    T_w2c_1 = self.gt_poses[src_idx]
                    T_rel = np.linalg.inv(T_w2c_0) @ T_w2c_1
                    inputs[("gt_pose", frame_id)] = torch.from_numpy(
                        T_rel.astype(np.float32))
        
        return inputs
