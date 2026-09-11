"""MultiSourceDataset: Weighted multi-dataset sampling."""

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiSourceDataset(Dataset):
    """Weighted concatenation of multiple datasets.
    
    Args:
        datasets: list of (dataset, weight) tuples.
                  Weight determines sampling probability.
    """
    
    def __init__(self, datasets):
        super().__init__()
        self.sub_datasets = [ds for ds, _ in datasets]
        weights = np.array([w for _, w in datasets], dtype=np.float64)
        self.weights = weights / weights.sum()
        self.cum_lengths = np.cumsum([len(ds) for ds in self.sub_datasets])
        self.total_len = int(self.cum_lengths[-1])
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        # Weighted sampling
        chosen = np.random.choice(len(self.sub_datasets), p=self.weights)
        ds = self.sub_datasets[chosen]
        sample_idx = np.random.randint(0, len(ds))
        sample = ds[sample_idx]
        # 补齐缺失的 depth_gt / depth_valid_mask，保证 batch collate 时 key 一致。
        # 无 GT 的子数据集填 0 (valid_mask=0)，训练时会被 valid_mask 过滤掉。
        if ("depth_gt", 0) not in sample:
            color = sample[("color", 0, 0)]
            h, w = color.shape[1], color.shape[2]
            sample[("depth_gt", 0)] = torch.zeros(1, h, w)
            sample[("depth_valid_mask", 0)] = torch.zeros(1, h, w)
        return sample
