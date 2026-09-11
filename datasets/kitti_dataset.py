"""KITTI dataset stubs for compatibility."""

from torch.utils.data import Dataset


class KITTIRAWDataset(Dataset):
    """Stub: KITTI raw dataset (not used in this project)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("KITTI dataset not available in this project")

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError


class KITTIOdomDataset(Dataset):
    """Stub: KITTI odometry dataset (not used in this project)."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("KITTI dataset not available in this project")

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise NotImplementedError
