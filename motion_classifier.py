"""
组织运动分类器网络
==================
双分支架构:
  - Patch CNN: 处理 16×16×4 (depth_i, depth_j, flow_x, flow_y) 输入
  - Flow MLP:   处理 (dx, dy, norm_x, norm_y) 稀疏流+位置特征
  - 融合头输出 P(moving) ∈ [0, 1]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCNN(nn.Module):
    """轻量 CNN 提取深度+光流 patch 的几何特征."""

    def __init__(self, in_channels=4, base_dim=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_dim)
        self.conv2 = nn.Conv2d(base_dim, base_dim * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_dim * 2)
        self.conv3 = nn.Conv2d(base_dim * 2, base_dim * 4, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_dim * 4)
        self.conv4 = nn.Conv2d(base_dim * 4, base_dim * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_dim * 4)
        self.pool = nn.MaxPool2d(2, 2)

        # After 3 pools: 16→8→4→2, 64 channels → 64*2*2 = 256
        self.fc = nn.Linear(base_dim * 4 * 2 * 2, 64)

    def forward(self, x):
        # x: (B, 4, 16, 16)  [depth_i, depth_j, flow_x, flow_y]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # (B, 16, 8, 8)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # (B, 32, 4, 4)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # (B, 64, 2, 2)
        x = F.relu(self.bn4(self.conv4(x)))               # (B, 64, 2, 2)
        x = x.reshape(x.size(0), -1)                        # (B, 256)
        x = F.relu(self.fc(x))                             # (B, 64)
        return x


class FlowMLP(nn.Module):
    """处理流向量和归一化位置."""

    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 16)
        self.fc2 = nn.Linear(16, hidden)

    def forward(self, x):
        # x: (B, 4) — dx, dy, norm_x, norm_y
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MotionClassifier(nn.Module):
    """组织运动二分类器（深度+光流输入版）."""

    def __init__(self, patch_base_dim=16, flow_hidden=32, dropout=0.3):
        super().__init__()
        self.patch_cnn = PatchCNN(in_channels=4, base_dim=patch_base_dim)  # → 64
        self.flow_mlp = FlowMLP(in_dim=4, hidden=flow_hidden)               # → 32
        self.dropout = nn.Dropout(dropout)
        self.fc_fuse = nn.Linear(64 + flow_hidden, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, depth_flow_patch, flow_feat):
        """
        Args:
            depth_flow_patch: (B, 4, 16, 16)  float32
                通道: [depth_i, depth_j, flow_x, flow_y]
                归一化: depth ∈ [0,1], flow ∈ [-1,1]
            flow_feat: (B, 4)  float32 [dx, dy, norm_x, norm_y]
                稀疏 LoFTR 流 + 归一化位置

        Returns:
            logits: (B, 1)  raw logits for BCEWithLogitsLoss
        """
        patch_feat = self.patch_cnn(depth_flow_patch)     # (B, 64)
        flow_feat_vec = self.flow_mlp(flow_feat)           # (B, 32)

        fused = torch.cat([patch_feat, flow_feat_vec], dim=1)  # (B, 96)
        fused = self.dropout(fused)
        fused = F.relu(self.fc_fuse(fused))                     # (B, 32)
        logits = self.fc_out(fused)                             # (B, 1)

        return logits

    def predict_proba(self, depth_flow_patch, flow_feat):
        """Return P(moving) ∈ [0, 1]."""
        logits = self.forward(depth_flow_patch, flow_feat)
        return torch.sigmoid(logits).squeeze(-1)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def test_model():
    """Sanity check."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MotionClassifier().to(device)
    print(f'Params: {model.num_params:,}')

    B = 4
    depth_flow_patch = torch.randn(B, 4, 16, 16).to(device)
    flow = torch.randn(B, 4).to(device)

    logits = model(depth_flow_patch, flow)
    print(f'Input: ({B}, 4, 16, 16) + ({B}, 4)')
    print(f'Output: {logits.shape}')
    print(f'Sigmoid: {torch.sigmoid(logits).squeeze()}')

    probs = model.predict_proba(depth_flow_patch, flow)
    print(f'Probs: {probs}')


if __name__ == '__main__':
    test_model()
