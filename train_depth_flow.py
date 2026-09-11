"""
深度+光流 运动分类器训练
========================
输入: (N, 4, 16, 16) → [depth_t, depth_{t+1}, flow_x, flow_y]
网络: 4层Conv2d(32-64-128-256) + BN + ReLU + MaxPool → GlobalAvgPool → FC(256→1) + Sigmoid
训练: 8个C2序列, 按序列7/1拆分训练/验证, 100 epochs
测试: c1_transverse1_t1_v2
"""
import os, sys, json, time, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import (roc_auc_score, precision_recall_fscore_support,
                              confusion_matrix, roc_curve, precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── 数据路径 ──
C2_DATASETS = [
    r'F:\dataset\c2_transverse1_t3_v1\generated\motion_train_data.npz',
    r'F:\dataset\c2_transverse1_t2_v1\generated\motion_train_data.npz',
    r'F:\dataset\c2_transverse1_t3_v2\generated\motion_train_data.npz',
    r'F:\dataset\c2_transverse2_t3_v2\generated\motion_train_data.npz',
    r'F:\dataset\c2_transverse2_t3_v1\generated\motion_train_data.npz',
    r'F:\dataset\c2_transverse2_t2_v1\generated\motion_train_data.npz',
    r'F:\dataset\c2_transverse2_t2_v2\generated\motion_train_data.npz',
    r'F:\dataset\c2_sigmoid_t2_v1\generated\motion_train_data.npz',
]
C1_TEST = r'F:\dataset\c1_transverse1_t1_v2\generated\motion_train_data.npz'
OUTPUT_DIR = r'E:\data1\monodepth2\motion_cls_data\depth_flow_output'

# ── 训练超参数 ──
BATCH_SIZE = 256
NUM_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
VAL_SEQ_INDEX = 0  # 用作验证的序列索引（0-7），指定则固定，-1则随机选择
N_TRAIN_SEQS_PER_EPOCH = 3  # 每个 epoch 随机加载的序列数，减少内存占用


# ══════════════════════════════════════════════
# 模型
# ══════════════════════════════════════════════

class DepthFlowClassifier(nn.Module):
    """4 层 Conv2d + GlobalAvgPool + FC + Sigmoid."""

    def __init__(self, in_channels=4, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16→8
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8→4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4→2
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 2→1
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, 4, 16, 16)
        x = self.conv1(x)   # (B, 32, 8, 8)
        x = self.conv2(x)   # (B, 64, 4, 4)
        x = self.conv3(x)   # (B, 128, 2, 2)
        x = self.conv4(x)   # (B, 256, 1, 1)
        x = x.mean(dim=[2, 3])  # GlobalAvgPool → (B, 256)
        x = self.dropout(x)
        x = self.fc(x)      # (B, 1)
        return torch.sigmoid(x).squeeze(-1)  # (B,)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════
# 数据集
# ══════════════════════════════════════════════

class MotionTrainDataset(Dataset):
    """单个序列的 motion_train_data.npz.

    数据格式:
      patches: (N, 16, 16, 4) HWC → __getitem__ 时转为 (4, 16, 16) CHW
      labels:  (N, 16, 16) per-pixel → 聚合为 (N,) scalar

    优化: 不预转置，保持 HWC 存于内存，减少 ~30% 内存占用。
    """

    def __init__(self, npz_path, augment=False):
        data = np.load(npz_path)
        self.samples = data['patches'].shape[0]

        # per-pixel labels → per-patch
        pixel_labels = data['labels'].astype(np.float32)
        self.labels = pixel_labels.max(axis=(1, 2))  # (N,)

        # 保持 HWC，不预转置
        self.patches = data['patches'].astype(np.float32)  # (N, 16, 16, 4)

        self.augment = augment
        n_moving = int(self.labels.sum())
        name = os.path.basename(npz_path)
        print(f'  {name}: {self.samples:,} samples, moving={n_moving:,} '
              f'({100*n_moving/max(self.samples,1):.1f}%)')

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        patch = self.patches[idx]  # (16, 16, 4) HWC
        label = self.labels[idx]

        # HWC → CHW, 转 contiguous 以便后续 inplace 操作
        patch = np.ascontiguousarray(patch.transpose(2, 0, 1))  # (4, 16, 16)

        if self.augment:
            # 1. 深度通道加高斯噪声 (σ=0.01)
            noise = np.random.randn(2, 16, 16).astype(np.float32) * 0.01
            patch[:2] += noise

            # 2. 光流通道加随机缩放 (±5%)
            flow_scale = 1.0 + np.random.uniform(-0.05, 0.05, size=(2, 1, 1)).astype(np.float32)
            patch[2:4] *= flow_scale

            # 3. 随机水平翻转 (50%)
            if np.random.rand() > 0.5:
                patch = patch[:, :, ::-1].copy()
                patch[2] = -patch[2]  # negate flow_x

        return torch.from_numpy(patch), torch.tensor(label, dtype=torch.float32)


# ══════════════════════════════════════════════
# 评估指标
# ══════════════════════════════════════════════

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0)
    acc = np.mean(preds == labels)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'accuracy': acc, 'auc': auc,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


# ══════════════════════════════════════════════
# 训练 / 验证
# ══════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, is_train, pos_weight_t):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_labels, all_probs = [], []

    for patches, labels in loader:
        patches = patches.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()
            probs = model(patches)
            # 手动加权 BCELoss
            sample_loss = criterion(probs, labels)
            weights = torch.where(labels > 0.5, pos_weight_t, torch.tensor(1.0).to(device))
            loss = (weights * sample_loss).mean()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                probs = model(patches)
                sample_loss = criterion(probs, labels)
                weights = torch.where(labels > 0.5, pos_weight_t, torch.tensor(1.0).to(device))
                loss = (weights * sample_loss).mean()

        total_loss += loss.item() * len(labels)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


# ══════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--val_seq', type=int, default=VAL_SEQ_INDEX,
                        help='验证集序列索引 (0-7, -1=随机)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # ── 预扫描：只读 labels，统计样本量和 moving 比例（不加载 patches）──
    print('Pre-scanning datasets (labels only)...')
    ds_meta = []  # [(path, n_samples, n_moving)]
    for path in C2_DATASETS:
        if not os.path.exists(path):
            print(f'  WARNING: {path} 不存在, 跳过')
            continue
        data = np.load(path)
        pixel_labels = data['labels'].astype(np.float32)
        labels = pixel_labels.max(axis=(1, 2))
        n_samples = len(labels)
        n_moving = int(labels.sum())
        ds_meta.append((path, n_samples, n_moving))
        print(f'  {os.path.basename(path)}: {n_samples:,} samples, '
              f'moving={n_moving:,} ({100*n_moving/max(n_samples,1):.1f}%)')

    if len(ds_meta) < 2:
        print('ERROR: 至少需要 2 个序列')
        return

    # ── 固定验证集 ──
    val_idx = args.val_seq
    if val_idx < 0 or val_idx >= len(ds_meta):
        val_idx = random.randint(0, len(ds_meta) - 1)
        print(f'\n随机选择验证序列: [{val_idx}]')
    else:
        print(f'\n固定验证序列: [{val_idx}]')

    val_path, val_n, val_moving = ds_meta[val_idx]
    train_meta = [m for i, m in enumerate(ds_meta) if i != val_idx]

    # 验证集只加载一次
    val_ds = MotionTrainDataset(val_path, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    print(f'Val:   {val_n:,} samples, moving={val_moving:,} '
          f'({100*val_moving/max(val_n,1):.1f}%)\n')

    # ── pos_weight: 从全量训练数据预计算 ──
    train_total_all = sum(m[1] for m in train_meta)
    train_moving_all = sum(m[2] for m in train_meta)
    pos_weight = train_total_all / max(train_moving_all, 1)
    pos_weight_t = torch.tensor([pos_weight]).to(device)
    print(f'Train (all): {train_total_all:,} samples, moving={train_moving_all:,} '
          f'({100*train_moving_all/max(train_total_all,1):.1f}%)')
    print(f'pos_weight: {pos_weight:.2f}')
    print(f'N_TRAIN_SEQS_PER_EPOCH: {N_TRAIN_SEQS_PER_EPOCH} / {len(train_meta)} total\n')

    # ── 模型 ──
    model = DepthFlowClassifier(dropout=DROPOUT).to(device)
    print(f'Model: {model.num_params:,} params\n')

    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # ── 训练 ──
    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    print(f'{"="*70}')
    print(f'Training: {args.epochs} epochs, lr={args.lr}, bs={args.batch_size}')
    print(f'Per-epoch: randomly select {N_TRAIN_SEQS_PER_EPOCH} training sequences')
    print(f'{"="*70}\n')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── 随机选 N 个训练序列，按需加载 ──
        n_pick = min(N_TRAIN_SEQS_PER_EPOCH, len(train_meta))
        chosen = random.sample(train_meta, n_pick)
        train_datasets = []
        ep_samples = 0
        for path, n_samples, n_moving in chosen:
            ds = MotionTrainDataset(path, augment=True)
            train_datasets.append(ds)
            ep_samples += n_samples
        train_ds = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

        chosen_names = '+'.join(os.path.basename(os.path.dirname(p))[:15]
                                for p, _, _ in chosen)
        print(f'[Epoch {epoch:3d}] seqs={chosen_names} ({ep_samples:,} samples)', end=' ', flush=True)

        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer, device, True, pos_weight_t)
        val_loss, val_m = run_epoch(model, val_loader, criterion, None, device, False, pos_weight_t)
        scheduler.step(val_loss)

        # ── 释放训练数据内存 ──
        del train_datasets, train_ds, train_loader

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_m['auc'])
        history['val_auc'].append(val_m['auc'])

        elapsed = time.time() - t0
        print(f'| Train Loss={train_loss:.4f} P={train_m["precision"]:.3f} '
              f'R={train_m["recall"]:.3f} F1={train_m["f1"]:.3f} '
              f'AUC={train_m["auc"]:.3f} | '
              f'Val Loss={val_loss:.4f} P={val_m["precision"]:.3f} '
              f'R={val_m["recall"]:.3f} F1={val_m["f1"]:.3f} '
              f'AUC={val_m["auc"]:.3f} | {elapsed:.1f}s', flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_m,
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f'  → Best model saved (Val Loss={best_val_loss:.4f})')

    print(f'\nBest epoch: {best_epoch}, Val Loss={best_val_loss:.4f}')

    # ── 训练曲线 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_auc'], label='Train')
    ax2.plot(history['val_auc'], label='Val')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('AUC')
    ax2.set_title('AUC Curves'); ax2.legend(); ax2.grid(True, alpha=0.3)

    curves_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.tight_layout(); plt.savefig(curves_path, dpi=100); plt.close()
    print(f'Training curves saved: {curves_path}')

    # ── 测试集评估 ──
    print(f'\n{"="*70}')
    print(f'Test set: {C1_TEST}')
    print(f'{"="*70}')

    if not os.path.exists(C1_TEST):
        print(f'  ERROR: 测试集不存在: {C1_TEST}')
        print(f'  (请先生成 C1 测试集的 motion_train_data.npz)')
    else:
        test_ds = MotionTrainDataset(C1_TEST, augment=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)

        # 加载最佳模型
        checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'),
                                map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        all_probs, all_labels = [], []
        with torch.no_grad():
            for patches, labels in test_loader:
                patches = patches.to(device)
                probs = model(patches)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        test_probs = np.concatenate(all_probs)
        test_labels = np.concatenate(all_labels)

        # 多阈值评估
        thresholds = [0.3, 0.5, 0.7]
        results = {}
        for th in thresholds:
            results[f'th{th:.1f}'] = compute_metrics(test_labels, test_probs, th)

        # 找最佳 F1 阈值
        best_f1, best_th = 0, 0.5
        for t in np.linspace(0.05, 0.95, 91):
            preds = (test_probs >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                test_labels, preds, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1, best_th = f1, t
        results['best'] = compute_metrics(test_labels, test_probs, best_th)

        # 输出报告
        report_lines = [
            f'MotionClassifier (Depth+Flow) Test Report',
            f'{"="*50}',
            f'Test set: c1_transverse1_t1_v2',
            f'Total samples: {len(test_labels):,}',
            f'Moving: {int(test_labels.sum()):,} '
            f'({100*test_labels.sum()/len(test_labels):.1f}%)',
            f'',
            f'--- 多阈值结果 ---',
        ]
        for th, r in results.items():
            label = f'th={best_th:.2f} (best F1)' if th == 'best' else f'th={float(th.replace("th","")):.1f}'
            report_lines.append(f'\n[{label}]')
            report_lines.append(f'  AUC:        {r["auc"]:.4f}')
            report_lines.append(f'  Accuracy:   {r["accuracy"]:.4f}')
            report_lines.append(f'  Precision:  {r["precision"]:.4f}')
            report_lines.append(f'  Recall:     {r["recall"]:.4f}')
            report_lines.append(f'  F1:         {r["f1"]:.4f}')
            report_lines.append(f'  TP={r["tp"]:,}  FP={r["fp"]:,}  '
                                f'TN={r["tn"]:,}  FN={r["fn"]:,}')

        # VO 管线视角
        r = results['th0.5']
        tnr = r['tn'] / max(r['tn'] + r['fp'], 1)  # 静止点保留率
        report_lines.extend([
            f'',
            f'--- VO 管线视角 (th=0.5) ---',
            f'  静止点保留率 (TNR): {tnr:.4f} ({100*tnr:.1f}%)',
            f'  PnP 输入干净度: {r["tn"]/max(r["tn"]+r["fn"],1):.4f}',
            f'  静止点损失: {r["fp"]:,} / {r["tn"]+r["fp"]:,}',
            f'  运动点漏网: {r["fn"]:,} / {r["tp"]+r["fn"]:,}',
        ])

        report_text = '\n'.join(report_lines)
        print(report_text)

        report_path = os.path.join(OUTPUT_DIR, 'test_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f'\nTest report saved: {report_path}')

    print(f'\nAll outputs: {OUTPUT_DIR}')
    print('Done!')


if __name__ == '__main__':
    main()
