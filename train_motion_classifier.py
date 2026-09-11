"""
组织运动分类器训练
==================
从 .npz 文件加载训练/验证数据, 训练 MotionClassifier.
"""
import os, sys, glob, argparse, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(__file__))
from motion_classifier import MotionClassifier

# ── Config ──
TRAIN_DIR = r'E:\data1\monodepth2\motion_cls_data\train_npy'
VAL_DIR = r'E:\data1\monodepth2\motion_cls_data\val_npy'
OUTPUT_DIR = r'E:\data1\monodepth2\motion_cls_data\checkpoints'

BATCH_SIZE = 256
NUM_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 4.0       # moving class weight (static: 80%, moving: 20%)
DROPOUT = 0.3
PATCH_DIM = 16


class MotionDataset(Dataset):
    """Loads pre-converted .npy files. Train: in-memory, Val: mmap."""

    def __init__(self, data_dir, augment=False, use_mmap=False):
        self.augment = augment
        meta = np.load(os.path.join(data_dir, 'meta.npz'))
        self._samples = int(meta['total'])

        if use_mmap:
            # Memory-mapped arrays (for val set, only accessed once per epoch)
            self.patches_i = np.memmap(os.path.join(data_dir, 'patches_i.npy'),
                                       dtype=np.uint8, mode='r',
                                       shape=(self._samples, 16, 16, 3))
            self.patches_j = np.memmap(os.path.join(data_dir, 'patches_j.npy'),
                                       dtype=np.uint8, mode='r',
                                       shape=(self._samples, 16, 16, 3))
            self.flows = np.memmap(os.path.join(data_dir, 'flows.npy'),
                                   dtype=np.float32, mode='r',
                                   shape=(self._samples, 2))
            self.pos = np.memmap(os.path.join(data_dir, 'pos.npy'),
                                 dtype=np.float32, mode='r',
                                 shape=(self._samples, 2))
            self.labels = np.memmap(os.path.join(data_dir, 'labels.npy'),
                                    dtype=np.float32, mode='r',
                                    shape=(self._samples,))
            self.disps = np.memmap(os.path.join(data_dir, 'displacements.npy'),
                                   dtype=np.float32, mode='r',
                                   shape=(self._samples,))
            mode_str = 'mmap'
        else:
            # Load into RAM for fast random access during training
            print(f'  Loading into RAM...')
            self.patches_i = np.array(np.memmap(
                os.path.join(data_dir, 'patches_i.npy'), dtype=np.uint8, mode='r',
                shape=(self._samples, 16, 16, 3)))
            self.patches_j = np.array(np.memmap(
                os.path.join(data_dir, 'patches_j.npy'), dtype=np.uint8, mode='r',
                shape=(self._samples, 16, 16, 3)))
            self.flows = np.array(np.memmap(
                os.path.join(data_dir, 'flows.npy'), dtype=np.float32, mode='r',
                shape=(self._samples, 2)))
            self.pos = np.array(np.memmap(
                os.path.join(data_dir, 'pos.npy'), dtype=np.float32, mode='r',
                shape=(self._samples, 2)))
            self.labels = np.array(np.memmap(
                os.path.join(data_dir, 'labels.npy'), dtype=np.float32, mode='r',
                shape=(self._samples,)))
            self.disps = np.array(np.memmap(
                os.path.join(data_dir, 'displacements.npy'), dtype=np.float32, mode='r',
                shape=(self._samples,)))
            mode_str = 'RAM'

        print(f'  {data_dir}: {self._samples:,} samples ({mode_str})')

    def __len__(self):
        return self._samples

    def __getitem__(self, idx):

        # Patches: uint8 [0,255] → float32 [0,1]
        patch_i = self.patches_i[idx].astype(np.float32) / 255.0
        patch_j = self.patches_j[idx].astype(np.float32) / 255.0

        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                patch_i = np.fliplr(patch_i).copy()
                patch_j = np.fliplr(patch_j).copy()
                # Flip flow x component
                flows = self.flows[idx].copy()
                flows[0] = -flows[0]
                pos = self.pos[idx].copy()
                pos[0] = 1.0 - pos[0]
            else:
                flows = self.flows[idx]
                pos = self.pos[idx]

            # Brightness jitter
            if np.random.rand() > 0.5:
                b = np.random.uniform(0.9, 1.1)
                patch_i = np.clip(patch_i * b, 0, 1)
                patch_j = np.clip(patch_j * b, 0, 1)
        else:
            flows = self.flows[idx]
            pos = self.pos[idx]

        # HWC → CHW
        patch_i = torch.from_numpy(patch_i.transpose(2, 0, 1)).float()
        patch_j = torch.from_numpy(patch_j.transpose(2, 0, 1)).float()

        # Flow features: (dx, dy, norm_x, norm_y)
        flow_feat = torch.tensor([
            flows[0], flows[1], pos[0], pos[1]
        ], dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return patch_i, patch_j, flow_feat, label


def compute_metrics(labels, probs, threshold=0.5):
    """Compute precision, recall, f1, accuracy, AUC."""
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0)

    acc = np.mean(preds == labels)

    # AUC
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs)
    else:
        auc = 0.5

    # Per-class breakdown
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'accuracy': acc, 'auc': auc,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_labels = []
    all_probs = []

    for patch_i, patch_j, flow_feat, labels in loader:
        patch_i = patch_i.to(device)
        patch_j = patch_j.to(device)
        flow_feat = flow_feat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(patch_i, patch_j, flow_feat).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    metrics = compute_metrics(all_labels, all_probs)

    return total_loss / len(all_labels), metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []

    for patch_i, patch_j, flow_feat, labels in loader:
        patch_i = patch_i.to(device)
        patch_j = patch_j.to(device)
        flow_feat = flow_feat.to(device)
        labels = labels.to(device)

        logits = model(patch_i, patch_j, flow_feat).squeeze(-1)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    metrics = compute_metrics(all_labels, all_probs)

    return total_loss / len(all_labels), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--pos_weight', type=float, default=POS_WEIGHT)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    # ── Datasets ──
    print('\nLoading datasets...', flush=True)
    train_ds = MotionDataset(TRAIN_DIR, augment=True, use_mmap=False)
    val_ds = MotionDataset(VAL_DIR, augment=False, use_mmap=True)
    print(f'  Train: {len(train_ds):,} samples', flush=True)
    print(f'  Val:   {len(val_ds):,} samples', flush=True)

    # Check class balance (labels is mmap, sum reads from disk efficiently)
    n_moving_train = int(train_ds.labels.sum())
    n_moving_val = int(val_ds.labels.sum())
    print(f'  Train moving: {n_moving_train:,} ({100*n_moving_train/len(train_ds):.1f}%)', flush=True)
    print(f'  Val moving:   {n_moving_val:,} ({100*n_moving_val/len(val_ds):.1f}%)', flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ──
    model = MotionClassifier(dropout=DROPOUT).to(device)
    print(f'\nModel params: {model.num_params:,}', flush=True)

    # ── Loss & Optimizer ──
    pos_weight_tensor = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ──
    best_val_f1 = 0
    best_epoch = 0
    history = []

    print(f'\n{"="*70}', flush=True)
    print(f'Training: {args.epochs} epochs, lr={args.lr}, pos_weight={args.pos_weight}', flush=True)
    print(f'{"="*70}', flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        # Log
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train Loss={train_loss:.4f} P={train_metrics["precision"]:.3f} R={train_metrics["recall"]:.3f} '
              f'F1={train_metrics["f1"]:.3f} AUC={train_metrics["auc"]:.3f} | '
              f'Val Loss={val_loss:.4f} P={val_metrics["precision"]:.3f} R={val_metrics["recall"]:.3f} '
              f'F1={val_metrics["f1"]:.3f} AUC={val_metrics["auc"]:.3f} | '
              f'{elapsed:.1f}s', flush=True)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_f1': train_metrics['f1'],
            'val_loss': val_loss, 'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_tp': int(val_metrics['tp']),
            'val_fp': int(val_metrics['fp']),
            'val_tn': int(val_metrics['tn']),
            'val_fn': int(val_metrics['fn']),
        })

        # Save best
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_metrics,
            }, os.path.join(OUTPUT_DIR, 'best_model.pt'))
            print(f'  → Best model saved (F1={best_val_f1:.4f})')

    # ── Final ──
    print(f'\nBest model: epoch {best_epoch}, Val F1={best_val_f1:.4f}')

    # Save last checkpoint and history
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
    }, os.path.join(OUTPUT_DIR, 'last_model.pt'))

    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nOutput: {OUTPUT_DIR}')
    print('Done!')


if __name__ == '__main__':
    main()
