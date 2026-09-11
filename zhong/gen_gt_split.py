#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 multi_c3vd_gt split: 过滤掉无 GT 深度的 EndoSLAM(@endo@) 样本。"""
import os, io

SRC = r'e:\data1\monodepth2\huifu\monodepth2-master\splits\multi_c3vd'
DSTS = [
    r'e:\data1\monodepth2\huifu\monodepth2-master\splits\multi_c3vd_gt',
    r'e:\data1\manydepth-master\splits\multi_c3vd_gt',
    r'E:\data1\Lite-Mono-main\splits\multi_c3vd_gt',
]

train = [l for l in io.open(os.path.join(SRC, 'train_files.txt'), encoding='utf-8').read().splitlines()]
val = [l for l in io.open(os.path.join(SRC, 'val_files.txt'), encoding='utf-8').read().splitlines()]

train_gt = [l for l in train if l.strip() and not l.split()[0].startswith('@endo@')]
val_gt = [l for l in val if l.strip() and not l.split()[0].startswith('@endo@')]

print(f'原始 train={len(train)} val={len(val)}')
print(f'过滤后 train={len(train_gt)} val={len(val_gt)}')

for d in DSTS:
    os.makedirs(d, exist_ok=True)
    with io.open(os.path.join(d, 'train_files.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_gt) + '\n')
    with io.open(os.path.join(d, 'val_files.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_gt) + '\n')
    print(f'[OK] {d}')
