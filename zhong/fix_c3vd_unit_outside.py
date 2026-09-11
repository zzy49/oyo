#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""修复工作区外两个 c3vd_dataset.py 的 GT 深度单位换算 bug (uint16→mm)。"""
import io

targets = [
    r'e:\data1\manydepth-master\manydepth\datasets\c3vd_dataset.py',
    r'E:\data1\Lite-Mono-main\datasets\c3vd_dataset.py',
]

OLD = ("        depth = np.array(depth_img, dtype=np.float32)\n"
       "        depth = np.clip(depth, 1.0, 500.0)\n")

NEW = ("        depth = np.array(depth_img, dtype=np.float32)\n"
       "        # uint16 编码 0-65535 线性映射到 0-100mm (与评估脚本 load_gt_depth 一致)\n"
       "        depth = depth * (100.0 / 65535.0)\n"
       "        depth = np.clip(depth, 1.0, 500.0)\n")

for p in targets:
    with io.open(p, 'r', encoding='utf-8') as f:
        s = f.read()
    n = s.count(OLD)
    if n == 0:
        print(f'[SKIP] 未找到目标片段: {p}')
        continue
    s = s.replace(OLD, NEW)
    with io.open(p, 'w', encoding='utf-8') as f:
        f.write(s)
    print(f'[OK] 已修复 {n} 处: {p}')
