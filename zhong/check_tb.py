#!/usr/bin/env python
"""解析 TensorBoard 训练日志, 提取 loss / 验证指标趋势"""
import sys, os
from collections import defaultdict
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("请安装 tensorboard: pip install tensorboard")
    sys.exit(1)

def parse_tb(log_dir, tag_filter=None):
    """加载 TensorBoard 日志, 返回 {tag: [(step, value), ...]}"""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    result = {}
    for tag in tags:
        if tag_filter and not any(t in tag for t in tag_filter):
            continue
        events = ea.Scalars(tag)
        result[tag] = [(e.step, e.value) for e in events]
    return result

MODELS = [
    ('Monodepth2', r'C:\Users\Administrator\tmp\c3vd_md2_full\train'),
    ('ManyDepth',  r'C:\Users\Administrator\tmp\c3vd_manydepth_full\train'),
    ('Lite-Mono',  r'C:\Users\Administrator\tmp\c3vd_litemono_full\train'),
]

KEY_TAGS = ['loss', 'abs_rel', 'rmse', 'a1']

for name, logdir in MODELS:
    print(f'\n{"="*60}')
    print(f'  {name} — {logdir}')
    print(f'{"="*60}')
    data = parse_tb(logdir, tag_filter=KEY_TAGS)
    
    for tag, events in data.items():
        if not events:
            continue
        # 每 ~20% epoch 采样
        n = len(events)
        step_size = max(1, n // 5)
        vals = []
        for i in range(0, n, step_size):
            step, val = events[i]
            vals.append(f'{step}:{val:.4f}')
        last_step, last_val = events[-1]
        print(f'  {tag}: {" → ".join(vals)} (final: {last_val:.4f})')

    if not data:
        print('  (无匹配 tag)')

# 也检查 val 日志
print(f'\n{"="*60}')
print(f'  Monodepth2 — Val Logs')
print(f'{"="*60}')
val_data = parse_tb(r'C:\Users\Administrator\tmp\c3vd_md2_full\val', tag_filter=KEY_TAGS)
for tag, events in val_data.items():
    if not events:
        continue
    n = len(events)
    step_size = max(1, n // 5)
    vals = []
    for i in range(0, n, step_size):
        step, val = events[i]
        vals.append(f'{step}:{val:.4f}')
    last_step, last_val = events[-1]
    print(f'  {tag}: {" → ".join(vals)} (final: {last_val:.4f})')
