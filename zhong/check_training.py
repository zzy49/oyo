"""检查长期优化训练结果"""
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logdir = r'C:\Users\Administrator\tmp\mdp\train'

print("Loading tensorboard events...")
ea = EventAccumulator(logdir, size_guidance={'scalars': 5000})
ea.Reload()

tags = ea.Tags()['scalars']
print(f"\nAvailable tags ({len(tags)}):")
for t in sorted(tags):
    print(f"  {t}")

# 提取关键 loss 曲线
key_tags = [t for t in tags if 'loss' in t.lower() or 'lr' in t.lower()]
print(f"\n{'='*60}")
print("Key loss curves (first 3 and last 3 values):")
print(f"{'='*60}")
for tag in sorted(key_tags):
    events = ea.Scalars(tag)
    if len(events) == 0:
        continue
    vals = [e.value for e in events]
    steps = [e.step for e in events]
    print(f"\n{tag} ({len(events)} points, step {steps[0]}->{steps[-1]}):")
    print(f"  First 3: {[f'{v:.4f}' for v in vals[:3]]}")
    print(f"  Last 3:  {[f'{v:.4f}' for v in vals[-3:]]}")
    print(f"  Min: {min(vals):.4f}  Max: {max(vals):.4f}  Trend: {'↓' if vals[-1] < vals[0] else '↑'}")
