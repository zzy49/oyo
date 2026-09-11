"""快速提取训练loss关键指标"""
import struct, os

logdir = r'C:\Users\Administrator\tmp\mdp\train'

# 只读最新的events文件
files = sorted([f for f in os.listdir(logdir) if f.startswith('events')])
files = [f for f in files if os.path.getsize(os.path.join(logdir, f)) > 100000]
print(f"Large event files: {files[-3:]}")

# 使用tensorflow的EventFileReader (更高效)
try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
    latest = os.path.join(logdir, files[-1])
    print(f"\nParsing: {files[-1]} ({os.path.getsize(latest)//1024//1024} MB)...")
    
    losses = {}
    for i, e in enumerate(summary_iterator(latest)):
        if i > 0 and i % 500000 == 0:
            print(f"  processed {i} events...")
        for v in e.summary.value:
            if 'loss' in v.tag.lower() or 'lr' in v.tag.lower():
                if v.tag not in losses:
                    losses[v.tag] = []
                losses[v.tag].append((e.step, v.simple_value))
    
    print(f"\nLoss trends ({len(losses)} tags):")
    for tag in sorted(losses.keys()):
        vals = losses[tag]
        if len(vals) < 2:
            continue
        steps = [s for s, _ in vals]
        values = [v for _, v in vals]
        print(f"  {tag}: {len(vals)} pts, step {steps[0]}->{steps[-1]}")
        print(f"    First 3: {[f'{v:.4f}' for v in values[:3]]}")
        print(f"    Last 3:  {[f'{v:.4f}' for v in values[-3:]]}")
        print(f"    Min={min(values):.4f} Max={max(values):.4f}", end="")
        if values[-1] < values[0]:
            print(f" ↓({(1-values[-1]/values[0])*100:.1f}%)")
        else:
            print(f" ↑")
except ImportError:
    print("tensorflow not available, trying alternative...")
    # Fallback: raw binary parse
    from struct import unpack
    latest = os.path.join(logdir, files[-1])
    with open(latest, 'rb') as f:
        data = f.read()
    print(f"Read {len(data)} bytes")
