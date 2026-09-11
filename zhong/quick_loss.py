"""快速检查训练loss (限制加载量)"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logdir = r'C:\Users\Administrator\tmp\mdp\train'
print("Loading (size_guidance=50)...")
ea = EventAccumulator(logdir, size_guidance={'scalars': 50})
ea.Reload()

tags = [t for t in ea.Tags()['scalars'] if 'loss' in t.lower()]
print(f"Loss tags found: {len(tags)}")
for t in sorted(tags):
    ev = ea.Scalars(t)
    if len(ev) >= 2:
        first, last = ev[0].value, ev[-1].value
        arrow = "↓" if last < first else "↑"
        pct = (1 - last/first)*100 if first != 0 else 0
        print(f"  {t:40s}: {first:.4f} -> {last:.4f} {arrow} {pct:+.1f}%")
