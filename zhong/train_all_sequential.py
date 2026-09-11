"""Sequential training: Monodepth2 -> ManyDepth -> Lite-Mono"""
import sys, os, subprocess, time
from datetime import datetime

ZHONG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ZHONG)
LOGDIR = os.path.join(ZHONG, 'logs')
os.makedirs(LOGDIR, exist_ok=True)

steps = [
    ('Monodepth2', os.path.join(ZHONG, 'train_md2_sup.py'), os.path.join(LOGDIR, 'md2_seq_train.log')),
    ('ManyDepth',  os.path.join(ZHONG, 'train_manydepth_sup.py'), os.path.join(LOGDIR, 'manydepth_seq_train.log')),
    ('Lite-Mono',  os.path.join(ZHONG, 'train_litemono_sup.py'), os.path.join(LOGDIR, 'litemono_seq_train.log')),
]

for name, script, log_path in steps:
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    divider = '=' * 70
    header = f'{divider}\n  [{stamp}] STARTING {name} training\n{divider}'
    print(header)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(header + '\n')

    proc = subprocess.Popen(
        [sys.executable, '-u', script],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )
    with open(log_path, 'a', encoding='utf-8') as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()
            # Also echo training progress lines to terminal
            if line.startswith('epoch') or 'STARTING' in line or 'COMPLETED' in line:
                print('  [' + name + '] ' + line.rstrip(), flush=True)
    ret = proc.wait()
    stamp_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if ret == 0:
        print(f'  [{stamp_end}] {name} COMPLETED successfully')
    else:
        print(f'  [{stamp_end}] {name} FAILED with code {ret}')
        sys.exit(ret)

print('\n' + '=' * 70)
print('  ALL THREE BASELINES TRAINING COMPLETED!')
print('=' * 70)
