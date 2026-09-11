#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""结束正在运行的 train_md2_sup.py 训练进程。"""
import subprocess, re

# 用 wmic 列出 python 进程的 PID 和命令行
out = subprocess.run(
    ['wmic', 'process', 'where', "name='python.exe'",
     'get', 'ProcessId,CommandLine', '/format:csv'],
    capture_output=True, text=True, encoding='utf-8', errors='replace'
).stdout

killed = []
for line in out.splitlines():
    if 'train_md2_sup' in line:
        m = re.search(r',(\d+)\s*$', line.strip())
        if m:
            pid = m.group(1)
            subprocess.run(['taskkill', '/F', '/PID', pid],
                           capture_output=True, text=True)
            killed.append(pid)
            print(f'[KILLED] PID {pid}')

if not killed:
    print('[INFO] 未找到 train_md2_sup 进程 (可能已结束)')
