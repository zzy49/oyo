"""一次性脚本: 重新运行 EndoSLAM 微调模型评估 (修复维度 bug 后)。

输出以原始字节写入 eval_endoslam_finetune.log, 保证 ASCII 标记 ALL_EVAL_DONE 可被检测。
"""
import subprocess, sys, os

PROJECT = r"e:\data1\monodepth2"
WEIGHTS = r"C:\Users\Administrator\tmp\endoslam_finetune\models\weights_4"

cmd = [sys.executable, os.path.join(PROJECT, "zhong", "eval_endoslam_finetune.py"),
       "--weights", WEIGHTS,
       "--max_frames", "0",
       "--pose_stride", "10",
       "--max_pose_pairs", "400"]

log_path = os.path.join(PROJECT, "zhong", "eval_endoslam_finetune.log")
with open(log_path, "wb") as f:
    p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=PROJECT)
print("exit_code:", p.returncode)
