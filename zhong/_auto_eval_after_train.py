"""后台自动化: 等待微调训练完成, 自动运行 EndoSLAM 微调模型评估。

用法:
    python zhong/_auto_eval_after_train.py

逻辑:
    1. 轮询训练进程 (endoslam_finetune 的 train 进程), 每 60s 检查一次
    2. 检测到训练结束 (进程退出 或 出现 weights_4) 后, 等 10s 让文件落盘
    3. 自动运行 zhong/eval_endoslam_finetune.py, 输出保存到日志
"""
import os, sys, time, subprocess, glob

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_dir)

LOG_DIR = r"C:\Users\Administrator\tmp\endoslam_finetune"
WEIGHTS_GLOB = os.path.join(LOG_DIR, "models", "weights_*")
OUT_LOG = os.path.join(_project_dir, "zhong", "eval_endoslam_finetune.log")

# 目标: 5 epoch → weights_4 (索引从 0 开始)
TARGET_WEIGHTS = os.path.join(LOG_DIR, "models", "weights_4")


def find_latest_weights():
    dirs = sorted(glob.glob(WEIGHTS_GLOB))
    return dirs[-1] if dirs else None


def main():
    print(f"[auto_eval] 监控训练: {LOG_DIR}")
    print(f"[auto_eval] 目标权重: {TARGET_WEIGHTS}")
    print(f"[auto_eval] 输出日志: {OUT_LOG}")

    waited = 0
    while True:
        if os.path.isdir(TARGET_WEIGHTS):
            print(f"[auto_eval] 检测到最终权重 weights_4 (已等 {waited//60} 分钟)")
            # 再等 15s 确保 checkpoint 完整写入
            time.sleep(15)
            break
        latest = find_latest_weights()
        print(f"[auto_eval] 等待中... ({waited//60} 分钟, 当前最新: {latest})")
        time.sleep(60)
        waited += 60
        if waited > 6 * 3600:  # 最多等 6 小时
            print("[auto_eval] 超时 6 小时, 退出")
            return

    # 运行评估
    weights = TARGET_WEIGHTS if os.path.isdir(TARGET_WEIGHTS) else find_latest_weights()
    cmd = [sys.executable, os.path.join(_project_dir, "zhong", "eval_endoslam_finetune.py"),
           "--weights", weights,
           "--max_frames", "0",
           "--pose_stride", "10",
           "--max_pose_pairs", "400"]
    print(f"[auto_eval] 启动评估: {' '.join(cmd)}")
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write(f"=== EndoSLAM 微调模型评估 (weights={weights}) ===\n\n")
        f.flush()
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=_project_dir)
    print(f"[auto_eval] 评估完成, exit_code={proc.returncode}, 日志: {OUT_LOG}")


if __name__ == "__main__":
    main()
