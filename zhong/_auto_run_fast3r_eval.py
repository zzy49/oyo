"""后台自动化: 微调训练+微调评估完成后, 自动运行 Endo-FASt3r 在 SCARED 上的深度+位姿评估。

用法:
    python zhong/_auto_run_fast3r_eval.py

逻辑:
    1. 等待微调训练完成 (weights_4 出现)
    2. 等待微调模型评估完成 (eval_endoslam_finetune.log 含 "完成")
    3. 依次运行 Endo-FASt3r 深度评估 + 位姿评估 seq1 + seq2
"""
import os, sys, time, subprocess

PROJECT = r"e:\data1\monodepth2"
FAST3R = os.path.join(PROJECT, "baselines", "Endo_FASt3r")
TRAIN_WEIGHTS4 = r"C:\Users\Administrator\tmp\endoslam_finetune\models\weights_4"
EVAL_LOG = os.path.join(PROJECT, "zhong", "eval_endoslam_finetune.log")
OUT_LOG = os.path.join(PROJECT, "zhong", "fast3r_eval.log")


def log(msg):
    line = f"[fast3r_eval] {msg}"
    print(line, flush=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd):
    log(f"运行: {cmd}")
    p = subprocess.run(cmd, shell=True, cwd=FAST3R,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", errors="replace")
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(out + "\n")
    log(f"退出码: {p.returncode}")
    return p.returncode


def main():
    log("=== Endo-FASt3r 评估自动化启动 ===")

    # 1. 等待微调训练完成
    waited = 0
    while not os.path.isdir(TRAIN_WEIGHTS4):
        log(f"等待微调训练完成(weights_4)... 已等 {waited} 分钟")
        time.sleep(60)
        waited += 1
        if waited > 360:
            log("等待训练超时 6 小时, 退出")
            return
    log("检测到 weights_4, 训练完成")

    # 2. 等待微调模型评估完成 (检测 ASCII 标记 ALL_EVAL_DONE, 避免编码问题)
    waited2 = 0
    while True:
        done = False
        if os.path.isfile(EVAL_LOG):
            try:
                with open(EVAL_LOG, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if "ALL_EVAL_DONE" in content:
                    done = True
            except Exception as e:
                log(f"读取评估日志失败: {e}")
        if done:
            log("微调评估已完成")
            break
        log(f"等待微调评估完成... 已等 {waited2} 分钟")
        time.sleep(60)
        waited2 += 1
        if waited2 > 180:
            log("等待微调评估超时 3 小时, 继续执行 Endo-FASt3r 评估")
            break

    # 3. 深度评估 (单目, median scaling)
    run("python evaluate_depth.py --data_path SCARED_Images_Resized "
        "--load_weights_folder best_weights --eval_mono --num_workers 4")

    # 4. 位姿评估 seq1 (batch_size 必须 >= 2, 因 prepare_images 的 squeeze)
    run("python evaluate_pose.py --data_path SCARED_Images_Resized "
        "--load_weights_folder best_weights --scared_pose_seq 1 "
        "--batch_size 2 --num_workers 4")

    # 5. 位姿评估 seq2
    run("python evaluate_pose.py --data_path SCARED_Images_Resized "
        "--load_weights_folder best_weights --scared_pose_seq 2 "
        "--batch_size 2 --num_workers 4")

    log("=== Endo-FASt3r 评估全部完成 ===")


if __name__ == "__main__":
    main()
