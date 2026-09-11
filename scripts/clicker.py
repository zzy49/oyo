"""
连点器（双圆心）：两个圆心各自圆内随机点击
打开后点击鼠标左键2次确定圆心位置，F4 启动/停止，ESC 退出
"""
import random
import time
import threading
import ctypes
import math

# Windows API 常量
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
VK_O = 0x4F
VK_F4 = 0x73
VK_ESCAPE = 0x1B

user32 = ctypes.windll.user32

# 配置
RADIUS = 53           # 圆的半径（像素）
MIN_INTERVAL = 16     # 长间隔最小（秒），原14+2
MAX_INTERVAL = 18     # 长间隔最大（秒），原16+2
SHORT_MIN = 4         # 短间隔最小（秒），原2+2
SHORT_MAX = 5         # 短间隔最大（秒），原3+2
POINT2_DELAY_MIN = 1  # 点1点击后→点2点击的延迟最小（秒）
POINT2_DELAY_MAX = 2  # 点1点击后→点2点击的延迟最大（秒）

running = False
stop_event = threading.Event()
center1 = None
center2 = None


def get_mouse_pos():
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def set_mouse_pos(x, y):
    user32.SetCursorPos(x, y)


def click_at(x, y):
    """移动到(x,y)并点击"""
    set_mouse_pos(x, y)
    time.sleep(0.05)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.05)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.1)


def random_point_in_circle(cx, cy, radius):
    """在圆内生成随机点（均匀分布）"""
    angle = random.uniform(0, 2 * math.pi)
    r = radius * math.sqrt(random.uniform(0, 1))
    x = int(cx + r * math.cos(angle))
    y = int(cy + r * math.sin(angle))
    return x, y


def do_point1_click(cx1, cy1, label):
    """在圆心1的圆内点击，返回偏移信息"""
    tx, ty = random_point_in_circle(cx1, cy1, RADIUS)
    click_at(tx, ty)
    offset_x = tx - cx1
    offset_y = ty - cy1
    dist = math.sqrt(offset_x ** 2 + offset_y ** 2)
    print(f"  {label}: ({tx}, {ty})  偏移: ({offset_x:+d}, {offset_y:+d})  距离: {dist:.0f}px")


def do_point2_click(cx2, cy2):
    """在圆心2的圆内点击，前面有1-2s延迟"""
    wait = random.uniform(POINT2_DELAY_MIN, POINT2_DELAY_MAX)
    if stop_event.wait(wait):
        return True  # 被中断
    tx, ty = random_point_in_circle(cx2, cy2, RADIUS)
    click_at(tx, ty)
    offset_x = tx - cx2
    offset_y = ty - cy2
    dist = math.sqrt(offset_x ** 2 + offset_y ** 2)
    print(f"           点2: ({tx}, {ty})  偏移: ({offset_x:+d}, {offset_y:+d})  距离: {dist:.0f}px")
    return False


def click_loop():
    global running, center1, center2

    cycle_count = 0
    while running:
        cx1, cy1 = center1
        cx2, cy2 = center2

        # === 第1次点击（点1）+ 点2跟进 ===
        do_point1_click(cx1, cy1, f"[周期 #{cycle_count + 1}] 点1-1")
        if do_point2_click(cx2, cy2):
            break

        # 长间隔 16-18s
        wait = random.uniform(MIN_INTERVAL, MAX_INTERVAL)
        print(f"  长间隔等待 {wait:.1f}s ...")
        if stop_event.wait(wait):
            break

        # 短间隔 4-5s + 连续2次（点1-2、点1-3），每次点1后跟点2
        for i in range(2):
            wait = random.uniform(SHORT_MIN, SHORT_MAX)
            print(f"    短间隔等待 {wait:.1f}s ...")
            if stop_event.wait(wait):
                return

            do_point1_click(cx1, cy1, f"  点1-{i + 2}")
            if do_point2_click(cx2, cy2):
                return

        # 点1-3之后，短间隔再回到点1-1
        wait = random.uniform(SHORT_MIN, SHORT_MAX)
        print(f"    短间隔等待 {wait:.1f}s (回到点1) ...")
        if stop_event.wait(wait):
            break

        cycle_count += 1


def toggle():
    global running
    if running:
        running = False
        stop_event.set()
        print("\n[停止] 连点器已停止")
    else:
        running = True
        stop_event.clear()
        t = threading.Thread(target=click_loop, daemon=True)
        t.start()
        print("\n[启动] 连点器已启动")


def wait_for_click(prompt):
    """等待用户按O键，记录当前鼠标位置为圆心"""
    print(f"  → {prompt}")
    pressed = False
    while True:
        if user32.GetAsyncKeyState(VK_O) & 0x8000:
            if not pressed:
                pressed = True
                pos = get_mouse_pos()
                print(f"  已记录: ({pos[0]}, {pos[1]})")
                time.sleep(0.3)  # 防抖
                return pos
        else:
            pressed = False
        time.sleep(0.05)


def main():
    global center1, center2, running

    print("=" * 45)
    print("  连点器（双圆心）")
    print("=" * 45)
    print(f"  圆半径: {RADIUS}px")
    print(f"  长间隔: {MIN_INTERVAL}-{MAX_INTERVAL}s")
    print(f"  短间隔: {SHORT_MIN}-{SHORT_MAX}s")
    print(f"  点2延迟: {POINT2_DELAY_MIN}-{POINT2_DELAY_MAX}s")
    print("-" * 45)
    print("  请移动鼠标到目标位置，按 O 键 2 次来设置圆心位置：")

    center1 = wait_for_click("等待第1次点击（圆心1）...")
    center2 = wait_for_click("等待第2次点击（圆心2）...")

    print("-" * 45)
    print("  两个圆心已就绪！")
    print(f"  圆心1: ({center1[0]}, {center1[1]})")
    print(f"  圆心2: ({center2[0]}, {center2[1]})")
    print(f"  F4  = 启动/停止")
    print(f"  ESC = 退出程序")
    print("=" * 45)

    # 全局热键循环
    f4_pressed = False
    while True:
        # 检测 F4
        if user32.GetAsyncKeyState(VK_F4) & 0x8000:
            if not f4_pressed:
                f4_pressed = True
                toggle()
        else:
            f4_pressed = False

        # 检测 ESC
        if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
            if running:
                running = False
                stop_event.set()
            print("\n[退出] 程序已退出")
            break

        time.sleep(0.05)


if __name__ == "__main__":
    main()
