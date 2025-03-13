import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# 读取数据
data = pd.read_csv('output_coordinates.csv')

# 提取必要列
frames = data['Frame']
ball_x = data['Ball_X']
ball_y = data['Ball_Y']
player1_x = data['Person1_X']
player1_y = data['Person1_Y']
player2_x = data['Person2_X']
player2_y = data['Person2_Y']
bounce_frames = data['Bounce']

# 替换 0 为 NaN 以便插值
ball_x[ball_x == 0] = np.nan
ball_y[ball_y == 0] = np.nan

# 使用线性插值填补缺失值
ball_x = ball_x.interpolate()
ball_y = ball_y.interpolate()

# 平滑球的 Y 轴数据
smoothed_ball_y = savgol_filter(ball_y, window_length=31, polyorder=3)
smoothed_ball_x = savgol_filter(ball_x, window_length=31, polyorder=3)

# 识别转折点（极大值和极小值）
peaks, _ = find_peaks(smoothed_ball_y, prominence=100, distance=30)  # 极大值
troughs, _ = find_peaks(-smoothed_ball_y, prominence=100, distance=30)  # 极小值
turning_points = np.sort(np.concatenate((peaks, troughs)))

# 增加距离约束，保留显著转折点
filtered_turning_points = []
min_distance = 800  # 设置极值点之间的最小距离
for i in range(len(turning_points) - 1):
    if abs(smoothed_ball_y[turning_points[i]] - smoothed_ball_y[turning_points[i + 1]]) > min_distance:
        filtered_turning_points.append(turning_points[i])
filtered_turning_points.append(turning_points[-1])  # 添加最后一个转折点

# 分组逻辑：奇数组和偶数组
group_1 = filtered_turning_points[::2]  # 奇数位置
group_2 = filtered_turning_points[1::2]  # 偶数位置

# 统计每组的整体偏向
def determine_group_side(group):
    top_count = sum(1 for point in group if smoothed_ball_y[point] < 1748)
    bottom_count = sum(1 for point in group if smoothed_ball_y[point] >= 1748)
    return 'top' if top_count > bottom_count else 'bottom'

group_1_side = determine_group_side(group_1)
group_2_side = determine_group_side(group_2)

# 根据分组结果对每个转折点分类
hit_points_x = []
hit_points_y = []
hit_point_frames = []
for i, point in enumerate(filtered_turning_points):
    # 判断当前点属于哪组
    if i % 2 == 0:  # 奇数索引，属于 group_1
        side = group_1_side
    else:  # 偶数索引，属于 group_2
        side = group_2_side

    # 记录击打点的坐标
    hit_point_frames.append(frames[point])
    if side == 'top':  # 玩家1击打
        hit_points_x.append(player1_x[point])
        hit_points_y.append(player1_y[point])
    else:  # 玩家2击打
        hit_points_x.append(player2_x[point])
        hit_points_y.append(player2_y[point])

# 确保击打点是浮点数
hit_points_x = [float(x) if x != 0 else None for x in hit_points_x]
hit_points_y = [float(y) if x != 0 else None for x, y in zip(hit_points_x, hit_points_y)]

# 标注弹跳点的位置
bounce_points_x = []
bounce_points_y = []
bounce_point_frames = []
original_bounce_points_x = []
original_bounce_points_y = []
for i, bounce in enumerate(bounce_frames):
    if bounce == 1:  # 标记为弹跳点的帧
        # 根据前后转折点插值计算位置
        previous_turning = max([pt for pt in filtered_turning_points if pt < i], default=None)
        next_turning = min([pt for pt in filtered_turning_points if pt > i], default=None)
        if previous_turning is not None and next_turning is not None:
            t_ratio = (i - previous_turning) / (next_turning - previous_turning)
            interpolated_x = hit_points_x[filtered_turning_points.index(previous_turning)] + t_ratio * (
                hit_points_x[filtered_turning_points.index(next_turning)] - hit_points_x[filtered_turning_points.index(previous_turning)])
            interpolated_y = hit_points_y[filtered_turning_points.index(previous_turning)] + t_ratio * (
                hit_points_y[filtered_turning_points.index(next_turning)] - hit_points_y[filtered_turning_points.index(previous_turning)])
            bounce_points_x.append(interpolated_x)
            bounce_points_y.append(interpolated_y)
            bounce_point_frames.append(frames[i])
            original_bounce_points_x.append(ball_x[i])
            original_bounce_points_y.append(ball_y[i])

# 绘制球在球场上的位置变化连线
plt.figure(figsize=(8, 12))  # 改为竖直长方形布局
plt.xlim(0, 1600)
plt.ylim(0, 3500)

# 绘制击打点之间的连线
for i in range(len(hit_points_x) - 1):
    if hit_points_x[i] is not None and hit_points_x[i + 1] is not None:
        plt.plot(
            [hit_points_x[i], hit_points_x[i + 1]],
            [hit_points_y[i], hit_points_y[i + 1]],
            color="blue", linestyle="-", linewidth=2, label="Ball Path" if i == 0 else ""
        )
    # 标注击打点和对应的帧号
    if hit_points_x[i] is not None and hit_points_y[i] is not None:
        plt.scatter(
            hit_points_x[i], hit_points_y[i], color="red", s=50, zorder=5, label="Hit Point" if i == 0 else ""
        )
        plt.text(
            hit_points_x[i], hit_points_y[i] + 50, f"Frame: {hit_point_frames[i]}", fontsize=8, color="black"
        )

# 标注最后一个击打点和帧号
if hit_points_x[-1] is not None and hit_points_y[-1] is not None:
    plt.scatter(hit_points_x[-1], hit_points_y[-1], color="red", s=50, zorder=5)
    plt.text(
        hit_points_x[-1], hit_points_y[-1] + 50, f"Frame: {hit_point_frames[-1]}", fontsize=8, color="black"
    )

# 绘制弹跳点
for i in range(len(bounce_points_x)):
    if bounce_points_x[i] is not None and bounce_points_y[i] is not None:
        plt.scatter(
            bounce_points_x[i], bounce_points_y[i], color="green", s=50, zorder=5, label="Bounce Point" if i == 0 else ""
        )
        plt.text(
            bounce_points_x[i], bounce_points_y[i] + 50, f"Frame: {bounce_point_frames[i]}", fontsize=8, color="darkgreen"
        )
        # 连线原始和推测的弹跳点
        plt.plot(
            [bounce_points_x[i], original_bounce_points_x[i]],
            [bounce_points_y[i], original_bounce_points_y[i]],
            color="purple", linestyle="--", linewidth=1, label="Bounce Deviation" if i == 0 else ""
        )

# 添加图例和标签
plt.title("Ball Trajectory Between Players with Bounce Points")
plt.xlabel("Court Width (X-Axis)")
plt.ylabel("Court Height (Y-Axis)")
plt.legend()
plt.grid()

# 显示图像
plt.show()

# 打印每个击打点的帧号及其归属侧
for i, frame in enumerate(hit_point_frames):
    side = group_1_side if i % 2 == 0 else group_2_side
    print(f"Hit Point {i + 1}: Frame {frame}, Side: {side}")

# 打印每个弹跳点的帧号
for i, frame in enumerate(bounce_point_frames):
    print(f"Bounce Point {i + 1}: Frame {frame}")

# 绘制球的位置随时间变化的图像
plt.figure(figsize=(10, 6))

# 绘制 X 轴位置随时间变化
plt.subplot(2, 1, 1)
plt.plot(frames, ball_x, label="Ball X Position", color="orange")
plt.title("Ball X-Axis Position Over Time")
plt.xlabel("Frame")
plt.ylabel("Court Width (X-Axis)")
plt.legend()
plt.grid()

# 绘制 Y 轴位置随时间变化
plt.subplot(2, 1, 2)
plt.plot(frames, ball_y, label="Ball Y Position", color="blue")
plt.title("Ball Y-Axis Position Over Time")
plt.xlabel("Frame")
plt.ylabel("Court Height (Y-Axis)")
plt.legend()
plt.grid()

# 调整子图间距并显示图像
plt.tight_layout()
plt.show()
