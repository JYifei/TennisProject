import cv2
import pandas as pd
import os
from scipy.signal import savgol_filter
import numpy as np

SEGMENT_DIR = "output_segments"
KEYPOINT_BASE_WIDTH = 1280
KEYPOINT_BASE_HEIGHT = 720
WINDOW_SIZE = 11
POLY_ORDER = 2
MARGIN = 80  # 留出球场边缘间隔
MARGIN_RATIO_LR = 0.08   # 左右遮罩使用球场宽度的 20%
MARGIN_RATIO_TOP = 0.4   # 上下遮罩使用球场高度的 50%
MARGIN_RATIO_BOT = 0.13

def smooth_keypoints(df):
    for col in df.columns:
        if col.startswith("KP") and df[col].dtype != 'object':
            series = df[col]
            if series.isna().sum() < len(series) - WINDOW_SIZE:
                smoothed = savgol_filter(series.interpolate(method="linear"), WINDOW_SIZE, POLY_ORDER)
                df[col] = smoothed
            else:
                print(f"跳过平滑 {col}（有效点数不足）")
    return df


def compute_kp34_distances(df, scale_x, scale_y):
    distances = []
    for _, row in df.iterrows():
        try:
            x3, y3 = float(row["KP3_X"]) * scale_x, float(row["KP3_Y"]) * scale_y
            x4, y4 = float(row["KP4_X"]) * scale_x, float(row["KP4_Y"]) * scale_y
            d = np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)
        except:
            d = np.nan
        distances.append(d)
    return distances


def detect_zoom_in_stable_point(distances, window_size=10, stability_frames=10, growth_thresh=0.2):
    series = pd.Series(distances).interpolate(limit_direction="both").fillna(method="bfill")
    moving_avg = series.rolling(window=window_size).mean()
    growth = moving_avg.diff()

    for i in range(window_size + stability_frames, len(growth)):
        recent_growth = growth[i - stability_frames:i]
        if recent_growth.abs().mean() < growth_thresh:
            return i
    return 0


def extend_line(p1, p2, height):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dy == 0:
        return (x1, 0), (x2, height)
    t1 = -y1 / dy
    t2 = (height - y1) / dy
    pt_top = (x1 + t1 * dx, 0)
    pt_bottom = (x1 + t2 * dx, height)
    return pt_top, pt_bottom


def offset_line_outward(p1, p2, offset, image_center_x):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx ** 2 + dy ** 2)
    if length == 0:
        return p1, p2
    nx = -dy / length
    ny = dx / length
    offset_vec = np.array([nx, ny]) * offset
    midpoint = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
    shifted_midpoint = midpoint + offset_vec
    if (midpoint[0] < image_center_x and shifted_midpoint[0] > midpoint[0]) or \
       (midpoint[0] > image_center_x and shifted_midpoint[0] < midpoint[0]):
        offset_vec = -offset_vec
    return (p1[0] + offset_vec[0], p1[1] + offset_vec[1]), (p2[0] + offset_vec[0], p2[1] + offset_vec[1])


def extend_line_horizontally(p1, p2, width):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return (0, y1), (width, y2)
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    pt_left = (0, b)
    pt_right = (width, k * width + b)
    return pt_left, pt_right


def draw_edges_on_video(video_path, csv_path, output_path, cropped_csv_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    df = pd.read_csv(csv_path)
    scale_x = 1
    scale_y = 1
    #scale_x = width / KEYPOINT_BASE_WIDTH
    #scale_y = height / KEYPOINT_BASE_HEIGHT

    distances = compute_kp34_distances(df, scale_x, scale_y)
    skip_frames = 0
    print(f"{os.path.basename(video_path)} 跳过前 {skip_frames} 帧")

    # 裁剪并保存 CSV
    df = df.iloc[skip_frames:].copy().reset_index(drop=True)
    df["Frame"] = range(len(df))
    #df.to_csv(cropped_csv_path, index=False)
    df = smooth_keypoints(df)

    # 跳过视频帧
    for _ in range(skip_frames):
        cap.read()

    for _, row in df.iterrows():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            x1, y1 = float(row["KP1_X"]) * scale_x, float(row["KP1_Y"]) * scale_y
            x2, y2 = float(row["KP2_X"]) * scale_x, float(row["KP2_Y"]) * scale_y
            x3, y3 = float(row["KP3_X"]) * scale_x, float(row["KP3_Y"]) * scale_y
            x4, y4 = float(row["KP4_X"]) * scale_x, float(row["KP4_Y"]) * scale_y
            
            width_top = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            width_bottom = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)

            court_width = (width_top + width_bottom) / 2
            
            height_left = abs(y3 - y1)
            height_right = abs(y4 - y2)

            court_height = (height_left + height_right) / 2
            
            margin_lr = MARGIN_RATIO_LR * court_width      # → 用于左右遮罩
            margin_top = MARGIN_RATIO_TOP * court_height     # → 用于上下遮罩
            margin_bot = MARGIN_RATIO_BOT * court_height

            
            #print(court_width)

            if not any(pd.isna([x1, y1, x3, y3])):
                p_top, p_bottom = extend_line((x1, y1), (x3, y3), height)
                p_top_offset, p_bottom_offset = offset_line_outward(p_top, p_bottom, margin_lr, width / 2)
                left_poly = np.array([[0, 0], p_top_offset, p_bottom_offset, [0, height]], dtype=np.int32)
                cv2.fillPoly(frame, [left_poly], (0, 0, 0))

            if not any(pd.isna([x2, y2, x4, y4])):
                p_top, p_bottom = extend_line((x2, y2), (x4, y4), height)
                p_top_offset, p_bottom_offset = offset_line_outward(p_top, p_bottom, margin_lr, width / 2)
                right_poly = np.array([[width, 0], p_top_offset, p_bottom_offset, [width, height]], dtype=np.int32)
                cv2.fillPoly(frame, [right_poly], (0, 0, 0))

            if not any(pd.isna([x3, y3, x4, y4])):
                p3, p4 = (x3, y3), (x4, y4)
                dx = p4[0] - p3[0]
                dy = p4[1] - p3[1]
                length = np.sqrt(dx ** 2 + dy ** 2)
                if length > 0:
                    ux, uy = dx / length, dy / length
                    nx, ny = -uy, ux
                    p3_shifted = (p3[0] + nx * margin_bot, p3[1] + ny * margin_bot)
                    p4_shifted = (p4[0] + nx * margin_bot, p4[1] + ny * margin_bot)
                    p_left, p_right = extend_line_horizontally(p3_shifted, p4_shifted, width)
                    p_left_down = (p_left[0], p_left[1] + height * 2)
                    p_right_down = (p_right[0], p_right[1] + height * 2)
                    bottom_poly = np.array([p_left, p_right, p_right_down, p_left_down], dtype=np.int32)
                    cv2.fillPoly(frame, [bottom_poly], (0, 0, 0))
            # 遮罩上方区域（KP1-KP2 向外偏移并水平延展）
            if not any(pd.isna([x1, y1, x2, y2])):
                p1, p2 = (x1, y1), (x2, y2)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = np.sqrt(dx ** 2 + dy ** 2)
                if length > 0:
                    # 单位方向与法向量
                    ux, uy = dx / length, dy / length
                    nx, ny = -uy, ux

                    # 向外偏移
                    p1_shift = (p1[0] - nx * margin_top, p1[1] - ny * margin_top)
                    p2_shift = (p2[0] - nx * margin_top, p2[1] - ny * margin_top)

                    # 在该偏移线上进行水平延展（左右到达整个画面边缘）
                    if p2_shift[0] != p1_shift[0]:
                        k = (p2_shift[1] - p1_shift[1]) / (p2_shift[0] - p1_shift[0])
                        b = p1_shift[1] - k * p1_shift[0]
                        left_pt = (0, b)
                        right_pt = (width, k * width + b)
                    else:
                        # 垂直线特判
                        left_pt = (p1_shift[0], 0)
                        right_pt = (p1_shift[0], height)

                    # 向外延长高度
                    extend = height * 2
                    left_pt_far = (left_pt[0] - nx * extend, left_pt[1] - ny * extend)
                    right_pt_far = (right_pt[0] - nx * extend, right_pt[1] - ny * extend)

                    top_poly = np.array([
                        [int(left_pt[0]), int(left_pt[1])],
                        [int(right_pt[0]), int(right_pt[1])],
                        [int(right_pt_far[0]), int(right_pt_far[1])],
                        [int(left_pt_far[0]), int(left_pt_far[1])]
                    ], dtype=np.int32)

                    cv2.fillPoly(frame, [top_poly], (0, 0, 0))


        except Exception as e:
            print(f"Frame error: {e}")

        out.write(frame)

    cap.release()
    out.release()
    print(f"输出完成: {output_path}")


if __name__ == '__main__':
    for file in os.listdir(SEGMENT_DIR):
        if file.endswith(".mp4") and "_with_edges" not in file and "annotated" not in file and "framed" not in file:
            video_path = os.path.join(SEGMENT_DIR, file)
            base = os.path.splitext(file)[0]
            csv_name = f"{base}_keypoints.csv"
            csv_path = os.path.join(SEGMENT_DIR, csv_name)
            if not os.path.exists(csv_path):
                print(f"找不到CSV: {csv_path}")
                continue
            output_path = os.path.join(SEGMENT_DIR, f"{base}_with_edges.mp4")
            cropped_csv_path = os.path.join(SEGMENT_DIR, f"{base}_cropped_keypoints.csv")
            print(f"正在处理: {video_path}")
            draw_edges_on_video(video_path, csv_path, output_path, cropped_csv_path)