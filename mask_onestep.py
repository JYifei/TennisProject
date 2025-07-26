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
MARGIN = 80  # buffer for court edge
MARGIN_RATIO_LR = 0.12   # 8% of court width for left/right mask
MARGIN_RATIO_TOP = 0.4   # 40% of court height for top mask
MARGIN_RATIO_BOT = 0.20  # 13% of court height for bottom mask

def smooth_keypoints(df):
    for col in df.columns:
        if col.startswith("KP") and df[col].dtype != 'object':
            series = df[col]
            if series.isna().sum() < len(series) - WINDOW_SIZE:
                smoothed = savgol_filter(series.interpolate(method="linear"), WINDOW_SIZE, POLY_ORDER)
                df[col] = smoothed
            else:
                print(f"Skipping smoothing {col} (insufficient valid points)")
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
    distances = compute_kp34_distances(df, scale_x, scale_y)
    skip_frames = 0

    df = df.iloc[skip_frames:].copy().reset_index(drop=True)
    df["Frame"] = range(len(df))
    df = smooth_keypoints(df)

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

            margin_lr = MARGIN_RATIO_LR * court_width
            margin_top = MARGIN_RATIO_TOP * court_height
            margin_bot = MARGIN_RATIO_BOT * court_height

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

            if not any(pd.isna([x1, y1, x2, y2])):
                p1, p2 = (x1, y1), (x2, y2)
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = np.sqrt(dx ** 2 + dy ** 2)
                if length > 0:
                    ux, uy = dx / length, dy / length
                    nx, ny = -uy, ux
                    p1_shift = (p1[0] - nx * margin_top, p1[1] - ny * margin_top)
                    p2_shift = (p2[0] - nx * margin_top, p2[1] - ny * margin_top)

                    if p2_shift[0] != p1_shift[0]:
                        k = (p2_shift[1] - p1_shift[1]) / (p2_shift[0] - p1_shift[0])
                        b = p1_shift[1] - k * p1_shift[0]
                        left_pt = (0, b)
                        right_pt = (width, k * width + b)
                    else:
                        left_pt = (p1_shift[0], 0)
                        right_pt = (p1_shift[0], height)

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
    print(f"Output complete: {output_path}")

if __name__ == '__main__':
    os.makedirs("masked_segments", exist_ok=True)

    for file in os.listdir(SEGMENT_DIR):
        if file.endswith(".mp4") and "_with_edges" not in file and "annotated" not in file and "framed" not in file:
            video_path = os.path.join(SEGMENT_DIR, file)
            base = os.path.splitext(file)[0]
            csv_name = f"{base}_keypoints.csv"

            csv_path = os.path.join("keypoints_data", csv_name)

            if not os.path.exists(csv_path):
                print(f"CSV not found: {csv_path}")
                continue

            output_path = os.path.join("masked_segments", f"{base}_masked.mp4")
            cropped_csv_path = os.path.join("masked_segments", f"{base}_cropped_keypoints.csv")

            print(f"Processing: {video_path}")
            draw_edges_on_video(video_path, csv_path, output_path, cropped_csv_path)
