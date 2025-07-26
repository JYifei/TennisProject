import os
import pandas as pd
import numpy as np
from glob import glob
from collections import deque, Counter
import cv2

# 常量
FOOT_PARTS = ['LAnkle', 'RAnkle']
ALL_KEYPOINTS = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder',
                 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip',
                 'LKnee', 'RKnee', 'LAnkle', 'RAnkle', 'Chest']
MAX_WINDOW = 10
STABILITY_WINDOW = 2  # 秒
STABILITY_THRESHOLD = 20  # px

def interpolate_keypoints(df, window=10):
    for col in df.columns:
        if '_x' in col or '_y' in col:
            values = df[col].values
            for i in range(len(values)):
                if values[i] == 0:
                    valid = []
                    for j in range(1, window + 1):
                        if i - j >= 0 and values[i - j] > 0:
                            valid.append(values[i - j])
                        if i + j < len(values) and values[i + j] > 0:
                            valid.append(values[i + j])
                    if valid:
                        values[i] = np.mean(valid)
            df[col] = values
    return df

def compute_center(row):
    coords = []
    for joint in FOOT_PARTS:
        x, y = row.get(f'{joint}_x', 0), row.get(f'{joint}_y', 0)
        if x > 0 and y > 0:
            coords.append([x, y])
    if not coords:
        return np.array([np.nan, np.nan])
    return np.nanmean(coords, axis=0)

def compute_keypoint_distance(row1, row2):
    dist = []
    for joint in ALL_KEYPOINTS:
        x1, y1 = row1.get(f'{joint}_x', 0), row1.get(f'{joint}_y', 0)
        x2, y2 = row2.get(f'{joint}_x', 0), row2.get(f'{joint}_y', 0)
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            dist.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    return np.mean(dist) if dist else np.inf

def get_initial_players_from_court_lines(segment_dir, court_kp_path, fps):
    FRAME_START = 0
    FRAME_WINDOW = 90
    df_court = pd.read_csv(court_kp_path)

    person_files = sorted(glob(os.path.join(segment_dir, '*_person*.csv')))
    all_data = {}
    for file in person_files:
        pid = os.path.basename(file).split('_person')[-1].split('.')[0]
        df = interpolate_keypoints(pd.read_csv(file), window=MAX_WINDOW)
        all_data[pid] = df

    top_counts, bottom_counts = {}, {}
    missing_kp = 0
    missing_foot = 0
    conflict_ids = 0
    total_frames = 0

    for frame in range(FRAME_START, FRAME_START + FRAME_WINDOW):
        total_frames += 1
        if frame >= len(df_court):
            continue
        try:
            top_y = np.nanmean([df_court.loc[frame, 'KP1_y'], df_court.loc[frame, 'KP2_y']])
            bottom_y = np.nanmean([df_court.loc[frame, 'KP3_y'], df_court.loc[frame, 'KP4_y']])
        except KeyError:
            missing_kp += 1
            continue
        if np.isnan(top_y) or np.isnan(bottom_y):
            missing_kp += 1
            continue

        candidates = []
        for pid, df in all_data.items():
            if frame >= len(df): continue
            cy = compute_center(df.iloc[frame])[1]
            if not np.isnan(cy):
                d_top = abs(cy - top_y)
                d_bottom = abs(cy - bottom_y)
                candidates.append((pid, d_top, d_bottom))

        if len(candidates) < 2:
            missing_foot += 1
            continue

        top_pid = min(candidates, key=lambda x: x[1])[0]
        bottom_pid = min(candidates, key=lambda x: x[2])[0]

        if top_pid == bottom_pid:
            conflict_ids += 1
            continue

        top_counts[top_pid] = top_counts.get(top_pid, 0) + 1
        bottom_counts[bottom_pid] = bottom_counts.get(bottom_pid, 0) + 1

    print(f"  [DEBUG] Total frames={total_frames}, missing_kp={missing_kp}, missing_foot={missing_foot}, conflict_ids={conflict_ids}")

    if top_counts and bottom_counts:
        top_final = max(top_counts.items(), key=lambda x: x[1])[0]
        bottom_final = max(bottom_counts.items(), key=lambda x: x[1])[0]
        if top_final != bottom_final:
            return [bottom_final, top_final], FRAME_START

    print("  [FALLBACK] Using Y-max-based foot center distance")
    centers = {}
    for pid, df in all_data.items():
        min_y = np.inf
        best_center = None
        for idx in range(FRAME_START, min(FRAME_START + FRAME_WINDOW, len(df))):
            cy = compute_center(df.iloc[idx])[1]
            if not np.isnan(cy) and cy < min_y:
                min_y = cy
                best_center = cy
        if best_center is not None:
            centers[pid] = best_center

    if len(centers) < 2:
        return [], 0

    sorted_by_y = sorted(centers.items(), key=lambda x: x[1], reverse=True)
    return [sorted_by_y[0][0], sorted_by_y[1][0]], FRAME_START


def track_players(segment_dir, player_ids, output_dir, output_video_path, start_frame, fps):
    person_files = sorted(glob(os.path.join(segment_dir, '*_person*.csv')))
    all_data = {}
    for file in person_files:
        pid = os.path.basename(file).split('_person')[-1].split('.')[0]
        df = interpolate_keypoints(pd.read_csv(file), window=MAX_WINDOW)
        all_data[pid] = df

    history = {0: deque(maxlen=3), 1: deque(maxlen=3)}
    tracked = {0: [], 1: []}
    fail_log = []

    video_file = next((f for f in os.listdir(segment_dir) if f.endswith('.mp4')), None)
    if not video_file:
        print(f"[ERROR] 视频文件不存在: {segment_dir}")
        return

    video_path = os.path.join(segment_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    if width == 0 or height == 0 or fps == 0:
        print(f"[ERROR] 视频属性读取失败: width={width}, height={height}, fps={fps}")
        return

    os.makedirs(output_dir, exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        print(f"[ERROR] VideoWriter 初始化失败: {output_video_path}")
        return

    # 向后查找直到两名玩家都出现为止
    new_start_frame = start_frame
    for i, pid in enumerate(player_ids):
        if pid not in all_data:
            tracked[i].append(None)
            continue

        df = all_data[pid]
        found = False
        for offset in range(start_frame, len(df)):
            cy = compute_center(df.iloc[offset])[1]
            if not np.isnan(cy):
                tracked[i].append(df.iloc[offset])
                history[i].append(df.iloc[offset])
                print(f"[INFO] 玩家 {pid} 初次出现于帧 {offset}")
                new_start_frame = max(new_start_frame, offset)  # 更新新的最早可追踪帧
                found = True
                break

        if not found:
            tracked[i].append(None)
            print(f"[WARN] 玩家 {pid} 在视频中找不到有效起始帧")

    start_frame = new_start_frame
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 从玩家可见的帧开始处理

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        candidates = {}
        for pid, df in all_data.items():
            if frame_idx + start_frame < len(df):
                candidates[pid] = df.iloc[frame_idx + start_frame]

        matched_ids = {}
        for i in [0, 1]:
            avg_template = pd.DataFrame(history[i]).mean()
            valid_candidates = []
            for pid, row in candidates.items():
                if pid in matched_ids.values():
                    continue
                dist = compute_keypoint_distance(avg_template, row)
                if dist < 200:
                    foot = compute_center(row)
                    if not np.isnan(foot[1]):
                        valid_candidates.append((pid, dist, foot[1], row))

            if valid_candidates:
                if i == 0:
                    best = min(valid_candidates, key=lambda x: x[2])  # 更靠上
                else:
                    best = max(valid_candidates, key=lambda x: x[2])  # 更靠下
                best_pid = best[0]
                matched_ids[i] = best_pid
                history[i].append(best[3])
                tracked[i].append(best[3])
                cx, cy = compute_center(best[3])
                if not np.isnan(cx) and not np.isnan(cy):
                    color = (0, 0, 255) if i == 0 else (255, 0, 0)
                    cv2.rectangle(frame, (int(cx) - 30, int(cy) - 30), (int(cx) + 30, int(cy) + 30), color, 2)
            else:
                tracked[i].append(None)
                fail_log.append((frame_idx + start_frame, i))

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    for i in [0, 1]:
        df_list = [pd.DataFrame([row]) for row in tracked[i] if row is not None]
        if df_list:
            pd.concat(df_list).to_csv(os.path.join(output_dir, f'player_{i}.csv'), index=False)
    with open(os.path.join(output_dir, 'match_failed_frames.txt'), 'w') as f:
        for fidx, pid in fail_log:
            f.write(f"Frame {fidx}: player {pid} match failed\n")

    if os.path.exists(output_video_path):
        print(f"[OK] 视频已保存: {output_video_path}")
    else:
        print(f"[WARN] 视频保存失败: {output_video_path}")


def process_all_segments(sports2d_root, court_kp_root, track_output_root):
    os.makedirs(track_output_root, exist_ok=True)
    segments = [d for d in os.listdir(sports2d_root) if os.path.isdir(os.path.join(sports2d_root, d))]
    for seg in sorted(segments):
        seg_path = os.path.join(sports2d_root, seg)
        print(f"Processing {seg} ...")

        video_file = next((f for f in os.listdir(seg_path) if f.endswith('.mp4')), None)
        if not video_file:
            print(" → No video found.")
            continue
        cap = cv2.VideoCapture(os.path.join(seg_path, video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        base_seg_name = seg.replace('_masked', '')
        court_kp_path = os.path.join(court_kp_root, f"{base_seg_name}_keypoints.csv")

        if not os.path.exists(court_kp_path):
            print(" → Court keypoints not found.")
            continue

        players, start_frame = get_initial_players_from_court_lines(seg_path, court_kp_path, fps)
        if len(players) < 2:
            print(" → Failed to detect two players")
            continue

        seg_output = os.path.join(track_output_root, seg)
        video_out_path = os.path.join(seg_output, 'player_tracking.mp4')
        track_players(seg_path, players, seg_output, video_out_path, start_frame, fps)
        print(f" → Finished {seg}\n")

if __name__ == "__main__":
    process_all_segments(
        sports2d_root='sports2d_results',
        court_kp_root='keypoints_data',
        track_output_root='selected_players'
    )
