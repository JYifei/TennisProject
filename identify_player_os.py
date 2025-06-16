import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import cv2

SEGMENT_DIR = "output_segments"
KEYPOINT_DIR = "keypoints_data"
TRACKED_DIR = "postprocessed_sports2D"
OUTPUT_DIR = "selected_players"
FRAME_COVERAGE_THRESHOLD = 0.5  # 比例阈值（50%）

def get_baselines_per_frame(keypoint_csv):
    df = pd.read_csv(keypoint_csv)
    y1_series, y2_series = [], []
    for idx in range(len(df)):
        baseline1_y = []
        baseline2_y = []
        for kp in ["KP1", "KP5", "KP7", "KP2"]:
            col = f"{kp}_Y"
            if col in df.columns and not pd.isna(df.loc[idx, col]):
                baseline1_y.append(df.loc[idx, col])
        for kp in ["KP3", "KP6", "KP8", "KP4"]:
            col = f"{kp}_Y"
            if col in df.columns and not pd.isna(df.loc[idx, col]):
                baseline2_y.append(df.loc[idx, col])
        y1_series.append(np.mean(baseline1_y) if baseline1_y else np.nan)
        y2_series.append(np.mean(baseline2_y) if baseline2_y else np.nan)
    return np.array(y1_series), np.array(y2_series)

def get_person_center_series(csv_path):
    df = pd.read_csv(csv_path)
    if "CHip_y" not in df.columns:
        return np.full(len(df), np.nan)
    return df["CHip_y"].values

def compute_total_movement(csv_path):
    df = pd.read_csv(csv_path)
    points = df[["CHip_x", "CHip_y"]].dropna().values
    if len(points) < 2:
        return 0
    return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

def identify_players():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for video_file in os.listdir(SEGMENT_DIR):
        if not video_file.endswith(".mp4"):
            continue

        base_name = Path(video_file).stem
        keypoint_csv = os.path.join(KEYPOINT_DIR, f"{base_name}_keypoints.csv")
        if not os.path.exists(keypoint_csv):
            print(f"[SKIP] Keypoint CSV not found for {base_name}")
            continue

        tracked_dir = os.path.join(TRACKED_DIR, f"{base_name}_masked")
        if not os.path.exists(tracked_dir):
            print(f"[SKIP] Tracked folder not found: {tracked_dir}")
            continue

        y1_series, y2_series = get_baselines_per_frame(keypoint_csv)
        video_len = len(y1_series)

        center_distances = {}
        movement_amounts = {}

        for file in os.listdir(tracked_dir):
            if not file.endswith(".csv"):
                continue
            path = os.path.join(tracked_dir, file)
            ch_y = get_person_center_series(path)
            n_valid = np.count_nonzero(~np.isnan(ch_y))
            coverage_ratio = n_valid / video_len

            if coverage_ratio < FRAME_COVERAGE_THRESHOLD:
                print(f"[DROP] {file} coverage {coverage_ratio:.2%} < {FRAME_COVERAGE_THRESHOLD:.0%}")
                continue

            n = min(len(ch_y), len(y1_series))
            dy_top = np.nansum(np.abs(ch_y[:n] - y1_series[:n]))
            dy_bottom = np.nansum(np.abs(ch_y[:n] - y2_series[:n]))

            center_distances[file] = (dy_top, dy_bottom)
            movement_amounts[file] = compute_total_movement(path)

        if not center_distances:
            print(f"[WARN] No valid persons found in {base_name}")
            continue

        top_file = min(center_distances.items(), key=lambda x: x[1][0])[0]
        bottom_file = min(center_distances.items(), key=lambda x: x[1][1])[0]
        selected_by_position = {top_file, bottom_file}

        sorted_movement = sorted(movement_amounts.items(), key=lambda x: -x[1])
        selected_by_motion = {sorted_movement[0][0], sorted_movement[1][0]} if len(sorted_movement) >= 2 else set()

        print(f"{base_name}:")
        print(f"  ➤ Position-based: {selected_by_position}")
        print(f"  ➤ Motion-based:   {selected_by_motion}")

        if selected_by_position != selected_by_motion:
            print(f"  ⚠️  Mismatch in player selection methods. Consider reviewing manually.")

        for filename in selected_by_position:
            src = os.path.join(tracked_dir, filename)
            dst = os.path.join(OUTPUT_DIR, f"{base_name}_{filename}")
            shutil.copyfile(src, dst)

if __name__ == '__main__':
    identify_players()
