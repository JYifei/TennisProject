import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

SEGMENT_DIR = "output_segments"
KEYPOINT_DIR = "keypoints_data"
TRACKED_DIR = "postprocessed_sports2D"
OUTPUT_DIR = "selected_players"

def get_baselines(keypoint_csv):
    df = pd.read_csv(keypoint_csv)
    baseline1_y = []
    baseline2_y = []
    for kp in ["KP1", "KP5", "KP7", "KP2"]:
        if not pd.isna(df.loc[0, f"{kp}_Y"]):
            baseline1_y.append(df.loc[0, f"{kp}_Y"])
    for kp in ["KP3", "KP6", "KP8", "KP4"]:
        if not pd.isna(df.loc[0, f"{kp}_Y"]):
            baseline2_y.append(df.loc[0, f"{kp}_Y"])
    y1 = np.mean(baseline1_y) if baseline1_y else np.inf
    y2 = np.mean(baseline2_y) if baseline2_y else np.inf
    return y1, y2

def get_person_center(csv_path):
    df = pd.read_csv(csv_path)
    sub_df = df.iloc[:90]  # First 3 seconds (assuming 30 FPS)
    center_points = sub_df[["CHip_x", "CHip_y"]].dropna().values
    if len(center_points) == 0:
        return np.array([np.nan, np.nan])
    return np.mean(center_points, axis=0)

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

        y1, y2 = get_baselines(keypoint_csv)

        # Record center distances and movement for each player
        center_distances = {}
        movement_amounts = {}

        for file in os.listdir(tracked_dir):
            if not file.endswith(".csv"):
                continue
            path = os.path.join(tracked_dir, file)
            center = get_person_center(path)
            move = compute_total_movement(path)

            if not np.isnan(center).any():
                dy_top = abs(center[1] - y1)
                dy_bottom = abs(center[1] - y2)
                center_distances[file] = (dy_top, dy_bottom)
                movement_amounts[file] = move

        if not center_distances:
            print(f"[WARN] No valid persons found in {base_name}")
            continue

        # Select by spatial proximity to top/bottom baselines
        top_file = min(center_distances.items(), key=lambda x: x[1][0])[0]
        bottom_file = min(center_distances.items(), key=lambda x: x[1][1])[0]
        selected_by_position = {top_file, bottom_file}

        # Select by total movement (top 2)
        sorted_movement = sorted(movement_amounts.items(), key=lambda x: -x[1])
        selected_by_motion = {sorted_movement[0][0], sorted_movement[1][0]} if len(sorted_movement) >= 2 else set()

        print(f"{base_name}:")
        print(f"  ➤ Position-based: {selected_by_position}")
        print(f"  ➤ Motion-based:   {selected_by_motion}")

        if selected_by_position != selected_by_motion:
            print(f"  ⚠️  Mismatch in player selection methods. Consider reviewing manually.")

        # Output only the players selected by position-based method
        for filename in selected_by_position:
            src = os.path.join(tracked_dir, filename)
            dst = os.path.join(OUTPUT_DIR, f"{base_name}_{filename}")
            shutil.copyfile(src, dst)

if __name__ == '__main__':
    identify_players()
