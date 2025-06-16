
import os
import pandas as pd
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from utils.main import get_court_img

SEGMENT_DIR = "detected_segments"
PLAYER_CSV_DIR = "selected_players"
MATRIX_DIR = "matrix_data"
FINAL_OUTPUT_DIR = "final_result"
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

def extract_player_number(filename):
    basename = os.path.basename(filename)
    try:
        return int(basename.split("_player_")[-1].split(".")[0])
    except:
        return -1

def process_segment(coord_csv_path):
    base_name = os.path.basename(coord_csv_path).replace("_ball_and_player_coordinates.csv", "")
    segment_prefix = base_name
    player_prefix = base_name.replace("_masked", "")

    print(f"Processing segment: {segment_prefix}")

    video_path = os.path.join("output_segments", f"{player_prefix}.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video {video_path}, skipping.")
        return
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    df_coord = pd.read_csv(coord_csv_path)

    raw_ball_csv = os.path.join(FINAL_OUTPUT_DIR, f"{segment_prefix}_ball_trajectory.csv")
    df_coord[["Frame", "Ball_Frame_X", "Ball_Frame_Y"]].to_csv(raw_ball_csv, index=False)

    matched_players = sorted(
        glob.glob(os.path.join(PLAYER_CSV_DIR, f"{player_prefix}_player_*.csv")),
        key=extract_player_number
    )
    if len(matched_players) < 2:
        print(f"Missing player CSVs for {segment_prefix}, skipping.")
        return

    df_player = [pd.read_csv(p) for p in matched_players[:2]]
    for i, df in enumerate(df_player):
        pid = i + 1
        df[f"Player{pid}_Ground_X"] = df[["LSmallToe_x", "LBigToe_x", "RSmallToe_x", "RBigToe_x"]].mean(axis=1)
        df[f"Player{pid}_Ground_Y"] = df[["LSmallToe_y", "LBigToe_y", "RSmallToe_y", "RBigToe_y"]].mean(axis=1)
        df.rename(columns={
            "RWrist_x": f"Player{pid}_RWrist_X", "RWrist_y": f"Player{pid}_RWrist_Y",
            "CHip_x": f"Player{pid}_Position_X", "CHip_y": f"Player{pid}_Position_Y"
        }, inplace=True)

    df_merged = df_coord.merge(df_player[0], on="Frame", how="left").merge(df_player[1], on="Frame", how="left")

    ball_valid = df_merged["Ball_Frame_X"].notna() & df_merged["Ball_Frame_Y"].notna()
    max_frame_gap = 5
    min_initial_segment_len = 20
    merge_gap_thresh = 10
    min_final_segment_len = 60

    segments = []
    current = []
    gap_count = 0
    for idx, v in zip(df_merged.index, ball_valid):
        if v:
            if gap_count > 0:
                current.extend(range(idx - gap_count, idx))
                gap_count = 0
            current.append(idx)
        else:
            if not current:
                continue
            gap_count += 1
            if gap_count > max_frame_gap:
                if len(current) >= min_initial_segment_len:
                    segments.append(current)
                current = []
                gap_count = 0
    if len(current) >= min_initial_segment_len:
        segments.append(current)

    merged_segments = []
    if segments:
        current = segments[0]
        for next_seg in segments[1:]:
            if next_seg[0] - current[-1] <= merge_gap_thresh:
                current += list(range(current[-1]+1, next_seg[0])) + next_seg
            else:
                merged_segments.append(current)
                current = next_seg
        merged_segments.append(current)

    final_segments = [seg for seg in merged_segments if len(seg) >= min_final_segment_len]

    trust_mask = pd.Series(False, index=df_merged.index)
    for seg in final_segments:
        trust_mask.iloc[seg] = True

    df_merged.loc[~trust_mask, ["Ball_Frame_X", "Ball_Frame_Y"]] = np.nan

    interp_cols = [
        "Ball_Frame_X", "Ball_Frame_Y",
        "Player1_RWrist_X", "Player1_RWrist_Y", "Player1_Ground_Y",
        "Player2_RWrist_X", "Player2_RWrist_Y", "Player2_Ground_Y"
    ]
    for col in interp_cols:
        if col in df_merged.columns:
            series = df_merged[col].copy()
            series.loc[~trust_mask] = np.nan
            series = series.interpolate(method='linear', limit_direction='both')
            df_merged[col] = series

    smooth_window = 5
    for col in ["Ball_Frame_X", "Ball_Frame_Y"]:
        if col in df_merged.columns:
            smoothed = df_merged[col].copy()
            smoothed.loc[trust_mask] = smoothed.loc[trust_mask].rolling(window=smooth_window, center=True, min_periods=1).mean()
            df_merged[col] = smoothed

    interpolated_csv = os.path.join(FINAL_OUTPUT_DIR, f"{segment_prefix}_ball_interpolated.csv")
    df_merged[["Frame", "Ball_Frame_X", "Ball_Frame_Y"]].to_csv(interpolated_csv, index=False)
    print(f"Saved interpolated ball trajectory: {interpolated_csv}")

    for i in [1, 2]:
        df_merged[f"Player{i}_Ball_Dist"] = np.sqrt(
            (df_merged["Ball_Frame_X"] - df_merged[f"Player{i}_RWrist_X"]) ** 2 +
            (df_merged["Ball_Frame_Y"] - df_merged[f"Player{i}_RWrist_Y"]) ** 2
        )
    df_merged["Distance_Diff"] = df_merged["Player1_Ball_Dist"] - df_merged["Player2_Ball_Dist"]
    df_merged["Impact_Frame"] = False

    extrema_max = argrelextrema(df_merged["Distance_Diff"].values, np.greater, order=5)[0]
    extrema_min = argrelextrema(df_merged["Distance_Diff"].values, np.less, order=5)[0]
    candidates = sorted(np.concatenate([extrema_max, extrema_min]))

    selected = []
    last = -1
    for idx in candidates:
        if np.abs(df_merged.at[idx, "Distance_Diff"]) < 200:
            continue
        if last == -1 or np.sign(df_merged.at[idx, "Distance_Diff"]) != np.sign(df_merged.at[last, "Distance_Diff"]):
            selected.append(idx)
        elif abs(df_merged.at[idx, "Distance_Diff"]) > abs(df_merged.at[last, "Distance_Diff"]):
            selected[-1] = idx
        last = idx
    df_merged.loc[selected, "Impact_Frame"] = True

    print(f"Impact frames: {df_merged.loc[df_merged['Impact_Frame'], 'Frame'].tolist()}")

    impact_rows = []
    for _, row in df_merged[df_merged["Impact_Frame"]].iterrows():
        player = 1 if row["Player1_Ball_Dist"] < row["Player2_Ball_Dist"] else 2
        impact_rows.append([
            int(row["Frame"]),
            row["Ball_Frame_X"],
            row[f"Player{player}_Ground_Y"],
            player
        ])
    df_impact = pd.DataFrame(impact_rows, columns=["Frame", "Ball_X", "Wrist_Y", "Player"])

    matrix_path = os.path.join(MATRIX_DIR, f"{player_prefix}_matrixes.npy")
    matrices = np.load(matrix_path, allow_pickle=True)
    court_img = get_court_img()
    court_img = cv2.cvtColor(court_img, cv2.COLOR_BGR2GRAY) if court_img.ndim == 3 else court_img
    visual = np.zeros((*court_img.shape, 3), dtype=np.uint8)
    visual[:] = (60, 160, 60)
    visual[court_img == 255] = (255, 255, 255)

    trajectory = []
    for _, row in df_impact.iterrows():
        frame_idx = int(row["Frame"])
        if frame_idx >= len(matrices): continue
        sx = row["Ball_X"] / original_width * 1280
        sy = row["Wrist_Y"] / original_height * 720
        pt = np.array([[sx, sy]], dtype=np.float32).reshape(1, 1, 2)
        transformed = cv2.perspectiveTransform(pt, matrices[frame_idx])[0][0]
        x, y = int(transformed[0]), int(transformed[1])
        trajectory.append((x, y))
        color = (0, 0, 255) if row["Player"] == 1 else (255, 0, 0)
        visual = cv2.circle(visual, (x, y), 20, color, -1)
        cv2.putText(visual, str(len(trajectory)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    if len(trajectory) > 1:
        cv2.polylines(visual, [np.array(trajectory)], False, (0, 0, 0), 5)

    out_csv = os.path.join(FINAL_OUTPUT_DIR, f"{segment_prefix}_impact_coordinates.csv")
    out_img = os.path.join(FINAL_OUTPUT_DIR, f"{segment_prefix}_court_reference.png")
    df_impact.to_csv(out_csv, index=False)
    cv2.imwrite(out_img, visual)
    print(f"Saved: {out_csv}, {out_img}, and distance plot")

    plt.figure(figsize=(10, 4))
    plt.plot(df_merged["Frame"], df_merged["Distance_Diff"], label="Distance Diff")
    plt.scatter(df_merged.loc[df_merged["Impact_Frame"], "Frame"],
                df_merged.loc[df_merged["Impact_Frame"], "Distance_Diff"],
                color='red', label='Impact Frames')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Frame")
    plt.ylabel("Distance Diff")
    plt.title(f"Distance Difference - {segment_prefix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"{segment_prefix}_distance_plot.png"))
    plt.close()

if __name__ == "__main__":
    for path in glob.glob(os.path.join(SEGMENT_DIR, "*_ball_and_player_coordinates.csv")):
        process_segment(path)
