import pandas as pd
import numpy as np
import os
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

def process_segment(coord_csv_path):
    base_name = os.path.basename(coord_csv_path).replace("_ball_and_player_coordinates.csv", "")
    segment_prefix = base_name
    player_prefix = base_name.replace("_masked", "")

    print(f"Processing segment: {segment_prefix}")

    player_csvs = [
        os.path.join(PLAYER_CSV_DIR, f"{player_prefix}_player_0.csv"),
        os.path.join(PLAYER_CSV_DIR, f"{player_prefix}_player_1.csv")
    ]
    if not all(os.path.exists(p) for p in player_csvs):
        print(f"Missing player CSVs for {segment_prefix}, skipping.")
        return

    df_coord = pd.read_csv(coord_csv_path)
    df_player = [pd.read_csv(p) for p in player_csvs]

    for i, df in enumerate(df_player):
        df[f"Player{i+1}_Ground_X"] = df[["LSmallToe_x", "LBigToe_x", "RSmallToe_x", "RBigToe_x"]].mean(axis=1)
        df[f"Player{i+1}_Ground_Y"] = df[["LSmallToe_y", "LBigToe_y", "RSmallToe_y", "RBigToe_y"]].mean(axis=1)
        df.rename(columns={
            "RWrist_x": f"Player{i+1}_RWrist_X", "RWrist_y": f"Player{i+1}_RWrist_Y",
            "CHip_x": f"Player{i+1}_Position_X", "CHip_y": f"Player{i+1}_Position_Y"
        }, inplace=True)

    df_merged = df_coord.merge(df_player[0], on="Frame", how="left") \
                        .merge(df_player[1], on="Frame", how="left")

    interp_cols = [
        "Ball_Frame_X", "Ball_Frame_Y",
        "Player1_RWrist_X", "Player1_RWrist_Y", "Player1_Ground_Y",
        "Player2_RWrist_X", "Player2_RWrist_Y", "Player2_Ground_Y"
    ]
    for col in interp_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].interpolate(method='linear', limit_direction='both')

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
        if last == -1 or np.sign(df_merged.at[idx, "Distance_Diff"]) != np.sign(df_merged.at[last, "Distance_Diff"]):
            selected.append(idx)
        elif abs(df_merged.at[idx, "Distance_Diff"]) > abs(df_merged.at[last, "Distance_Diff"]):
            selected[-1] = idx
        last = idx
    df_merged.loc[selected, "Impact_Frame"] = True

    print(f"Impact frames: {df_merged.loc[df_merged['Impact_Frame'], 'Frame'].tolist()}")

    # 绘制距离差图
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
        pt = np.array([[row["Ball_X"], row["Wrist_Y"]]], dtype=np.float32).reshape(1, 1, 2)
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

if __name__ == "__main__":
    for path in glob.glob(os.path.join(SEGMENT_DIR, "*_ball_and_player_coordinates.csv")):
        process_segment(path)
