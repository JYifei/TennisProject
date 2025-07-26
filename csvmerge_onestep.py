import os
import pandas as pd
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from utils.main import get_court_img
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d

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

def smooth_outliers(series, threshold=200, window=10):
    values = series.values.copy()
    for i in range(1, len(values)-1):
        if not np.isnan(values[i]):
            local_window = series[max(i-window, 0):min(i+window+1, len(series))].dropna()
            if len(local_window) >= 3:
                median = np.median(local_window)
                if abs(values[i] - median) > threshold:
                    values[i] = median
    return pd.Series(values, index=series.index)

def process_segment(coord_csv_path):
    base_name = os.path.basename(coord_csv_path).replace("_ball_and_player_coordinates.csv", "")
    segment_prefix = base_name
    player_prefix = base_name.replace("_masked", "")

    print(f"Processing segment: {segment_prefix}")

    video_path = os.path.join("masked_segments", f"{player_prefix}_masked.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video {video_path}, skipping.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    max_gap = int(fps * 1)

    df_coord = pd.read_csv(coord_csv_path)
    x = df_coord["Ball_Frame_X"].copy()
    y = df_coord["Ball_Frame_Y"].copy()

    trust_mask = pd.Series(False, index=df_coord.index)
    last_valid = df_coord[df_coord["Ball_Frame_X"].notna() & df_coord["Ball_Frame_Y"].notna()].index.max()
    if pd.isna(last_valid):
        print("[ERROR] 找不到任何有效的球坐标。")
        return
    i = last_valid
    while i >= 0:
        if pd.notna(x.iloc[i]) and pd.notna(y.iloc[i]):
            trust_mask.iloc[i] = True
            i -= 1
        else:
            gap_end = i
            gap_start = i
            while gap_start >= 0 and (pd.isna(x.iloc[gap_start]) or pd.isna(y.iloc[gap_start])):
                gap_start -= 1
            gap_len = gap_end - gap_start
            if gap_len > max_gap or gap_start < 0:
                print(f"[停止] gap 超过 {max_gap} 帧，无数据可接，插值终止于 {gap_end}")
                break
            else:
                trust_mask.iloc[gap_start + 1:gap_end + 1] = True
                i = gap_start

    for col in ["Ball_Frame_X", "Ball_Frame_Y"]:
        series = df_coord[col].copy()
        series[~trust_mask] = np.nan
        df_coord[col] = series.interpolate(method="linear", limit_direction="forward")

    df_coord["Ball_Frame_Y"] = smooth_outliers(df_coord["Ball_Frame_Y"], window=int(fps/5))

    raw_ball_csv = os.path.join(FINAL_OUTPUT_DIR, f"{segment_prefix}_ball_trajectory.csv")
    df_coord[["Frame", "Ball_Frame_X", "Ball_Frame_Y"]].to_csv(raw_ball_csv, index=False)

    segment_folder = os.path.join(PLAYER_CSV_DIR, segment_prefix)
    matched_players = sorted(glob.glob(os.path.join(segment_folder, "player_*.csv")), key=extract_player_number)
    if len(matched_players) < 2:
        print(f"Missing player CSVs for {segment_prefix}, skipping.")
        return

    df_player = [pd.read_csv(p) for p in matched_players[:2]]
    for i, df in enumerate(df_player):
        pid = i + 1
        df[f"Player{pid}_Ground_X"] = df[["LSmallToe_x", "LBigToe_x", "RSmallToe_x", "RBigToe_x"]].mean(axis=1)
        df[f"Player{pid}_Ground_Y"] = df[["LSmallToe_y", "LBigToe_y", "RSmallToe_y", "RBigToe_y"]].mean(axis=1)
        df[f"Player{pid}_LHeel_y"] = df["LHeel_y"]
        df[f"Player{pid}_RHeel_y"] = df["RHeel_y"]
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

    segments, current, gap_count = [], [], 0
    for idx, v in zip(df_merged.index, ball_valid):
        if v:
            if gap_count > 0:
                current.extend(range(idx - gap_count, idx))
                gap_count = 0
            current.append(idx)
        else:
            if not current: continue
            gap_count += 1
            if gap_count > max_frame_gap:
                if len(current) >= min_initial_segment_len:
                    segments.append(current)
                current, gap_count = [], 0
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

    selected, last = [], -1
    for idx in candidates:
        if np.abs(df_merged.at[idx, "Distance_Diff"]) < 500:
            continue
        if last == -1 or np.sign(df_merged.at[idx, "Distance_Diff"]) != np.sign(df_merged.at[last, "Distance_Diff"]):
            selected.append(idx)
        elif abs(df_merged.at[idx, "Distance_Diff"]) > abs(df_merged.at[last, "Distance_Diff"]):
            selected[-1] = idx
        last = idx
    df_merged.loc[selected, "Impact_Frame"] = True
    
    # 原始击球帧索引
    selected = sorted(df_merged.index[df_merged["Impact_Frame"]].tolist())
    to_remove = set()
    i = 0
    while i + 2 < len(selected):
        a, b, c = selected[i], selected[i+1], selected[i+2]
        if c - a <= fps * 1.5:
            sign_ab = np.sign(df_merged.at[a, "Distance_Diff"]) != np.sign(df_merged.at[b, "Distance_Diff"])
            sign_bc = np.sign(df_merged.at[b, "Distance_Diff"]) != np.sign(df_merged.at[c, "Distance_Diff"])
            if sign_ab or sign_bc:
                # 删除 b
                to_remove.add(b)
                # a vs c 之间距离差更小的也删掉
                diff_a = abs(df_merged.at[a, "Distance_Diff"])
                diff_c = abs(df_merged.at[c, "Distance_Diff"])
                to_remove.add(a if diff_a < diff_c else c)
                i += 3  # 跳过下一组，防止重叠处理
            else:
                i += 1
        else:
            i += 1

    # 清除不合格击球点
    df_merged.loc[list(to_remove), "Impact_Frame"] = False
    print(f"Removed {len(to_remove)} suspicious impact frames (cluster removal)")


    print(f"Impact frames: {df_merged.loc[df_merged['Impact_Frame'], 'Frame'].tolist()}")

    PIXELS_PER_M_X = (1379 - 286) / 10.97
    PIXELS_PER_M_Y = (2935 - 561) / 23.77
    NET_CENTER_X, NET_CENTER_Y = 832.5, 1748

    def pixel_to_world(x, y):
        dx = x - NET_CENTER_X
        dy = NET_CENTER_Y - y
        return round(dx / PIXELS_PER_M_X, 3), round(dy / PIXELS_PER_M_Y, 3)

    matrix_path = os.path.join(MATRIX_DIR, f"{player_prefix}_matrixes.npy")
    matrices = np.load(matrix_path, allow_pickle=True)
    court_img = get_court_img()
    court_img = cv2.cvtColor(court_img, cv2.COLOR_BGR2GRAY) if court_img.ndim == 3 else court_img
    visual = np.zeros((*court_img.shape, 3), dtype=np.uint8)
    visual[:] = (60, 160, 60)
    visual[court_img == 255] = (255, 255, 255)

    trajectory = []
    impact_rows = []

    for _, row in df_merged[df_merged["Impact_Frame"]].iterrows():
        player = 1 if row["Player1_Ball_Dist"] < row["Player2_Ball_Dist"] else 2
        heel_avg = np.nanmean([row[f"Player{player}_LHeel_y"], row[f"Player{player}_RHeel_y"]])
        frame_idx = int(row["Frame"])
        if frame_idx >= len(matrices): continue

        sx = row["Ball_Frame_X"] / original_width * 1280
        sy = heel_avg / original_height * 720
        pt = np.array([[sx, sy]], dtype=np.float32).reshape(1, 1, 2)
        transform_mat = np.array(matrices[frame_idx], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, transform_mat)[0][0]

        x, y = int(transformed[0]), int(transformed[1])
        mx, my = pixel_to_world(x, y)

        impact_rows.append([
            int(row["Frame"]),
            row["Ball_Frame_X"],
            heel_avg,
            player,
            mx,
            my
        ])

        trajectory.append((x, y))
        color = (0, 0, 255) if player == 1 else (255, 0, 0)
        visual = cv2.circle(visual, (x, y), 20, color, -1)
        label = f"{len(trajectory)} ({mx}m,{my}m)"
        cv2.putText(visual, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        print(f"Impact {len(trajectory)}: Player {player} → Ball_X: {row['Ball_Frame_X']:.1f}, Heel_Y: {heel_avg:.1f} → Court XY: ({x}, {y}), Real XY: ({mx}m, {my}m)")

    df_impact = pd.DataFrame(impact_rows, columns=["Frame", "Ball_X", "Heel_Y", "Player", "Court_X_m", "Court_Y_m"])

    if len(trajectory) > 1:
        cv2.polylines(visual, [np.array(trajectory)], False, (0, 0, 0), 5)

    out_csv = os.path.join(FINAL_OUTPUT_DIR, f"{segment_prefix}_impact_coordinates.csv")
    out_img = os.path.join(FINAL_OUTPUT_DIR, f"{segment_prefix}_court_reference.png")
    df_impact.to_csv(out_csv, index=False)
    cv2.imwrite(out_img, visual)
    print(f"Saved: {out_csv}, {out_img}, and distance plot")

    
    mpl.rcParams['agg.path.chunksize'] = 10000
    plt.style.use('seaborn-v0_8-whitegrid')  # 美观风格

    # 平滑处理
    smoothed_diff = gaussian_filter1d(df_merged["Distance_Diff"].fillna(method='ffill').values, sigma=2)
    smoothed_p1 = gaussian_filter1d(df_merged["Player1_Ball_Dist"].fillna(method='ffill').values, sigma=2)
    smoothed_p2 = gaussian_filter1d(df_merged["Player2_Ball_Dist"].fillna(method='ffill').values, sigma=2)

    # 图1：Distance Difference
    plt.figure(figsize=(12, 5), dpi=150)
    plt.plot(df_merged["Frame"], smoothed_diff, label="Smoothed Distance Diff", color='orange', linewidth=2)
    plt.scatter(df_merged.loc[df_merged["Impact_Frame"], "Frame"],
                df_merged.loc[df_merged["Impact_Frame"], "Distance_Diff"],
                color='red', label='Impact Frames', zorder=5, s=30)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Distance Diff", fontsize=12)
    plt.title(f"Distance Difference - {segment_prefix}", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"{segment_prefix}_distance_plot.png"))
    plt.close()

    # 图2：Player 1 距离球
    plt.figure(figsize=(12, 5), dpi=150)
    plt.plot(df_merged["Frame"], smoothed_p1, label="Player 1 to Ball", color='blue', linewidth=2)
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Distance (pixels)", fontsize=12)
    plt.title(f"Player 1 Ball Distance - {segment_prefix}", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"{segment_prefix}_player1_distance.png"))
    plt.close()

    # 图3：Player 2 距离球
    plt.figure(figsize=(12, 5), dpi=150)
    plt.plot(df_merged["Frame"], smoothed_p2, label="Player 2 to Ball", color='green', linewidth=2)
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Distance (pixels)", fontsize=12)
    plt.title(f"Player 2 Ball Distance - {segment_prefix}", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FINAL_OUTPUT_DIR, f"{segment_prefix}_player2_distance.png"))
    plt.close()

if __name__ == "__main__":
    for path in glob.glob(os.path.join(SEGMENT_DIR, "*_ball_and_player_coordinates.csv")):
        process_segment(path)
