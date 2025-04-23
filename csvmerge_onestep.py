import pandas as pd
import numpy as np
import glob
import os
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from main import get_court_img

SEGMENT_DIR = "output_segments"
SPORTS2D_DIR = "sports2d_results"

def csv_processing(df):
    filled_df = df.copy()
    cols = ["Person1_X", "Person1_Y", "Person2_X", "Person2_Y", "Ball_Frame_X", "Ball_Frame_Y"]
    for col in cols:
        mask = filled_df[col].notna()
        if mask.sum() >= 2:
            f_interp = interp1d(
                filled_df.loc[mask, 'Frame'],
                filled_df.loc[mask, col],
                kind='cubic',
                fill_value="extrapolate"
            )
            missing = filled_df[col].isna()
            filled_df.loc[missing, col] = f_interp(filled_df.loc[missing, 'Frame'])
    return filled_df

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def process_segment(coordinate_csv_path):
    print(f"Processing segment: {coordinate_csv_path}")
    file_prefix = os.path.splitext(os.path.basename(coordinate_csv_path))[0].replace("_output_coordinates", "")

    sports2d_subdir = os.path.join(SPORTS2D_DIR, file_prefix, f"{file_prefix}_Sports2D")
    sports2d_files = glob.glob(os.path.join(sports2d_subdir, "*px_person*.trc.csv"))

    if len(sports2d_files) < 2:
        print("Missing necessary input files.")
        return

    output_coordinates_df = pd.read_csv(coordinate_csv_path)
    output_coordinates_df = csv_processing(output_coordinates_df)

    person_avg_positions = {}
    for file in sports2d_files:
        df = pd.read_csv(file)
        person_avg_positions[file] = (df["CHip_x"].mean(), df["CHip_y"].mean())

    person1_avg = (output_coordinates_df["Person1_X"].mean(), output_coordinates_df["Person1_Y"].mean())
    person2_avg = (output_coordinates_df["Person2_X"].mean(), output_coordinates_df["Person2_Y"].mean())
    distances = {file: (euclidean_distance(person_avg_positions[file], person1_avg), euclidean_distance(person_avg_positions[file], person2_avg)) for file in sports2d_files}
    selected_files = sorted(distances.items(), key=lambda x: min(x[1]))[:2]
    selected_files = [file[0] for file in selected_files]

    selected_sports2d_dfs = [pd.read_csv(file) for file in selected_files]
    for i, df in enumerate(selected_sports2d_dfs):
        df = df[[
            "Frame", 
            "RWrist_x", "RWrist_y", "LWrist_x", "LWrist_y",
            "CHip_x", "CHip_y",
            "LSmallToe_x", "LSmallToe_y", "LBigToe_x", "LBigToe_y",
            "RSmallToe_x", "RSmallToe_y", "RBigToe_x", "RBigToe_y"
        ]].rename(columns={
            "RWrist_x": f"Player{i+1}_RWrist_X", "RWrist_y": f"Player{i+1}_RWrist_Y",
            "LWrist_x": f"Player{i+1}_LWrist_X", "YLWrist_y": f"Player{i+1}_LWrist_Y",
            "CHip_x": f"Player{i+1}_Position_X", "CHip_y": f"Player{i+1}_Position_Y",
            "LSmallToe_x": f"Player{i+1}_LSmallToe_X", "LSmallToe_y": f"Player{i+1}_LSmallToe_Y",
            "LBigToe_x": f"Player{i+1}_LBigToe_X", "LBigToe_y": f"Player{i+1}_LBigToe_Y",
            "RSmallToe_x": f"Player{i+1}_RSmallToe_X", "RSmallToe_y": f"Player{i+1}_RSmallToe_Y",
            "RBigToe_x": f"Player{i+1}_RBigToe_X", "RBigToe_y": f"Player{i+1}_RBigToe_Y"
        })
        df[f"Player{i+1}_Ground_X"] = df[[f"Player{i+1}_LSmallToe_X", f"Player{i+1}_LBigToe_X", f"Player{i+1}_RSmallToe_X", f"Player{i+1}_RBigToe_X"]].mean(axis=1)
        df[f"Player{i+1}_Ground_Y"] = df[[f"Player{i+1}_LSmallToe_Y", f"Player{i+1}_LBigToe_Y", f"Player{i+1}_RSmallToe_Y", f"Player{i+1}_RBigToe_Y"]].mean(axis=1)
        selected_sports2d_dfs[i] = df

    merged_sports2d_df = pd.merge(selected_sports2d_dfs[0], selected_sports2d_dfs[1], on="Frame", how="inner")
    final_merged_df = pd.merge(output_coordinates_df, merged_sports2d_df, on="Frame", how="inner")

    for i in [1, 2]:
        final_merged_df[f"Player{i}_Ball_Dist"] = np.sqrt((final_merged_df["Ball_Frame_X"] - final_merged_df[f"Player{i}_RWrist_X"])**2 + (final_merged_df["Ball_Frame_Y"] - final_merged_df[f"Player{i}_RWrist_Y"])**2)

    final_merged_df["Distance_Diff"] = final_merged_df["Player1_Ball_Dist"] - final_merged_df["Player2_Ball_Dist"]

    local_max_indices = argrelextrema(final_merged_df["Distance_Diff"].values, np.greater, order=5)[0]
    local_min_indices = argrelextrema(final_merged_df["Distance_Diff"].values, np.less, order=5)[0]
    candidate_impact_frames = sorted(np.concatenate((local_max_indices, local_min_indices)))
    final_merged_df["Impact_Frame"] = False

    selected_impact_frames = []
    last_frame = -1
    for idx in candidate_impact_frames:
        if last_frame == -1 or np.sign(final_merged_df.at[idx, "Distance_Diff"]) != np.sign(final_merged_df.at[last_frame, "Distance_Diff"]):
            selected_impact_frames.append(idx)
        elif abs(final_merged_df.at[idx, "Distance_Diff"]) > abs(final_merged_df.at[last_frame, "Distance_Diff"]):
            selected_impact_frames[-1] = idx
        last_frame = idx

    final_merged_df.loc[selected_impact_frames, "Impact_Frame"] = True
    impact_frames = final_merged_df.loc[final_merged_df["Impact_Frame"] == True, "Frame"].tolist()
    print(f"Frame of impacts: {impact_frames}")

    new_coords = []
    for frame in final_merged_df.loc[final_merged_df["Impact_Frame"] == True, "Frame"]:
        row = final_merged_df[final_merged_df["Frame"] == frame].iloc[0]
        player = 1 if row["Player1_Ball_Dist"] < row["Player2_Ball_Dist"] else 2
        selected_y = row[f"Player{player}_Ground_Y"]
        new_coords.append([frame, row["Ball_Frame_X"], selected_y, player])

    new_coords_df = pd.DataFrame(new_coords, columns=["Frame", "Ball_X", "Wrist_Y", "Player"])
    matrix_file = os.path.join(SEGMENT_DIR, f"{file_prefix}_matrixes.npy")
    homography_matrices = np.load(matrix_file, allow_pickle=True)

    court_img = get_court_img()
    if len(court_img.shape) == 2:
        gray = court_img
    else:
        gray = cv2.cvtColor(court_img, cv2.COLOR_BGR2GRAY)
    green = (60, 160, 60)
    colored_court = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    colored_court[:] = green
    colored_court[gray == 255] = (255, 255, 255)
    court_img = colored_court

    trajectory_points = []
    for i, row in new_coords_df.iterrows():
        frame_idx = int(row["Frame"])
        if frame_idx >= len(homography_matrices):
            continue
        H = homography_matrices[frame_idx]
        point = np.array([[row["Ball_X"] / 1.5, row["Wrist_Y"] / 1.5]], dtype=np.float32).reshape(1, 1, 2)
        transformed_point = cv2.perspectiveTransform(point, H)
        x_transformed, y_transformed = int(transformed_point[0, 0, 0]), int(transformed_point[0, 0, 1])
        new_coords_df.at[i, "Transformed_X"] = x_transformed
        new_coords_df.at[i, "Transformed_Y"] = y_transformed
        trajectory_points.append((x_transformed, y_transformed))
        color = (0, 0, 255) if row["Player"] == 1 else (255, 0, 0)
        court_img = cv2.circle(court_img, (x_transformed, y_transformed), radius=20, color=color, thickness=-1)
        cv2.putText(court_img, str(len(trajectory_points)), (x_transformed, y_transformed), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

    if len(trajectory_points) > 1:
        cv2.polylines(court_img, [np.array(trajectory_points, dtype=np.int32)], isClosed=False, color=(0, 0, 0), thickness=5, lineType=cv2.LINE_AA)

    impact_csv_path = os.path.join(SEGMENT_DIR, f"{file_prefix}_impact_coordinates.csv")
    court_image_path = os.path.join(SEGMENT_DIR, f"{file_prefix}_court_reference.png")
    new_coords_df.to_csv(impact_csv_path, index=False)
    cv2.imwrite(court_image_path, court_img)
    print(f"Saved: {impact_csv_path} and {court_image_path}")

if __name__ == '__main__':
    for segment_file in glob.glob(os.path.join(SEGMENT_DIR, "*_output_coordinates.csv")):
        process_segment(segment_file)
