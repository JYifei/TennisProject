import pandas as pd
import numpy as np
import glob
import os
import cv2
import re
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from main import get_court_img


def csv_processing(df):
    filled_df = df.copy()

    # List of coordinate columns to fill missing values (do not modify 'Frame')
    cols = ["Person1_X", "Person1_Y", "Person2_X", "Person2_Y", "Ball_Frame_X", "Ball_Frame_Y"]

    # For each coordinate column, fill in only the missing values using cubic interpolation
    for col in cols:
        mask = filled_df[col].notna()
        if mask.sum() >= 2:  # Perform interpolation only if there are at least 2 non-missing data points
            f_interp = interp1d(
                filled_df.loc[mask, 'Frame'],
                filled_df.loc[mask, col],
                kind='cubic',
                fill_value="extrapolate"
            )
            missing = filled_df[col].isna()
            filled_df.loc[missing, col] = f_interp(filled_df.loc[missing, 'Frame'])

    return filled_df  # Return the processed DataFrame instead of saving



import glob
input_folder = "."  
csv_files = glob.glob(os.path.join(input_folder, "*_output_coordinates.csv"))
print(f"Glob searching in: {os.path.abspath(input_folder)}")
print("Glob found files:", csv_files)


csv_files = glob.glob("*_output_coordinates.csv")



# **read csv and process**
if len(csv_files) == 1:
    output_coordinates_file = csv_files[0]
    print(f"Found file: {output_coordinates_file}")
    
    file_prefix = output_coordinates_file.replace("_output_coordinates.csv", "")
    
    output_coordinates_df = pd.read_csv(output_coordinates_file)
    output_coordinates_df = csv_processing(output_coordinates_df)

    sports2d_folder = "outputTRC"
    sports2d_pattern = os.path.join(sports2d_folder, "*px_person*.trc.csv")
    sports2d_files = glob.glob(sports2d_pattern)

    print(f"Processed DataFrame is ready. Found {len(sports2d_files)} sports2d files.")
    
elif len(csv_files) > 1:
    print("找到多个符合条件的 `_output_coordinates.csv` 文件，请手动指定！")
else:
    print("未找到 `_output_coordinates.csv` 文件，请检查文件是否存在！")

# calculate average position of player (CHip_x, CHip_y)
person_avg_positions = {}
for file in sports2d_files:
    df = pd.read_csv(file)
    person_avg_positions[file] = (df["CHip_x"].mean(), df["CHip_y"].mean())

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# pair sports2d data with csv 
person1_avg = (output_coordinates_df["Person1_X"].mean(), output_coordinates_df["Person1_Y"].mean())
person2_avg = (output_coordinates_df["Person2_X"].mean(), output_coordinates_df["Person2_Y"].mean())
distances = {file: (euclidean_distance(person_avg_positions[file], person1_avg), euclidean_distance(person_avg_positions[file], person2_avg)) for file in sports2d_files}
selected_files = sorted(distances.items(), key=lambda x: min(x[1]))[:2]
selected_files = [file[0] for file in selected_files]


selected_sports2d_dfs = [pd.read_csv(file) for file in selected_files]
for i, df in enumerate(selected_sports2d_dfs):
    df = df[[
        "Frame", 
        "RWrist_x", "RWrist_y", "LWrist_x", "YLWrist_y",
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
    df[f"Player{i+1}_Ground_X"] = df[
        [f"Player{i+1}_LSmallToe_X", f"Player{i+1}_LBigToe_X",
         f"Player{i+1}_RSmallToe_X", f"Player{i+1}_RBigToe_X"]
    ].mean(axis=1)

    df[f"Player{i+1}_Ground_Y"] = df[
        [f"Player{i+1}_LSmallToe_Y", f"Player{i+1}_LBigToe_Y",
         f"Player{i+1}_RSmallToe_Y", f"Player{i+1}_RBigToe_Y"]
    ].mean(axis=1)
    selected_sports2d_dfs[i] = df

# merge sports2d data
merged_sports2d_df = pd.merge(selected_sports2d_dfs[0], selected_sports2d_dfs[1], on="Frame", how="inner")
final_merged_df = pd.merge(output_coordinates_df, merged_sports2d_df, on="Frame", how="inner")

# calculate distance of hand(wrist) and ball
for i in [1, 2]:
    final_merged_df[f"Player{i}_Ball_Dist"] = np.sqrt((final_merged_df["Ball_Frame_X"] - final_merged_df[f"Player{i}_RWrist_X"])**2 + (final_merged_df["Ball_Frame_Y"] - final_merged_df[f"Player{i}_RWrist_Y"])**2)

# find difference between the distances
final_merged_df["Distance_Diff"] = final_merged_df["Player1_Ball_Dist"] - final_merged_df["Player2_Ball_Dist"]

# find hits
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

#print impact frames
impact_frames = final_merged_df.loc[final_merged_df["Impact_Frame"] == True, "Frame"].tolist()
print(f"击球发生的帧: {impact_frames}")

# keep useful values
columns_to_keep = ["Frame", "Bounce", "Ball_Frame_X", "Ball_Frame_Y", "Impact_Frame", "Player1_Ball_Dist", "Player2_Ball_Dist", "Distance_Diff"] + [col for col in final_merged_df.columns if "RWrist" in col or "LWrist" in col or "Position" in col or "Ground" in col]
final_merged_df = final_merged_df[columns_to_keep]


plt.figure(figsize=(12, 6))
plt.plot(final_merged_df["Frame"], final_merged_df["Player1_Ball_Dist"], label="Player 1 Distance")
plt.plot(final_merged_df["Frame"], final_merged_df["Player2_Ball_Dist"], label="Player 2 Distance")

#mark impact frames
for frame in impact_frames:
    plt.axvline(x=frame, color='red', linestyle='--', alpha=0.6)

plt.xlabel("Frame")
plt.ylabel("Distance to Ball")
plt.title("Player Distance to Ball Over Time")
plt.xticks(np.arange(min(final_merged_df["Frame"]), max(final_merged_df["Frame"]), 50), rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


# final_merged_df.to_csv("merged_player_ball_data.csv", index=False)
# print("merged data saved as merged_player_ball_data.csv")


new_coords = []
for frame in final_merged_df.loc[final_merged_df["Impact_Frame"] == True, "Frame"]:
    row = final_merged_df[final_merged_df["Frame"] == frame].iloc[0]
    
    if row["Player1_Ball_Dist"] < row["Player2_Ball_Dist"]:
        player = 1
    else:
        player = 2
    
    rwrist_dist = row[f"Player{player}_Ball_Dist"]
    lwrist_dist = np.sqrt((row["Ball_Frame_X"] - row[f"Player{player}_LWrist_X"])**2 + (row["Ball_Frame_Y"] - row[f"Player{player}_LWrist_Y"])**2)
    
    if rwrist_dist < lwrist_dist:
        selected_y = row[f"Player{player}_Ground_Y"]
    else:
        selected_y = row[f"Player{player}_Ground_Y"]
    
    new_coords.append([frame, row["Ball_Frame_X"], selected_y, player])

# save as csv
new_coords_df = pd.DataFrame(new_coords, columns=["Frame", "Ball_X", "Wrist_Y", "Player"])


output_filled_filename = f"{file_prefix}_impact_coordinates.csv"

new_coords_df.to_csv(output_filled_filename, index=False)



court_img = get_court_img()

filename = f"{file_prefix}_perspective_matrices.npy"

# read matrix
homography_matrices = np.load(filename, allow_pickle=True)

new_coords_df["Transformed_X"] = np.nan
new_coords_df["Transformed_Y"] = np.nan

# 4. Interpolation
import numpy as np
import cv2


trajectory_points = []

for i, row in new_coords_df.iterrows():
    frame_idx = int(row["Frame"])  
    if frame_idx >= len(homography_matrices):
        continue 
    
    H = homography_matrices[frame_idx]  # read Homography matrix of the frame

    # change into numpy array
    point = np.array([[row["Ball_X"] / 1.5, row["Wrist_Y"] / 1.5]], dtype=np.float32).reshape(1, 1, 2)

    # transform
    transformed_point = cv2.perspectiveTransform(point, H)

    # get transformed point in pixel
    x_transformed, y_transformed = int(transformed_point[0, 0, 0]), int(transformed_point[0, 0, 1])
    new_coords_df.at[i, "Transformed_X"] = x_transformed
    new_coords_df.at[i, "Transformed_Y"] = y_transformed


    # save trace points
    trajectory_points.append((x_transformed, y_transformed))
    
    if row["Player"] == 1:
        court_img = cv2.circle(court_img, (int(transformed_point[0, 0, 0]), int(transformed_point[0, 0, 1])),
                                                       radius=0, color=(0, 255, 255), thickness=50)
    else:
        court_img = cv2.circle(court_img, (int(transformed_point[0, 0, 0]), int(transformed_point[0, 0, 1])),
                                                       radius=0, color=(255, 255, 0), thickness=50)
    cv2.putText(court_img, str(len(trajectory_points)), (x_transformed, y_transformed),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)  # 红色编号
# draw trace
if len(trajectory_points) > 1:
    cv2.polylines(court_img, [np.array(trajectory_points, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=3)


# cv2.imshow("Court Image with Trajectory", court_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 5. save data
#new_coords_df.to_csv("transformed_coords.csv", index=False)
cv2.imwrite("court_reference.png", court_img)
print(new_coords_df.head())



