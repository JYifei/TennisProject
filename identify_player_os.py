import os
import pandas as pd
import numpy as np
from pathlib import Path

SEGMENT_DIR = "output_segments"
SPORTS2D_DIR = "sports2d_results"

# 获取底线关键点（KP1,5,7,2 和 KP3,6,8,4）的平均位置，跳过缺失值
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
    #print(y1,y2)
    return y1, y2

# 计算人体中心点，这里用 CHip 点（中央髋关节）作为代表
def get_person_center(csv_path):
    df = pd.read_csv(csv_path)
    # 只取前三秒内的数据（假设fps为30）
    sub_df = df.iloc[:120]
    center_points = sub_df[["CHip_x", "CHip_y"]].dropna().values
    if len(center_points) == 0:
        return np.array([np.nan, np.nan])
    center_mean = np.mean(center_points, axis=0)
    #print(f"{os.path.basename(csv_path)} center: {center_mean}")
    return center_mean

# 主逻辑：输出每个视频挑出的两个玩家编号
def identify_players():
    for video_file in os.listdir(SEGMENT_DIR):
        if not video_file.endswith("_with_edges.mp4"):
            continue

        base_name = Path(video_file).stem
        keypoint_csv = os.path.join(SEGMENT_DIR, f"{base_name.replace('_with_edges', '')}_keypoints.csv")
        if not os.path.exists(keypoint_csv):
            print(f"Keypoint CSV not found for {base_name}")
            continue

        y1, y2 = get_baselines(keypoint_csv)

        video_result_dir = os.path.join(SPORTS2D_DIR, base_name, f"{base_name}_Sports2D")
        if not os.path.exists(video_result_dir):
            print(f"Sports2D folder not found: {video_result_dir}")
            continue

        closest_top = (None, np.inf)
        closest_bottom = (None, np.inf)

        for file in os.listdir(video_result_dir):
            if file.endswith(".trc.csv") and "px_person" in file:
                person_path = os.path.join(video_result_dir, file)
                center = get_person_center(person_path)
                if np.isnan(center).any():
                    continue
                dy_top = abs(center[1] - y1)
                dy_bottom = abs(center[1] - y2)
                person_id = Path(file).stem

                if dy_top < closest_top[1]:
                    closest_top = (person_id, dy_top)
                if dy_bottom < closest_bottom[1]:
                    closest_bottom = (person_id, dy_bottom)

        final_players = {closest_top[0], closest_bottom[0]}
        print(f"{base_name}: {sorted(final_players)}")


if __name__ == '__main__':
    identify_players()
