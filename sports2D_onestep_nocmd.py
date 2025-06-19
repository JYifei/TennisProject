import os
from pathlib import Path
import shutil
from trc import TRCData
import pandas as pd
import numpy as np
import toml
from Sports2D import Sports2D

SEGMENT_DIR = "masked_segments"
SPORTS2D_OUTPUT_DIR = "sports2d_results"

def convert_trc_to_csv(trc_file):
    mocap_data = TRCData()
    mocap_data.load(trc_file)
    rows = int(mocap_data['NumFrames'])
    num_markers = int(mocap_data['NumMarkers'])
    cols = num_markers * 3
    frames = np.zeros((rows, cols + 5))

    for i in range(rows):
        num_frames = mocap_data[i]
        col_vec = np.array(num_frames[1], ndmin=3)
        size = col_vec.shape[1]
        x = col_vec[0, range(size), 0]
        y = col_vec[0, range(size), 1]
        z = col_vec[0, range(size), 2]
        b = np.array([
            mocap_data['CameraRate'],
            mocap_data['NumMarkers'],
            mocap_data['NumFrames'],
            num_frames[0],
            i + 1
        ])
        a = np.concatenate((b, x, y, z), axis=0)
        frames[i, range(0, size * 3 + 5)] = a

    data = pd.DataFrame(frames)
    data.columns = [
        'CameraRate', 'NumMarkers', 'NumFrames', 'Time', 'Frame',
        'CHip_x','RHip_x','RKnee_x','RAnkle_x','RBigToe_x','RSmallToe_x','RHeel_x','LHip_x','LKnee_x','LAnkle_x','LBigToe_x','LSmallToe_x','LHeel_x','Neck_x','X_Head','Nose_x','RShoulder_x','RElbow_x','RWrist_x','LShoulder_x','LElbow_x','LWrist_x', 
        'CHip_y','RHip_y','RKnee_y','RAnkle_y','RBigToe_y','RSmallToe_y','RHeel_y','LHip_y','LKnee_y','LAnkle_y','LBigToe_y','LSmallToe_y','LHeel_y','Neck_y','Y_Head','Nose_y','RShoulder_y','RElbow_y','RWrist_y','LShoulder_y','LElbow_y','LWrist_y',
        'CHip_z','RHip_z','RKnee_z','RAnkle_z','RBigToe_z','RSmallToe_z','RHeel_z','LHip_z','LKnee_z','LAnkle_z','LBigToe_z','LSmallToe_z','LHeel_z','Neck_z','Z_Head','Nose_z','RShoulder_z','RElbow_z','RWrist_z','LShoulder_z','LElbow_z','LWrist_z'
    ]
    csv_path = trc_file + ".csv"
    data.to_csv(csv_path, index=False)
    print(f"Converted {trc_file} to CSV")

def run_one_video_with_config(full_path):
    video_name = Path(full_path).stem
    output_subdir = os.path.join(SPORTS2D_OUTPUT_DIR, video_name)
    os.makedirs(output_subdir, exist_ok=True)

    # 手动构造完整 config（跳过 bug）
    config_dict = toml.load('Config_demo.toml')
    config_dict['base']['video_input'] = str(full_path)
    config_dict['base']['result_dir'] = output_subdir
    config_dict['base']['save_vid'] = True
    config_dict['base']['save_img'] = False
    config_dict['base']['show_realtime_results'] = False

    Sports2D.process(config_dict)

    # 后处理
    sports2d_inner = os.path.join(output_subdir, f"{video_name}_Sports2D")
    if os.path.exists(sports2d_inner):
        for f in os.listdir(sports2d_inner):
            if f.lower().endswith(".trc"):
                convert_trc_to_csv(os.path.join(sports2d_inner, f))

        for f in os.listdir(sports2d_inner):
            src = os.path.join(sports2d_inner, f)
            dst = os.path.join(output_subdir, f)
            if os.path.exists(dst): os.remove(dst)
            shutil.move(src, dst)
        try:
            os.rmdir(sports2d_inner)
        except Exception as e:
            print(f"Failed to remove subfolder: {e}")
    else:
        print(f"[WARNING] No inner folder found: {sports2d_inner}")

    for f in os.listdir(output_subdir):
        if not (f.endswith(".mp4") or ("px_person" in f and f.endswith(".trc.csv"))):
            try:
                os.remove(os.path.join(output_subdir, f))
            except Exception as e:
                print(f"[WARNING] Could not delete {f}: {e}")

def run_all_segments():
    os.makedirs(SPORTS2D_OUTPUT_DIR, exist_ok=True)
    for root, _, files in os.walk(SEGMENT_DIR):
        for file in files:
            if file.endswith(".mp4") and "masked" in file:
                full_path = os.path.abspath(os.path.join(root, file))
                print(f"[INFO] Processing {full_path}")
                run_one_video_with_config(full_path)

if __name__ == "__main__":
    run_all_segments()
