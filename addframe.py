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


def annotate_video_with_frame_index(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Add frame index text to the top-left corner
        cv2.putText(
            frame,
            f"Frame: {frame_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    print(f"Annotated video saved: {output_path}")


if __name__ == '__main__':
    for file in os.listdir(SEGMENT_DIR):
        if file.endswith("_annotated.mp4"):
            video_path = os.path.join(SEGMENT_DIR, file)
            output_path = os.path.join(SEGMENT_DIR, file.replace("_annotated.mp4", "_annotated_framed.mp4"))
            annotate_video_with_frame_index(video_path, output_path)

            # 删除原始 annotated 视频
            os.remove(video_path)
            print(f"Deleted original video: {video_path}")
