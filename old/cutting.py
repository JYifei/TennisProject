import cv2
import argparse
import torch
import os
import numpy as np
import csv
import shutil
from court_detection_net import CourtDetectorNet
from tqdm import tqdm
import contextlib, io

def save_keypoints_csv(keypoints, video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = f"{video_name}_court_keypoints.csv"
    header = ["Frame"] + [f"KP{i+1}_{axis}" for i in range(14) for axis in ("X","Y")]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for idx, kps in enumerate(keypoints):
            if kps is None:
                row = [idx] + [None] * (14*2)
            else:
                coords = [float(val) for coord in kps for val in np.array(coord).flatten()]
                row = [idx] + coords
            writer.writerow(row)
    print(f"Court keypoints saved to {csv_path}")

def split_segments(keypoints, threshold):
    segments, start, no_court = [], None, 0
    for i, kps in enumerate(keypoints):
        if kps is not None:
            if start is None:
                start = i
            no_court = 0
        elif start is not None:
            no_court += 1
            if no_court >= threshold:
                segments.append((start, i - no_court + 1))
                start, no_court = None, 0
    if start is not None:
        segments.append((start, len(keypoints)))
    return segments

def process_video(path_court_model, path_input_video, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for inference: {device}")

    cap = cv2.VideoCapture(path_input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = CourtDetectorNet(path_court_model, device)
    detector.model.to(device)
    detector.model.eval()
    print(f"Model parameters on device: {next(detector.model.parameters()).device}")

    keypoints_scaled = []
    matrixes_all = []
    buffer = []

    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames), desc='Detecting court', unit='frame'):
            ret, frame = cap.read()
            if not ret:
                break
            buffer.append(frame)

            if len(buffer) == batch_size or frame_idx == total_frames - 1:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    matrixes_batch, kps_batch = detector.infer_model(buffer)
                for kps, mat in zip(kps_batch, matrixes_batch):
                    keypoints_scaled.append(kps if kps is not None else None)
                    matrixes_all.append(mat if mat is not None else None)
                buffer = []

    cap.release()

    save_keypoints_csv(keypoints_scaled, path_input_video)

    threshold_seconds = 2  # 秒数阈值
    threshold_frames = int(threshold_seconds * fps)
    segments = split_segments(keypoints_scaled, threshold=threshold_frames)

    video_name = os.path.splitext(os.path.basename(path_input_video))[0]
    segments_dir = os.path.join('.', f"{video_name}_segments")
    os.makedirs(segments_dir, exist_ok=True)

    for idx, (start, end) in enumerate(segments, 1):
        cap = cv2.VideoCapture(path_input_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        writer = None
        for frame_no in tqdm(range(start, end), desc=f'Saving segment {idx}', unit='frame'):
            ret, frame = cap.read()
            if not ret:
                break
            if writer is None:
                height, width = frame.shape[:2]
                segment_path = os.path.join(segments_dir, f"{video_name}_segment_{idx}.mp4")
                writer = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
            writer.write(frame)
        if writer:
            writer.release()
            print(f"Saved segment: {segment_path}")
        cap.release()

        # 保存对应的 matrixes
        segment_matrixes = matrixes_all[start:end]
        matrix_path = os.path.join(segments_dir, f"{video_name}_segment_{idx}_matrixes.npy")
        np.save(matrix_path, segment_matrixes)
        print(f"Saved matrixes: {matrix_path}")
        
        # 保存对应的关键点坐标为 CSV
        segment_keypoints = keypoints_scaled[start:end]
        keypoint_csv_path = os.path.join(segments_dir, f"{video_name}_segment_{idx}_keypoints.csv")
        header = ["Frame"] + [f"KP{i+1}_{axis}" for i in range(14) for axis in ("X", "Y")]
        with open(keypoint_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for rel_idx, kps in enumerate(segment_keypoints):
                if kps is None:
                    row = [rel_idx] + [None] * 28
                else:
                    coords = [float(val) for coord in kps for val in np.array(coord).flatten()]
                    row = [rel_idx] + coords
                writer.writerow(row)
        print(f"Saved keypoints CSV: {keypoint_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Court detection video splitter')
    parser.add_argument('--path_court_model', type=str, required=True)
    parser.add_argument('--path_input_video', type=str, required=True)
    #parser.add_argument('--path_output_video', type=str, required=True)
    args = parser.parse_args()
    process_video(args.path_court_model, args.path_input_video)