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
                coords = [float(val) for coord in kps for val in coord]
                row = [idx] + coords
            writer.writerow(row)
    print(f"Court keypoints saved to {csv_path}")

def split_segments(keypoints, threshold=30):
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

def process_video(path_court_model, path_input_video, path_output_video, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for inference: {device}")

    cap = cv2.VideoCapture(path_input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_w, target_h = 1280, 720
    scale_x, scale_y = width_orig/target_w, height_orig/target_h

    detector = CourtDetectorNet(path_court_model, device)
    detector.model.to(device)
    detector.model.eval()
    print(f"Model parameters on device: {next(detector.model.parameters()).device}")

    keypoints_scaled = []
    buffer = []

    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames), desc='Detecting court', unit='frame'):
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (target_w, target_h))
            buffer.append(resized)

            if len(buffer) == batch_size or frame_idx == total_frames - 1:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    _, kps_batch = detector.infer_model(buffer)
                for kps in kps_batch:
                    if kps is None:
                        keypoints_scaled.append(None)
                    else:
                        keypoints_scaled.append([(kp[0,0]*scale_x, kp[0,1]*scale_y) for kp in kps])
                buffer.clear()

                #if device.type == 'cuda':
                #    print(f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
    cap.release()

    save_keypoints_csv(keypoints_scaled, path_input_video)
    shutil.copy(path_input_video, path_output_video)

    segments = split_segments(keypoints_scaled)
    video_name = os.path.splitext(os.path.basename(path_input_video))[0]
    segments_dir = os.path.join(os.path.dirname(path_output_video) or '.', f"{video_name}_segments")
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Court detection video splitter')
    parser.add_argument('--path_court_model', type=str, required=True)
    parser.add_argument('--path_input_video', type=str, required=True)
    parser.add_argument('--path_output_video', type=str, required=True)
    args = parser.parse_args()
    process_video(args.path_court_model, args.path_input_video, args.path_output_video)