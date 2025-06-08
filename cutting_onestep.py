import cv2
import torch
import os
import numpy as np
import csv
import shutil
from utils.court_detection_net import CourtDetectorNet
from tqdm import tqdm
import contextlib, io

INPUT_DIR = "input_videos"
OUTPUT_DIR = "output_segments"
PROCESSED_DIR = "processed_videos"
COURT_MODEL_PATH = "models/model_tennis_court_det.pt"

KEYPOINT_DIR = "keypoints_data"
MATRIX_DIR = "matrix_data"

def save_keypoints_csv(keypoints, video_path, scale_x, scale_y):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(KEYPOINT_DIR, f"{video_name}_court_keypoints.csv")
    header = ["Frame"] + [f"KP{i+1}_{axis}" for i in range(14) for axis in ("X","Y")]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for idx, kps in enumerate(keypoints):
            if kps is None:
                row = [idx] + [None] * (14*2)
            else:
                coords = []
                for coord in kps:
                    arr = np.array(coord).flatten()
                    if len(arr) >= 2:
                        x, y = arr[:2]
                        coords.extend([x * scale_x, y * scale_y])
                    else:
                        coords.extend([None, None])  # fallback
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

def process_video(path_input_video, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for inference: {device}")

    cap = cv2.VideoCapture(path_input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = CourtDetectorNet(COURT_MODEL_PATH, device)
    input_height, input_width = 720, 1280 
    scale_x = width_orig / input_width
    scale_y = height_orig / input_height

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
    #save_keypoints_csv(keypoints_scaled, path_input_video, scale_x, scale_y)

    threshold_seconds = 2
    threshold_frames = int(threshold_seconds * fps)
    segments = split_segments(keypoints_scaled, threshold=threshold_frames)

    video_name = os.path.splitext(os.path.basename(path_input_video))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
                segment_path = os.path.join(OUTPUT_DIR, f"{video_name}_segment_{idx}.mp4")
                writer = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
            writer.write(frame)
        if writer:
            writer.release()
            print(f"Saved segment: {segment_path}")
        cap.release()

        segment_matrixes = matrixes_all[start:end]
        matrix_path = os.path.join(MATRIX_DIR, f"{video_name}_segment_{idx}_matrixes.npy")
        np.save(matrix_path, segment_matrixes)
        print(f"Saved matrixes: {matrix_path}")

        segment_keypoints = keypoints_scaled[start:end]
        keypoint_csv_path = os.path.join(KEYPOINT_DIR, f"{video_name}_segment_{idx}_keypoints.csv")
        header = ["Frame"] + [f"KP{i+1}_{axis}" for i in range(14) for axis in ("X", "Y")]
        with open(keypoint_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for rel_idx, kps in enumerate(segment_keypoints):
                if kps is None:
                    row = [rel_idx] + [None] * 28
                else:
                    coords = []
                    for coord in kps:
                        arr = np.array(coord).flatten()
                        if len(arr) >= 2:
                            x, y = arr[:2]
                            coords.extend([x * scale_x, y * scale_y])
                        else:
                            coords.extend([None, None])  # fallback
                    row = [rel_idx] + coords
                writer.writerow(row)
        print(f"Saved keypoints CSV: {keypoint_csv_path}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    shutil.move(path_input_video, os.path.join(PROCESSED_DIR, os.path.basename(path_input_video)))
    print(f"Moved processed video to {PROCESSED_DIR}")

if __name__ == '__main__':
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(KEYPOINT_DIR, exist_ok=True)
    os.makedirs(MATRIX_DIR, exist_ok=True)

    videos = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4') or f.endswith('.avi')]
    if not videos:
        print("No input videos found in input_videos/")
    for vid in videos:
        full_path = os.path.join(INPUT_DIR, vid)
        print(f"\n=== Processing {full_path} ===")
        process_video(full_path)
