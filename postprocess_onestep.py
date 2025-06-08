import os
import cv2
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict
from tqdm import tqdm

INPUT_DIR = "sports2d_results"
VIDEO_DIR = "output_segments"
OUTPUT_DIR = "postprocessed_sports2D"
MISSING_THRESHOLD = 0.25
RGB_WEIGHT = 0.5
POSE_WEIGHT = 0.5
PATCH_SIZE = 25
TORSO_KEYS = ['CHip', 'Neck', 'LShoulder', 'RShoulder']

def load_and_filter_csvs(segment_path):
    files = sorted(glob(os.path.join(segment_path, "*.csv")))
    valid_persons = []
    for f in files:
        df = pd.read_csv(f)
        df = df.drop(columns=[col for col in df.columns if col.endswith('_z')], errors='ignore')
        df = df[df.iloc[:, 1:].apply(lambda x: not (x == 0).all(), axis=1)]
        if df.empty:
            continue
        missing_ratio = df.isna().mean().mean()
        if missing_ratio < MISSING_THRESHOLD:
            valid_persons.append((f, df.fillna(0)))
        else:
            print(f"Filtered out {os.path.basename(f)} due to missing ratio {missing_ratio:.2f}")
    return valid_persons

def extract_keypoints(df, frame_idx):
    if frame_idx >= len(df):
        return None
    row = df.iloc[frame_idx]
    return row.values.astype(np.float32)

def compute_average_rgb(img, keypoints, cols, torso_keys):
    h, w, _ = img.shape
    avg_colors = []
    for key in torso_keys:
        if f"{key}_x" in cols and f"{key}_y" in cols:
            x, y = int(keypoints[cols.index(f"{key}_x")]), int(keypoints[cols.index(f"{key}_y")])
            if 0 <= x < w and 0 <= y < h:
                patch = img[max(0, y - PATCH_SIZE):min(h, y + PATCH_SIZE),
                            max(0, x - PATCH_SIZE):min(w, x + PATCH_SIZE)]
                if patch.size > 0:
                    avg_color = patch.mean(axis=(0, 1))
                    avg_colors.append(avg_color)
    return np.mean(avg_colors, axis=0) if avg_colors else np.zeros(3)

def compute_combined_distance(p1, p2, rgb1, rgb2):
    pos_dist = np.linalg.norm(p1 - p2)
    rgb_dist = np.linalg.norm(rgb1 - rgb2)
    return POSE_WEIGHT * pos_dist + RGB_WEIGHT * rgb_dist

def load_video_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def track_ids_with_rgb(person_dfs, video_path, max_ids):
    n_frames = person_dfs[0][1].shape[0]
    header = person_dfs[0][1].columns.tolist()
    id_tracks = defaultdict(list)
    prev_features = {}
    current_id = 0

    for frame_idx in tqdm(range(n_frames), desc="Tracking"):
        frame = load_video_frame(video_path, frame_idx)
        if frame is None:
            continue
        candidates = []
        for _, df in person_dfs:
            keypoints = extract_keypoints(df, frame_idx)
            if keypoints is None or not np.any(keypoints):
                continue
            rgb_feat = compute_average_rgb(frame, keypoints, header, TORSO_KEYS)
            candidates.append((keypoints, rgb_feat))

        used_prev_ids, used_cands, matches = set(), set(), {}

        for c_idx, (pfeat, prgb) in enumerate(candidates):
            best_id, min_dist = None, float('inf')
            for pid, (prev_feat, prev_rgb) in prev_features.items():
                dist = compute_combined_distance(pfeat, prev_feat, prgb, prev_rgb)
                if dist < min_dist:
                    min_dist, best_id = dist, pid
            if best_id is not None and best_id not in used_prev_ids:
                matches[c_idx] = best_id
                used_prev_ids.add(best_id)
                used_cands.add(c_idx)

        for i, (feat, rgb) in enumerate(candidates):
            if i not in used_cands and current_id < max_ids:
                matches[i] = current_id
                current_id += 1

        prev_features.clear()
        for i, pid in matches.items():
            prev_features[pid] = candidates[i]
            id_tracks[pid].append((frame_idx, candidates[i][0]))

    return id_tracks, header

def save_tracks(segment_name, id_tracks, header):
    segment_out_dir = os.path.join(OUTPUT_DIR, segment_name)
    os.makedirs(segment_out_dir, exist_ok=True)
    for pid, records in id_tracks.items():
        rows = [[f_idx] + joints.tolist() for f_idx, joints in records]
        expected_cols = len(rows[0])
        output_header = ["Frame"] + header[:expected_cols - 1]
        df = pd.DataFrame(rows, columns=output_header)
        df.to_csv(os.path.join(segment_out_dir, f"player_{pid}.csv"), index=False)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for segment in os.listdir(INPUT_DIR):
        seg_path = os.path.join(INPUT_DIR, segment)
        print(f"\n=== Processing segment: {segment} ===")
        person_dfs = load_and_filter_csvs(seg_path)
        if not person_dfs:
            print(f"No valid persons found in {segment}, skipping.")
            continue

        max_ids = len(person_dfs)
        video_name = segment.replace("_masked", "")
        video_path = os.path.join(VIDEO_DIR, f"{video_name}.mp4")

        if not os.path.exists(video_path):
            print(f"Warning: video {video_path} not found, skipping.")
            continue

        id_tracks, header = track_ids_with_rgb(person_dfs, video_path, max_ids)
        save_tracks(segment, id_tracks, header)
        print(f"Saved {len(id_tracks)} tracks to {segment}")

if __name__ == "__main__":
    main()
