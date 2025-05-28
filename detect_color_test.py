import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque

VIDEO_DIR = "output_segments_test"
TRC_DIR = "sports2d_results"
COLOR_WEIGHT = 1.0
POSITION_WEIGHT = 2.0
COLOR_HISTORY = 5

def extract_tshirt_color(frame, keypoints, box_halfsize=15):
    try:
        shoulder_mid = np.mean([keypoints['LShoulder'], keypoints['RShoulder']], axis=0)
        hip_mid = np.mean([keypoints['CHip'], keypoints.get('MidHip', keypoints['CHip'])], axis=0)
        center = np.mean([shoulder_mid, hip_mid], axis=0).astype(int)
        cx, cy = center
        x_min = max(0, cx - box_halfsize)
        x_max = min(frame.shape[1], cx + box_halfsize)
        y_min = max(0, cy - box_halfsize)
        y_max = min(frame.shape[0], cy + box_halfsize)
        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None
        mean_color = crop.mean(axis=(0, 1))
        if np.mean(mean_color) < 10:
            return None
        return mean_color
    except:
        return None

def person_center(keypoints):
    shoulder_mid = np.mean([keypoints['LShoulder'], keypoints['RShoulder']], axis=0)
    hip_mid = np.mean([keypoints['CHip'], keypoints.get('MidHip', keypoints['CHip'])], axis=0)
    return np.mean([shoulder_mid, hip_mid], axis=0)

def hsv_distance(c1, c2):
    c1_hsv = cv2.cvtColor(np.uint8([[c1]]), cv2.COLOR_BGR2HSV)[0][0]
    c2_hsv = cv2.cvtColor(np.uint8([[c2]]), cv2.COLOR_BGR2HSV)[0][0]
    dh = min(abs(int(c1_hsv[0]) - int(c2_hsv[0])), 180 - abs(int(c1_hsv[0]) - int(c2_hsv[0])))
    ds = abs(int(c1_hsv[1]) - int(c2_hsv[1]))
    dv = abs(int(c1_hsv[2]) - int(c2_hsv[2]))
    return np.sqrt(dh**2 + ds**2 + dv**2)

def load_all_keypoints(trc_folder):
    data = {}
    person_files = []
    for file in sorted(os.listdir(trc_folder)):
        if file.endswith(".trc.csv") and "px_person" in file:
            person_files.append(file)
            df = pd.read_csv(os.path.join(trc_folder, file))
            for i, row in df.iterrows():
                if i not in data:
                    data[i] = {}
                keypoints = {}
                for part in ["LShoulder", "RShoulder", "CHip", "MidHip"]:
                    try:
                        x = row[f"{part}_x"]
                        y = row[f"{part}_y"]
                        if not np.isnan(x) and not np.isnan(y):
                            keypoints[part] = (x, y)
                    except:
                        continue
                if keypoints:
                    data[i][file] = keypoints
    return data, person_files

def average_color(history):
    if not history:
        return None
    return np.mean(np.array(history), axis=0)

def find_best_match(global_color, global_center, candidates, used_indexes):
    best_idx = None
    best_cost = float("inf")
    for idx, (center, color) in candidates.items():
        if idx in used_indexes:
            continue
        pos_dist = np.linalg.norm(global_center - center)
        col_dist = hsv_distance(global_color, color)
        cost = POSITION_WEIGHT * pos_dist + COLOR_WEIGHT * col_dist
        if cost < best_cost:
            best_cost = cost
            best_idx = idx
    return best_idx

def process_always_assign_best_match():
    output_rows = []

    for video_name in sorted(os.listdir(VIDEO_DIR)):
        if not video_name.endswith("with_edges.mp4"):
            continue

        base = Path(video_name).stem
        video_path = os.path.join(VIDEO_DIR, video_name)
        trc_folder = os.path.join(TRC_DIR, base, f"{base}_Sports2D")
        if not os.path.isfile(video_path) or not os.path.isdir(trc_folder):
            print(f"❌ Skip {video_name}")
            continue

        print(f"🎬 Processing {video_name}")
        cap = cv2.VideoCapture(video_path)
        keypoints_by_frame, person_files = load_all_keypoints(trc_folder)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        person_to_index = {file: i for i, file in enumerate(person_files)}
        N = len(person_files)

        global_matrix = {i: [""] * frame_count for i in range(N)}
        global_tracks = {
            i: {
                "center": None,
                "color_history": deque(maxlen=COLOR_HISTORY),
                "avg_color": None,
                "last_index": None
            } for i in range(N)
        }

        for frame_idx in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_candidates = {}
            if frame_idx in keypoints_by_frame:
                for file, kps in keypoints_by_frame[frame_idx].items():
                    center = person_center(kps)
                    color = extract_tshirt_color(frame, kps)
                    if color is not None:
                        frame_candidates[person_to_index[file]] = (center, color)

            used_indexes = set()
            for gid in range(N):
                track = global_tracks[gid]
                if track["center"] is None:
                    continue
                match_idx = find_best_match(track["avg_color"], track["center"], frame_candidates, used_indexes)
                if match_idx is not None:
                    center, color = frame_candidates[match_idx]
                    track["center"] = center
                    track["color_history"].append(color)
                    track["avg_color"] = average_color(track["color_history"])
                    track["last_index"] = match_idx
                    global_matrix[gid][frame_idx] = str(match_idx)
                    used_indexes.add(match_idx)

            for idx, (center, color) in frame_candidates.items():
                if idx in used_indexes:
                    continue
                for gid in range(N):
                    if global_tracks[gid]["center"] is None:
                        global_tracks[gid]["center"] = center
                        global_tracks[gid]["color_history"].append(color)
                        global_tracks[gid]["avg_color"] = average_color(global_tracks[gid]["color_history"])
                        global_tracks[gid]["last_index"] = idx
                        global_matrix[gid][frame_idx] = str(idx)
                        used_indexes.add(idx)
                        break

        cap.release()

        for gid, frames in global_matrix.items():
            output_rows.append([f"{video_name}_ID{gid}"] + frames)

    max_len = max(len(row) for row in output_rows)
    columns = ["global_id"] + [f"frame_{i}" for i in range(max_len - 1)]
    df = pd.DataFrame(output_rows, columns=columns)
    df.to_csv("global_id_always_assign_best.csv", index=False)
    print("✅ Saved to global_id_always_assign_best.csv")
    return df

# Run it
if __name__ == "__main__":
    df_result = process_always_assign_best_match()
