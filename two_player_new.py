import os
import pandas as pd
import numpy as np
from glob import glob
from collections import deque, Counter
import cv2

# 常量
FOOT_PARTS = ['LAnkle', 'RAnkle']
ALL_KEYPOINTS = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder',
                 'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip',
                 'LKnee', 'RKnee', 'LAnkle', 'RAnkle', 'Chest']
MAX_WINDOW = 10
STABILITY_WINDOW = 2  # 秒
STABILITY_THRESHOLD = 20  # px
MANUAL_SELECT = True        # True = enable manual selection
PREVIEW_DURATION = 3        # scan first 3 seconds to find best frame


def interpolate_keypoints(df, window=10):
    for col in df.columns:
        if '_x' in col or '_y' in col:
            values = df[col].values
            for i in range(len(values)):
                if values[i] == 0:
                    valid = []
                    for j in range(1, window + 1):
                        if i - j >= 0 and values[i - j] > 0:
                            valid.append(values[i - j])
                        if i + j < len(values) and values[i + j] > 0:
                            valid.append(values[i + j])
                    if valid:
                        values[i] = np.mean(valid)
            df[col] = values
    return df

def compute_center(row):
    coords = []
    for joint in FOOT_PARTS:
        x, y = row.get(f'{joint}_x', 0), row.get(f'{joint}_y', 0)
        if x > 0 and y > 0:
            coords.append([x, y])
    if not coords:
        return np.array([np.nan, np.nan])
    return np.nanmean(coords, axis=0)

def compute_keypoint_distance(row1, row2):
    dist = []
    for joint in ALL_KEYPOINTS:
        x1, y1 = row1.get(f'{joint}_x', 0), row1.get(f'{joint}_y', 0)
        x2, y2 = row2.get(f'{joint}_x', 0), row2.get(f'{joint}_y', 0)
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            dist.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    return np.mean(dist) if dist else np.inf

def get_initial_players_from_court_lines(segment_dir, court_kp_path, fps):
    FRAME_START = 0
    FRAME_WINDOW = 90
    df_court = pd.read_csv(court_kp_path)

    person_files = sorted(glob(os.path.join(segment_dir, '*_person*.csv')))
    all_data = {}
    for file in person_files:
        pid = os.path.basename(file).split('_person')[-1].split('.')[0]
        df = interpolate_keypoints(pd.read_csv(file), window=MAX_WINDOW)
        all_data[pid] = df

    top_counts, bottom_counts = {}, {}
    missing_kp = 0
    missing_foot = 0
    conflict_ids = 0
    total_frames = 0

    for frame in range(FRAME_START, FRAME_START + FRAME_WINDOW):
        total_frames += 1
        if frame >= len(df_court):
            continue
        try:
            top_y = np.nanmean([df_court.loc[frame, 'KP1_y'], df_court.loc[frame, 'KP2_y']])
            bottom_y = np.nanmean([df_court.loc[frame, 'KP3_y'], df_court.loc[frame, 'KP4_y']])
        except KeyError:
            missing_kp += 1
            continue
        if np.isnan(top_y) or np.isnan(bottom_y):
            missing_kp += 1
            continue

        candidates = []
        for pid, df in all_data.items():
            if frame >= len(df): continue
            cy = compute_center(df.iloc[frame])[1]
            if not np.isnan(cy):
                d_top = abs(cy - top_y)
                d_bottom = abs(cy - bottom_y)
                candidates.append((pid, d_top, d_bottom))

        if len(candidates) < 2:
            missing_foot += 1
            continue

        top_pid = min(candidates, key=lambda x: x[1])[0]
        bottom_pid = min(candidates, key=lambda x: x[2])[0]

        if top_pid == bottom_pid:
            conflict_ids += 1
            continue

        top_counts[top_pid] = top_counts.get(top_pid, 0) + 1
        bottom_counts[bottom_pid] = bottom_counts.get(bottom_pid, 0) + 1

    print(f"  [DEBUG] Total frames={total_frames}, missing_kp={missing_kp}, missing_foot={missing_foot}, conflict_ids={conflict_ids}")

    if top_counts and bottom_counts:
        top_final = max(top_counts.items(), key=lambda x: x[1])[0]
        bottom_final = max(bottom_counts.items(), key=lambda x: x[1])[0]
        if top_final != bottom_final:
            return [bottom_final, top_final], FRAME_START

    print("  [FALLBACK] Using Y-max-based foot center distance")
    centers = {}
    for pid, df in all_data.items():
        min_y = np.inf
        best_center = None
        for idx in range(FRAME_START, min(FRAME_START + FRAME_WINDOW, len(df))):
            cy = compute_center(df.iloc[idx])[1]
            if not np.isnan(cy) and cy < min_y:
                min_y = cy
                best_center = cy
        if best_center is not None:
            centers[pid] = best_center

    if len(centers) < 2:
        return [], 0

    sorted_by_y = sorted(centers.items(), key=lambda x: x[1], reverse=True)
    return [sorted_by_y[0][0], sorted_by_y[1][0]], FRAME_START


def track_players(segment_dir, player_ids, output_dir, output_video_path, start_frame, fps):
    """
    Track two selected players throughout the video.
    
    - If MANUAL_SELECT=True → trust the given start_frame, directly start tracking.
    - If MANUAL_SELECT=False → scan further from start_frame to find first valid appearances.
    """

    global MANUAL_SELECT  # read global manual/auto mode switch

    # --- Load all person keypoints ---
    person_files = sorted(glob(os.path.join(segment_dir, '*_person*.csv')))
    all_data = {}
    for file in person_files:
        pid = os.path.basename(file).split('_person')[-1].split('.')[0]
        df = interpolate_keypoints(pd.read_csv(file), window=MAX_WINDOW)
        all_data[pid] = df

    # --- Prepare tracking states ---
    history = {0: deque(maxlen=3), 1: deque(maxlen=3)}  # recent frames smoothing
    tracked = {0: [], 1: []}  # final tracked keypoints
    fail_log = []  # failed matches

    # --- Load video ---
    video_file = next((f for f in os.listdir(segment_dir) if f.endswith('.mp4')), None)
    if not video_file:
        print(f"[ERROR] No video file in: {segment_dir}")
        return

    video_path = os.path.join(segment_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    if width == 0 or height == 0 or fps == 0:
        print(f"[ERROR] Failed to read video properties: width={width}, height={height}, fps={fps}")
        return

    # --- Prepare output video writer ---
    os.makedirs(output_dir, exist_ok=True)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        print(f"[ERROR] VideoWriter init failed: {output_video_path}")
        return

    # --- Decide start_frame ---
    if MANUAL_SELECT:
        # Manual mode: trust the selected best_frame_idx directly
        print(f"[INFO] Manual mode: starting tracking directly from frame {start_frame}")
    else:
        # Auto mode: scan further until both players appear
        print(f"[INFO] Auto mode: scanning forward from initial frame {start_frame} to ensure both players appear")
        new_start_frame = start_frame
        for i, pid in enumerate(player_ids):
            if pid not in all_data:
                tracked[i].append(None)
                continue

            df = all_data[pid]
            found = False
            for offset in range(start_frame, len(df)):
                cy = compute_center(df.iloc[offset])[1]
                if not np.isnan(cy):
                    tracked[i].append(df.iloc[offset])
                    history[i].append(df.iloc[offset])
                    print(f"[INFO] Player {pid} first valid frame found at {offset}")
                    new_start_frame = max(new_start_frame, offset)
                    found = True
                    break

            if not found:
                tracked[i].append(None)
                print(f"[WARN] Player {pid} has no valid starting frame")

        start_frame = new_start_frame
        print(f"[INFO] Auto mode adjusted start_frame to {start_frame}")

    # --- Start reading video from start_frame ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Collect all candidate keypoints for this frame ---
        candidates = {}
        for pid, df in all_data.items():
            if frame_idx + start_frame < len(df):
                candidates[pid] = df.iloc[frame_idx + start_frame]

        # --- Match players ---
        matched_ids = {}
        for i in [0, 1]:
            avg_template = pd.DataFrame(history[i]).mean() if len(history[i]) > 0 else None
            valid_candidates = []

            for pid, row in candidates.items():
                if pid in matched_ids.values():
                    continue
                # If no history yet, allow initial match
                dist = compute_keypoint_distance(avg_template, row) if avg_template is not None else 0
                if dist < 200 or avg_template is None:
                    foot = compute_center(row)
                    if not np.isnan(foot[1]):
                        valid_candidates.append((pid, dist, foot[1], row))

            if valid_candidates:
                # i=0 = top player (smaller y), i=1 = bottom player (larger y)
                if i == 0:
                    best = min(valid_candidates, key=lambda x: x[2])  # choose smallest y
                else:
                    best = max(valid_candidates, key=lambda x: x[2])  # choose largest y
                best_pid = best[0]
                matched_ids[i] = best_pid
                history[i].append(best[3])
                tracked[i].append(best[3])
                cx, cy = compute_center(best[3])
                if not np.isnan(cx) and not np.isnan(cy):
                    color = (0, 0, 255) if i == 0 else (255, 0, 0)
                    cv2.rectangle(frame, (int(cx) - 30, int(cy) - 30),
                                  (int(cx) + 30, int(cy) + 30), color, 2)
            else:
                tracked[i].append(None)
                fail_log.append((frame_idx + start_frame, i))

        out.write(frame)
        frame_idx += 1

    # --- Close everything ---
    cap.release()
    out.release()

    # --- Save tracked CSV ---
    for i in [0, 1]:
        df_list = [pd.DataFrame([row]) for row in tracked[i] if row is not None]
        if df_list:
            pd.concat(df_list).to_csv(os.path.join(output_dir, f'player_{i}.csv'), index=False)

    # --- Save failed frames log ---
    with open(os.path.join(output_dir, 'match_failed_frames.txt'), 'w') as f:
        for fidx, pid in fail_log:
            f.write(f"Frame {fidx}: player {pid} match failed\n")

    # --- Post-process interpolation ---
    post_interpolate_tracked_csv(output_dir)

    if os.path.exists(output_video_path):
        print(f"[OK] Tracking video saved: {output_video_path}")
    else:
        print(f"[WARN] Failed to save video: {output_video_path}")



def process_all_segments(sports2d_root, court_kp_root, track_output_root):
    os.makedirs(track_output_root, exist_ok=True)
    segments = [d for d in os.listdir(sports2d_root) if os.path.isdir(os.path.join(sports2d_root, d))]
    for seg in sorted(segments):
        seg_path = os.path.join(sports2d_root, seg)
        print(f"Processing {seg} ...")

        video_file = next((f for f in os.listdir(seg_path) if f.endswith('.mp4')), None)
        if not video_file:
            print(" → No video found.")
            continue
        cap = cv2.VideoCapture(os.path.join(seg_path, video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        base_seg_name = seg.replace('_masked', '')
        court_kp_path = os.path.join(court_kp_root, f"{base_seg_name}_keypoints.csv")

        if MANUAL_SELECT:
            players, start_frame = manual_select_players(seg_path, preview_duration=3)
            if len(players) < 2:
                print(" → Manual selection cancelled or failed.")
                continue
        else:
            if not os.path.exists(court_kp_path):
                print(" → Court keypoints not found.")
                continue
            players, start_frame = get_initial_players_from_court_lines(seg_path, court_kp_path, fps)
            if len(players) < 2:
                print(" → Failed to detect two players automatically.")
                continue

        seg_output = os.path.join(track_output_root, seg)
        video_out_path = os.path.join(seg_output, 'player_tracking.mp4')
        track_players(seg_path, players, seg_output, video_out_path, start_frame, fps)
        print(f" → Finished {seg}\n")

def post_interpolate_tracked_csv(output_dir):
    """
    Post-process tracked CSV files to fill missing frames with interpolation.
    It reads player_0.csv and player_1.csv, applies linear interpolation for missing x/y,
    and then forward/backward fills for head and tail missing values.
    """
    for pid in [0, 1]:
        csv_path = os.path.join(output_dir, f'player_{pid}.csv')
        if not os.path.exists(csv_path):
            print(f"[WARN] {csv_path} does not exist, skipping interpolation.")
            continue

        df = pd.read_csv(csv_path)

        # Only interpolate x/y columns
        xy_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]

        # Linear interpolation for missing values (fills middle gaps)
        df[xy_cols] = df[xy_cols].interpolate(method='linear', limit_direction='both')

        # Fill head/tail missing values with nearest valid values
        df[xy_cols] = df[xy_cols].fillna(method='bfill').fillna(method='ffill')

        df.to_csv(csv_path, index=False)
        print(f"[OK] Interpolation completed and saved: {csv_path}")

def manual_select_players(segment_dir, preview_duration=3):
    """
    Manual player selection based on a single preview frame.
    
    1. Scan first `preview_duration` seconds of the video to find the frame with the most detected players.
    2. Display that frame with all candidate pids and bounding boxes.
    3. Allow mouse clicks to select 2 players:
        - Left click selects the nearest pid (max 2, cannot overwrite).
        - Press 'r' to reset selection.
        - Press 'c' to confirm when 2 players are selected.
        - Press ESC to cancel and return [].
    4. Returns a list of 2 selected pids, or [] if cancelled.
    """

    # --- Load video and fps ---
    video_file = next((f for f in os.listdir(segment_dir) if f.endswith('.mp4')), None)
    if not video_file:
        print(f"[ERROR] No video found in {segment_dir}")
        return []
    video_path = os.path.join(segment_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), fps * preview_duration))

    # --- Load all person csv files ---
    person_files = sorted(glob(os.path.join(segment_dir, '*_person*.csv')))
    all_data = {}
    for f in person_files:
        pid = os.path.basename(f).split('_person')[-1].split('.')[0]
        df = pd.read_csv(f)
        all_data[pid] = df

    # --- Function to compute foot center for a pid at a given frame ---
    def compute_center(row):
        parts = ['LAnkle', 'RAnkle']
        coords = []
        for joint in parts:
            x, y = row.get(f'{joint}_x', 0), row.get(f'{joint}_y', 0)
            if x > 0 and y > 0:
                coords.append((x, y))
        if not coords:
            return None
        coords = np.array(coords)
        return np.mean(coords[:,0]), np.mean(coords[:,1])

    # --- Find the best frame with the most players ---
    best_frame_idx = 0
    best_count = 0
    frame_candidate_map = {}  # frame_idx -> [(pid,cx,cy)]

    for idx in range(total_frames):
        candidates = []
        for pid, df in all_data.items():
            if idx >= len(df): 
                continue
            c = compute_center(df.iloc[idx])
            if c is not None:
                candidates.append((pid, *c))
        frame_candidate_map[idx] = candidates
        if len(candidates) > best_count:
            best_count = len(candidates)
            best_frame_idx = idx

    if best_count == 0:
        print("[ERROR] No valid players detected in first few seconds.")
        return []
    
    
    print(f"[INFO] Displaying best preview frame: {best_frame_idx}")

    # --- Extract best frame image ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
    ret, preview_frame = cap.read()
    cap.release()
    if not ret:
        print(f"[ERROR] Cannot read frame {best_frame_idx}")
        return []

    # --- Get candidates on best frame ---
    candidates = frame_candidate_map[best_frame_idx]
    centers = {pid: (int(cx), int(cy)) for pid, cx, cy in candidates}

    selected_ids = []

    # --- Mouse callback ---
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_ids
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(selected_ids) >= 2:
                print("[INFO] Already selected 2 players. Press 'r' to reset.")
                return
            min_pid, min_dist = None, 1e9
            for pid, (cx, cy) in centers.items():
                d = np.hypot(cx - x, cy - y)
                if d < min_dist:
                    min_pid, min_dist = pid, d
            if min_pid and min_pid not in selected_ids:
                selected_ids.append(min_pid)
                print(f"[INFO] Selected PID: {min_pid}")

    # --- OpenCV setup ---
    cv2.namedWindow("Manual Player Selection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Manual Player Selection", mouse_callback)

    confirmed = False

    while True:
        frame_display = preview_frame.copy()


        box_size = 60  # bigger box
        for pid, (cx, cy) in centers.items():
            color = (255, 255, 255)
            if pid in selected_ids:
                color = (0, 255, 0)  # green highlight
            cv2.rectangle(frame_display,
                        (cx - box_size, cy - box_size),
                        (cx + box_size, cy + box_size),
                        color, 4)
            cv2.putText(frame_display, f"PID:{pid}", (cx - 30, cy - box_size - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        # --- create hint area (white background) ---
        height, width = frame_display.shape[:2]
        hint_area = np.ones((80, width, 3), dtype=np.uint8) * 255  # white bar

        # write hints on hint_area
        hint_text = "Click 2 players | [r] Reset | [c] Confirm | [ESC] Cancel"
        cv2.putText(hint_area, hint_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        cv2.putText(hint_area, f"Selected: {len(selected_ids)}/2", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # combine frame + hint_area
        frame_with_hint = np.vstack([frame_display, hint_area])
        cv2.imshow("Manual Player Selection", frame_with_hint)


        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            print("[INFO] Manual selection cancelled.")
            break
        elif key == ord('r'):
            selected_ids = []
            print("[INFO] Selection reset.")
        elif key == ord('c'):
            if len(selected_ids) == 2:
                confirmed = True
                print(f"[INFO] Confirmed selection: {selected_ids}")
                break
            else:
                print("[WARN] You must select 2 players before confirming.")

    cv2.destroyWindow("Manual Player Selection")
    return (selected_ids if confirmed else [], best_frame_idx)

if __name__ == "__main__":
    process_all_segments(
        sports2d_root='sports2d_results',
        court_kp_root='keypoints_data',
        track_output_root='selected_players'
    )
