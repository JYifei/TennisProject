# Sports Video Analysis Pipeline

This project provides a complete end-to-end pipeline for analyzing racket sports videos (e.g., badminton, tennis). It includes rally segmentation, masking, player and ball tracking, hit-point detection, and trajectory visualization. Each stage is handled by a dedicated `*_onestep.py` script.

---

## Input Preparation

Place raw full-length match videos into:
```
input_videos/
```

---

## Processing Pipeline

### 1. Rally Cutting & Court Keypoints
```bash
python cutting_onestep.py
```
- Splits full videos into rallies
- Outputs to `output_segments/`
- Saves:
  - `*_keypoints.csv` → in `keypoints_data/`
  - `*_matrixes.npy` → in `matrix_data/`
- Original video is moved to `processed_videos/`

---

### 2. Court Masking (Optional)
```bash
python mask_onestep.py
```
- Applies court masks to each rally
- Outputs masked videos to `masked_segments/`

---

### 3. Sports2D Player Detection
```bash
python sports2D_onestep.py
```
- Runs Sports2D on each masked clip
- Saves results to `sports2d_results/segment_name/`
- Converts `.trc` to `.csv`

---

### 4. Postprocess Sports2D Tracking
```bash
python postprocess_onestep.py
```
- Filters noisy tracking results
- Matches player IDs frame-by-frame using position and color features
- Outputs consistent tracking results to `postprocessed_sports2D/`
- Save video with bounding boxes to `postprocessed_sports2D_annotated/`

---

### 5. Player Selection
```bash
python identify_player_os.py
```
- Selects the two main players in each clip
- Based on proximity to baseline + track length
- Saves filtered `.csv` for each player to `selected_players/`

---

### 6. Ball & Player Detection
```bash
python main_onestep.py
```
- Detects ball position and bounce points
- Tracks player bounding boxes
- Outputs per-frame result video + coordinates CSV
- Saves to `detected_segments/`

---

### 7. Hit Point Detection & Court Projection
```bash
python csvmerge_onestep.py
```
- Merges player and ball data
- Identifies hitting frames (based on wrist-ball distance variation)
- Projects hit points onto court image using homography
- Translates pixel coordinates into real world meters(use the midpoint of the net as original)
- Outputs:
  - `*_impact_coordinates.csv`
  - `*_court_reference.png`
  - `*_distance_plot.png`
- Saves to `final_result/`

---

## Output Structure Summary

| Folder                    | Description |
|--------                   |-------------|
| `input_videos/`           | Raw input videos |
| `output_segments/`        | Rally clips after cutting |
| `masked_segments/`        | Videos with masked background |
| `sports2d_results/`       | Raw Sports2D output (`.trc.csv`) |
| `postprocessed_sports2D/` | Tracking results with global IDs |
| `selected_players/`       | Cleaned `.csv` for the two players |
| `detected_segments/`      | Frame-level ball + player coordinates |
| `matrix_data/`            | Homography matrices per rally |
| `keypoints_data/`         | Detected keypoints per rally |
| `final_result/`           | Final court projection & CSV hits |
| `postprocessed_sports2D_annotated/` | Video with bounding boxes and player ids |
---

## Notes
- All `*_onestep.py` scripts support batch-processing.
- Models used:
  - Sports2D for 2D pose estimation - https://github.com/davidpagnon/Sports2D
  - TrackNet ball detector (`models/model_best.pt`) - https://github.com/yastrebksv/TennisProject
  - CatBoostRegressor bounce detector (`models/ctb_regr_bounce.cbm`) - https://github.com/yastrebksv/TennisProject
- You can tune ID-matching logic, masking shape, and hit-point criteria independently.

---

## Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

Ensure model files are placed in the `models/` directory.
