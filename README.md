# Sports Video Analysis Pipeline

This project provides a complete pipeline for processing sports videos (e.g., badminton or tennis), including rally segmentation, court masking, player tracking, and hit-point detection. Each stage is encapsulated in a single `*_onestep.py` script to simplify batch processing.

---

## Input Preparation

Place the raw full-length videos into the following folder:

```
input_videos/
```

---

## Processing Steps

### 1. Rally Cutting & Court Keypoints

Run:
```bash
python cutting_onestep.py
```

Functionality:
- Automatically split full-length videos into short rally clips
- Output segmented clips to `output_segments/`
- Generate for each clip:
  - Court keypoints (`*_keypoints.csv`)
  - Homographic transformation matrices (`*_homography.npy`)
- The original video after cutting is moved to `processed_videos/`

---

### 2. Court Masking (Optional)

Run:
```bash
python mask_onestep.py
```

Functionality:
- Apply court-based masks to remove irrelevant regions (e.g., spectators, referees)
- Output masked clips as `*_with_edges.mp4` files inside `output_segments/`

---

### 3. Player Detection with Sports2D

Run:
```bash
python sports2D_onestep.py
```

Functionality:
- Detect and track players in each rally clip using the Sports2D model
- Save results to `sports2d_results/` as `.trc.csv` files (one per player)

---

### 4. Postprocessing Sports2D Tracking (Work in Progress)

Run:
```bash
python postprocess_onestep.py
```

Functionality:
- Refine Sports2D tracking data to correct ID swaps
- Assign consistent global IDs across frames
- Identify the two main players for downstream analysis

---

### 5. Player & Ball Detection

Run:
```bash
python main_onestep.py
```

Functionality:
- Detect ball bounce locations
- Extract per-frame coordinates of the two selected players

---

### 6. Merge Data for Hit Point Extraction

Run:
```bash
python csvmerge_onestep.py
```

Functionality:
- Merge outputs from keypoints, Sports2D, and player-ball detection
- Produce unified CSV files containing hit point information, including:
  - Frame index
  - Ball bounce position
  - Player involved

---

## Output Summary

| Folder | Description |
|--------|-------------|
| `input_videos/`        | Raw full-length match videos (input) |
| `processed_videos/`    | Original videos after cutting |
| `output_segments/`     | Segmented rally clips and intermediate results |
| `sports2d_results/`    | Sports2D tracking results (per player `.trc.csv`) |
| `*_keypoints.csv`      | Court keypoints per rally |
| `*_homography.npy`     | Homography matrix per rally |
| Final CSVs             | Integrated data with player and hit point annotations |

---

## Notes

- All `*_onestep.py` scripts support batch-processing all relevant files in their folders.
- You can customize court masking and player selection logic in `mask_onestep.py` and `postprocess_onestep.py` if needed.

