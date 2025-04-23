import cv2
import numpy as np
import csv
import os
import torch
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from utils import scene_detect

SEGMENT_DIR = "output_segments"
BALL_MODEL_PATH = r"models\model_best.pt"
BOUNCE_MODEL_PATH = r"models\ctb_regr_bounce.cbm"

def read_video(path_video):
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps, width, height

def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img

def load_kps_and_matrix(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    folder = os.path.dirname(video_path)
    matrix_path = os.path.join(folder, f"{base_name}_matrixes.npy")
    kps_path = os.path.join(folder, f"{base_name}_keypoints.csv")

    matrices = np.load(matrix_path, allow_pickle=True)
    keypoints = []
    with open(kps_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            kps = []
            for i in range(14):
                x = row[1 + i * 2]
                y = row[1 + i * 2 + 1]
                if x == '' or y == '':
                    kps.append(None)
                else:
                    pt = np.array([float(x), float(y)])
                    kps.append(pt)
            keypoints.append(kps if any(k is not None for k in kps) else None)
    return matrices, keypoints

def process_and_save(frames, fps, scenes, bounces, ball_track, homography_matrices, kps_court, persons_top, persons_bottom, width_original, height_original, output_csv_path, output_video_path):
    width_rate = width_original / 1280
    height_rate = height_original / 720
    csv_data = []
    imgs_res = []
    width_minimap = 166
    height_minimap = 350
    is_track = [x is not None for x in homography_matrices]
    for num_scene in range(len(scenes)):
        sum_track = sum(is_track[scenes[num_scene][0]:scenes[num_scene][1]])
        len_track = scenes[num_scene][1] - scenes[num_scene][0]
        eps = 1e-15
        scene_rate = sum_track/(len_track+eps)
        if (scene_rate > 0.5):
            court_img = get_court_img()
            for i in range(scenes[num_scene][0], scenes[num_scene][1]):
                img_res = frames[i]
                inv_mat = homography_matrices[i]
                if ball_track[i][0]:
                    img_res = cv2.circle(img_res , (int(ball_track[i][0]), int(ball_track[i][1])), radius=5,
                                             color=(0, 255, 0), thickness=2)
                if kps_court[i] is not None:
                    for pt in kps_court[i]:
                        if pt is not None:
                            x, y = int(pt[0]), int(pt[1])
                            img_res = cv2.circle(img_res, (x, y), radius=0, color=(0, 0, 255), thickness=10)
                height, width, _ = img_res.shape
                if i in bounces and inv_mat is not None:
                    ball_point = ball_track[i]
                    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                                       radius=0, color=(0, 255, 255), thickness=50)
                minimap = court_img.copy()
                persons = persons_top[i] + persons_bottom[i]
                for j, person in enumerate(persons):
                    if len(person[0]) > 0:
                        person_bbox = list(person[0])
                        img_res = cv2.rectangle(img_res, (int(person_bbox[0]), int(person_bbox[1])),
                                                (int(person_bbox[2]), int(person_bbox[3])), [255, 0, 0], 2)
                person_coor = [[None, None], [None, None]]
                if len(persons_top[i]) > 0:
                    pt = persons_top[i][0][1]
                    person_coor[0] = [pt[0]*width_rate, pt[1]*height_rate]
                if len(persons_bottom[i]) > 0:
                    pt = persons_bottom[i][0][1]
                    person_coor[1] = [pt[0]*width_rate, pt[1]*height_rate]
                bounce_flag = int(i in bounces)
                ball_x = float(ball_track[i][0] * width_rate) if ball_track[i][0] else None
                ball_y = float(ball_track[i][1] * height_rate) if ball_track[i][1] else None
                csv_data.append([i, bounce_flag, *person_coor[0], *person_coor[1], ball_x, ball_y])
                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap
                imgs_res.append(img_res)
        else:
            imgs_res = imgs_res + frames[scenes[num_scene][0]:scenes[num_scene][1]]
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Bounce", "Person1_X", "Person1_Y", "Person2_X", "Person2_Y", "Ball_Frame_X", "Ball_Frame_Y"])
        writer.writerows(csv_data)
    height, width = imgs_res[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for frame in imgs_res:
        out.write(frame)
    out.release()

def run_all_segments():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for file in os.listdir(SEGMENT_DIR):
        if file.endswith(".mp4"):
            input_path = os.path.join(SEGMENT_DIR, file)
            base_name = os.path.splitext(file)[0]
            output_video_path = os.path.join(SEGMENT_DIR, f"{base_name}_annotated.mp4")
            output_csv_path = os.path.join(SEGMENT_DIR, f"{base_name}_output_coordinates.csv")

            print(f"Processing {input_path}")
            frames, fps, width_original, height_original = read_video(input_path)
            frames = [cv2.resize(f, (1280, 720)) for f in frames]
            scenes = scene_detect(input_path)

            print('ball detection')
            ball_detector = BallDetector(BALL_MODEL_PATH, device)
            ball_track = ball_detector.infer_model(frames)

            print('loading saved court detection results')
            homography_matrices, kps_court = load_kps_and_matrix(input_path)

            print('person detection')
            person_detector = PersonDetector(device)
            persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)

            print('bounce detection')
            bounce_detector = BounceDetector(BOUNCE_MODEL_PATH)
            x_ball = [x[0] for x in ball_track]
            y_ball = [x[1] for x in ball_track]
            bounces = bounce_detector.predict(x_ball, y_ball)

            process_and_save(frames, fps, scenes, bounces, ball_track, homography_matrices, kps_court,
                             persons_top, persons_bottom, width_original, height_original,
                             output_csv_path, output_video_path)
            print(f"Saved: {output_video_path}\n")

if __name__ == '__main__':
    run_all_segments()