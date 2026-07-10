from ultralytics import YOLO
import cv2
import os
import csv


ROOT_FOLDER = r"G:\Sac"                                  #Enter correct path here
MODEL_PATH = "yolo11n.pt"
OUTPUT_NAME = "vehicle_density.csv"
FAIL_LOG = "failed_videos.txt"

vehicle_classes = [2, 3, 5, 7]

model = YOLO(MODEL_PATH)

camera_results = {}

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Cannot open video")

    total = 0
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=vehicle_classes, verbose=False)
        total += len(results[0].boxes)
        frames += 1

    cap.release()

    return total / frames if frames > 0 else 0


for root, dirs, files in os.walk(ROOT_FOLDER):

    for file in files:

        if not file.lower().endswith(".mp4"):
            continue

        video_path = os.path.join(root, file)

        rel = os.path.relpath(video_path, ROOT_FOLDER)
        camera = rel.split(os.sep)[0]
        subfolder = os.path.basename(root)

        if camera not in camera_results:
            camera_results[camera] = []

        try:
            avg = process_video(video_path)

            camera_results[camera].append([
                file,
                subfolder,
                round(avg, 4)
            ])

            print(f"[OK] {file} | {avg:.3f}")

        except Exception as e:
            print(f"[FAIL] {video_path} -> {e}")

            with open(FAIL_LOG, "a") as f:
                f.write(video_path + "\n")


for camera, rows in camera_results.items():

    camera_path = os.path.join(ROOT_FOLDER, camera)
    os.makedirs(camera_path, exist_ok=True)

    csv_path = os.path.join(camera_path, OUTPUT_NAME)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "Subfolder", "Avg_Vehicles_Per_Frame"])
        writer.writerows(rows)

    print(f"[CSV SAVED] {csv_path}")

print("DONE")