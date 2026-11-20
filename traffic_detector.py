from __future__ import print_function
import cv2 as cv
import numpy as np



frame_number = 0
prev_gray = None
prev_pts = None
MAX_FRAMES = 200

def brightness_variation(video_path):
    cap = cv.VideoCapture(video_path)
    vals = []
    i = 0

    if not cap.isOpened():
        return None

    while i < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        vals.append(hsv[:, :, 2].mean())
        i += 1

    cap.release()

    if len(vals) < 2:
        return None

    return np.std(np.diff(vals))


def collect_raw_areas_from_video(video_path, is_night, sample_step=5, max_frames=200):
    cap = cv.VideoCapture(video_path)

    areas_night = []
    areas_day = []

    frame_id = 0

    if is_night:
        backSub = cv.createBackgroundSubtractorKNN(dist2Threshold=800.0, detectShadows=False)

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        lower_yellow = np.array([20, 50, 220])
        upper_yellow = np.array([35, 120, 255])

        while frame_id < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % sample_step == 0:
                fgMask = backSub.apply(frame)
                fgMask = cv.inRange(fgMask, 200, 255)

                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                mask_white = cv.inRange(hsv, lower_white, upper_white)
                mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
                headlightMask = cv.bitwise_and(cv.bitwise_or(mask_white, mask_yellow), fgMask)


                contours, _ = cv.findContours(headlightMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    area = cv.contourArea(c)
                    if area > 10:
                        areas_night.append(area)

            frame_id += 1

    else:
        backSub = cv.createBackgroundSubtractorKNN(dist2Threshold=800.0, detectShadows=False)

        while frame_id < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % sample_step == 0:
                fgMask = cv.inRange(backSub.apply(frame), 200, 255)

                contours, _ = cv.findContours(fgMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    area = cv.contourArea(c)
                    if area > 10:
                        areas_day.append(area)

            frame_id += 1

    cap.release()
    return areas_night, areas_day


def process_single_video(input_path, min_area, max_area, is_night, resize_factor=1.0):
    global frame_number, prev_gray, prev_pts

    capture = cv.VideoCapture(input_path)
    occupancies = []

    ret, frame = capture.read()
    if not ret:
        return

    capture.set(cv.CAP_PROP_POS_FRAMES, 0)

    prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_number = 0

    if is_night:
        backSub = cv.createBackgroundSubtractorKNN(dist2Threshold=800.0, detectShadows=False)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        lower_yellow = np.array([20, 50, 220])
        upper_yellow = np.array([35, 120, 255])

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            if resize_factor != 1.0:
                frame = cv.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

            fgMask = backSub.apply(frame)
            fgMask = cv.inRange(fgMask, 200, 255)

            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask_white = cv.inRange(hsv, lower_white, upper_white)
            mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
            headlightMask = cv.bitwise_and(cv.bitwise_or(mask_white, mask_yellow), fgMask)

            contours, _ = cv.findContours(headlightMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            filtered_contours = [
                c for c in contours
                if min_area <= cv.contourArea(c) <= max_area
            ]

            occupancies.append(len(filtered_contours))
            frame_number += 1

    else:
        backSub = cv.createBackgroundSubtractorKNN(dist2Threshold=800.0, detectShadows=False)

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            if resize_factor != 1.0:
                frame = cv.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)

            fgMask1 = cv.inRange(backSub.apply(frame), 200, 255)
            contours, _ = cv.findContours(fgMask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            filtered_contours = [
                c for c in contours
                if min_area <= cv.contourArea(c) <= max_area
            ]

            occupancies.append(len(filtered_contours))
            frame_number += 1

    capture.release()

    if occupancies:
        return sum(occupancies) / len(occupancies)
    return 0