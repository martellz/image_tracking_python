import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from src.tracker.multi_templates_tracker import Multi_templates_matching_based_tracker
from src.detector.orb_detector import Orb_detector

fovy = 45.0 * np.pi / 180   # 45 in radian. field of view vertical



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='model.bmp')
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--tracker_file', type=str, default='tracker.npy')
    parser.add_argument('--tracker_json', type=str, default='tracker.json')
    parser.add_argument('--save_video', type=str, default=None)
    parser.add_argument('--learn', action='store_true')

    args = parser.parse_args()

    is_detected = False

    image = cv2.imread(args.image)
    detector = Orb_detector(image, 1000)
    tracker = Multi_templates_matching_based_tracker()

    f = None
    projectionTransform = None

    tracker_file = Path(args.tracker_file) if args.tracker_file is not None else Path('multi_tracker.npy')
    if args.learn:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if tracker.learn(gray, 5, 80, 20, 20, 80, 80, 10000):
            tracker.save(args.tracker_file, args.tracker_json)
            print('Learning succeeded!')
            exit(0)
        else:
            print('Learning failed!')
            exit(1)

    if tracker_file.exists():
        tracker.load(args.tracker_file)
    else:
        print('Please learn tracker first!')
        exit(1)

    if args.video is not None:
        capture = cv2.VideoCapture(args.video)
    else:
        capture = cv2.VideoCapture(0)

    out = None
    last_frame = None
    last_homo = None
    while capture.isOpened():
        t0 = time.time()
        ret, frame = capture.read()
        if not ret:
            print('capture.read() failed!')
            break

        h, w = frame.shape[:2]
        if h > 720:
            frame = cv2.resize(frame, (int(720 * w / h), 720))

        if args.save_video is not None and out is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(args.save_video, fourcc, 30.0, (frame.shape[1], frame.shape[0]))

        if f is None or projectionTransform is None:
            f = frame.shape[0] / 2 / np.tan(fovy / 2)
            projectionTransform = np.array([
                [f, 0, frame.shape[1]],
                [0, f, frame.shape[0]],
                [0, 0, 1]
            ])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not is_detected:
            try:
                H, dst = detector.detect(gray)
            except:
                continue
            dst = dst.squeeze()
            print(H, dst)

            if tracker.init_with_detector(*dst[0], *dst[3], *dst[2], *dst[1]):
                result = tracker.track(frame)
                if result:
                    is_detected = True
                else:
                    is_detected = False  # not necessary

                    # use dst to draw line
                    dst = dst.astype(np.int32)
                    cv2.line(frame, (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]), (255, 0, 0), 2)
                    cv2.line(frame, (dst[1][0], dst[1][1]), (dst[2][0], dst[2][1]), (255, 0, 0), 2)
                    cv2.line(frame, (dst[2][0], dst[2][1]), (dst[3][0], dst[3][1]), (255, 0, 0), 2)
                    cv2.line(frame, (dst[3][0], dst[3][1]), (dst[0][0], dst[0][1]), (255, 0, 0), 2)
        else:
            # track
            try:
                result = tracker.track(frame)
            except:
                last_homo = None
                last_frame = None
                is_detected = False
                continue
            if not result:
                last_homo = None
                last_frame = None
                is_detected = False

        t1 = time.time()
        print('time: ', t1 - t0)
        cv2.imshow('frame', frame)
        if out is not None:
            out.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    if out is not None:
        out.release()