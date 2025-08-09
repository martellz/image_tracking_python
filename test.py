import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from src.tracker.template_matching_based_tracker import Template_matching_based_tracker
from src.detector.orb_detector import Orb_detector


lk_params = dict(winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ShiTomasi角点检测的参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

near = 10
far = 100000
fovy = 45.0 * np.pi / 180   # 45 in radian. field of view vertical


def draw_tracked_locations(frame: np.ndarray, tracker: Template_matching_based_tracker):
  for i in range(tracker.nx * tracker.ny):
    x1, y1 = tracker.f.transform_point(tracker.m[2 * i], tracker.m[2 * i + 1])
    cv2.circle(frame, (int(x1), int(y1)), 3, (0, 255, 0), 1)

def draw_reproject_location(frame: np.ndarray, tracker: Template_matching_based_tracker, projectionTransform: np.ndarray, modelViewMatrix: np.ndarray):
    # modelViewMatrix: 4x4
    # projectionTransform: 3x3
    p = np.array(tracker.m).reshape(-1, 2)
    p = np.concatenate([p, np.zeros((p.shape[0], 1)), np.ones((p.shape[0], 1))], axis=1)
    p = np.matmul(modelViewMatrix, p.T).T[:, :3]
    p = np.matmul(projectionTransform, p.T).T
    p = p / p[:, 2:]
    p = p.astype(np.int32)
    for i in range(tracker.nx * tracker.ny):
        cv2.circle(frame, (p[i, 0], p[i, 1]), 3, (255, 0, 0), 1)

def homography_to_model_view_matrix(H):
    K = projectionTransform
    KInv = np.linalg.inv(K)

    _KInvH = np.matmul(KInv, H)
    norm1 = np.linalg.norm(_KInvH[:, 0])
    norm2 = np.linalg.norm(_KInvH[:, 1])
    tnorm = (norm1 + norm2) / 2.0

    R = np.zeros((3, 3))
    R[:, 0] = _KInvH[:, 0] / norm1
    R[:, 1] = _KInvH[:, 1] / norm2
    R[:, 2] = np.cross(R[:, 0], R[:, 1])

    norm3 = np.sqrt(np.sum(R[:, 2] ** 2))
    R[:, 2] = R[:, 2] / norm3

    tran = _KInvH[:, 2] / tnorm

    M = np.zeros((4, 4))
    M[:3, :3] = R
    M[:3, 3] = tran
    M[3, 3] = 1.0

    return M


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='model.bmp')
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--tracker_file', type=str, default='tracker.npy')
    parser.add_argument('--save_video', type=str, default=None)

    args = parser.parse_args()
    print(args.image)

    is_detected = False

    image = cv2.imread(args.image)
    detector = Orb_detector(image, 1000)
    tracker = Template_matching_based_tracker()

    f = None
    projectionTransform = None

    tracker_file = Path(args.tracker_file)
    if tracker_file.exists():
        tracker.load(args.tracker_file)
    else:
        # u_corner = [184, 426, 438, 183]
        # v_corner = [241, 241, 440, 443]
        # gray = cv2.imread(args.image, 0)
        # tracker.learn(gray, 5, 40, 20, 20,
        #             u_corner[0], v_corner[1], u_corner[2], v_corner[2],
        #             40, 40, 10000)
        # tracker.save('tracker.npy')
        print('tracker learning is not implemented!')
        exit(1)

    tracker.init()

    if args.video is not None:
        capture = cv2.VideoCapture(args.video)
    else:
        capture = cv2.VideoCapture(0)


    out = None

    # virtual camera first frame ground truth, tested
    # tracker.init_with_detector(184, 228, 437, 242, 438, 440, 182, 441)
    # tracker.init_with_detector(190, 39, 503, 40, 503, 524, 190, 525)

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
            H, dst = detector.detect(gray)
            dst = dst.squeeze()
            print(H, dst)

            if tracker.init_with_detector(*dst[0], *dst[3], *dst[2], *dst[1]):
                result = tracker.track(gray)
                if result:
                    draw_tracked_locations(frame, tracker)

                    tracker.u = tracker.u.astype(np.int32)
                    cv2.line(frame, (tracker.u[0], tracker.u[1]), (tracker.u[2], tracker.u[3]), (0, 255, 0), 2)
                    cv2.line(frame, (tracker.u[2], tracker.u[3]), (tracker.u[4], tracker.u[5]), (0, 255, 0), 2)
                    cv2.line(frame, (tracker.u[4], tracker.u[5]), (tracker.u[6], tracker.u[7]), (0, 255, 0), 2)
                    cv2.line(frame, (tracker.u[6], tracker.u[7]), (tracker.u[0], tracker.u[1]), (0, 255, 0), 2)

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
            result = tracker.track(gray)
            if result:
                draw_tracked_locations(frame, tracker)

                tracker.u = tracker.u.astype(np.int32)
                cv2.line(frame, (tracker.u[0], tracker.u[1]), (tracker.u[2], tracker.u[3]), (0, 255, 0), 2)
                cv2.line(frame, (tracker.u[2], tracker.u[3]), (tracker.u[4], tracker.u[5]), (0, 255, 0), 2)
                cv2.line(frame, (tracker.u[4], tracker.u[5]), (tracker.u[6], tracker.u[7]), (0, 255, 0), 2)
                cv2.line(frame, (tracker.u[6], tracker.u[7]), (tracker.u[0], tracker.u[1]), (0, 255, 0), 2)

                modelViewMatrix = homography_to_model_view_matrix(tracker.f.homo)
                for i in range(tracker.nx * tracker.ny):
                    x1, y1 = tracker.f.transform_point(tracker.m[2 * i], tracker.m[2 * i + 1])
                    cv2.circle(frame, (int(x1), int(y1)), 3, (0, 255, 0), 1)

                # re-project tracker.m to frame to visualize the error of homography_to_model_view_matrix
                # draw_reproject_location(frame, tracker, projectionTransform, modelViewMatrix)

                # use tracker.m as optical flow tracking points
                # if last_frame is not None and last_homo is not None:
                #     kps = []
                #     for i in range(tracker.nx * tracker.ny):
                #         x1, y1 = tracker.f.transform_point(tracker.m[2 * i], tracker.m[2 * i + 1])
                #         kps.append([x1, y1])
                #     kps = np.array(kps, dtype=np.float32).reshape(-1, 1, 2)
                #     p1, st, err = cv2.calcOpticalFlowPyrLK(last_frame, gray, kps, None, **lk_params)
                #     good_new = p1[st == 1]
                #     good_old = kps[st == 1]
                #     H1, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 2.0)
                #     f2 = Homography(H1)
                #     kps = kps.squeeze()
                #     for i in range(tracker.nx * tracker.ny):
                #         x1, y1 = f2.transform_point(kps[i, 0], kps[i, 1])
                #         cv2.circle(frame, (int(x1), int(y1)), 3, (0, 0, 255), 1)

                # use cv2.goodFeaturesToTrack to get tracking points in target region
                # if last_frame is not None and last_homo is not None:
                #     p1, st, err = cv2.calcOpticalFlowPyrLK(last_frame, gray, p0, None, **lk_params)
                #     good_new = p1[st == 1]
                #     good_old = p0[st == 1]

                #     if (good_new.shape[0] >= 10):
                #         H1, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 2.0)
                #         f1 = Homography(last_homo)
                #         f2 = Homography(H1)
                #         kps = []
                #         for i in range(tracker.nx * tracker.ny):
                #             x1, y1 = f1.transform_point(tracker.m[2 * i], tracker.m[2 * i + 1])
                #             kps.append([x1, y1])
                #         kps = np.array(kps, dtype=np.float32).reshape(-1, 2)
                #         for i in range(tracker.nx * tracker.ny):
                #             x1, y1 = f2.transform_point(kps[i, 0], kps[i, 1])
                #             cv2.circle(frame, (int(x1), int(y1)), 3, (0, 0, 255), 1)

                #         p0 = good_new.reshape(-1, 1, 2)
                #     else:
                #         last_frame = None
                #         last_homo = None

                #         print('re-init OF')
                #         mask = (cv2.warpPerspective(image, tracker.f.homo, (gray.shape[1], gray.shape[0]))[..., 0] > 0).astype(np.uint8)
                #         p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
                # else:
                #     # init optical flow
                #     print('init OF')
                #     mask = (cv2.warpPerspective(image, tracker.f.homo, (gray.shape[1], gray.shape[0]))[..., 0] > 0).astype(np.uint8)
                #     p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)

                # last_homo = tracker.f.homo.copy()
                # last_frame = gray.copy()
            else:
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