from typing import Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


from ..tracker.template_matching_based_tracker import Template_matching_based_tracker
from ..homography.homography_estimator import estimate_homography

class POT280Data:
    # POT280Data contains folder named from 'V01' to 'Vxx', here we use 'V01' to 'V10'
    # in each folder, there are several videos, for example, 'V01_1.avi', 'V01_2.avi', 'V01_3.avi'
    # we only use 'Vxx_1.avi' in each folder
    # the annotation files are named 'Vxx_1_flag.txt', 'Vxx_1_gt_homography.txt', 'Vxx_1_gt_points.txt'
    # the annotation files are in another folder

    def __init__(self, dataset_dir, annotation_dir):
        self.dataset_dir = Path(dataset_dir)
        self.annotation_dir = Path(annotation_dir)

    def _get_video_path(self, video_id):
        video_path = self.dataset_dir / 'V{:02d}'.format(video_id) / 'V{:02d}_1.avi'.format(video_id)
        return video_path

    def _get_annotation_path(self, video_id):
        # flag = self.annotation_dir / 'V{:02d}_1_flag.txt'.format(video_id)    # flag is useless
        homography = self.annotation_dir / 'V{:02d}_1_gt_homography.txt'.format(video_id)
        points = self.annotation_dir / 'V{:02d}_1_gt_points.txt'.format(video_id)
        return homography, points

    def get_video_with_annotation(self, video_id):
        video_path = self._get_video_path(video_id)
        homography, points = self._get_annotation_path(video_id)
        return video_path, homography, points


class VideoData:
    index: int = 0
    video: cv2.VideoCapture
    warpHomo: np.ndarray = np.identity(3, dtype=float)
    tracker: Template_matching_based_tracker = Template_matching_based_tracker()

    def __init__(self, video_path: Path, homography_path: Path, points_path: Path):
        self.video_path = video_path
        self.homography_path = homography_path
        self.points_path = points_path
        self.checkpoint_path = video_path.parent / 'checkpoint.npy'

        self.video = cv2.VideoCapture(video_path.as_posix())
        self.homography = np.loadtxt(homography_path.as_posix())
        self.points = np.loadtxt(points_path.as_posix())

    def _read(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        ret, frame = self.video.read()
        if not ret:
            return None
        assert frame is not None and self.index < len(self.points), 'frame is None or index out of range'

        self.index += 1
        homography = self.homography[self.index]
        points = self.points[self.index]
        return frame, homography, points

    @property
    def valid(self) -> bool:
        return self.video_path.exists() and self.homography_path.exists() and self.points_path.exists()

    def init_first_frame(self) -> bool:
        ret, first_frame = self.video.read()
        assert ret, 'Cannot read video from {}'.format(self.video_path)

        corner_points = self.points[0]  # [x0, y0, x1, y1, x2, y2, x3, y3]

        # method 1: select a axis-aligned area in target as the initial template
        # x_left = max(corner_points[0], corner_points[6])
        # x_right = min(corner_points[2], corner_points[4])
        # y_top = max(corner_points[1], corner_points[3])
        # y_bottom = min(corner_points[5], corner_points[7])

        # template = first_frame[y_top:y_bottom, x_left:x_right]

        # tracker.learn(template, 5, max_motion, 20, 20,
        #             corner_points[0], corner_points[3], corner_points[4], corner_points[5],
        #             max_motion, max_motion, 10000)

        # method 2: warp the target to a rectangle as the initial template
        # I simply use the aabb's shape as the template's shape
        x_left = int(min(corner_points[0], corner_points[6]))
        x_right = int(max(corner_points[2], corner_points[4]))
        y_top = int(min(corner_points[1], corner_points[3]))
        y_bottom = int(max(corner_points[5], corner_points[7]))

        homo = estimate_homography(np.array(corner_points), np.array([x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom]))
        assert (homo is not None)
        self.warpHomo = homo

        # comment these part to see debug plot
        # debug_image = first_frame.copy()
        # aabb = np.array([x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom])
        # # for i in range(4):
        # #     cv2.circle(debug_image, (int(corner_points[2*i]), int(corner_points[2*i+1])), 1, (0, 255, 0), 2)
        # #     cv2.circle(debug_image, (int(aabb[2*i]), int(aabb[2*i+1])), 1, (255, 0, 0), 2)
        # cv2.line(debug_image, (int(corner_points[0]), int(corner_points[1])), (int(corner_points[2]), int(corner_points[3])), (255, 0, 0), 2)
        # cv2.line(debug_image, (int(corner_points[2]), int(corner_points[3])), (int(corner_points[4]), int(corner_points[5])), (255, 0, 0), 2)
        # cv2.line(debug_image, (int(corner_points[4]), int(corner_points[5])), (int(corner_points[6]), int(corner_points[7])), (255, 0, 0), 2)
        # cv2.line(debug_image, (int(corner_points[6]), int(corner_points[7])), (int(corner_points[0]), int(corner_points[1])), (255, 0, 0), 2)
        # cv2.line(debug_image, (int(aabb[0]), int(aabb[1])), (int(aabb[2]), int(aabb[3])), (0, 255, 0), 2)
        # cv2.line(debug_image, (int(aabb[2]), int(aabb[3])), (int(aabb[4]), int(aabb[5])), (0, 255, 0), 2)
        # cv2.line(debug_image, (int(aabb[4]), int(aabb[5])), (int(aabb[6]), int(aabb[7])), (0, 255, 0), 2)
        # cv2.line(debug_image, (int(aabb[6]), int(aabb[7])), (int(aabb[0]), int(aabb[1])), (0, 255, 0), 2)
        # plt.imshow(debug_image)
        # plt.show()

        warp_template = cv2.warpPerspective(first_frame, homo, (first_frame.shape[1], first_frame.shape[0]))[y_top: y_bottom+1, x_left: x_right+1]
        gray = cv2.cvtColor(warp_template, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # in case a edge is too small
        while min(h, w) < 400:
            h = h * 2
            w = w * 2
            print('resize to {}x{}'.format(w, h))
            gray = cv2.resize(gray, (w, h))

        u_corner = [0, w - 1, w - 1, 0]
        v_corner = [0, 0, h - 1, h - 1]
        max_motion = max(w, h) // 10

        if self.checkpoint_path.exists():
            self.tracker.load(self.checkpoint_path.as_posix())
        else:
            if not self.tracker.learn(gray, 5, max_motion, 20, 20,
                                u_corner[0], v_corner[1], u_corner[2], v_corner[2],
                                max_motion, max_motion, 10000):
                return False
            self.tracker.save(self.checkpoint_path.as_posix(), None)

        if not self.tracker.init_with_detector(*corner_points):
            return False

        # for debug, save warp_template to video dir
        warp_template = cv2.resize(warp_template, (gray.shape[1], gray.shape[0]))
        cv2.imwrite((self.video_path.parent / 'template.jpg').as_posix(), warp_template)
        return True

    def track(self, visualize: bool = True) -> Tuple[bool, int, int, np.ndarray]:
        # tracker should be initialized.
        # during tracking, if track fails, use the ground truth to re-init

        count = 0
        failure_count = 0
        diffs = []

        while 1:
            input_data = self._read()
            if input_data is None:
                break

            frame, homography, points = input_data
            points = np.array(points)
            if np.linalg.norm(points) < 1e-3:
                # the annotation of points is somehow [0.0, 0.0, 0.0, ...], skip it for now
                continue

            count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = self.tracker.track(gray)
            if not result:
                print('track failed, re-init')
                failure_count += 1
                self.tracker.init_with_detector(*points)
            else:
                if visualize:
                    draw_tracked_locations(frame, self.tracker)
                    u = self.tracker.u.astype(np.int32)
                    cv2.line(frame, (u[0], u[1]), (u[2], u[3]), (0, 255, 0), 2)
                    cv2.line(frame, (u[2], u[3]), (u[4], u[5]), (0, 255, 0), 2)
                    cv2.line(frame, (u[4], u[5]), (u[6], u[7]), (0, 255, 0), 2)
                    cv2.line(frame, (u[6], u[7]), (u[0], u[1]), (0, 255, 0), 2)

                # calculate the difference between ground truth points and predicted u
                diff = np.sqrt((points.reshape(-1, 2) - self.tracker.u.reshape(-1, 2)) ** 2).mean()
                if diff > 20:
                    print('diff is too large, re-init')
                    failure_count += 1
                    self.tracker.init_with_detector(*points)
                else:
                    diffs.append(diff)

            if visualize:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            if failure_count > 100:
                print('too many failure, regard as failed')
                return False, count, failure_count, np.array(diffs)

        return True, count, failure_count, np.array(diffs)

    def dispose(self):
        cv2.destroyAllWindows()
        self.video.release()


def draw_tracked_locations(frame: np.ndarray, tracker: Template_matching_based_tracker):
    for i in range(tracker.nx * tracker.ny):
        x1, y1 = tracker.f.transform_point(tracker.m[2 * i], tracker.m[2 * i + 1])
        cv2.circle(frame, (int(x1), int(y1)), 3, (0, 255, 0), 1)