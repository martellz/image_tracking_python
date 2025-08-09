from typing import Optional, Tuple, Sequence
import cv2
import numpy as np

class Orb_detector:

    detector: cv2.ORB
    matcher: cv2.BFMatcher

    kp1: Sequence[cv2.KeyPoint]
    des1: np.ndarray

    h: int
    w: int

    def __init__(self, image: np.ndarray, maxFeatures: int = 1000):
        self.detector = cv2.ORB_create()
        self.detector.setMaxFeatures(maxFeatures)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.kp1, self.des1 = self.detector.detectAndCompute(image, None)
        self.h, self.w = image.shape[:2]

    def detect(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        kp2, des2 = self.detector.detectAndCompute(frame, None)
        matches = self.matcher.match(self.des1, des2)
        # matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 4:
            return None

        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        pts = np.float32([[0, 0], [0, self.h - 1], [self.w - 1, self.h - 1],
                            [self.w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        return H, dst

