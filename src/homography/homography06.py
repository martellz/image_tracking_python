import cv2
import numpy as np

class Homography:

    homo: np.ndarray

    def __init__(self, homo: np.ndarray | None = None):
        if homo is None:
            homo = np.eye(3, dtype=np.float64)
        self.homo = homo

    def transform_point(self, x, y):
        inv_k = 1.0 / (self.homo[2, 0] * x + self.homo[2, 1] * y + self.homo[2, 2])
        up = inv_k * (self.homo[0, 0] * x + self.homo[0, 1] * y + self.homo[0, 2]) + 0.5
        vp = inv_k * (self.homo[1, 0] * x + self.homo[1, 1] * y + self.homo[1, 2]) + 0.5
        return up, vp

    def transform_points_array(self, points):
        points = np.asarray(points)
        x, y = points[0::2], points[1::2]
        inv_k = 1.0 / (self.homo[2, 0] * x + self.homo[2, 1] * y + self.homo[2, 2])
        up = inv_k * (self.homo[0, 0] * x + self.homo[0, 1] * y + self.homo[0, 2]) + 0.5
        vp = inv_k * (self.homo[1, 0] * x + self.homo[1, 1] * y + self.homo[1, 2]) + 0.5
        return up, vp

    def transform_points_matrix(self, points: np.ndarray) -> np.ndarray:
        # points.shape = (N, 2)
        p = (self.homo @ points.T).T
        return p

    def toPose(self, K):
        _, Rs, Ts, Ns = cv2.decomposeHomographyMat(self.homo, K)
        return Rs, Ts