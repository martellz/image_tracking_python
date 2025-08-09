from typing import Self, Tuple, List, Optional, override
import cv2
import numpy as np
from tqdm import tqdm

from ..homography.homography_estimator import estimate_homography
from ..homography.homography06 import Homography
from .template_matching_based_tracker import Template_matching_based_tracker

# for debug and test
import matplotlib.pyplot as plt



class Template:
    number_of_levels = 5    # level of movement, from big to small

    # linear predictor
    As: List[np.ndarray] # (N, 8)

    # The pose of a template is parameterized using 4 corner points.
    u0: np.ndarray = np.zeros((4, 2))

    #
    occlusion_state: int
    H: np.ndarray

    def __init__(self, As: List[np.ndarray], u0: np.ndarray, i0: np.ndarray, m: np.ndarray, H) -> None:
        self.As = As
        self.u0 = u0
        self.i0 = i0
        self.m = m

        # recording H for updating top level template
        self.H = H

    def add_neighbors(self, neighbors: List['Template']):   # This typing is for python<3.7. But vscode uses it!
        self.neighbors = neighbors

    def track(self, image: np.ndarray, f: Homography) -> Optional[np.ndarray]:
        h, w = image.shape
        for level in range(self.number_of_levels):
            for _ in range(3):
                xs, ys = f.transform_points_array(self.m)
                xs = xs.astype(np.int32)
                ys = ys.astype(np.int32)
                if np.any(xs < 0) or np.any(xs >= w) or np.any(ys < 0) or np.any(ys >= h):
                    return None

                coords = np.asarray([xs, ys]).transpose(1, 0)   # (4, 2)
                i1 = self.normalize(image[coords])
                di = i1 - self.i0
                u2 = self.u0 - np.matmul(self.As[level], di)

                fs_homo = estimate_homography(self.u0, u2)
                return fs_homo

    def apply_homography_to_u(self, homo: np.ndarray):
        return (homo[:2, :2] @ self.u0.T).T

    def normalize(self, v: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        mean = v.mean()
        sum2 = (v ** 2).sum()
        if mean < 10:
            # print('not enough contrast, better not put this sample into the training set')
            return (False, None)

        temp = np.sqrt(sum2 / v.shape[0] - mean ** 2)
        if temp > -1e-4 and temp < 1e-4:
            # print('not enough contrast, better not put this sample into the training set')
            return (False, None)

        inv_sigma = 1.0 / temp
        return (True, inv_sigma * (v - mean))


class ALP(Template_matching_based_tracker):
    # this is top level template class, using subsets to detect occlusion
    # also adding online updating

    # use 2 low level subsets for occlusion detection
    low_level_templates: Tuple[List[List['Template']], List[List['Template']]]
    low_level_occlusion_maps: Tuple[np.ndarray, np.ndarray]   # dtype is bool for indexing
    low_level_tracking_points: Tuple[np.ndarray, np.ndarray]

    def __init__(self) -> None:
        super(Template_matching_based_tracker, self).__init__()

    @override
    def track(self, image: np.ndarray):
        for templates in self.low_level_templates:
            # templates is 2d structure
            pass

    def learn(self, image: np.ndarray):
        pass