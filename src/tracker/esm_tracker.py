from typing import Tuple, List, Optional
import cv2
import numpy as np

from ..homography.homography_estimator import HomographyEstimator
from ..homography.homography06 import Homography


class ESM_tracker:
