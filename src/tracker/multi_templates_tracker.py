from typing import Tuple, List, Optional
import cv2
import numpy as np

from .template_matching_based_tracker import Template_matching_based_tracker
from ..homography.homography_estimator import estimate_homography
from ..homography.homography06 import Homography

# for debug and test
import matplotlib.pyplot as plt


class Multi_templates_matching_based_tracker:
  trackers: List[Template_matching_based_tracker] = []
  f: Homography = Homography()  # do we need this?
  width: int = 0
  height: int = 0

  def __init__(self):
    pass

  def load(self, ckpt_fn: str):
    self.trackers = []
    ckpt = np.load(ckpt_fn, allow_pickle=True).item()
    params = ckpt['trackers']
    for param in params:
      tracker = Template_matching_based_tracker()
      tracker.u0 = param['u0']
      tracker.nx = param['nx']
      tracker.ny = param['ny']
      tracker.m = param['m']
      tracker.i0 = np.array(param['i0']).reshape(-1, 1)
      tracker.number_of_levels = param['number_of_levels']
      tracker.As = param['As']
      tracker.width = param['width']
      tracker.height = param['height']

      tracker.du = np.zeros((8, 1)) # shape: (8, 1)
      tracker.DI = np.zeros((tracker.nx * tracker.ny, 1))
      tracker.i1 = np.zeros((tracker.nx * tracker.ny, 1))

      self.trackers.append(tracker)

    self.width = ckpt['width']
    self.height = ckpt['height']

  def learn(self, image: np.ndarray, number_of_levels: int, max_motion: int, nx: int, ny: int, bx: int, by: int, Ns: int) -> bool:
    success = False
    # 1 template is the whole image
    # 4 sub-templates is the 1 * 0.75 part of the image

    # TODO: also set different max_motion, nx and ny for different templates
    h, w = image.shape[:2]
    xULs = [0,  0,        0.25 * w, 0,        0]
    yULs = [0,  0,        0,        0,        0.25 * h]
    xBRs = [w,  0.75 * w, w,        w,        w]
    yBRs = [h,  h,        h,        0.75 * h, h]
    self.width = w
    self.height = h

    for i in range(len(xULs)):
      tracker = Template_matching_based_tracker()
      success = tracker.learn(image, number_of_levels, max_motion, nx, ny, int(xULs[i]),
                              int(yULs[i]), int(xBRs[i]), int(yBRs[i]), bx, by, Ns) or success
      self.trackers.append(tracker)

    return success

  def save(self, fn: str, json_fn: str):
    params = []
    for tracker in self.trackers:
      param = {
        'u0': tracker.u0,
        'nx': tracker.nx,
        'ny': tracker.ny,
        'm': tracker.m,
        'i0': tracker.i0.tolist(),
        'number_of_levels': tracker.number_of_levels,
        'As': tracker.As,
        'width': tracker.width,
        'height': tracker.height
      }
      params.append(param)

    ckpt = {
      'trackers': params,
      'width': self.width,
      'height': self.height,
      'version': '1.0.2'
    }

    np.save(fn, np.asanyarray(ckpt))

    import json
    if json_fn is not None:
      for param in params:
        for k, v in param.items():
          if isinstance(v, np.ndarray):
            param[k] = v.tolist()
      with open(json_fn, 'w') as f:
        json.dump(ckpt, f)

  '''
  determine whether the u0 is the image corner or the template region corner
  in this function, the u0 is the template corner (for many tiny templates)
  '''
  def init_with_detector_origin(self, x0, y0, x1, y1, x2, y2, x3, y3, image) -> bool:
    u0 = np.array([0, 0, self.width - 1, 0, self.width - 1, self.width - 1, 0, self.width - 1])
    u = np.array([x0, y0, x1, y1, x2, y2, x3, y3])
    homo = estimate_homography(u0, u)
    f = Homography(homo)
    if homo is None:
      return False

    w = self.width
    h = self.height
    xULs = [0.25 * w, 0, 0, 0.5 * w, 0.5 * w]
    yULs = [0.25 * h, 0, 0.5 * h, 0, 0.5 * h]
    xBRs = [0.75 * w, 0.5 * w, 0.5 * w, w, w]
    yBRs = [0.75 * h, 0.5 * h, h, 0.5 * h, h]

    debug_image = image.copy()

    for i, tracker in enumerate(self.trackers):
      tracker_u0 = np.array([xULs[i], yULs[i], xBRs[i], yULs[i], xBRs[i], yBRs[i], xULs[i], yBRs[i]], dtype=np.float32)
      x0, y0 = f.transform_point(tracker_u0[0], tracker_u0[1])
      x1, y1 = f.transform_point(tracker_u0[2], tracker_u0[3])
      x2, y2 = f.transform_point(tracker_u0[4], tracker_u0[5])
      x3, y3 = f.transform_point(tracker_u0[6], tracker_u0[7])
      tracker.init_with_detector(x0, y0, x1, y1, x2, y2, x3, y3)

    #   cv2.circle(debug_image, (int(x0), int(y0)), 3, (0, 255, 0), 1)
    #   cv2.circle(debug_image, (int(x1), int(y1)), 3, (0, 255, 0), 1)
    #   cv2.circle(debug_image, (int(x2), int(y2)), 3, (0, 255, 0), 1)
    #   cv2.circle(debug_image, (int(x3), int(y3)), 3, (0, 255, 0), 1)

    # u = u.astype(np.int32)
    # cv2.line(debug_image, (u[0], u[1]), (u[2], u[3]), (255, 0, 0), 2)
    # cv2.line(debug_image, (u[2], u[3]), (u[4], u[5]), (255, 0, 0), 2)
    # cv2.line(debug_image, (u[4], u[5]), (u[6], u[7]), (255, 0, 0), 2)
    # cv2.line(debug_image, (u[6], u[7]), (u[0], u[1]), (255, 0, 0), 2)

    # plt.imshow(debug_image)
    # plt.show()

    return True

  def init_with_detector(self, x0, y0, x1, y1, x2, y2, x3, y3) -> bool:
    success = False
    for tracker in self.trackers:
      success = tracker.init_with_detector(x0, y0, x1, y1, x2, y2, x3, y3) or success
    return success

  def track(self, input_frame: np.ndarray) -> bool:
    status = [False for i in range(len(self.trackers))]
    gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)
    u = None
    for i, tracker in enumerate(self.trackers):
      status[i] = tracker.track(gray)
      tracker.visualize_points(input_frame, (0, 255, 0) if status[i] else (0, 0, 255))
      if status[i]:
        u = tracker.u.squeeze()
        tracker.visualize_edges(input_frame, (0, 255, 0))
        break

    result = u is not None
    if result:
      self.init_with_detector(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7])

    return result
