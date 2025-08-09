from typing import Tuple, List, Optional
import cv2
import numpy as np
from tqdm import tqdm

from ..homography.homography_estimator import HomographyEstimator
from ..homography.homography06 import Homography

# for debug and test
import matplotlib.pyplot as plt

class Template_matching_based_tracker:
    width: int
    height: int

    # homography06
    f: Homography

    # store u, v
    m: List[int]

    he: HomographyEstimator

    As: np.ndarray

    u0: np.ndarray  # shape: (8, )
    u: np.ndarray   # shape: (8, )
    i0: np.ndarray  # shape: (nx * ny, 1)
    du: np.ndarray  # shape: (8, 1)
    DI: np.ndarray  # shape: (nx * ny, 1)
    i1: np.ndarray  # shape: (nx * ny, 1)

    err: float  # record the error of last tracking

    # TODO: these variables are data.fl in c++ codes, so remove these
    # u0: List[float]
    # u: List[float]
    # i0: List[float]
    # du: List[float]
    # i1: List[float]

    number_of_levels: int
    nx: int
    ny: int

    def __init__(self):
        self.f = Homography(None)
        self.he = HomographyEstimator()
        self.u = np.zeros((8, ))

    def init(self):
        self.u = self.u0.copy()
        self.f.homo = self.he.estimate(self.u0, self.u)
        if self.f.homo is not None:
            print('init', self.f.homo)
            return True
        else:
            print('init failed')
            return False

    def init_with_detector(self, x0, y0, x1, y1, x2, y2, x3, y3):
        self.u = np.array([x0, y0, x1, y1, x2, y2, x3, y3])
        self.f.homo = self.he.estimate(self.u0, self.u)
        if self.f.homo is not None:
            # print('init with detector', self.f.homo)
            return True
        else:
            print('init_with_detector failed')
            return False

    def move(self, x: int, y: int, amp: int) -> Tuple[int, int]:
        d = np.random.randint(0, amp)
        a = np.random.rand() * np.pi * 2.0

        x2 = x + d * np.cos(a)
        y2 = y + d * np.sin(a)
        return (x2, y2)

    def normalize(self, v: np.ndarray, counts = None) -> Tuple[bool, Optional[np.ndarray]]:
        if counts is not None:
            mean = v.sum() / counts
        else:
            counts = v.shape[0]
            mean = v.mean()
        sum2 = (v ** 2).sum()
        if mean < 10:
            # print('not enough contrast, better not put this sample into the training set')
            return (False, None)

        temp = np.sqrt(sum2 / counts - mean ** 2)
        if temp > -1e-4 and temp < 1e-4:
            # print('not enough contrast, better not put this sample into the training set')
            return (False, None)

        inv_sigma = 1.0 / temp
        return (True, inv_sigma * (v - mean))

    def add_noise(self, input: np.ndarray):
        v = input.copy()
        gamma = 0.5 + 2.3 * np.random.rand()

        v_max = v.max()
        if v_max > 25:
            # v is from 0 to 255
            v = np.power(v, gamma) + np.random.rand(*v.shape) * 10 - 5
        else:
            # v is from 0 to 1
            v = v * 255
            v = np.power(v, gamma) + np.random.rand(*v.shape) * 10 - 5
        v = np.clip(v, 0, 255)
        return v

    def compute_gradient(self, image: np.ndarray):
        dx = np.zeros_like(image, dtype=np.float32)
        dy = np.zeros_like(image, dtype=np.float32)

        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        dx = dx ** 2
        dy = dy ** 2
        return dx + dy

    # TODO: vectorize this function
    def get_local_maximum(self, G: np.ndarray, xc: int, yc: int, w: int, h: int) -> Tuple[int, int]:
        max = -1
        xm = ym = -1
        for v in range(int(yc), int(yc + h + 1)):
            for u in range(int(xc), int(xc + w + 1)):
                if G[v, u] > max:
                    max = G[v, u]
                    xm = u
                    ym = v
        return (xm, ym)

    def find_2d_points(self, image: np.ndarray, bx: int, by: int, xUL, yUL, xBR, yBR):
        gradient = self.compute_gradient(image)

        stepx = float(xBR - xUL - 2 * bx) / self.nx
        stepy = float(yBR - yUL - 2 * by) / self.ny
        for j in range(self.ny):
            for i in range(self.nx):
                xm, ym = self.get_local_maximum(
                    gradient,
                    int(xUL + bx + i * stepx),
                    int(yUL + by + j * stepy),
                    int(stepx), int(stepy),
                )
                # using grid points is not good as using local maximum, but may be more convenient
                # xm = int(self.u0[0] + bx + i * stepx + 0.5)
                # ym = int(self.u0[1] + by + j * stepy + 0.5)
                self.m[2 * (j * self.nx + i)] = xm
                self.m[2 * (j * self.nx + i) + 1] = ym

        # for debug, draw (xm, ym) on image, and it looks good
        image_copy = image.copy()
        for i in range(self.nx * self.ny):
            cv2.circle(image_copy, (self.m[2 * i], self.m[2 * i + 1]), 1, (255, 0, 0), 2)
        cv2.imwrite('debug_find_2d_points.jpg', image_copy)
        # plt.imshow(image_copy)
        # plt.show()


    def compute_As_matrices(self, image, max_motion: int, Ns: int):
        self.As = []

        '''
        CvMat * Y = cvCreateMat(8, Ns, CV_32F);
        CvMat * H = cvCreateMat(nx * ny, Ns, CV_32F);
        CvMat * HHt = cvCreateMat(nx * ny, nx * ny, CV_32F);
        CvMat * HHt_inv = cvCreateMat(nx * ny, nx * ny, CV_32F);
        CvMat * Ht_HHt_inv = cvCreateMat(Ns, nx * ny, CV_32F);
        '''
        Y = np.zeros((8, Ns), dtype=np.float32)
        H = np.zeros((self.nx * self.ny, Ns), dtype=np.float32)
        # HHt = np.zeros((self.nx * self.ny, self.nx * self.ny), dtype=np.float32)

        for level in range(self.number_of_levels):
            n = 0
            pbar = tqdm(total=Ns, desc='Level: {} - {} training samples generated.'.format(level, n))
            k = np.exp(1. / (self.number_of_levels - 1) * np.log(5.0 / max_motion))
            amp = np.power(k, level) * max_motion
            while n < Ns:
                u1 = np.zeros((8, ), dtype=np.float32)
                self.i1 = np.zeros((self.nx * self.ny, 1), dtype=np.float32)

                for i in range(4):
                    u1[2 * i], u1[2 * i + 1] = self.move(self.u0[2 * i], self.u0[2 * i + 1], amp)

                Y[:, n] = u1 - self.u0

                ft = Homography(self.he.estimate(self.u0, u1))
                if ft.homo is None:
                    return False

                x1s, y1s = ft.transform_points_array(self.m)
                x1s = np.clip(x1s.astype(np.int32), 0, image.shape[1] - 1)
                y1s = np.clip(y1s.astype(np.int32), 0, image.shape[0] - 1)
                self.i1 = image[y1s, x1s]

                self.i1 = self.add_noise(self.i1)
                ok, self.i1 = self.normalize(self.i1)
                if ok:
                    H[:, n] = (self.i1 - self.i0).squeeze()
                    n += 1
                    pbar.update(1)

            self.As.append(self.compute_As_by_original_method(Y, H))
            # self.As.append(self.compute_As_by_reformulated_method(Y, H))  # quicker, but decrease acc

            # debug codes
            # debug_image = image.copy()
            # for i in range(len(y1s)):
            #     x = x1s[i]
            #     y = y1s[i]
            #     x0 = self.m[2 * i]
            #     y0 = self.m[2 * i + 1]
            #     cv2.circle(debug_image, (x, y), 3, (0, 255, 0))
            #     cv2.circle(debug_image, (x0, y0), 3, (0, 0, 255))
            # print(u1)
            # plt.imshow(debug_image)
            # plt.show()

        self.As = np.array(self.As)

        # I want to use last level's H to estimate max error
        # but it seems not matching the real situation
        # so I comment this part and use 20 as the threshold in tracking
        '''
        errs = np.sqrt((H ** 2).sum(axis=0))
        print(errs.shape)
        for err_threshold in range(int(errs.max())):
            err_counts = np.count_nonzero(errs <= err_threshold)
            print(err_counts, err_threshold)
            if err_counts >= Ns * 0.95:
                print('err_threshold', err_threshold)
                break
        '''

    def compute_As_by_original_method(self, Y: np.ndarray, H: np.ndarray):
        HHt = np.matmul(H, H.T)     # same as np.dot(H, H.T) ... # shape: (nx * ny, nx * ny)

        HHt_inv = np.linalg.inv(HHt)
        # ok, HHt_inv = cv2.invert(HHt, cv2.DECOMP_SVD)

        Ht_HHt_inv = np.matmul(H.T, HHt_inv)     # shape: (Ns, nx * ny)
        # Ht_HHt_inv_ = cv2.gemm(H, HHt_inv, 1.0, 0, 0.0, None, cv2.GEMM_1_T)

        return np.matmul(Y, Ht_HHt_inv)    # shape: (8, nx * ny)

    def compute_As_by_reformulated_method(self, Y: np.ndarray, H: np.ndarray):
        B = H @ Y.T @ np.linalg.inv(Y @ Y.T)
        return np.linalg.inv(B.T @ B) @ B.T

    def learn(self, image: np.ndarray, number_of_levels: int, max_motion: int, nx: int, ny: int,
              xUL, yUL, xBR, yBR, bx: int, by: int, Ns: int) -> bool:
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.number_of_levels = number_of_levels
        self.nx = nx
        self.ny = ny

        self.m = [0 for _ in range(2 * nx * ny)]
        self.u0 = np.array([0, 0, self.width - 1, 0, self.width - 1, self.width - 1, 0, self.width - 1])

        # looks good
        self.find_2d_points(image, bx, by, xUL, yUL, xBR, yBR)

        self.u = np.zeros((8, ))

        self.i0 = np.zeros((nx * ny))
        self.i0[:] = image[self.m[1::2], self.m[0::2]]   # not be tested!

        ok, self.i0 = self.normalize(self.i0)
        if(not ok):
            print('normalize failed')
            return False

        self.i1 = np.zeros((nx * ny, 1))
        self.DI = np.zeros((nx * ny, 1))
        self.du = np.zeros((8, 1))

        self.compute_As_matrices(image, max_motion, Ns)
        return True

    def track(self, input_frame: np.ndarray) -> bool:
        self.err = 1000
        points_out_of_frame = 0     # here just for typing warning
        for level in range(self.number_of_levels):
            for iter in range(3):
                points_out_of_frame = 0
                out_of_frame_indices = []
                As = self.As[level].copy()
                for i in range(self.nx * self.ny):
                    x1, y1 = self.f.transform_point(self.m[2 * i], self.m[2 * i + 1])
                    if x1 < 0 or y1 < 0 or x1 >= input_frame.shape[1] or y1 >= input_frame.shape[0]:
                        self.i1[i] = 0
                        As[:, i] = 0
                        out_of_frame_indices.append(i)
                        points_out_of_frame += 1
                    else:
                        self.i1[i] = input_frame[int(y1), int(x1)]

                ok, res = self.normalize(self.i1, self.nx * self.ny - points_out_of_frame)
                if not ok or res is None:
                    print('norm i1 failed')
                    return False
                self.i1 = res

                if points_out_of_frame > 0:
                    if points_out_of_frame > self.nx * self.ny / 4:
                        return False
                    print('points_out_of_frame', points_out_of_frame)
                    self.i1[out_of_frame_indices] = self.i0[out_of_frame_indices]
                self.DI = self.i1 - self.i0

                self.du = np.matmul(As, self.DI).squeeze() * (1 - points_out_of_frame / (self.nx * self.ny))
                if (self.du > 80).any():
                    return False
                u2 = self.u0 - self.du

                fs_homo = self.he.estimate(self.u0, u2)
                if fs_homo is None:
                    return False

                self.f.homo = np.matmul(self.f.homo, fs_homo)
                self.f.homo = self.f.homo / np.sqrt((self.f.homo ** 2).sum())

        err = np.sqrt((self.DI ** 2).sum()) * (1 - points_out_of_frame / (self.nx * self.ny))
        self.err = err
        # print('err by DI', err)
        if err > 20:
            return False

        self.u[0], self.u[1] = self.f.transform_point(self.u0[0], self.u0[1])
        self.u[2], self.u[3] = self.f.transform_point(self.u0[2], self.u0[3])
        self.u[4], self.u[5] = self.f.transform_point(self.u0[4], self.u0[5])
        self.u[6], self.u[7] = self.f.transform_point(self.u0[6], self.u0[7])

        return True

    def save(self, fn: str, json_fn: Optional[str]):
        param = {
            'u0': self.u0,
            'nx': self.nx,
            'ny': self.ny,
            'm': self.m,
            'i0': self.i0.tolist(),
            'number_of_levels': self.number_of_levels,
            'As': self.As,
            'width': self.width,
            'height': self.height
        }

        np.save(fn, np.asanyarray(param))   # use np.asanyarray for tying warning

        import json

        if json_fn is not None:
            for k, v in param.items():
                if isinstance(v, np.ndarray):
                    param[k] = v.tolist()
            with open(json_fn, 'w') as f:
                json.dump(param, f)

    def load(self, fn: str):
        param = np.load(fn, allow_pickle=True).item()
        self.u0 = param['u0']
        self.nx = param['nx']
        self.ny = param['ny']
        self.m = param['m']
        self.i0 = np.array(param['i0']).reshape(-1, 1)
        self.number_of_levels = param['number_of_levels']
        self.As = param['As']

        self.du = np.zeros((8, 1)) # shape: (8, 1)
        self.DI = np.zeros((self.nx * self.ny, 1))
        self.i1 = np.zeros((self.nx * self.ny, 1))

    def visualize_points(self, frame, color=(0, 255, 0)):
        for i in range(self.nx * self.ny):
            x1, y1 = self.f.transform_point(self.m[2 * i], self.m[2 * i + 1])
            cv2.circle(frame, (int(x1), int(y1)), 3, color, 1)

    def visualize_edges(self, frame, color=(0, 255, 0)):
        self.u = self.u.astype(np.int32)
        cv2.line(frame, (self.u[0], self.u[1]), (self.u[2], self.u[3]), color, 2)
        cv2.line(frame, (self.u[2], self.u[3]), (self.u[4], self.u[5]), color, 2)
        cv2.line(frame, (self.u[4], self.u[5]), (self.u[6], self.u[7]), color, 2)
        cv2.line(frame, (self.u[6], self.u[7]), (self.u[0], self.u[1]), color, 2)

    def __repr__(self):
        return 'Template_matching_based_tracker: u0: {}, nx: {}, ny: {}, len(m): {}, i0: {}, number_of_levels: {}, As: {}'.format(
            self.u0.shape, self.nx, self.ny, len(self.m), self.i0.shape, self.number_of_levels, self.As.shape
        )
