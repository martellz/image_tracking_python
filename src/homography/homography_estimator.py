from typing import Optional
import cv2
import numpy as np

class HomographyEstimator:
    def __init__(self):
        pass

    def estimate_by_cv2(self, u0: np.ndarray, u: np.ndarray) -> np.ndarray:
        # u0: shape: (8, ), u: shape: (8, )
        H, _ = cv2.findHomography(u0.reshape(4, 2), u.reshape(4, 2), cv2.RANSAC, 3.0)
        return H

    def estimate(self, u0: np.ndarray, u: np.ndarray) -> Optional[np.ndarray]:
        # u0: shape: (8, ), u: shape: (8, )

        R = np.zeros((3, 3))

        a,b,c,d = u0.reshape((4, 2))
        x,y,z,w = u.reshape((4, 2))
        Hr = homography_from_4pt(a,b,c,d)
        Hl = homography_from_4pt(x,y,z,w)

        # the following code computes R = Hl * inverse Hr
        t2 = Hr[1][1]-Hr[2][1]*Hr[1][2]
        t4 = Hr[0][0]*Hr[1][1]
        t5 = Hr[0][0]*Hr[1][2]
        t7 = Hr[1][0]*Hr[0][1]
        t8 = Hr[0][2]*Hr[1][0]
        t10 = Hr[0][1]*Hr[2][0]
        t12 = Hr[0][2]*Hr[2][0]
        t15 = 1/(t4-t5*Hr[2][1]-t7+t8*Hr[2][1]+t10*Hr[1][2]-t12*Hr[1][1])
        t18 = -Hr[1][0]+Hr[1][2]*Hr[2][0]
        t23 = -Hr[1][0]*Hr[2][1]+Hr[1][1]*Hr[2][0]
        t28 = -Hr[0][1]+Hr[0][2]*Hr[2][1]
        t31 = Hr[0][0]-t12
        t35 = Hr[0][0]*Hr[2][1]-t10
        t41 = -Hr[0][1]*Hr[1][2]+Hr[0][2]*Hr[1][1]
        t44 = t5-t8
        t47 = t4-t7
        t48 = t2*t15
        t49 = t28*t15
        t50 = t41*t15
        R[0][0] = Hl[0][0]*t48+Hl[0][1]*(t18*t15)-Hl[0][2]*(t23*t15)
        R[0][1] = Hl[0][0]*t49+Hl[0][1]*(t31*t15)-Hl[0][2]*(t35*t15)
        R[0][2] = -Hl[0][0]*t50-Hl[0][1]*(t44*t15)+Hl[0][2]*(t47*t15)
        R[1][0] = Hl[1][0]*t48+Hl[1][1]*(t18*t15)-Hl[1][2]*(t23*t15)
        R[1][1] = Hl[1][0]*t49+Hl[1][1]*(t31*t15)-Hl[1][2]*(t35*t15)
        R[1][2] = -Hl[1][0]*t50-Hl[1][1]*(t44*t15)+Hl[1][2]*(t47*t15)
        R[2][0] = Hl[2][0]*t48+Hl[2][1]*(t18*t15)-t23*t15
        R[2][1] = Hl[2][0]*t49+Hl[2][1]*(t31*t15)-t35*t15
        R[2][2] = -Hl[2][0]*t50-Hl[2][1]*(t44*t15)+t47*t15

        if eps_compare(homography_transform(a, R), x) and eps_compare(homography_transform(b, R), y) and eps_compare(homography_transform(c, R), z) and eps_compare(homography_transform(d, R), w):
            return R
        else:
            return None

def homography_transform(a, H):
    z = H[2][0]*a[0] + H[2][1]*a[1] + H[2][2]
    r0 = (H[0][0]*a[0] + H[0][1]*a[1] + H[0][2])/z
    r1 = (H[1][0]*a[0] + H[1][1]*a[1] + H[1][2])/z
    return (r0, r1)

def eps_compare(a, b):
    eps = 1e-4
    dx = a[0]-b[0]
    dy = a[1]-b[1]
    # print(dx, dy)
    res = (dx**2 <eps and dy**2<eps)
    if not res:
        print(a, b)
    return res

def homography_from_4pt(x, y, z, w) -> np.ndarray:
    cgret = np.ones((3, 3))
    t1 = x[0]
    t2 = z[0]
    t4 = y[1]
    t5 = t1 * t2 * t4
    t6 = w[1]
    t7 = t1 * t6
    t8 = t2 * t7
    t9 = z[1]
    t10 = t1 * t9
    t11 = y[0]
    t14 = x[1]
    t15 = w[0]
    t16 = t14 * t15
    t18 = t16 * t11
    t20 = t15 * t11 * t9
    t21 = t15 * t4
    t24 = t15 * t9
    t25 = t2 * t4
    t26 = t6 * t2
    t27 = t6 * t11
    t28 = t9 * t11
    t30 = 0.1e1 / (-t24 + t21 - t25 + t26 - t27 + t28)
    t32 = t1 * t15
    t35 = t14 * t11
    t41 = t4 * t1
    t42 = t6 * t41
    t43 = t14 * t2
    t46 = t16 * t9
    t48 = t14 * t9 * t11
    t51 = t4 * t6 * t2
    t55 = t6 * t14
    cgret[0, 0] = -(-t5 + t8 + t10 * t11 - t11 * t7 - t16 * t2 + t18 - t20 + t21 * t2) * t30
    cgret[0, 1] = (t5 - t8 - t32 * t4 + t32 * t9 + t18 - t2 * t35 + t27 * t2 - t20) * t30
    cgret[0, 2] = t1
    cgret[1, 0] = (-t9 * t7 + t42 + t43 * t4 - t16 * t4 + t46 - t48 + t27 * t9 - t51) * t30
    cgret[1, 1] = (-t42 + t41 * t9 - t55 * t2 + t46 - t48 + t55 * t11 + t51 - t21 * t9) * t30
    cgret[1, 2] = t14
    cgret[2, 0] = (-t10 + t41 + t43 - t35 + t24 - t21 - t26 + t27) * t30
    cgret[2, 1] = (-t7 + t10 + t16 - t43 + t27 - t28 - t21 + t25) * t30

    return cgret

def estimate_homography(u0: np.ndarray, u: np.ndarray) -> Optional[np.ndarray]:
    # u0: shape: (8, ), u: shape: (8, )

    R = np.zeros((3, 3))

    a,b,c,d = u0.reshape((4, 2))
    x,y,z,w = u.reshape((4, 2))
    Hr = homography_from_4pt(a,b,c,d)
    Hl = homography_from_4pt(x,y,z,w)

    # the following code computes R = Hl * inverse Hr
    t2 = Hr[1][1]-Hr[2][1]*Hr[1][2]
    t4 = Hr[0][0]*Hr[1][1]
    t5 = Hr[0][0]*Hr[1][2]
    t7 = Hr[1][0]*Hr[0][1]
    t8 = Hr[0][2]*Hr[1][0]
    t10 = Hr[0][1]*Hr[2][0]
    t12 = Hr[0][2]*Hr[2][0]
    t15 = 1/(t4-t5*Hr[2][1]-t7+t8*Hr[2][1]+t10*Hr[1][2]-t12*Hr[1][1])
    t18 = -Hr[1][0]+Hr[1][2]*Hr[2][0]
    t23 = -Hr[1][0]*Hr[2][1]+Hr[1][1]*Hr[2][0]
    t28 = -Hr[0][1]+Hr[0][2]*Hr[2][1]
    t31 = Hr[0][0]-t12
    t35 = Hr[0][0]*Hr[2][1]-t10
    t41 = -Hr[0][1]*Hr[1][2]+Hr[0][2]*Hr[1][1]
    t44 = t5-t8
    t47 = t4-t7
    t48 = t2*t15
    t49 = t28*t15
    t50 = t41*t15
    R[0][0] = Hl[0][0]*t48+Hl[0][1]*(t18*t15)-Hl[0][2]*(t23*t15)
    R[0][1] = Hl[0][0]*t49+Hl[0][1]*(t31*t15)-Hl[0][2]*(t35*t15)
    R[0][2] = -Hl[0][0]*t50-Hl[0][1]*(t44*t15)+Hl[0][2]*(t47*t15)
    R[1][0] = Hl[1][0]*t48+Hl[1][1]*(t18*t15)-Hl[1][2]*(t23*t15)
    R[1][1] = Hl[1][0]*t49+Hl[1][1]*(t31*t15)-Hl[1][2]*(t35*t15)
    R[1][2] = -Hl[1][0]*t50-Hl[1][1]*(t44*t15)+Hl[1][2]*(t47*t15)
    R[2][0] = Hl[2][0]*t48+Hl[2][1]*(t18*t15)-t23*t15
    R[2][1] = Hl[2][0]*t49+Hl[2][1]*(t31*t15)-t35*t15
    R[2][2] = -Hl[2][0]*t50-Hl[2][1]*(t44*t15)+t47*t15

    if eps_compare(homography_transform(a, R), x) and eps_compare(homography_transform(b, R), y) and eps_compare(homography_transform(c, R), z) and eps_compare(homography_transform(d, R), w):
        return R
    else:
        return None