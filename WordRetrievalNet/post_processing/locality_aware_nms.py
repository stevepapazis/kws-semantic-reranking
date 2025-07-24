#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import numpy as np
# from shapely.geometry import Polygon
from numba import njit
import numba

@njit
def intersection(g, p):
    gx1, gy1, gx2, gy2 = g[0], g[1], g[4], g[5]
    px1, py1, px2, py2 = p[0], p[1], p[4], p[5]

    ix1 = max(gx1,px1)
    iy1 = max(gy1,py1)
    ix2 = min(gx2,px2)
    iy2 = min(gy2,py2)

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    intersection = (ix2-ix1)*(iy2-iy1)
    g_area = (gx2-gx1)*(gy2-gy1)
    p_area = (px2-px1)*(py2-py1)
    union = g_area + p_area - intersection

    if union == 0:
        return 0.0
    else:
        return intersection/union


@njit
def weighted_merge(g, p):
    res = np.zeros_like(g)
    res[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    res[8] = (g[8] + p[8])
    return res


@njit
def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = np.zeros(len(S),dtype=np.bool_)#[]
    while order.size > 0:
        i = order[0]
        keep[i] = True # keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    # keep = np.array(keep)
    return S[keep], keep


@njit
def nms_locality(polys, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = np.zeros_like(polys)#[]

    last_ind = 0
    if len(polys) > 0:
        p = polys[0]
        for g in polys[1:]:
            if intersection(g, p) > thres:
                p = weighted_merge(g, p)
            else:
                S[last_ind] = p #S.append(p)
                last_ind += 1
                p = g
        S[last_ind] = p #S.append(p)
        last_ind += 1

    return standard_nms(S[:last_ind], thres)


nms_locality(np.zeros(shape=(0,9), dtype=np.float64), thres=0.3) # numba compiles the function during this call

# if __name__ == '__main__':
#     # 343,350,448,135,474,143,369,359
#     print(Polygon(np.array([[343, 350], [448, 135],
#                             [474, 143], [369, 359]])).area)
