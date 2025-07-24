#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-


import tqdm
import numpy as np
import scipy.sparse as spa
from multiprocessing import Pool
from shapely.geometry import Polygon
from scipy.spatial.distance import cdist
# from post_processing import lanms
# import lanms
# from post_processing import locality_aware_nms as nms_locality
from post_processing.locality_aware_nms import nms_locality

from numba import njit

@njit
def average_precision_segfree(res, t, o, sinds, n_relevant, ot):
    """
        Computes the average precision
        res: sorted list of labels of the proposals, aka the results of a query.
        t: transcription of the query
        o: overlap matrix between the proposals and gt_boxes.
        sinds: The gt_box with which the proposals overlaps the most.
        n_relevant: The number of relevant retrievals in ground truth dataset
        ot: overlap_threshold
    """
    
    correct_label = res == t

    last_ind = 0
    tmp = np.zeros_like(res, dtype=np.float64)
    covered = np.zeros_like(res, dtype=np.bool_)
    
    for i in range(len(res)):
        if covered[sinds[i]] == 0: # if a ground truth box has been covered, mark proposal as irrelevant
            tmp[last_ind] = o[i]
            if o[i] >= ot and correct_label[i]:
                covered[sinds[i]] = 1
        last_ind += 1
        
    relevance = (correct_label * (tmp >= ot)).astype(np.float64)
    rel_cumsum = np.cumsum(relevance)#, dtype=np.float32)
    precision = rel_cumsum / np.arange(1, relevance.size + 1)

    if n_relevant > 0:
        ap = (precision * relevance).sum() / n_relevant
    else:
        ap = 0.0
    return ap, sinds[covered]
    


def hh(arg):
    query, t, db, db_targets, joint_boxes, query_nms_overlap, all_overlaps, inds, gt_targets, ot, qw = arg
    
    count = np.sum(db_targets == t)
    if count == 0:  # i.e., we have missed this word completely
        return 0.0, 0.0
    
    dists = np.squeeze(cdist(query[np.newaxis, :], db, metric="cosine"))
    sim = (np.nanmax(dists)) - dists
    if np.all(np.isnan(sim)):
        sim = np.zeros_like(sim)
        
    dets = np.hstack((joint_boxes, sim[:, np.newaxis]))
    dets = np.float64(dets)
    
    nms_dets, pick = nms_locality(dets, query_nms_overlap)

    I = np.argsort(dists[pick])
    res = db_targets[pick][I]  # Sort results after distance to query image
    o = all_overlaps[pick][I]
    
    sinds = inds[pick][I]
    n_relevant = np.sum(gt_targets == t)
   
    ap, covered = average_precision_segfree(res, t, o, sinds, n_relevant, ot)
    
    r = float(np.unique(covered).shape[0]) / n_relevant if n_relevant > 0 else 0.0
    
    return ap, r


def cal_map(queries, qtargets, db, db_targets, gt_targets, joint_boxes, all_overlaps, query_nms_overlap, ot, qbs_words, num_workers):
    # if not all_overlaps: return np.zeros(2, dtype=np.float64)
    inds = all_overlaps.argmax(axis=1)
    x_inds = np.arange(len(all_overlaps))
    all_overlaps = all_overlaps[np.s_[x_inds, inds]]
    # all_overlaps = spa.csr_matrix(all_overlaps)
    args = [(q, t, db, db_targets, joint_boxes, query_nms_overlap, all_overlaps, inds, gt_targets, ot, qw)
            for q, t, qw in zip(queries, qtargets, qbs_words)]
    # print(f"There are {len(args)} queries to calculate the average precision of")
    # num_workers = 0
    if num_workers == 0:  #single thread
        res = np.zeros((len(args),2))
        for i,arg in enumerate(tqdm.tqdm(args)):
            res[i] = hh(arg)
    else:  #Multithreading
        # print("Starting threads")
        with Pool(num_workers) as p:
            res = p.map(hh, tqdm.tqdm(args))
        # print("WHAT IS HAPPENING??")
    return np.mean(np.array(res), axis=0)

from post_processing.locality_aware_nms import intersection 

#TODO check this once again
@njit
def cal_overlap(ab, tb):
    overlaps = np.zeros((len(ab), len(tb)))
    for i in range(len(ab)):
        b1 = ab[i]
        for j in range(len(tb)):
            b2 = tb[j]
            overlaps[i,j] = intersection(b1, b2)
    # overlaps = np.array([[overlap(b1, b2) for b2 in tb] for b1 in ab])
    return overlaps

