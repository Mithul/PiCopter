# -*- coding: utf-8 -*-

import os, sys
import tempfile
import time
import logging
from subprocess import call
import numpy as np
import json
import uuid
import cv2
import csfm

from recon3d import context


logger = logging.getLogger(__name__)


def resized_image(image, config):
    feature_process_size = config.get('feature_process_size', -1)
    size = np.array(image.shape[0:2])
    if 0 < feature_process_size < size.max():
        new_size = size * feature_process_size / size.max()
        return cv2.resize(image, dsize=(new_size[1], new_size[0]))
    else:
        return image

def root_feature_surf(desc, l2_normalization=False, partial=False):
    """
    Experimental square root mapping of surf-like feature, only work for 64-dim surf now
    """
    if desc.shape[1] == 64:
        if l2_normalization:
            s2 = np.linalg.norm(desc, axis=1)
            desc = (desc.T/s2).T
        if partial:
            ii = np.array([i for i in xrange(64) if (i%4==2 or i%4==3)])
        else:
            ii = np.arange(64)
        desc_sub = np.abs(desc[:, ii])
        desc_sub_sign = np.sign(desc[:, ii])
        # s_sub = np.sum(desc_sub, 1)  # This partial normalization gives slightly better results for AKAZE surf
        s_sub = np.sum(np.abs(desc), 1)
        desc_sub = np.sqrt(desc_sub.T/s_sub).T
        desc[:, ii] = desc_sub*desc_sub_sign
    return desc

def normalized_image_coordinates(pixel_coords, width, height):
    size = max(width, height)
    p = np.empty((len(pixel_coords), 2))
    p[:, 0] = (pixel_coords[:, 0] + 0.5 - width / 2.0) / size
    p[:, 1] = (pixel_coords[:, 1] + 0.5 - height / 2.0) / size
    return p

def mask_and_normalize_features(points, desc, colors, width, height, config):
    masks = np.array(config.get('masks',[]))
    for mask in masks:
        top = mask['top'] * height
        left = mask['left'] * width
        bottom = mask['bottom'] * height
        right = mask['right'] * width
        ids  = np.invert ( (points[:,1] > top) *
                           (points[:,1] < bottom) *
                           (points[:,0] > left) *
                           (points[:,0] < right) )
        points = points[ids]
        desc = desc[ids]
        colors = colors[ids]
    points[:, :2] = normalized_image_coordinates(points[:, :2], width, height)
    return points, desc, colors

def extract_features_surf(image, config):
    surf_hessian_threshold = config.get('surf_hessian_threshold', 1000)
    detector = cv2.xfeatures2d.SURF_create()
    descriptor = detector
    detector.setHessianThreshold(surf_hessian_threshold)
    detector.setNOctaves(config.get('surf_n_octaves', 4))
    detector.setNOctaveLayers(config.get('surf_n_octavelayers', 2))
    detector.setUpright(config.get('surf_upright', 0))
    
    while True:
        logger.debug('Computing surf with threshold {0}'.format(surf_hessian_threshold))
        t = time.time()
        if context.OPENCV3:
            detector.setHessianThreshold(surf_hessian_threshold)
        else:
            detector.setDouble("hessianThreshold", surf_hessian_threshold)  # default: 0.04
        points = detector.detect(image)
        logger.debug('Found {0} points in {1}s'.format( len(points), time.time()-t ))
        if len(points) < config.get('feature_min_frames', 0) and surf_hessian_threshold > 0.0001:
            surf_hessian_threshold = (surf_hessian_threshold * 2) / 3
            logger.debug('reducing threshold')
        else:
            logger.debug('done')
            break

    points, desc = descriptor.compute(image, points)
    if config.get('feature_root', False): desc = root_feature_surf(desc, partial=True)
    points = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in points])
    return points, desc


def extract_features(color_image, config):
    assert len(color_image.shape) == 3
    color_image = resized_image(color_image, config)
    image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    points, desc = extract_features_surf(image, config)
    
    xs = points[:,0].round().astype(int)
    ys = points[:,1].round().astype(int)
    colors = color_image[ys, xs]

    return mask_and_normalize_features(points, desc, colors, image.shape[1], image.shape[0], config)
    # return points,desc,colors

def build_flann_index(features, config):
    FLANN_INDEX_LINEAR          = 0
    FLANN_INDEX_KDTREE          = 1
    FLANN_INDEX_KMEANS          = 2
    FLANN_INDEX_COMPOSITE       = 3
    FLANN_INDEX_KDTREE_SINGLE   = 4
    FLANN_INDEX_HIERARCHICAL    = 5
    FLANN_INDEX_LSH             = 6

    if features.dtype.type is np.float32:
        FLANN_INDEX_METHOD = FLANN_INDEX_KMEANS
    else:
        FLANN_INDEX_METHOD = FLANN_INDEX_LSH

    flann_params = dict(algorithm=FLANN_INDEX_METHOD,
                        branching=config.get('flann_branching', 16),
                        iterations=config.get('flann_iterations', 20))

    # flann_Index = cv2.flann.Index
    return cv2.flann.Index(features, flann_params)
