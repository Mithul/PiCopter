# -*- coding: utf-8 -*-

import math
import random

import numpy as np
import cv2

from recon3d import transformations as tf
from recon3d import csfm


def nullspace(A):
    '''Compute the null space of A.

    Return the smallest sigular value and the corresponding vector.
    '''
    u, s, vh = np.linalg.svd(A)
    return s[-1], vh[-1]

#add extra column of 1s to matrix to make it homogeneous
def homogeneous(x):
    s = x.shape[:-1] + (1,)
    return np.hstack((x, np.ones(s)))


def homogeneous_vec(x):
    '''Add a column of zeros to x.
    '''
    s = x.shape[:-1] + (1,)
    return np.hstack((x, np.zeros(s)))


def euclidean(x):
    '''Divide by last column and drop it.
    '''
    return x[..., :-1] / x[..., -1:]

#calculate projection matrix from Camera matrix, Rotation matrix and translation vector
#formula from Multiview geometry - Hartley and Zisserman
#projection matrix is 3 X 4; P = K[R|t]
def P_from_KRt(K, R, t):
    P = np.empty((3, 4))
    P[:, :3] = np.dot(K, R)
    P[:, 3] = np.dot(K, t)
    return P

def KRt_from_P(P):
    K, R = rq(P[:, :3])
    #force K to have a positive diagonal
    T = np.diag(np.sign(np.diag(K)))
    K = np.dot(K, T)
    R = np.dot(T, R)
    t = np.linalg.solve(K, P[:,3])
    if np.linalg.det(R) < 0:         # ensure det(R) = 1
        R = -R
        t = -t
    K /= K[2, 2]                     # normalize K

    return K, R, t

#RQ Decomposition using QR decomposition - decomposition of matrix into upper triangular matrix(K) and orthogonal matrix (R)
def rq(A):
    Q, R = np.linalg.qr(np.flipud(A).T) 
    R = np.flipud(R.T)
    Q = Q.T
    #flip R from left to right
    return R[:,::-1], Q[::-1,:]


def vector_angle(u, v):
    cos = np.dot(u, v) / math.sqrt(np.dot(u,u) * np.dot(v,v))
    if cos >= 1.0: return 0.0
    else: return math.acos(cos)


def decompose_similarity_transform(T):
    ''' Decompose the similarity transform to scale, rotation and translation
    '''
    m, n = T.shape[0:2]
    assert(m==n)
    A, b = T[:(m-1),:(m-1)], T[:(m-1),(m-1)]
    s = np.linalg.det(A)**(1./(m-1))
    A /= s
    return s, A, b

#RANSAC - RANdom SAmple Consensus is used to robustly fit data to a model by detecting inliers 
def ransac_max_iterations(kernel, inliers, failure_probability):
    if len(inliers) >= kernel.num_samples():
        return 0
    inlier_ratio = float(len(inliers)) / kernel.num_samples()
    n = kernel.required_samples
    return math.log(failure_probability) / math.log(1.0 - inlier_ratio**n)


def ransac(kernel, threshold):
    max_iterations = 1000
    best_error = float('inf')
    best_model = None
    best_inliers = []
    i = 0
    while i < max_iterations:
        try:
            samples = kernel.sampling()
        except AttributeError:
            samples = random.sample(xrange(kernel.num_samples()),
                                kernel.required_samples)
        models = kernel.fit(samples)
        for model in models:
            errors = kernel.evaluate(model)
            inliers = np.flatnonzero(np.fabs(errors) < threshold)
            error = np.fabs(errors).clip(0, threshold).sum()
            if len(inliers) and error < best_error:
                best_error = error
                best_model = model
                best_inliers = inliers
                max_iterations = min(max_iterations,
                    ransac_max_iterations(kernel, best_inliers, 0.01))
        i += 1
    return best_model, best_inliers, best_error


class TestLinearKernel:
    required_samples = 1

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def num_samples(self):
        return len(self.x)

    def fit(self, samples):
        x = self.x[samples[0]]
        y = self.y[samples[0]]
        return [y / x]

    def evaluate(self, model):
        return self.y - model * self.x


class PlaneKernel:
    '''
    A kernel for estimating plane from on-plane points and vectors
    '''

    def __init__(self, points, vectors, verticals, point_threshold=1.0, vector_threshold=5.0):
        self.points = points
        self.vectors = vectors
        self.verticals = verticals
        self.required_samples = 3
        self.point_threshold = point_threshold
        self.vector_threshold = vector_threshold

    def num_samples(self):
        return len(self.points)

    def sampling(self):
        samples = {}
        if len(self.vectors)>0:
            samples['points'] = self.points[random.sample(xrange(len(self.points)), 2),:]
            samples['vectors'] = [self.vectors[i] for i in random.sample(xrange(len(self.vectors)), 1)]
        else:
            samples['points'] = self.points[:,random.sample(xrange(len(self.points)), 3)]
            samples['vectors'] = None
        return samples

    def fit(self, samples):
        model = fit_plane(samples['points'], samples['vectors'], self.verticals)
        return [model]

    def evaluate(self, model):
        # only evaluate on points
        normal = model[0:3]
        normal_norm = np.linalg.norm(normal)+1e-10
        point_error = np.abs(model.T.dot(homogeneous(self.points).T))/normal_norm
        vectors = np.array(self.vectors)
        vector_norm = np.sum(vectors*vectors, axis=1)
        vectors = (vectors.T / vector_norm).T
        vector_error = abs(np.rad2deg(abs(np.arccos(vectors.dot(normal)/normal_norm)))-90)
        vector_error[vector_error<self.vector_threshold] = 0.0
        vector_error[vector_error>=self.vector_threshold] = self.point_threshold+0.1
        point_error[point_error<self.point_threshold] = 0.0
        point_error[point_error>=self.point_threshold] = self.point_threshold+0.1
        errors = np.hstack((point_error, vector_error))
        return errors


def fit_plane_ransac(points, vectors, verticals, point_threshold=1.2, vector_threshold=5.0):
    vectors = [v/math.pi*180.0 for v in vectors]
    kernel = PlaneKernel(points - points.mean(axis=0), vectors, verticals, point_threshold, vector_threshold)
    p, inliers, error = ransac(kernel, point_threshold)
    num_point = points.shape[0]
    points_inliers = points[inliers[inliers<num_point],:]
    vectors_inliers = [vectors[i-num_point] for i in inliers[inliers>=num_point]]
    p = fit_plane(points_inliers - points_inliers.mean(axis=0), vectors_inliers, verticals)
    return p, inliers, error


def fit_plane(points, vectors, verticals):
    '''Estimate a plane fron on-plane points and vectors.

    >>> x = [[0,0,0], [1,0,0], [0,1,0]]
    >>> p = fit_plane(x, None, None)
    >>> np.allclose(p, [0,0,1,0]) or np.allclose(p, [0,0,-1,0])
    True
    >>> x = [[0,0,0], [0,1,0]]
    >>> v = [[1,0,0]]
    >>> p = fit_plane(x, v, None)
    >>> np.allclose(p, [0,0,1,0]) or np.allclose(p, [0,0,-1,0])
    True
    >>> vert = [[0,0,1]]
    >>> p = fit_plane(x, v, vert)
    >>> np.allclose(p, [0,0,1,0])
    True
    '''
    # (x 1) p = 0
    # (v 0) p = 0
    points = np.array(points)
    s = 1. / max(1e-8, points.std())           # Normalize the scale to improve conditioning.
    x = homogeneous(s * points)
    if vectors:
        v = homogeneous_vec(s * np.array(vectors))
        A = np.vstack((x, v))
    else:
        A = x
    _, p = nullspace(A)
    p[3] /= s

    if np.allclose(p[:3], [0,0,0]):
        return np.array([0.0, 0.0, 1.0, 0])

    # Use verticals to decide the sign of p
    if verticals:
        d = 0
        for vertical in verticals:
            d += p[:3].dot(vertical)
        p *= np.sign(d)
    return p


def plane_horizontalling_rotation(p):
    '''Compute a rotation that brings p to z=0

    >>> p = [1.,2.,3.]
    >>> R = plane_horizontalling_rotation(p)
    >>> np.allclose(R.dot(p), [0,0,np.linalg.norm(p)])
    True
    '''
    v0 = p[:3]
    v1 = [0,0,1.0]
    angle = tf.angle_between_vectors(v0, v1)
    if angle > 0:
        return tf.rotation_matrix(angle,
                                  tf.vector_product(v0, v1)
                                  )[:3,:3]
    else:
        return np.eye(3)


def fit_similarity_transform(p1, p2, max_iterations=1000, threshold=1):
    ''' Fit a similarity transform between two points sets
    '''
    # TODO (Yubin): adapt to RANSAC class

    num_points, dim = p1.shape[0:2]

    assert(p1.shape[0]==p2.shape[0])

    best_inliers= 0

    for i in xrange(max_iterations):

        rnd = np.random.permutation(num_points)
        rnd = rnd[0:dim]

        T = tf.affine_matrix_from_points(p1[rnd,:].T, p2[rnd,:].T, shear=False)
        p1h = homogeneous(p1)
        p2h = homogeneous(p2)

        errors = np.sqrt(np.sum( ( p2h.T - np.dot(T, p1h.T) ) ** 2 , axis=0 ) )

        inliers = np.argwhere(errors < threshold)[:,0]

        num_inliers = len(inliers)

        if num_inliers >= best_inliers:
            best_inliers = num_inliers
            best_T = T.copy()
            inliers = np.argwhere(errors < threshold)[:,0]

    # Estimate similarity transform with inliers
    if len(inliers)>dim+3:
        best_T = tf.affine_matrix_from_points(p1[inliers,:].T, p2[inliers,:].T, shear=False)

    return best_T, inliers


def K_from_camera(camera):
    f = float(camera['focal'])
    return np.array([[f, 0., 0.],
                     [0., f, 0.],
                     [0., 0., 1.]])


def focal_from_homography(H):
    '''Solve for w = H w H^t, with w = diag(a, a, b)

    >>> K = np.diag([0.8, 0.8, 1])
    >>> R = cv2.Rodrigues(np.array([0.3, 0, 0]))[0]
    >>> H = K.dot(R).dot(np.linalg.inv(K))
    >>> f = focal_from_homography(3 * H)
    >>> np.allclose(f, 0.8)
    True
    '''
    H = H / np.linalg.det(H)**(1.0 / 3.0)
    A = np.array([
        [H[0,0] * H[0,0] + H[0,1] * H[0,1] - 1, H[0,2] * H[0,2]    ],
        [H[0,0] * H[1,0] + H[0,1] * H[1,1]    , H[0,2] * H[1,2]    ],
        [H[0,0] * H[2,0] + H[0,1] * H[2,1]    , H[0,2] * H[2,2]    ],
        [H[1,0] * H[1,0] + H[1,1] * H[1,1] - 1, H[1,2] * H[1,2]    ],
        [H[1,0] * H[2,0] + H[1,1] * H[2,1]    , H[1,2] * H[2,2]    ],
        [H[2,0] * H[2,0] + H[2,1] * H[2,1]    , H[2,2] * H[2,2] - 1],
    ])
    _, (a,b) = nullspace(A)
    focal = np.sqrt(a / b)
    return focal


def R_from_homography(H, f1, f2):
    K1 = np.diag([f1, f1, 1])
    K2 = np.diag([f2, f2, 1])
    K2inv = np.linalg.inv(K2)
    R = K2inv.dot(H).dot(K1)
    R = project_to_rotation_matrix(R)
    return R


def count_focal_homography_inliers(f1, f2, H, p1, p2, threshold=0.02):
    R = R_from_homography(f1, f2, H)
    if R is None:
        return 0
    H = K1.dot(R).dot(K2inv)
    return count_homography_inliers(H, p1, p2, threshold)


def count_homography_inliers(H, p1, p2, threshold=0.02):
    p2map = euclidean(H.dot(homogeneous(p1).T).T)
    d = p2 - p2map
    return np.sum((d * d).sum(axis=1) < threshold**2)


def project_to_rotation_matrix(A):
    try:
        u, d, vt = np.linalg.svd(A)
    except np.linalg.linalg.LinAlgError:
        return None
    return u.dot(vt)
