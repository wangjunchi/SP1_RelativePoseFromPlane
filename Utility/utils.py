import math

import numpy as np
import cv2

def cheirality_check(R, t, K, points_1, points_2):
    # check the cheirality of the camera pose
    # R: 3x3
    # t: 3x1
    # K: 3x3
    # points_1: 2xN
    # points_2: 2xN
    # return: True if the camera pose is valid
    #         False if the camera pose is invalid

    # construct projection matrix
    t = t*2
    E1 = np.zeros((3, 4))
    E1[:3, :3] = np.eye(3)
    M1 = K @ E1

    E2 = np.zeros((3,4))
    E2[:3,:3] = R
    E2[:,3] = t[:,0]
    M2 = K @ E2

    points3d = cv2.triangulatePoints(M1, M2, points_1.T, points_2.T)
    points3d = points3d / points3d[3]
    points3d = points3d[:3]

    if np.sum(points3d[2] > 0) > 0.9 * points_1.shape[0]:
        return True
    else:
        return False


def compute_reprojection_error(E, points_1, points_2):
    # given essential matrix, compute the reprojection error
    # E: 3x3
    # points_1: 2*N
    # points_2: 2*N
    # return: reprojection error

    # sample points
    points_1 = np.concatenate([points_1, np.ones((1, points_1.shape[1]))], axis=0)
    points_2 = np.concatenate([points_2, np.ones((1, points_2.shape[1]))], axis=0)

    # compute reprojection error
    error_1 = (points_2.T @ E @ points_1).diagonal()
    error_2 = (points_1.T @ E.T @ points_2).diagonal()
    # error2 = np.sum(points_1.T @ E.T @ E @ points_1, axis=1)
    error = np.abs(error_1) + np.abs(error_2)
    return error

def compute_reprojection_error2(R, t, K, points_1, points_2):
    t = t * 2
    E1 = np.zeros((3, 4))
    E1[:3, :3] = np.eye(3)
    M1 = K @ E1

    E2 = np.zeros((3, 4))
    E2[:3, :3] = R
    E2[:, 3] = t[:, 0]
    M2 = K @ E2

    points3d = cv2.triangulatePoints(M1, M2, points_1, points_2)
    points3d = points3d / points3d[3]
    # points3d = points3d[:3]

    # project points to image 1
    points_1_proj = M1 @ points3d
    points_1_proj = points_1_proj / points_1_proj[2]
    points_1_proj = points_1_proj[:2]

    # compute reprojection error
    error = np.sum(np.abs(points_1_proj - points_1), axis=0)

    return error


def sample_points(pts_1, pts_2, num=500):
    # sample points from the matching points
    # pts_1: 2xN
    # pts_2: 2xN
    # num: number of points to sample
    # return: sampled points
    num_pts = pts_1.shape[0]
    if num_pts < num:
        return pts_1, pts_2
    else:
        idx = np.random.choice(num_pts, num, replace=False)
        return pts_1[idx, :], pts_2[idx, :]

def filter_and_sort_planes(labels):
    num_planes = int(labels.max())
    # order label according to their area
    w, h = labels.shape
    total_area = w * h
    ordered_label = np.zeros_like(labels)
    existing_labels = np.unique(labels)
    area_label = []
    for label in existing_labels:
        if label != 0:
            area = np.sum(labels == label)
            area_label.append((area, label))
    area_label.sort(key=lambda x: x[0], reverse=True)

    for i in range(len(existing_labels) - 1):
        if i >= 20:
            break  # at most 20 non-plane labels, the same as PlaneAE
        # filter plane with area less than 4% of the image
        if area_label[i][0] / total_area < 0.04:
            break
        new_label = i + 1
        old_label = area_label[i][1]
        ordered_label[labels == old_label] = new_label

    return ordered_label


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap

def map_label_to_color(labels):
    # map label to color
    # labels: HxW
    # return: HxWx3
    colors = labelcolormap(256)
    color_seg = np.stack([colors[labels, 0],
                          colors[labels, 1],
                          colors[labels, 2]], axis=2)
    return color_seg

def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q

def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()

    return err_q, err_t

def skew(x):
    # skew symmetric matrix
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
