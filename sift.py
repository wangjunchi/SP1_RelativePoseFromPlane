import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from Dataset.hypersim_image_pair_dataset import HypersimImagePairDataset

fx, fy = 886.81, 886.81
cx, cy = 512, 384

MIN_MATCH_COUNT = 8
FLANN_INDEX_KDTREE = 1

def getBestMatches(ref_des, q_des, ratio=0.9):
    bf = cv2.BFMatcher(crossCheck=True)

    matches = bf.match(ref_des, q_des)  # first k best matches

    return matches


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

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


def process_scene(scene_name):
    # load dataset
    dataset = HypersimImagePairDataset(data_dir="/cluster/project/infk/cvg/students/junwang/hypersim", scene_name=scene_name)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    err_q_list = []
    err_t_list = []
    for step, batch in tqdm(enumerate(loader), total=len(loader)):
        # if step == 5:
        #     break
        sample_1 = batch['sample_1']
        sample_2 = batch['sample_2']
        gt_match = batch['gt_match']

        # visualize the image pair using plt
        image_1 = sample_1['image'][0].numpy() * 255
        image_1 = image_1.astype(np.uint8).transpose(1, 2, 0)
        image_2 = sample_2['image'][0].numpy() * 255
        image_2 = image_2.astype(np.uint8).transpose(1, 2, 0)

        # convert to gray scale
        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        # get gt camera pose
        gt_pose_1 = sample_1['camera_pose'][0].numpy()
        gt_pose_2 = sample_2['camera_pose'][0].numpy()
        gt_relative_pose = np.matmul(np.linalg.inv(gt_pose_2), gt_pose_1)

        # using sift to extract features
        sift = cv2.SIFT_create()
        ref_kp, ref_des = sift.detectAndCompute(gray_1, None)
        q_kp, q_des = sift.detectAndCompute(gray_2, None)
        best_matches = getBestMatches(ref_des, q_des, ratio=0.7)

        img3 = cv2.drawMatches(image_1, ref_kp, image_2, q_kp, best_matches[:], None)
        # plt.imshow(img3)
        # plt.show()
        # pass

        # compute the essential matrix
        # Camera Intrisics matrix
        K = np.float64([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])

        src_pts = np.float32([ref_kp[m.queryIdx].pt for m in best_matches])
        dst_pts = np.float32([q_kp[m.trainIdx].pt for m in best_matches])

        if len(dst_pts) < 5:
            continue

        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.99, threshold=0.5)

        # if step % 10 == 0:
        #     # draw the matching points that pass the RANSAC
        #     draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
        #                     singlePointColor=None,
        #                     matchesMask=mask.ravel().tolist(),  # draw only inliers
        #                     flags=2)
        #     img4 = cv2.drawMatches(image_1, ref_kp, image_2, q_kp, best_matches[:], None, **draw_params)
        #     plt.imshow(img4)
        #     plt.savefig("sift_preview/{}_{}.jpg".format(scene_name, step), dpi=300)
        #     # plt.show()
        #     # pass

        points, R_est, t_est, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, K, mask)

        # evaluate the estimated pose
        err_q, err_t = evaluate_R_t(gt_relative_pose[:3, :3], gt_relative_pose[:3, 3], R_est, t_est)
        err_q = err_q / np.pi * 180
        err_t = err_t / np.pi * 180
        err_q_list.append(err_q)
        err_t_list.append(err_t)
        print('sample {}: err_q: {:.4f}, err_t: {:.4f}'.format(step, err_q, err_t))
        # pass

    print('mean err_q: {:.4f}, mean err_t: {:.4f}'.format(np.mean(err_q_list), np.mean(err_t_list)))
    print('median err_q: {:.4f}, median err_t: {:.4f}'.format(np.median(err_q_list), np.median(err_t_list)))

    return {'err_q' : err_q_list, 
            'err_t': err_t_list}



if __name__ == '__main__':
    root_dir = "/cluster/project/infk/cvg/students/junwang/hypersimLite"
    # scene_name = "ai_001_010"
    # scene_list = ['ai_001_001', 'ai_001_002', 'ai_001_010']
    test_list_path = os.path.join(root_dir, "test_scenes.txt")
    with open(test_list_path, "r") as f:
        test_scene = f.read().splitlines()
    
    err_q = []
    err_t = []
    count = 1
    for scene in test_scene:
        # if count == 3:
        #     break
        print("Processing scene {}".format(scene))
        print("No {} / {}".format(count, len(test_scene)))
        err = process_scene(scene)
        err_q.append(err['err_q'])
        err_t.append(err['err_t'])
        count += 1
    
    err_q = np.concatenate(err_q)
    err_t = np.concatenate(err_t)
    print('mean err_q: {:.4f}, mean err_t: {:.4f}'.format(np.mean(err_q), np.mean(err_t)))
    print('median err_q: {:.4f}, median err_t: {:.4f}'.format(np.median(err_q), np.median(err_t)))



