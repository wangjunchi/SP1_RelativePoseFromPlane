import numpy as np
import os
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset.hypersim_image_pair_dataset import HypersimImagePairDataset
from SuperPoint.predict_points import load_sp_model
from Utility.utils import *
fx, fy = 886.81, 886.81
cx, cy = 512, 384

MIN_MATCH_COUNT = 8
FLANN_INDEX_KDTREE = 1

def getBestMatches(ref_des, q_des, ratio=0.9):
    bf = cv2.BFMatcher(crossCheck=True)

    ref_des = ref_des.transpose()
    q_des = q_des.transpose()
    matches = bf.match(ref_des, q_des)  # first k best matches

    return matches


def draw_matches(img1, kp1, img2, kp2, matches, color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: ndarray [n1, 2]
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: ndarray [n2, 2]
        matches: ndarray [n_match, 2]
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 4
    thickness = 2
    if type(color) is not None:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if type(color) is None:
            c = np.random.randint(0, 256, 3) if len(img1.shape) == 3 else np.random.randint(0, 256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        c = tuple(map(int, c))
        end1 = tuple(np.round(kp1[m[0]]).astype(int))
        end2 = tuple(np.round(kp2[m[1]]).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    return new_img

def process_scene(scene_name, sp_model):
    # load dataset
    dataset = HypersimImagePairDataset(data_dir="/cluster/project/infk/cvg/students/junwang/hypersim", scene_name=scene_name)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    err_q_list = []
    err_t_list = []
    for step, batch in tqdm(enumerate(loader), total=len(loader)):
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
        gray_1 = gray_1.astype(np.float32) / 255.0
        gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        gray_2 = gray_2.astype(np.float32) / 255.0

        # get gt camera pose
        gt_pose_1 = sample_1['camera_pose'][0].numpy()
        gt_pose_2 = sample_2['camera_pose'][0].numpy()
        gt_relative_pose = np.matmul(np.linalg.inv(gt_pose_2), gt_pose_1)

        # using superpoint to get keypoints and descriptors
        ref_kp, ref_des, _ = sp_model.run(gray_1)
        mask_1 = ref_kp[2, :] > 0.2
        ref_kp = ref_kp[:2, mask_1].transpose()
        ref_des = ref_des[:, mask_1]

        q_kp, q_des, _ = sp_model.run(gray_2)
        mask_2 = q_kp[2, :] > 0.2
        q_kp = q_kp[:2, mask_2].transpose()
        q_des = q_des[:, mask_2]

        best_matches = getBestMatches(ref_des, q_des, ratio=0.95)

        # compute the essential matrix
        # Camera Intrisics matrix
        K = np.float64([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])

        src_pts = np.float32([ref_kp[m.queryIdx] for m in best_matches])
        dst_pts = np.float32([q_kp[m.trainIdx] for m in best_matches])

        if len(src_pts) <= 5:
            continue

        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.99, threshold=0.5)

        # draw the matching points that pass the RANSAC
        if step % 10 ==0:
            valid_src_pts = src_pts[mask.ravel() == 1]
            valid_dst_pts = dst_pts[mask.ravel() == 1]
            matching = np.concatenate([[np.arange(len(valid_dst_pts))], [np.arange(len(valid_dst_pts))]], axis=0).T
            img4 = draw_matches(image_1, valid_src_pts, image_2, valid_dst_pts, matching, np.array([0, 255, 0]))    
            plt.imshow(img4)
            plt.savefig("sp_preview/{}_{}.jpg".format(scene_name, step))
            # plt.show()
        pass

        points, R_est, t_est, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, K)

        # evaluate the estimated pose
        err_q, err_t = evaluate_R_t(gt_relative_pose[:3, :3], gt_relative_pose[:3, 3], R_est, t_est)
        err_q = err_q / np.pi * 180
        err_t = err_t / np.pi * 180
        err_q_list.append(err_q)
        err_t_list.append(err_t)
        print('err_q: {:.4f}, err_t: {:.4f}'.format(err_q, err_t))
        # pass

    print('mean err_q: {:.4f}, mean err_t: {:.4f}'.format(np.mean(err_q_list), np.mean(err_t_list)))
    print('median err_q: {:.4f}, median err_t: {:.4f}'.format(np.median(err_q_list), np.median(err_t_list)))
    
    return {'err_q' : err_q_list, 
            'err_t': err_t_list}


if __name__ == '__main__':
    root_dir = "/cluster/project/infk/cvg/students/junwang/hypersimLite"
    sp_model = load_sp_model()
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
        err = process_scene(scene, sp_model)
        err_q.append(err['err_q'])
        err_t.append(err['err_t'])
        count += 1
    
    err_q = np.concatenate(err_q)
    err_t = np.concatenate(err_t)
    print('mean err_q: {:.4f}, mean err_t: {:.4f}'.format(np.mean(err_q), np.mean(err_t)))
    print('median err_q: {:.4f}, median err_t: {:.4f}'.format(np.median(err_q), np.median(err_t)))

    # process_scene('ai_001_001')



