import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset.hypersim_image_pair_dataset import HypersimImagePairDataset
from PlaneDetection.predict_planes import predict_image, load_plane_detector
from PlaneMatching.predict_matching import predict_matching, load_gem_model
from SuperPoint.predict_points import load_sp_model
from Utility.utils import *

fx, fy = 886.81, 886.81
cx, cy = 512, 384

MIN_MATCH_COUNT = 8
FLANN_INDEX_KDTREE = 1

def getBestMatches(ref_des, q_des, ratio=0.9):
    bf = cv2.BFMatcher()
    # return  bf.match(ref_des, q_des)
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    ref_des = ref_des.transpose()
    q_des = q_des.transpose()
    matches = bf.knnMatch(ref_des, q_des, k=2)  # first k best matches

    best_matches = []
    best_matches_numpy = []

    # from Lowe's
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            best_matches_numpy.append((m.queryIdx, m.trainIdx))
            best_matches.append(m)

    return best_matches, best_matches_numpy


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
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
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


def main():
    # load dataset
    dataset = HypersimImagePairDataset(data_dir="/home/junchi/sp1/dataset/hypersim", scene_name='ai_001_001')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    pec_weights_path = "/home/junchi/sp1/project_junchi/pythonProject/whole_pipeline/PlaneDetection/trained_models/pec_junchi.tar"
    plane_model = load_plane_detector(pec_weights_path)
    gem_weights_path = '/home/junchi/sp1/project_junchi/pythonProject/whole_pipeline/PlaneMatching/trained_models/gl18-tl-resnet50-gem-w-83fdc30.pth'
    gem_model = load_gem_model(gem_weights_path)

    sp_model = load_sp_model()
    torch.cuda.empty_cache()

    err_q_list = []
    err_t_list = []
    for step, batch in tqdm(enumerate(loader), total=len(loader)):
        sample_1 = batch['sample_1']
        sample_2 = batch['sample_2']
        gt_match = batch['gt_match']

        image_1 = sample_1['image'][0].numpy().transpose(1, 2, 0)
        image_2 = sample_2['image'][0].numpy().transpose(1, 2, 0)

        # predict planes
        labels_1, depth_1, normals_1 = predict_image(sample_1['image'], plane_model)
        labels_2, depth_2, normals_2 = predict_image(sample_2['image'], plane_model)
        labels_1 = labels_1[0, 0, :, :].cpu().numpy()
        labels_2 = labels_2[0, 0, :, :].cpu().numpy()

        # resize the segmentation to the same size as the image
        labels_1 = cv2.resize(labels_1, (image_1.shape[1], image_1.shape[0]), interpolation=cv2.INTER_NEAREST)
        labels_2 = cv2.resize(labels_2, (image_2.shape[1], image_2.shape[0]), interpolation=cv2.INTER_NEAREST)

        # filter and sort planes
        labels_1 = filter_and_sort_planes(labels_1)
        labels_2 = filter_and_sort_planes(labels_2)

        # predict matching
        plane_mask_1 = torch.from_numpy(labels_1)
        plane_mask_2 = torch.from_numpy(labels_2)
        image_1 = torch.from_numpy(image_1).permute(2, 0, 1)
        image_2 = torch.from_numpy(image_2).permute(2, 0, 1)

        pred_match = predict_matching(image_1, image_2, plane_mask_1, plane_mask_2, gem_model)

        # change back to numpy
        image_1 = image_1.permute(1, 2, 0).numpy()
        grey_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
        image_2 = image_2.permute(1, 2, 0).numpy()
        grey_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

        # get gt camera pose
        gt_pose_1 = sample_1['camera_pose'][0].numpy()
        gt_pose_2 = sample_2['camera_pose'][0].numpy()
        gt_relative_pose = np.matmul(np.linalg.inv(gt_pose_2), gt_pose_1)

        # using superpoint to get keypoints and descriptors
        ref_kp, ref_des, _ = sp_model.run(grey_1)
        mask_1 = ref_kp[2, :] > 0.2
        ref_kp = ref_kp[:2, mask_1].transpose()
        ref_des = ref_des[:, mask_1]

        q_kp, q_des, _ = sp_model.run(grey_2)
        mask_2 = q_kp[2, :] > 0.2
        q_kp = q_kp[:2, mask_2].transpose()
        q_des = q_des[:, mask_2]

        points_src = []
        points_dst = []
        for i in range(len(pred_match)):
            sift = cv2.SIFT_create()
            matching_label = int(pred_match[i]) + 1
            if matching_label != 0:
                plane_mask_1 = labels_1 == i + 1
                plane_mask_2 = labels_2 == matching_label

                # filter keypoints and descriptor inside the plane
                ref_kp_in_plane = ref_kp[plane_mask_1[ref_kp[:, 1].astype(int), ref_kp[:, 0].astype(int)]]
                ref_des_in_plane = ref_des[:, plane_mask_1[ref_kp[:, 1].astype(int), ref_kp[:, 0].astype(int)]]
                q_kp_in_plane = q_kp[plane_mask_2[q_kp[:, 1].astype(int), q_kp[:, 0].astype(int)]]
                q_des_in_plane = q_des[:, plane_mask_2[q_kp[:, 1].astype(int), q_kp[:, 0].astype(int)]]

                if len(ref_kp_in_plane) < 4 or len(q_kp_in_plane) < 4:
                    continue
                best_matches, best_matches_numpy = getBestMatches(ref_des_in_plane, q_des_in_plane, ratio=0.95)
                best_matches_numpy = np.array(best_matches_numpy)

                src_pts = np.float32([ref_kp_in_plane[m.queryIdx] for m in best_matches])
                dst_pts = np.float32([q_kp_in_plane[m.trainIdx] for m in best_matches])

                # for debug
                img1_color = (image_1*255).astype(np.uint8)
                img2_color = (image_2*255).astype(np.uint8)
                img3 = draw_matches(img1_color, ref_kp_in_plane, img2_color, q_kp_in_plane, best_matches_numpy[:], None)
                # plt.imshow(img3)
                # plt.show()
                # pass

                points_src.append(src_pts)
                points_dst.append(dst_pts)
        # concat the sampled points
        K = np.float64([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])
        points_src_matching = np.concatenate(points_src, axis=0)
        points_dst_matching = np.concatenate(points_dst, axis=0)
        E, mask = cv2.findEssentialMat(points_src_matching, points_dst_matching, K, method=cv2.RANSAC, prob=0.99, threshold=0.5)

        points, R_est, t_est, mask_pose = cv2.recoverPose(E, points_src_matching, points_dst_matching, K)

        # evaluate the estimated pose
        err_q, err_t = evaluate_R_t(gt_relative_pose[:3, :3], gt_relative_pose[:3, 3], R_est, t_est)
        err_q_list.append(err_q)
        err_t_list.append(err_t)
        print('err_q: {:.4f}, err_t: {:.4f}'.format(err_q, err_t))
        pass

    print('mean err_q: {:.4f}, mean err_t: {:.4f}'.format(np.mean(err_q_list), np.mean(err_t_list)))
    print('median err_q: {:.4f}, median err_t: {:.4f}'.format(np.median(err_q_list), np.median(err_t_list)))




if __name__ == '__main__':
    main()


