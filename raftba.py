# run the whole pipeline
# step 1: read images
# step 2: extract planes
# step 3: matching planes
# step 4: estimate homography
# step 5: extract camera pose
import argparse
import math

import cv2
import os
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.transform import Rotation


from Dataset.hypersim_image_pair_dataset import HypersimImagePairDataset
from PlaneDetection.predict_planes import predict_image, load_plane_detector
from PlaneMatching.predict_matching import predict_matching, load_gem_model
from RaftBA.predict_homography import load_model, predict_homography, extract_plane_patch, restoreHomography
# from HomographyEstimation.predict_homography import predict_homography, extract_plane_patch, load_model, restoreHomography
from Utility.utils import *


def process_scene(scene_name, plane_model, gem_model, raft_model):
    # load dataset
    dataset = HypersimImagePairDataset(data_dir="/cluster/project/infk/cvg/students/junwang/hypersim", scene_name=scene_name)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # evaluation
    err_q_essential = []
    err_t_essential = []
    err_q_best_homography = []
    err_t_best_homography = []
    err_q_ranked_homography = []
    err_t_ranked_homography = []
    for step, batch in tqdm(enumerate(loader), total=len(loader), disable=False):
        # if step>5:
        #     break

        sample_1 = batch['sample_1']
        sample_2 = batch['sample_2']
        gt_match = batch['gt_match']

        # visualize the image pair using plt
        image_1 = sample_1['image'][0].numpy().transpose(1, 2, 0)
        image_2 = sample_2['image'][0].numpy().transpose(1, 2, 0)

        segmentation_1 = sample_1['planes'][0, :,:,1].numpy()
        segmentation_2 = sample_2['planes'][0, :,:,1].numpy()
        # map label to color
        segmentation_1 = map_label_to_color(segmentation_1.astype(np.int32))
        segmentation_2 = map_label_to_color(segmentation_2.astype(np.int32))
        # plt.subplot(2, 2, 1)
        # plt.imshow(image_1)
        # plt.subplot(2, 2, 2)
        # plt.imshow(image_2)
        # plt.subplot(2, 2, 3)
        # plt.imshow(segmentation_1)
        # plt.subplot(2, 2, 4)
        # plt.imshow(segmentation_2)
        # plt.show()
        pass

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

        labels_1_color = map_label_to_color(labels_1.astype(np.int32))
        labels_2_color = map_label_to_color(labels_2.astype(np.int32))
        # visualize the plane mask using plt
        # plt.subplot(2, 2, 1)
        # plt.imshow(image_1)
        # plt.subplot(2, 2, 2)
        # plt.imshow(image_2)
        # plt.subplot(2, 2, 3)
        # plt.imshow(labels_1_color)
        # plt.subplot(2, 2, 4)
        # plt.imshow(labels_2_color)
        # plt.show()

        # predict matching
        plane_mask_1 = torch.from_numpy(labels_1)
        plane_mask_2 = torch.from_numpy(labels_2)
        image_1 = torch.from_numpy(image_1).permute(2, 0, 1)
        image_2 = torch.from_numpy(image_2).permute(2, 0, 1)

        pred_match = predict_matching(image_1, image_2, plane_mask_1, plane_mask_2, gem_model)

        # visualize the matching using plt
        # change the label of the plane mask to the same as the matching
        new_labels_2 = np.zeros_like(labels_2)
        for i in range(len(pred_match)):
            matching_label = int(pred_match[i]) + 1
            if matching_label != 0:
                new_labels_2[labels_2 == matching_label] = i + 1

        new_labels_1_color = map_label_to_color(labels_1.astype(np.int32))
        new_labels_2_color = map_label_to_color(new_labels_2.astype(np.int32))
        # plt.subplot(2, 2, 1)
        # plt.imshow(image_1.permute(1, 2, 0).numpy())
        # plt.subplot(2, 2, 2)
        # plt.imshow(image_2.permute(1, 2, 0).numpy())
        # plt.subplot(2, 2, 3)
        # plt.imshow(new_labels_1_color)
        # plt.subplot(2, 2, 4)
        # plt.imshow(new_labels_2_color)
        # plt.show()
        pass

        # get gt camera pose
        gt_pose_1 = sample_1['camera_pose'][0].numpy()
        gt_pose_2 = sample_2['camera_pose'][0].numpy()
        gt_relative_pose = np.matmul(np.linalg.inv(gt_pose_2), gt_pose_1)

        # load model
        model = raft_model
        model.eval()
        model.cuda()
        # resize image and plane mask to 192*256
        image_1 = cv2.resize(image_1.permute(1, 2, 0).numpy(), (512, 384))
        image_2 = cv2.resize(image_2.permute(1, 2, 0).numpy(), (512, 384))
        labels_1 = cv2.resize(labels_1, (512, 384), interpolation=cv2.INTER_NEAREST)
        labels_2 = cv2.resize(labels_2, (512, 384), interpolation=cv2.INTER_NEAREST)
        K = np.array([[886.81, 0, 512], [0, 886.81, 384], [0, 0, 1]])

        gt_r = gt_relative_pose[:3, :3]
        gt_angle = Rotation.from_matrix(gt_r)
        gt_angle = gt_angle.as_euler('xyz', degrees=True)

        print('gt_angle: ', gt_angle)
        points_src = []
        points_dst = []
        points_src_all = []
        points_dst_all = []
        proposed_r = []
        proposed_t = []
        err_q_homo = []
        err_t_homo = []
        err_pose = []
        min_num_points = 128*128 + 1

        best_homo_id = -1
        best_pose_id = -1
        best_homo_metric = 1000
        best_pose_metric = 1000
        err_q_essential = []
        err_t_essential = []
        err_q_best_homography = []
        err_t_best_homography = []
        err_q_best_pose = []
        err_t_best_pose = []
        err_q_reproj_homography = []
        err_t_reproj_homography = []
        err_q_epipo_homography = []
        err_t_epipo_homography = []
        id = 0
        pose_id = 0
        for i in range(len(pred_match)):
            matching_label = int(pred_match[i]) + 1
            if matching_label != 0:
                plane_mask_1 = labels_1 == i + 1
                plane_mask_2 = labels_2 == matching_label
                image_patch_1, mask_1, origin_x_1, origin_y_1 = extract_plane_patch(image_1, plane_mask_1)
                image_patch_2, mask_2, origin_x_2, origin_y_2 = extract_plane_patch(image_2, plane_mask_2)

                # # mask out the background
                image_patch_1 = image_patch_1 * mask_1[..., None]
                image_patch_2 = image_patch_2 * mask_2[..., None]
                estimates_grid, H = predict_homography(model, image_patch_1, image_patch_2, mask_1)
                estimates_grid = estimates_grid[0]

                # sample points
                h, w, _ = image_patch_1.shape
                X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                                   np.linspace(0, h - 1, h))
                X, Y = X.flatten(), Y.flatten()
                pts_src = np.stack([X, Y], axis=1)
                pts_src = pts_src+np.array([origin_x_1, origin_y_1])
                pts_src = pts_src * 2

                pts_dst = estimates_grid.reshape(2, -1).detach().cpu().numpy()
                pts_dst = np.transpose(pts_dst) + pts_src

                sampled_pts_src, sampled_pts_dst = sample_points(pts_src, pts_dst, 100)

                points_src.append(sampled_pts_src)
                points_dst.append(sampled_pts_dst)
                points_src_all.append(pts_src)
                points_dst_all.append(pts_dst)
            
                origin_1 = (origin_x_1, origin_y_1)
                origin_2 = (origin_x_2, origin_y_2)
                H_full = restoreHomography(H, origin_1, origin_2)
                n, Rs, Ts, _ = cv2.decomposeHomographyMat(H_full, K)
                for i in range(n):
                    print('check solution ', i)
                    R = Rs[i]
                    T = Ts[i]
                    # if cheirality_check(R, T, K, ref_point_src, ref_point_dst):
                    if True:
                        print('R: ', R)
                        print('T: ', T)
                        angle = Rotation.from_matrix(R)
                        angle = angle.as_euler('xyz', degrees=True)
                        proposed_r.append(R)
                        proposed_t.append(T)
                        # print('valid homo angle: ', angle)
                        # print('valid homo translation: ', T)

                        # evaluate
                        err_q, err_t = evaluate_R_t(gt_r, gt_relative_pose[:3, 3], R, T)
                        err_q_homo.append(err_q)
                        err_t_homo.append(err_t)
                        print('err_q for homo {}: '.format(i), err_q)
                        print('err_t: for homo {}: '.format(i), err_t)

                        E = np.linalg.inv(K.T) @ R @ skew(T[:,0]) @ np.linalg.inv(K)
    #         # convert E to homogenous coordinates

                        error_epi = compute_reprojection_error(E, sampled_pts_src.T, sampled_pts_dst.T)
                        error_epi = np.mean(error_epi)
                        print("error_proj = ", error_epi)
                        err_pose.append(error_epi)

                        if (err_q+err_t/1.0) < best_homo_metric:
                            best_homo_id = id
                            best_homo_metric = err_q+err_t/1.0
                        id += 1

                        if error_epi<best_pose_metric:
                            best_pose_id = pose_id
                            best_pose_metric = error_epi
                        pose_id += 1
                    else:
                        angle = Rotation.from_matrix(R)
                        angle = angle.as_euler('xyz', degrees=True)
                        # print('not valid homo angle', angle)
                        # print('not valid homo translation: ', T)
                    # convert rotation matrix to euler angles
                    # r = Rotation.from_matrix(R)
                    # angles = r.as_euler("xyz", degrees=True)
                    # print('est angles from homography: ', angles)
        if best_homo_id >= 0:
            err_q_best_homography.append(err_q_homo[best_homo_id])
            err_t_best_homography.append(err_t_homo[best_homo_id])
            print('err_q for best homo ', err_q_homo[best_homo_id])
            print('err_t for best homo ', err_t_homo[best_homo_id])
        
        if best_pose_id >=0:
            err_q_best_pose.append(err_q_homo[best_pose_id])
            err_t_best_pose.append(err_t_homo[best_pose_id])
            print('err_q for best pose ', err_q_homo[best_pose_id])
            print('err_t for best pose ', err_t_homo[best_pose_id])


    #     # concat the sampled points
    #     if len(points_src) == 0:
    #         continue

    #     points_src_matching = np.concatenate(points_src, axis=0)
    #     points_dst_matching = np.concatenate(points_dst, axis=0)
    #     E, mask = cv2.findEssentialMat(points_src_matching, points_dst_matching, K, cv2.RANSAC, 0.999, 0.5, None)
    #     # restore pose from essential matrix
    #     points_src_matching = points_src_matching[mask.flatten() == 1]
    #     points_dst_matching = points_dst_matching[mask.flatten() == 1]
    #     _, R, T, _ = cv2.recoverPose(E, points_src_matching, points_dst_matching, K)

    #     # print('est angle from matching: ', Rotation.from_matrix(R).as_euler('xyz', degrees=True))
    #     err_q, err_t = evaluate_R_t(gt_r, gt_relative_pose[:3, 3], R, T)
    #     err_q_essential.append(err_q)
    #     err_t_essential.append(err_t)
    #     print('from essential: err_q: ', err_q)
    #     print('from essential: err_t: ', err_t)

    #     # compute reprojection error for each proposed solution
    #     points_src_homo = []
    #     points_dst_homo = []
    #     for i in range(len(points_src)):
    #         plane_points_src = points_src_all[i]
    #         plane_points_dst = points_dst_all[i]
    #         sampled_pts_src, sampled_pts_dst = sample_points(plane_points_src, plane_points_dst, 100)
    #         points_src_homo.append(sampled_pts_src)
    #         points_dst_homo.append(sampled_pts_dst)

    #     points_src_homo = np.concatenate(points_src_homo, axis=0)
    #     points_dst_homo = np.concatenate(points_dst_homo, axis=0)


    #     num_proposed = len(proposed_r)
    #     assert num_proposed == len(proposed_t), 'number of proposed rotation and translation should be the same'
    #     reprojection_error = []
    #     epipolar_error = []
    #     for i in range(num_proposed):
    #         R = proposed_r[i]
    #         T = proposed_t[i]
    #         # recover essential matrix
    #         E = np.linalg.inv(K.T) @ R @ skew(T[:,0]) @ np.linalg.inv(K)
    #         # convert E to homogenous coordinates

    #         error_epi = compute_reprojection_error(E, points_src_homo.T, points_dst_homo.T)
    #         error_proj = compute_reprojection_error2(R, T, K, points_src_homo.T, points_dst_homo.T)
    #         error_epi = np.mean(error_epi)
    #         error_proj = np.mean(error_proj)
    #         reprojection_error.append(error_proj)
    #         epipolar_error.append(error_epi)

    #     min_proj_error_index = np.argmin(reprojection_error)
    #     min_epi_error_index = np.argmin(epipolar_error)

    #     # evaluate
    #     err_q, err_t = evaluate_R_t(gt_r, gt_relative_pose[:3, 3], proposed_r[min_proj_error_index], proposed_t[min_proj_error_index])
    #     err_q_reproj_homography.append(err_q)
    #     err_t_reproj_homography.append(err_t)
    #     print('from minimal reprojection error homography: err_q: ', err_q)
    #     print('from minimal reprojection error homography: err_t: ', err_t)

    #     err_q, err_t = evaluate_R_t(gt_r, gt_relative_pose[:3, 3], proposed_r[min_epi_error_index], proposed_t[min_epi_error_index])
    #     err_q_epipo_homography.append(err_q)
    #     err_t_epipo_homography.append(err_t)
    #     print('from minimal epipolar error homography: err_q: ', err_q)
    #     print('from minimal epipolar error homography: err_t: ', err_t)

    # # print('err_q_essential: ', np.mean(err_q_essential))
    # # print('err_t_essential: ', np.mean(err_t_essential))
    # print('err_q_best_homography: ', np.mean(err_q_best_homography))
    # print('err_t_best_homography: ', np.mean(err_t_best_homography))
    # # print('err_q_ranked_homography: ', np.mean(err_q_ranked_homography))
    # # print('err_t_ranked_homography: ', np.mean(err_t_ranked_homography))

    # return {
    #     'err_q_essential' : err_q_essential,
    #     'err_t_essential' : err_t_essential,
    #     'err_q_best_homography' : err_q_best_homography,
    #     'err_t_best_homography' : err_t_best_homography,
    #     'err_q_ranked_homography' : err_q_ranked_homography,
    #     'err_t_ranked_homography' : err_t_ranked_homography
    # }

    return {
        'err_q_best_homography' : err_q_best_homography,
        'err_t_best_homography' : err_t_best_homography,
        # 'err_q_essential': err_q_essential,
        # 'err_t_essential': err_t_essential,
        'err_q_reproj_homography': err_q_best_pose,
        'err_t_reproj_homography': err_t_best_pose,
        # 'err_q_epipo_homography': err_q_epipo_homography,
        # 'err_t_epipo_homography': err_t_epipo_homography,
    }

def main():
    # raft args
    parser = argparse.ArgumentParser(description='Raft evaluation on HPatches')
    # Paths
    parser.add_argument('--cfg-file', type=str, default='/cluster/project/infk/cvg/students/junwang/SP1_wholePipeline/HomographyEstimation/config/s-coco/raft-orig-eval.yaml',
                        help='path to training transformation csv folder')
    parser.add_argument('--dataset_dir', type=str,
                        default='/cluster/project/infk/cvg/students/junwang/hypersim',
                        help='path to folder containing training images')
    parser.add_argument('--ckpt', type=str, default='/cluster/project/infk/cvg/students/junwang/SP1_wholePipeline/HomographyEstimation/trained_models/model_raft_scoco.pth',
                        help='Checkpoint to use')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='evaluation batch size')
    parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

    raft_args = parser.parse_args()
    with open(raft_args.cfg_file, 'r') as file:
        config = yaml.full_load(file)

    pec_weights_path = "/cluster/project/infk/cvg/students/junwang/SP1_wholePipeline/PlaneDetection/trained_models/pec_junchi.tar"
    plane_model = load_plane_detector(pec_weights_path)
    raft_weight_path = "/cluster/project/infk/cvg/students/junwang/SP1_wholePipeline/RaftBA/trained_models/hypersim_model_24_epoch.pth"
    raft_model = load_model(raft_weight_path)
    network_path = '/cluster/project/infk/cvg/students/junwang/SP1_wholePipeline/PlaneMatching/trained_models/gl18-tl-resnet50-gem-w-83fdc30.pth'
    gem_model = load_gem_model(network_path)

    root_dir = "/cluster/project/infk/cvg/students/junwang/hypersimLite"
    test_list_path = os.path.join(root_dir, "test_scenes.txt")
    with open(test_list_path, "r") as f:
        test_scene = f.read().splitlines()
    
    err_q_essential = []
    err_t_essential = []
    err_q_best_homography = []
    err_t_best_homography = []
    err_q_reproj_homography = []
    err_t_reproj_homography = []
    err_q_epipo_homography = []
    err_t_epipo_homography = []
    count = 1
    for scene in test_scene:
        # if count == 3:
        #     break
        print("---------------------------------------------------------------")
        print("Processing scene {}".format(scene))
        print("No {} / {}".format(count, len(test_scene)))
        err = process_scene(scene, plane_model=plane_model, gem_model=gem_model, raft_model=raft_model)
        # err_q_essential.append(err['err_q_essential'])
        # err_t_essential.append(err['err_t_essential'])
        err_q_best_homography.append(err['err_q_best_homography'])
        err_t_best_homography.append(err['err_t_best_homography'])
        err_q_reproj_homography.append(err['err_q_reproj_homography'])
        err_t_reproj_homography.append(err['err_t_reproj_homography'])
        # err_q_epipo_homography.append(err['err_q_epipo_homography'])
        # err_t_epipo_homography.append(err['err_t_epipo_homography'])
        # err_q_ranked_homography.append(err['err_q_ranked_homography'])
        # err_t_ranked_homography.append(err['err_t_ranked_homography'])
        count += 1


    # err_q_essential = np.concatenate(err_q_essential)
    # err_t_essential = np.concatenate(err_t_essential)
    err_q_best_homography = np.concatenate(err_q_best_homography)
    err_t_best_homography = np.concatenate(err_t_best_homography)
    err_q_reproj_homography = np.concatenate(err_q_reproj_homography)
    err_t_reproj_homography = np.concatenate(err_t_reproj_homography)
    # err_q_ranked_homography = np.concatenate(err_q_ranked_homography)
    # err_t_ranked_homography = np.concatenate(err_t_ranked_homography)

    # print('mean err_q_essential: {:.4f}, mean err_t_essential: {:.4f}'.format(np.mean(err_q_essential), np.mean(err_t_essential)))
    print('mean err_q_best_homography: {:.4f}, mean err_t_best_homography: {:.4f}'.format(np.mean(err_q_best_homography), np.mean(err_t_best_homography)))
    print('mean err_q_reproj_homography: {:.4f}, mean err_t_reproj_homography: {:.4f}'.format(np.mean(err_q_reproj_homography), np.mean(err_t_reproj_homography)))
    
    # print('median err_q_essential: {:.4f}, median err_t_essential: {:.4f}'.format(np.median(err_q_essential), np.median(err_t_essential)))
    print('median err_q_best_homography: {:.4f}, median err_t_best_homography: {:.4f}'.format(np.median(err_q_best_homography), np.median(err_t_best_homography)))
    print('median err_q_reproj_homography: {:.4f}, median err_t_reproj_homography: {:.4f}'.format(np.median(err_q_reproj_homography), np.median(err_t_reproj_homography)))



if __name__ == '__main__':
    main()



