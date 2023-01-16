# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os.path as path
import os
import random

import cv2
import numpy as np
import pandas as pd
import h5py

import open3d as o3d
from scipy.spatial import ConvexHull
import itertools

from tqdm import tqdm


def convert_distance_to_depth_map(depth, focal_length):
    """This function transformed the distance to the center to a more commonly known depth map.

    Function from: https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697

    Parameters:
        depth (np.ndarray): Depth to be transformed
        focal_length (float): Focal length of the scene.
    """
    height, width = np.array(depth).shape
    npy_image_plane_x = np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width) \
                            .reshape(1, width).repeat(height, 0).astype(np.float32)[:, :, None]
    npy_image_plane_y = np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height) \
                            .reshape(height, 1).repeat(width, 1).astype(np.float32)[:, :, None]
    npy_image_plane_z = np.full([height, width, 1], focal_length, np.float32)
    npy_image_plane = np.concatenate([npy_image_plane_x, npy_image_plane_y, npy_image_plane_z], 2)

    return depth / np.linalg.norm(npy_image_plane, 2, 2) * focal_length


def get_sample_pd(sample_list_file, root_dir):
    image_information = pd.read_csv(path.join(root_dir, sample_list_file))
    # all_files = image_information
    all_files = image_information[(image_information["split_partition_name"] == "train") | (image_information["split_partition_name"] == "test")]

    # check which folders are downloaded
    directories = []
    for root, dirs, files in os.walk(root_dir):
        # print("Found dir: ", dirs)
        directories = dirs
        break
    # print(directories)
    return all_files[all_files["scene_name"].isin(directories)].copy().reset_index()

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

def generate_color(idx):
    cmap = np.zeros((1, 3), dtype=np.uint8)
    r = 0
    g = 0
    b = 0
    id = idx
    for j in range(7):
        str_id = uint82bin(id)
        r = r ^ (np.uint8(str_id[-1]) << (7 - j))
        g = g ^ (np.uint8(str_id[-2]) << (7 - j))
        b = b ^ (np.uint8(str_id[-3]) << (7 - j))
        id = id >> 3
    cmap[0, 0] = b
    cmap[0, 1] = g
    cmap[0, 2] = r
    return cmap/255.0

# def intersect_bbox(box1, box2):
#     # calculate intersection of bounding boxes
#
plane_list = []

def check_overlap(sample_1, sample_2):
    # create point cloud from depth using open3d
    K = o3d.camera.PinholeCameraIntrinsic(1024, 768, 886.81, 886.81, 512, 384)
    # H = np.linalg.inv(h)
    H = sample_1["e_matrix"]
    depth_1 = o3d.geometry.Image(sample_1["depth"])
    pcd_1 = o3d.geometry.PointCloud.create_from_depth_image(depth_1, K)
    # convert to world coordinates
    pcd_1 = pcd_1.transform(H)
    # convert to img2 coordinates
    H2 = sample_2["e_matrix"]
    H2 = np.linalg.inv(H2)
    pcd_1 = pcd_1.transform(H2)
    # visualize
    pcd_2 = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(sample_2["depth"]), K)
    # o3d.visualization.draw_geometries([pcd_1, pcd_2])

    # convert points to numpy
    points_1 = np.asarray(pcd_1.points)
    points_2 = np.asarray(pcd_2.points)
    # project points to image
    points_1_img = np.dot(K.intrinsic_matrix, points_1.T).T
    points_2_img = np.dot(K.intrinsic_matrix, points_2.T).T
    # normalize
    points_1_img = points_1_img / points_1_img[:, 2:]
    points_2_img = points_2_img / points_2_img[:, 2:]
    points_1_img = points_1_img[:, :2]
    points_2_img = points_2_img[:, :2]

    h, w = sample_1["depth"].shape
    valid = ((points_1_img[:, 0] >= 0)
             & (points_1_img[:, 1] >= 0)
             & (points_1_img[:, 0] <= w - 1)
             & (points_1_img[:, 1] <= h - 1))

    overlap_ratio = np.sum(valid) / len(valid)

    return overlap_ratio


def load_sample(row):
    image_path_t = "{}/images/scene_{}_final_preview/frame.{}.tonemap.jpg"
    depth_path_t = "{}/images/scene_{}_geometry_hdf5/frame.{}.depth_meters.hdf5"
    camera_position_path_t = "{}/_detail/{}/camera_keyframe_positions.hdf5"
    camera_orientation_path_t = "{}/_detail/{}/camera_keyframe_orientations.hdf5"
    unit_path_t = "{}/_detail/metadata_scene.csv"

    image_path = path.join(root_dir, image_path_t.format(scene_name, row["camera_name"], str(row["frame_id"]).zfill(4)))
    image = cv2.imread(image_path)

    depth_path = path.join(root_dir, depth_path_t.format(scene_name, row["camera_name"], str(row["frame_id"]).zfill(4)))
    depth_data = h5py.File(depth_path, 'r')
    depth = np.array(depth_data["dataset"]).astype('float32')
    depth_data.close()

    camera_position_path = path.join(root_dir, camera_position_path_t.format(scene_name, row["camera_name"]))
    camera_position_data = h5py.File(camera_position_path, 'r')
    camera_position = np.array(camera_position_data["dataset"]).astype('float32')
    camera_position_data.close()

    camera_orientation_path = path.join(root_dir, camera_orientation_path_t.format(scene_name, row["camera_name"]))
    camera_orientation_data = h5py.File(camera_orientation_path, 'r')
    camera_orientation = np.array(camera_orientation_data["dataset"]).astype('float32')
    camera_orientation_data.close()

    r = camera_orientation[row["frame_id"]]
    # matrix from right hand to left hand
    t = np.array([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]])

    r = np.matmul(t, r)
    r = np.matmul(r, np.linalg.inv(t))
    t = camera_position[row["frame_id"]]
    unit_file = path.join(root_dir, unit_path_t.format(scene_name))
    unit = pd.read_csv(unit_file).to_numpy()[0][1]
    t = t * unit
    t[1] = -t[1]
    t[2] = -t[2]
    h = np.concatenate([r, t.reshape(3, 1)], axis=1)
    h = np.concatenate([h, np.array([[0, 0, 0, 1.]])], axis=0)

    return image, depth, h


def process_scene(scene_name, root_dir, num_samples=100):
    focal_length = 886.81
    image_path_t = "{}/images/scene_{}_final_preview/frame.{}.tonemap.jpg"
    depth_path_t = "{}/images/scene_{}_geometry_hdf5/frame.{}.depth_meters.hdf5"

    sample_pd = get_sample_pd("hypersim_split.csv", root_dir)
    # drop index column
    sample_pd = sample_pd.drop(columns=["index"])
    # select all samples with specific scene name
    scene_pd = sample_pd[sample_pd["scene_name"] == scene_name]
    camear_names = scene_pd["camera_name"].unique()
    # read all image  in the scene_pd

    # read camera position
    # camera_keyframe_path = "{}/_detail/{}/camera_keyframe_frame_indices.hdf5"
    camera_position_path_t = "{}/_detail/{}/camera_keyframe_positions.hdf5"
    camera_orientation_path_t = "{}/_detail/{}/camera_keyframe_orientations.hdf5"

    p_list = []
    q_list = []
    camera_num = len(camear_names)
    for camera_name in camear_names:
        print("processing camera {}".format(camera_name))
        camera_pd = scene_pd[scene_pd["camera_name"] == camera_name]
        all_pairs = itertools.combinations(camera_pd.iterrows(), 2)
        # shuffle all pairs
        all_pairs = list(all_pairs)
        random.shuffle(all_pairs)

        count = 0
        for (id1, row1), (id2, row2) in tqdm(all_pairs, total=len(camera_pd) * (len(camera_pd) - 1) / 2):
            if count == int(num_samples / camera_num):
                break
            # count += 1
            # if index != 2:
            #     continue
            image_1, depth_1, e_matrix_1 = load_sample(row1)
            image_2, depth_2, e_matrix_2 = load_sample(row2)

            depth_1 = convert_distance_to_depth_map(depth_1, focal_length)
            depth_2 = convert_distance_to_depth_map(depth_2, focal_length)

            sample_1 = {"image": image_1, "depth": depth_1, "e_matrix": e_matrix_1}
            sample_2 = {"image": image_2, "depth": depth_2, "e_matrix": e_matrix_2}

            overlap_ratio_1 = check_overlap(sample_1, sample_2)
            overlap_ratio_2 = check_overlap(sample_2, sample_1)
            overlap_ratio = min(overlap_ratio_1, overlap_ratio_2)
            if overlap_ratio > 0.7:
                count += 1
                print("count = {}, overlap_ratio: {}, for image {} and {}".format(count, overlap_ratio, row1["frame_id"], row2["frame_id"]))
                p_list.append(row1.to_frame().T)
                q_list.append(row2.to_frame().T)

    print("Finish scene {}, total number of pairs: {}".format(scene_name,len(p_list)))

    p_df = pd.concat(p_list, ignore_index=True)
    q_df = pd.concat(q_list, ignore_index=True)
    p_path = path.join(root_dir, "{}/p.csv".format(scene_name))
    q_path = path.join(root_dir, "{}/q.csv".format(scene_name))
    p_df.to_csv(p_path)
    q_df.to_csv(q_path)
        # process_sample(depth, gt_segmentation, existing_planes_bbox, h)
        # for plane in existing_planes_bbox:
        #     pass
        # break


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root_dir = "/cluster/project/infk/cvg/students/junwang/hypersim"
    # scene_name = "ai_001_010"
    test_scene = ['ai_001_001', 'ai_001_002', 'ai_001_010']
    # test_list_path = os.path.join(root_dir, "test_scenes.txt")
    # with open(test_list_path, "r") as f:
    #     test_scene = f.read().splitlines()
    for scene_name in test_scene:
        print("processing scene {}".format(scene_name))
        process_scene(scene_name, root_dir)

