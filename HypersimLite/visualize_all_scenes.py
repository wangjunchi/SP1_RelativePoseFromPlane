import h5py
import tqdm
import sys
import numpy as np

import cv2
import os
import pandas as pd


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

def main():
    csv_path = '/home/junchi/sp1/dataset/hypersim/hypersim_split.csv'
    dataset_path = '/home/junchi/sp1/dataset/hypersim'
    image_path_t = "{}/images/scene_{}_final_preview/frame.{}.tonemap.jpg"
    planes_path_t = "{}/images/scene_{}_geometry_hdf5/frame.{}.planes.hdf5"
    image_save_path_t = "./preview/{}/"
    df = pd.read_csv(csv_path)
    # filter all train scenes
    train_df = df[df['split_partition_name'] == 'train']
    test_df = df[df['split_partition_name'] == 'test']
    # all training scenes
    train_scenes = train_df['scene_name'].unique()
    # all testing scenes
    test_scenes = test_df['scene_name'].unique()
    all_scenes = np.concatenate((train_scenes, test_scenes))

    colors = labelcolormap(256)
    # iterate over all scenes
    # check folder
    if not os.path.exists(image_save_path_t.format('train')):
        os.makedirs(image_save_path_t.format('train'))
    if not os.path.exists(image_save_path_t.format('test')):
        os.makedirs(image_save_path_t.format('test'))
    if not os.path.exists(image_save_path_t.format('all')):
        os.makedirs(image_save_path_t.format('all'))

    for scene in tqdm.tqdm(all_scenes):
        # sample 5 images from each scene
        cam_id = 'cam_00'
        for frame_id in range(0, 50, 10):
            # read the image
            frame_id = str(frame_id).zfill(4)
            img_path = os.path.join(dataset_path, image_path_t.format(scene, cam_id, frame_id))
            img = cv2.imread(img_path)
            # read the planes
            try:
                planes_path = os.path.join(dataset_path, planes_path_t.format(scene, cam_id, frame_id))
                planes_data = h5py.File(planes_path, 'r')
                planes = np.array(planes_data["dataset"][:, :, 1].squeeze()).astype('int')
                planes_data.close()
                pd_planes_img = np.stack([colors[planes, 0], colors[planes, 1], colors[planes, 2]], axis=2)
            except Exception as e:
                print(e)
                continue

            # save the image
            img_save_path = os.path.join(image_save_path_t.format('all'), "{}_cam00_frame_{}.jpg".format(scene, frame_id))
            cv2.imwrite(img_save_path, img)
            plane_save_path = os.path.join(image_save_path_t.format('all'), "{}_cam00_frame_{}_planes.jpg".format(scene, frame_id))
            cv2.imwrite(plane_save_path, pd_planes_img)



if __name__ == '__main__':
    main()


