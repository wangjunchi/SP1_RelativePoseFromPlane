import cv2
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import h5py
import torch
import pandas as pd
import itertools
import random
import matplotlib.pyplot as plt
import ast

class PlaneDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.scenes = self.get_dirs()
        self.samples = self.get_sample_list()

        self.image_path = "{}/{}/{}/original.png"
        self.seg_path = "{}/{}/{}/plane_seg.npy"

    def get_dirs(self):
        dirs = []
        for root, dirs, files in os.walk(self.data_dir):
            print("Found dir: ", dirs)
            break
        return dirs

    def get_sample_list(self):
        scenes = []
        for scene in self.scenes:
            scene_path = os.path.join(self.data_dir, scene)
            scene_list = pd.read_csv(os.path.join(scene_path, "homography.csv"))
            scenes.append(scene_list)
        sample_list = pd.concat(scenes)
        return sample_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        scene_name = row["scene_name"]

        camera_id_1 = row["camera_id_1"]
        frame_id_1 = row["frame_id_1"]
        plane_id_1 = row["plane_id_1"]

        camera_id_2 = row["camera_id_2"]
        frame_id_2 = row["frame_id_2"]
        plane_id_2 = row["plane_id_2"]

        homography = row["homography"]
        homography = np.fromstring(homography[1:-1], dtype=float, sep=',').reshape([3, 3])

        image_1 = self.load_image(scene_name, camera_id_1, frame_id_1)
        segment_1 = self.load_segment(scene_name, camera_id_1, frame_id_1, plane_id_1)
        # image_1 *= segment_1[:, :, None]

        segment_2 = self.load_segment(scene_name, camera_id_2, frame_id_2, plane_id_2)
        image_2 = self.load_image(scene_name, camera_id_2, frame_id_2)

        # resize image and segment to 192*256
        image_1 = cv2.resize(image_1, (256, 192))
        image_2 = cv2.resize(image_2, (256, 192))
        segment_1 = cv2.resize(segment_1, (256, 192), interpolation=cv2.INTER_NEAREST)
        segment_2 = cv2.resize(segment_2, (256, 192), interpolation=cv2.INTER_NEAREST)

        s1 = np.array([[1/4.0, 0, 0], [0, 1/4.0, 0], [0, 0, 1]])
        s2 = np.array([[1/4.0, 0, 0], [0, 1/4.0, 0], [0, 0, 1]])
        homography = np.matmul(s2, np.matmul(homography, np.linalg.inv(s1)))

        # create square bounding box from mask
        mask = segment_1.astype('bool')
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        center = np.array([(y0 + y1) / 2, (x0 + x1) / 2]).astype('int')
        origin_x_1 = max(0, center[1] - 64)
        origin_y_1 = max(0, center[0] - 64)
        image_1_patch = image_1[origin_y_1:origin_y_1+128, origin_x_1:origin_x_1+128, :]
        # padding to 128*128
        if image_1_patch.shape[0] < 128:
            image_1_patch = np.pad(image_1_patch, ((0, 128-image_1_patch.shape[0]), (0, 0), (0, 0)), 'constant')
        if image_1_patch.shape[1] < 128:
            image_1_patch = np.pad(image_1_patch, ((0, 0), (0, 128-image_1_patch.shape[1]), (0, 0)), 'constant')
        segment_1 = segment_1[origin_y_1:origin_y_1+128, origin_x_1:origin_x_1+128]
        # padding to 128*128
        if segment_1.shape[0] < 128:
            segment_1 = np.pad(segment_1, ((0, 128-segment_1.shape[0]), (0, 0)), 'constant')
        if segment_1.shape[1] < 128:
            segment_1 = np.pad(segment_1, ((0, 0), (0, 128-segment_1.shape[1])), 'constant')

        mask = segment_2.astype('bool')
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        center = np.array([(y0 + y1) / 2, (x0 + x1) / 2]).astype('int')
        origin_x_2 = max(0, center[1] - 64)
        origin_y_2 = max(0, center[0] - 64)
        image_2_patch = image_2[origin_y_2:origin_y_2+128, origin_x_2:origin_x_2+128, :]
        # padding to 128*128
        if image_2_patch.shape[0] < 128:
            image_2_patch = np.pad(image_2_patch, ((0, 128-image_2_patch.shape[0]), (0, 0), (0, 0)), 'constant')
        if image_2_patch.shape[1] < 128:
            image_2_patch = np.pad(image_2_patch, ((0, 0), (0, 128-image_2_patch.shape[1]), (0, 0)), 'constant')
        segment_2 = segment_2[origin_y_2:origin_y_2+128, origin_x_2:origin_x_2+128]
        # padding to 128*128
        if segment_2.shape[0] < 128:
            segment_2 = np.pad(segment_2, ((0, 128-segment_2.shape[0]), (0, 0)), 'constant')
        if segment_2.shape[1] < 128:
            segment_2 = np.pad(segment_2, ((0, 0), (0, 128-segment_2.shape[1])), 'constant')

        t2 = np.array([[1, 0, -origin_x_2], [0, 1, -origin_y_2], [0, 0, 1]])
        t1 = np.array([[1, 0, -origin_x_1], [0, 1, -origin_y_1], [0, 0, 1]])
        homography_new = np.matmul(t2, np.matmul(homography, np.linalg.inv(t1)))

        # generate matching
        gt_match, mask = self.generate_gt_match(image_1_patch, homography_new)

        # convert to tensor
        image_1_patch = torch.from_numpy(image_1_patch).permute(2, 0, 1).float()
        image_2_patch = torch.from_numpy(image_2_patch).permute(2, 0, 1).float()

        return {'patch_1': image_1_patch,
                'patch_2': image_2_patch,
                'target': gt_match,
                'gt_homography': homography_new,
                'mask_1': segment_1,
                'mask_2': segment_2,
                'origin_1': np.array([origin_x_1, origin_y_1]),
                'origin_2': np.array([origin_x_2, origin_y_2])}




    def __getitem__2(self, idx):
        row = self.samples.iloc[idx]
        scene_name = row["scene_name"]

        camera_id_1 = row["camera_id_1"]
        frame_id_1 = row["frame_id_1"]
        plane_id_1 = row["plane_id_1"]

        camera_id_2 = row["camera_id_2"]
        frame_id_2 = row["frame_id_2"]
        plane_id_2 = row["plane_id_2"]

        homography = row["homography"]

        image_1 = self.load_image(scene_name, camera_id_1, frame_id_1)
        segment_1 = self.load_segment(scene_name, camera_id_1, frame_id_1, plane_id_1)
        # image_1 *= segment_1[:, :, None]


        segment_2 = self.load_segment(scene_name, camera_id_2, frame_id_2, plane_id_2)
        image_2 = self.load_image(scene_name, camera_id_2, frame_id_2)



        # image_2 *= segment_2[:, :, None]

        # create square bounding box from mask
        mask = segment_1.astype('bool')
        coords = np.argwhere(mask)
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        l1 = max(x1 - x0, y1 - y0)
        if l1 % 2 == 1:
            l1 -= 1
        org_1 = np.array((x0, y0, 1))
        # segment_1 = segment_1[x0:x1, y0:y1]
        image_1_patch = image_1[x0 : x0+l1, y0 : y0 + l1]
        # padding the image to make it square
        image_1_patch = np.pad(image_1_patch, ((0, l1 - image_1_patch.shape[0]), (0, l1 - image_1_patch.shape[1]), (0, 0)), 'constant')

        mask = segment_2.astype('bool')
        coords = np.argwhere(mask)
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        l2 = max(x1 - x0, y1 - y0)
        if l2 % 2 == 1:
            l2 -= 1
        org_2 = np.array((x0, y0, 1))
        # segment_2 = segment_2[x0:x1, y0:y1]
        image_2_patch = image_2[x0 : x0+l2, y0 : y0 + l2]
        # padding the image to make it square
        image_2_patch = np.pad(image_2_patch, ((0, l2 - image_2_patch.shape[0]), (0, l2 - image_2_patch.shape[1]), (0, 0)), 'constant')

        # change the homography to the new coordinate system
        homography  = np.fromstring(homography[1:-1], dtype=float, sep=',').reshape([3, 3])
        t2 = np.array([[1, 0, -org_2[1]], [0, 1, -org_2[0]], [0, 0, 1]])
        t1 = np.array([[1, 0, -org_1[1]], [0, 1, -org_1[0]], [0, 0, 1]])
        homography_new = np.matmul(t2, np.matmul(homography, np.linalg.inv(t1)))
        new_size = 128
        s1 = np.array([[new_size / l1, 0, 0], [0, new_size / l1, 0], [0, 0, 1]])
        s2 = np.array([[new_size / l2, 0, 0], [0, new_size / l2, 0], [0, 0, 1]])
        homography_new = np.matmul(s2, np.matmul(homography_new, np.linalg.inv(s1)))
        # resize the image patch
        image_1_patch = cv2.resize(image_1_patch, (new_size, new_size))
        image_2_patch = cv2.resize(image_2_patch, (new_size, new_size))

        segment_1 = cv2.resize(segment_1, (new_size, new_size), interpolation=cv2.INTER_NEAREST)
        segment_2 = cv2.resize(segment_2, (new_size, new_size), interpolation=cv2.INTER_NEAREST)

        # generate matching
        gt_match, mask = self.generate_gt_match(image_1_patch, homography_new)

        # convert to tensor
        image_1_patch = torch.from_numpy(image_1_patch).permute(2, 0, 1).float()
        image_2_patch = torch.from_numpy(image_2_patch).permute(2, 0, 1).float()
        # gt_match = torch.from_numpy(gt_match).permute(2, 0, 1).float()

        return {'patch_1': image_1_patch,
                'patch_2': image_2_patch,
                'target': gt_match,
                'gt_homography': homography_new,
                'mask_1': segment_1,
                'mask_2': segment_2,}

    def load_image(self, scene_name, camera_id, frame_id):
        frame_id = str(frame_id).zfill(4)
        img_path = os.path.join(self.data_dir, self.image_path.format(scene_name, camera_id, frame_id))
        assert os.path.exists(img_path), "Image file does not exist: {}".format(img_path)
        image_data = cv2.imread(img_path).astype('float32')
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        return image_data

    def load_segment(self, scene_name, camera_id, frame_id, plane_id):
        frame_id = str(frame_id).zfill(4)
        seg_path = os.path.join(self.data_dir, self.seg_path.format(scene_name, camera_id, frame_id))
        assert os.path.exists(seg_path), "Segment file does not exist: {}".format(seg_path)
        seg_data = np.load(seg_path)
        seg_data[seg_data != plane_id] = 0
        seg_data[seg_data == plane_id] = 1
        return seg_data

    def generate_gt_match(self, patch_1, H):
        h, w, _ = patch_1.shape
        # inverse homography matrix
        H_inv = np.linalg.inv(H)

        # estimate the grid
        X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                           np.linspace(0, h - 1, h))
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # multiply Hinv to XYhom to find the warped grid
        XYwarpHom = np.matmul(H, XYhom)

        # vector representation
        XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
        YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
        ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

        # Xwarp = \
        #     (2 * XwarpHom / (ZwarpHom + 1e-8) / (w_scale - 1) - 1)
        # Ywarp = \
        #     (2 * YwarpHom / (ZwarpHom + 1e-8) / (h_scale - 1) - 1)
        Xwarp = (XwarpHom / (ZwarpHom + 1e-8))
        Ywarp = (YwarpHom / (ZwarpHom + 1e-8))
        # and now the grid
        grid_gt = torch.stack([Xwarp.view(h, w),
                               Ywarp.view(h, w)], dim=-1)

        # mask
        mask = grid_gt.ge(0) & grid_gt.le(w)
        mask = mask[:, :, 0] & mask[:, :, 1]

        return grid_gt, mask