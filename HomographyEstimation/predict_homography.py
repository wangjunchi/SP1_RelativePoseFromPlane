import importlib

import cv2
import numpy as np
from os import path as osp
import yaml

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import argparse

from Dataset.hypersim_image_pair_dataset import HypersimImagePairDataset

class ModelWrapper(torch.nn.Sequential):
    def __init__(self, *args):
        super(ModelWrapper, self).__init__(*args)

    def predict_homography(self, data):
        for idx, m in enumerate(self):
            data = m.predict_homography(data)
        return data

    def predict_corresponding(self, data):
        for idx, m in enumerate(self):
            data = m.predict_corresponding(data)
        return data

def restoreHomography(H, origin_1, origin_2):
    t2 = np.array([[1, 0, origin_2[0]], [0, 1, origin_2[1]], [0, 0, 1]])
    t1 = np.array([[1, 0, origin_1[0]], [0, 1, origin_1[1]], [0, 0, 1]])
    H_full = np.matmul(t2, np.matmul(H, np.linalg.inv(t1)))
    s1 = np.array([[4.0, 0, 0], [0, 4.0, 0], [0, 0, 1]])
    s2 = np.array([[4.0, 0, 0], [0, 4.0, 0], [0, 0, 1]])
    H_full = np.matmul(s2, np.matmul(H_full, np.linalg.inv(s1)))
    H_full = H_full / H_full[2, 2]
    return H_full


def predict_homography(model, image_1, image_2):
    # check two image have the same size
    assert image_1.shape == image_2.shape

    # check image size is 128*128
    assert image_1.shape == (128, 128, 3)

    if image_1.max() <= 1:
        image_1 = (image_1 * 255).astype('float')
    if image_2.max() <= 1:
        image_2 = (image_2 * 255).astype('float')
    # convert to tensor
    image_1 = torch.from_numpy(image_1).float().permute(2, 0, 1).unsqueeze(0)
    image_2 = torch.from_numpy(image_2).float().permute(2, 0, 1).unsqueeze(0)


    # send to device
    image_1 = image_1.cuda()
    image_2 = image_2.cuda()

    data = {'patch_1': image_1, 'patch_2': image_2}

    estimates_grid, estimates_grid_ransca, _h = model.predict_corresponding(data)

    return estimates_grid, _h


def extract_plane_patch(image, plane_mask):
    mask = plane_mask.astype('bool')
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    center = np.array([(y0 + y1) / 2, (x0 + x1) / 2]).astype('int')
    origin_x = max(0, center[1] - 64)
    origin_y = max(0, center[0] - 64)
    image_patch = image[origin_y:origin_y + 128, origin_x:origin_x + 128, :]
    # padding to 128*128
    if image_patch.shape[0] < 128:
        image_patch = np.pad(image_patch, ((0, 128 - image_patch.shape[0]), (0, 0), (0, 0)), 'constant')
    if image_patch.shape[1] < 128:
        image_patch = np.pad(image_patch, ((0, 0), (0, 128 - image_patch.shape[1]), (0, 0)), 'constant')
    # padding to 128*128
    plane_mask = plane_mask[origin_y:origin_y + 128, origin_x:origin_x + 128]
    if plane_mask.shape[0] < 128:
        plane_mask = np.pad(plane_mask, ((0, 128 - plane_mask.shape[0]), (0, 0)), 'constant')
    if plane_mask.shape[1] < 128:
        plane_mask = np.pad(plane_mask, ((0, 0), (0, 128 - plane_mask.shape[1])), 'constant')

    return image_patch, plane_mask, origin_x, origin_y


def load_model(args, config):
    # Model
    checkpoint_fname = args.ckpt
    if not osp.isfile(checkpoint_fname):
        raise ValueError('check the snapshots path')

    # Import model
    backbone_module = importlib.import_module('HomographyEstimation.src.backbones.{}'.format(config['MODEL']['BACKBONE']['NAME']))
    backbone_class_to_call = getattr(backbone_module, 'Model')

    # Create model class
    backbone = backbone_class_to_call(**config['MODEL']['BACKBONE'])

    ###########################################################################
    # Import and create the head
    ###########################################################################

    # Import backbone
    head_module = importlib.import_module('HomographyEstimation.src.heads.{}'.format(config['MODEL']['HEAD']['NAME']))
    head_class_to_call = getattr(head_module, 'Model')

    # Create backbone class
    head = head_class_to_call(backbone, **config['MODEL']['HEAD'])

    ###########################################################################
    # Import and create the head
    ###########################################################################

    model = ModelWrapper(backbone, head)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(checkpoint_fname, map_location=device)['model'])
    # model = nn.DataParallel(model)

    return model

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Raft evaluation on HPatches')
    # Paths
    parser.add_argument('--cfg-file', type=str, default='config/s-coco/raft-orig-eval.yaml',
                        help='path to training transformation csv folder')
    parser.add_argument('--dataset_dir', type=str,
                        default='/home/junchi/sp1/dataset/hypersim',
                        help='path to folder containing training images')
    parser.add_argument('--ckpt', type=str, default='./trained_models/model_raft_scoco.pth',
                        help='Checkpoint to use')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='evaluation batch size')
    parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(args.cfg_file, 'r') as file:
        config = yaml.full_load(file)

    # Model
    checkpoint_fname = args.ckpt
    if not osp.isfile(checkpoint_fname):
        raise ValueError('check the snapshots path')

    # Import model
    backbone_module = importlib.import_module('src.backbones.{}'.format(config['MODEL']['BACKBONE']['NAME']))
    backbone_class_to_call = getattr(backbone_module, 'Model')

    # Create model class
    backbone = backbone_class_to_call(**config['MODEL']['BACKBONE'])

    ###########################################################################
    # Import and create the head
    ###########################################################################

    # Import backbone
    head_module = importlib.import_module('src.heads.{}'.format(config['MODEL']['HEAD']['NAME']))
    head_class_to_call = getattr(head_module, 'Model')

    # Create backbone class
    head = head_class_to_call(backbone, **config['MODEL']['HEAD'])

    ###########################################################################
    # Import and create the head
    ###########################################################################

    model = ModelWrapper(backbone, head)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(checkpoint_fname, map_location=device)['model'])
    # model = nn.DataParallel(model)
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        dataset = HypersimImagePairDataset(args.dataset_dir, scene_name='ai_001_001')
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        for i, data in enumerate(dataloader):
            sample_1 = data['sample_1']
            sample_2 = data['sample_2']
            gt_match = data['gt_match'][0]

            image_1 = sample_1['image'][0].numpy().transpose(1, 2, 0)
            image_2 = sample_2['image'][0].numpy().transpose(1, 2, 0)

            segmentation_1 = sample_1['planes'][0, :, :, 1].numpy()
            segmentation_2 = sample_2['planes'][0, :, :, 1].numpy()

            # resize image and segment to 192*256
            image_1 = cv2.resize(image_1, (256, 192))
            image_2 = cv2.resize(image_2, (256, 192))
            segmentation_1 = cv2.resize(segmentation_1, (256, 192), interpolation=cv2.INTER_NEAREST)
            segmentation_2 = cv2.resize(segmentation_2, (256, 192), interpolation=cv2.INTER_NEAREST)

            # iterate over all plane matches
            for i in range(gt_match.shape[0]):
                if gt_match[i] != -1:
                    plane_mask_1 = np.copy(segmentation_1)
                    plane_mask_2 = np.copy(segmentation_2)
                    plane_mask_1[plane_mask_1 != i+1] = 0
                    matching = int(gt_match[i])+1
                    plane_mask_2[plane_mask_2 != matching] = 0

                    # extract plane patch
                    image_patch_1, plane_mask_1, origin_x_1, origin_y_1 = extract_plane_patch(image_1, plane_mask_1)
                    image_patch_2, plane_mask_2, origin_x_2, origin_y_2 = extract_plane_patch(image_2, plane_mask_2)

                    # shift image_patch_1 to 10 pixels right
                    image_patch_3 = np.roll(image_patch_1, shift=10, axis=1)
                    image_patch_3[:, :10, :] = 0


                    # visualize plane patch using plt
                    plt.subplot(1, 2, 1)
                    plt.imshow(image_patch_1)
                    plt.subplot(1, 2, 2)
                    plt.imshow(image_patch_3)
                    plt.show()

                    # pixel corresponding
                    # image_1 (0,0) -> image_2 (10,0)
                    # estimates_grid is the matching from image_1 to image_2
                    # the homography H is the projection from image_1 to image_2 (H*image_1 = image_2)
                    estimates_grid, H = predict_homography(model, image_patch_1, image_patch_3)

                    # h, w, _ = image_patch_1.shape
                    # X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                    #                    np.linspace(0, h - 1, h))
                    # X, Y = X.flatten(), Y.flatten()
                    # pts_src = np.stack([X, Y], axis=1)
                    # pts_dst = estimates_grid[0]
                    # cv2.findEssentialMat(pts_src, pts_dst, cameraMatrix=

                    print("H: ", H)

                    # estimate homography using dense matching
                    estimates_grid = estimates_grid[0]
                    h, w, _ = image_patch_1.shape
                    mask = plane_mask_1

                    dst_pts = estimates_grid[mask.flatten() != 0]

                    X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                                       np.linspace(0, h - 1, h))
                    X, Y = X.flatten(), Y.flatten()
                    src_dst = np.stack([X, Y], axis=1)
                    src_dst = src_dst[mask.flatten() != 0]

                    H2, _ = cv2.findHomography(src_dst, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=1)
                    print("H2: ", H2)
                    pass







