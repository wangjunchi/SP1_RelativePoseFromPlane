import os
import pickle

import h5py
import numpy as np
from PIL import Image
import torch
from scipy import spatial
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop, resize
from torch.utils.model_zoo import load_url
from torchvision import transforms
from tqdm import tqdm

from PlaneMatching.networks.imageretrievalnet import init_network, extract_vectors, extract_ss, extract_ms
from skimage.segmentation import expand_labels

from Dataset.hypersim_image_pair_dataset import HypersimImagePairDataset

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [num_instances, height, width]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = torch.zeros([4])

    # Bounding box.
    horizontal_indicies = torch.where(torch.any(mask, dim=0))[0]
    vertical_indicies = torch.where(torch.any(mask, dim=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    box = torch.tensor([x1, y1, x2, y2])
    return box


def extract_features(net, img, seg):
    num_planes = int(torch.max(seg))
    # extract vector for each plane
    plane_vectors = torch.zeros((net.meta['outputdim'], num_planes))
    # extract global vector
    # global_vector = extract_ss(net, image.unsqueeze(0).cuda())
    for i in range(1, num_planes + 1):
        plane_mask = seg == i
        box = extract_bboxes(plane_mask)
        plane_mask = plane_mask.unsqueeze(0).float()
        # plane_mask[plane_mask == 0] = 0.1
        plane_mask = expand_labels(plane_mask, 10)

        masked_img = img * plane_mask
        # masked_img = resize(masked_img, (384, 512))

        # enlarge the box by 20 pixels
        box[0] = max(0, box[0] - 20)
        box[1] = max(0, box[1] - 20)
        box[2] = min(img.shape[2], box[2] + 20)
        box[3] = min(img.shape[1], box[3] + 20)
        masked_img = crop(masked_img, int(box[1]), int(box[0]), int(box[3] - box[1]), int(box[2] - box[0]))
        # scale the image
        h, w = masked_img.shape[1], masked_img.shape[2]
        # masked_img = resize(masked_img, (h//2, w//2))
        masked_img = masked_img.unsqueeze(0).cuda()
        # plane_mask = plane_mask.unsqueeze(0).float()
        # # extract plane vector
        # masked_img = img * plane_mask
        # masked_img = masked_img.unsqueeze(0).cuda()
        # plane_vec = extract_ss(net, masked_img)
        # msp = net.pool.p.item()
        plane_vec = extract_ms(net, masked_img, [1, 2**(1/2), 1/2], msp=1)
        # normalize
        # plane_vec = plane_vec / torch.norm(plane_vec)
        plane_vectors[:, i - 1] = plane_vec

        # for debug
        masked_img_debug = masked_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        pass

    return plane_vectors


def load_gem_model(model_path):
    state = torch.load(model_path)
    # parsing net params from meta
    # architecture, pooling, mean, std required
    # the rest has default values, in case that is doesnt exist
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', False)
    net_params['mean'] = state['meta']['mean']
    net_params['std'] = state['meta']['std']
    net_params['pretrained'] = False

    # load network
    net = init_network(net_params)
    net.load_state_dict(state['state_dict'])
    net.cuda()

    return net

def predict_matching(image_1, image_2, seg_1, seg_2, net):

    # load the network
    # network_path = '/home/junchi/sp1/project_junchi/pythonProject/whole_pipeline/PlaneMatching/trained_models/gl18-tl-resnet50-gem-w-83fdc30.pth'
    # state = torch.load(network_path)
    # parsing net params from meta
    # architecture, pooling, mean, std required
    # the rest has default values, in case that is doesnt exist
    # net_params = {}
    # net_params['architecture'] = state['meta']['architecture']
    # net_params['pooling'] = state['meta']['pooling']
    # net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    # net_params['regional'] = state['meta'].get('regional', False)
    # net_params['whitening'] = state['meta'].get('whitening', False)
    # net_params['mean'] = state['meta']['mean']
    # net_params['std'] = state['meta']['std']
    # net_params['pretrained'] = False

    # # load network
    # net = init_network(net_params)
    # net.load_state_dict(state['state_dict'])
    # net.cuda()

    # extract features
    plane_vectors_1 = extract_features(net, image_1, seg_1)
    plane_vectors_2 = extract_features(net, image_2, seg_2)

    num_planes_1 = plane_vectors_1.shape[-1]
    num_planes_2 = plane_vectors_2.shape[-1]

    similarity_matrix = np.zeros((num_planes_1, num_planes_2))
    for i in range(num_planes_1):
        for j in range(num_planes_2):
            similarity_matrix[i, j] = 1 - spatial.distance.cosine(plane_vectors_1[:, i], plane_vectors_2[:, j])

    distance_matrix = 1 - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # store all the matching pair
    pred_match = np.ones(num_planes_1) * -1
    mean_similarity = 0
    count = 0
    for (r, c) in zip(row_ind, col_ind):
        mean_similarity += similarity_matrix[r, c]
        count += 1

    for (i, j) in zip(row_ind, col_ind):
        if similarity_matrix[i, j] > 0.4:
            pred_match[i] = int(j)

    return pred_match


if __name__ == '__main__':
    dataset = HypersimImagePairDataset(data_dir="/home/junchi/sp1/dataset/hypersim", scene_name='ai_001_001')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for step, batch in tqdm(enumerate(loader)):
        sample_1 = batch['sample_1']
        sample_2 = batch['sample_2']
        gt_match = batch['gt_match']

        image_1 = sample_1['image'][0]
        image_2 = sample_2['image'][0]

        segmentation_1 = sample_1['planes'][0, :, :, 1]
        segmentation_2 = sample_2['planes'][0, :, :, 1]

        pred_match = predict_matching(image_1, image_2, segmentation_1, segmentation_2)
        print("pred_match: ", pred_match)
        print("gt_match: ", gt_match)
        pass


