import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from HomographyEstimation.src.backbones.raft_block.update import BasicUpdateBlock, SmallUpdateBlock
from HomographyEstimation.src.backbones.raft_block.extractor import BasicEncoder, SmallEncoder
from HomographyEstimation.src.backbones.raft_block.corr import CorrBlock, AlternateCorrBlock
from HomographyEstimation.src.backbones.raft_block.utils import bilinear_sampler, coords_grid, upflow8
from HomographyEstimation.src.backbones.raft_block.flow_viz import flow_to_image

from easydict import EasyDict as edict
import cv2

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


def viz(img1, img2, flow12):
    img1 = img1[0].permute(1, 2, 0).cpu().numpy()
    img2 = img2[0].permute(1, 2, 0).cpu().numpy()
    flow12 = flow12[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    # flo = flow_to_image(flow12)
    img1 = (img1+1)/2 * 255
    img1 = np.ascontiguousarray(img1, dtype=np.uint8)
    img1 = cv2.resize(img1, (512, 512))
    flow_up = cv2.resize(flow12, (512, 512))
    flow_up = flow_up * -1
    # img1 = img1.astype(np.uint8)
    # normalize the flow
    flow_dire = flow_up[:, :, :2] / np.linalg.norm(flow_up[:, :, :2], axis=2, keepdims=True)
    # drwa arrows on img1 according to flow12
    for i in range(0, flow_dire.shape[0], 40):
        for j in range(0, flow_dire.shape[1], 40):
            cv2.arrowedLine(img1, (j, i), (int(j+flow_dire[j, i, 0]*20), int(i+flow_dire[j, i, 1]*20)), (255, 0, 0), 2, tipLength=0.3)

    img2 = (img2+1)/2 * 255
    img2 = np.ascontiguousarray(img2, dtype=np.uint8)
    img2 = cv2.resize(img2, (512, 512))
    img = np.concatenate([img1, img2], axis=1)
    # cv2.imshow('img', img)
    # cv2.waitKey()
    pass
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)


class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.corr_radius = 4
        self.corr_radius = 3
        self.dropout = 0.0

        self.patch_keys = kwargs['PATCH_KEYS']
        self.target_keys = kwargs['TARGET_KEYS']

        self.test_mode = True

        self.args = edict({'small': False, 'mixed_precision': False, 'alternate_corr': False})

        if self.args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.args.corr_levels = 4
            self.args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.args.corr_levels = 4
            self.args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if self.args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=self.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def predict_homography(self, data):
        (e1, e2) = self.patch_keys

        # Oneline
        o1 = self.target_keys[0]
        image1 = data[e1]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = data[e2]
        image2 = 2 * (image2 / 255.0) - 1.0

        debug_image1 = image1.cpu().numpy()[0].transpose(1, 2, 0)
        debug_image2 = image2.cpu().numpy()[0].transpose(1, 2, 0)

        # tarnslate image1 to 10 pixels to the right
        # image3 = torch.roll(image1, shifts=10, dims=3)
        # image3[:, :, :, :10] = 0
        image3 = torch.roll(image1, shifts=10, dims=2)
        image3[:, :, :10, :] = 0

        debug_image3 = image3.cpu().numpy()[0].transpose(1, 2, 0)

        # flow_up: 2*128*128, the first dimension is the flow in x direction,
        # the second dimension is the flow in y direction
        flow_down, flow_up = self._forward(image1, image2, iters=4, test_mode=True)

        # viz(image1, image2, flow_up)
        # flow_up = flow_up * -1
        data[o1] = flow_up

        return data

    def predict_corresponding(self, data):
        return self.predict_homography(data)

    def forward(self, data):
        (e1, e2) = self.patch_keys

        # Oneline
        o1 = self.target_keys[0]
        image1 = data[e1]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = data[e2]
        image2 = 2 * (image2 / 255.0) - 1.0

        flow_predictions = self._forward(image1, image2, iters=12, test_mode=False)
        data[o1] = flow_predictions

        return data

    def _forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        # image1 = image1.repeat(1, 3, 1, 1)
        # image2 = image2.repeat(1, 3, 1, 1)

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
