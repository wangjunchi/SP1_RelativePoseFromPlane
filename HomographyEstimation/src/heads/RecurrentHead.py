import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.utils import four_point_to_homography
from torch.nn.functional import smooth_l1_loss

class SmallEncoder(nn.Module):
    def __init__(self, output_dim=64):
        super(SmallEncoder, self).__init__()
        # cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convf1 = nn.Conv2d(16, 128, 3, padding=1)
        self.convf2 = nn.Conv2d(128, output_dim, 3, padding=1)

    def forward(self, feature):
        feature = F.relu(self.convf1(feature))
        feature = F.relu(self.convf2(feature))
        return feature

class SmallMotionEncoder(nn.Module):
    def __init__(self):
        super(SmallMotionEncoder, self).__init__()
        # cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(16, 64, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 3, padding=1)
        self.convf2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv = nn.Conv2d(128, 64, 3, padding=1)

    def forward(self, feature, pf):
        feat = F.relu(self.convc1(feature))
        pf = F.relu(self.convf1(pf))
        pf = F.relu(self.convf2(pf))
        fear_pf = torch.cat([feat, pf], dim=1)
        out = F.relu(self.conv(fear_pf))
        return out

class FlowHead(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=64, input_dim=64):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SmallUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder()
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, hidden, feature, pf_hat):
        motion_features = self.encoder(feature, pf_hat)
        # input_x = torch.cat([input_x, motion_features], dim=1)
        hidden = self.gru(hidden, motion_features)
        delta_pf = self.flow_head(hidden)

        return hidden, delta_pf

def sequence_loss(pf_preds, pf_gt, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(pf_preds)
    seq_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = smooth_l1_loss(pf_preds[i], pf_gt, reduction='mean')
        seq_loss += i_weight * i_loss

    return seq_loss

class Model(nn.Module):

    def __init__(self, backbone, **kwargs):
        super(Model, self).__init__()

        # Target gen: '4_points' or 'all_points'
        self.target_gen = kwargs['TARGET_GEN']

        # ['target', 'pfs_hat_12' ,'delta', 'pf_hat_12']
        self.learning_keys = kwargs['LEARNING_KEYS']

        self.hdim = 64
        self.cnet = SmallEncoder(output_dim=self.hdim)
        self.update_block = SmallUpdateBlock(hidden_dim=self.hdim)
        self.iterations = 5


    def forward(self, data):
        feature_map = data['feature_map']
        pf_hat_12 = data['pf_hat_12']
        pfs = [pf_hat_12]
        hidden = self.cnet(feature_map)

        for i in range(self.iterations):
            pf_hat_12 = pf_hat_12.detach()
            hidden, delta_pf = self.update_block(hidden, feature_map, pf_hat_12)
            pf_hat_12 = pf_hat_12 + delta_pf
            pfs.append(pf_hat_12)

        loss = sequence_loss(pfs, data['target'])

        # ret = []
        # # First 3 elements could be fetched directly
        # for key in self.learning_keys[:-1]:
        #     if key == 'pfs_hat_12':
        #         ret.append(pfs)
        #     else:
        #         ret.append(data[key])

        # Last element could require transformation
        delta_gt = data['delta']
        if self.target_gen == '4_points':
            delta_hat = (data[self.learning_keys[-1]])
            # ret.append(data[self.learning_keys[-1]])
        elif self.target_gen == 'all_points':

            target_hat = pfs[-1]
            delta_hat = torch.zeros((target_hat.shape[0], 4, 2), device=target_hat.device, dtype=target_hat.dtype)
            h, w = target_hat.shape[-2:]

            delta_hat[:, 0, 0] = target_hat[:, 0, 0, 0]             # top left x
            delta_hat[:, 0, 1] = target_hat[:, 1, 0, 0]             # top left y

            delta_hat[:, 1, 0] = target_hat[:, 0, 0, w - 1]         # top right x
            delta_hat[:, 1, 1] = target_hat[:, 1, 0, w - 1]         # top right y

            delta_hat[:, 2, 0] = target_hat[:, 0, h - 1, w - 1]     # bottom right x
            delta_hat[:, 2, 1] = target_hat[:, 1, h - 1, w - 1]     # bottom right y

            delta_hat[:, 3, 0] = target_hat[:, 0, h - 1, 0]         # bottom left x
            delta_hat[:, 3, 1] = target_hat[:, 1, h - 1, 0]         # bottom left y

        else:
            assert False, 'I didnt understand that!'

        return loss, delta_gt, delta_hat

    def predict_homography(self, data):
        if self.target_gen == '4_points':

            # Get corners
            if 'corners' in data:
                corners = data['corners']
            else:
                assert False, 'How to handle it?'

            # Estimate homography
            delta_hat = data[self.learning_keys[3]]
            homography_hat = four_point_to_homography(corners=corners, deltas=delta_hat, crop=False)

            # Return the patch
            return delta_hat, homography_hat

        elif self.target_gen == 'all_points':
            return self._postprocess(data[self.learning_keys[1]][-1])

    @staticmethod
    def _postprocess(perspective_field):

        # Move data to numpy?
        if torch.is_tensor(perspective_field):
            perspective_field = perspective_field.cpu().detach().numpy()

        # Create field of the coordinates
        y_patch_grid, x_patch_grid = np.mgrid[0:perspective_field.shape[-2], 0:perspective_field.shape[-1]]
        x_patch_grid = np.tile(x_patch_grid.reshape(1, -1), (perspective_field.shape[0], 1))
        y_patch_grid = np.tile(y_patch_grid.reshape(1, -1), (perspective_field.shape[0], 1))
        coordinate_field = np.stack((x_patch_grid, y_patch_grid), axis=1).transpose(0, 2, 1)

        # Create prediction field
        pf_x1_img_coord, pf_y1_img_coord = np.split(perspective_field, 2, axis=1)
        pf_x1_img_coord = pf_x1_img_coord.reshape(perspective_field.shape[0], -1)
        pf_y1_img_coord = pf_y1_img_coord.reshape(perspective_field.shape[0], -1)
        mapping_field = coordinate_field + np.stack((pf_x1_img_coord, pf_y1_img_coord), axis=1).transpose(0, 2, 1)

        # Find best homography fit and its delta
        predicted_h = []
        predicted_delta = []
        four_points = [[0, 0], [perspective_field.shape[-1], 0], [perspective_field.shape[-1],
                                                                  perspective_field.shape[-2]],
                       [0, perspective_field.shape[-2]]]
        for i in range(perspective_field.shape[0]):
            h = cv2.findHomography(np.float32(coordinate_field[i]), np.float32(mapping_field[i]), cv2.RANSAC, 10)[0]
            delta = cv2.perspectiveTransform(np.asarray([four_points], dtype=np.float32), h).squeeze() - four_points
            predicted_h.append(h)
            predicted_delta.append(delta)
        predicted_h = np.array(predicted_h)
        predicted_delta = np.array(predicted_delta)

        # Find delta
        return predicted_delta, predicted_h
