import argparse
import glob
import numpy as np
import os
import time

import cv2
from SuperPoint.model import SuperPointFrontend
import torch


def load_sp_model():
    """ Load the SuperPoint model. """
    weights_path = '/cluster/project/infk/cvg/students/junwang/SP1_wholePipeline/SuperPoint/trained_models/superpoint_v1.pth'
    nms_dist = 4
    conf_thresh = 0.015
    nn_thresh = 0.7
    cuda = True
    print('==> Loading pre-trained network.')
    model = SuperPointFrontend(weights_path=weights_path,
                            nms_dist=nms_dist,
                            conf_thresh=conf_thresh,
                            nn_thresh=nn_thresh,
                            cuda=cuda)
    print('==> Successfully loaded pre-trained network.')
    return model