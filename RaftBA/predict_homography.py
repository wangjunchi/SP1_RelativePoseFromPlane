import numpy as np
import cv2
import yaml
import torch

from RaftBA.Models.Raft import Model as RAFT
from RaftBA.metrics import compute_homography

def extract_plane_patch(image, plane_mask):
    mask = plane_mask.astype('bool')
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    center = np.array([(y0 + y1) / 2, (x0 + x1) / 2]).astype('int')
    origin_x = max(0, center[1] - 160)
    origin_y = max(0, center[0] - 120)
    image_patch = image[origin_y:origin_y + 240, origin_x:origin_x + 320, :]
    # padding to 240*320
    if image_patch.shape[0] < 240:
        image_patch = np.pad(image_patch, ((0, 240 - image_patch.shape[0]), (0, 0), (0, 0)), 'constant')
    if image_patch.shape[1] < 320:
        image_patch = np.pad(image_patch, ((0, 0), (0, 320 - image_patch.shape[1]), (0, 0)), 'constant')
    # padding to 240*320
    plane_mask = plane_mask[origin_y:origin_y + 240, origin_x:origin_x + 320]
    if plane_mask.shape[0] < 240:
        plane_mask = np.pad(plane_mask, ((0, 240 - plane_mask.shape[0]), (0, 0)), 'constant')
    if plane_mask.shape[1] < 320:
        plane_mask = np.pad(plane_mask, ((0, 0), (0, 320 - plane_mask.shape[1])), 'constant')

    kernel = np.ones((3, 3), np.uint8)
    plane_mask = plane_mask.astype('uint8')
    plane_mask = cv2.dilate(plane_mask, kernel, iterations=10)

    return image_patch, plane_mask, origin_x, origin_y


def restoreHomography(H, origin_1, origin_2):
    t2 = np.array([[1, 0, origin_2[0]], [0, 1, origin_2[1]], [0, 0, 1]])
    t1 = np.array([[1, 0, origin_1[0]], [0, 1, origin_1[1]], [0, 0, 1]])
    H_full = np.matmul(t2, np.matmul(H, np.linalg.inv(t1)))
    s1 = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1]])
    s2 = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1]])
    H_full = np.matmul(s2, np.matmul(H_full, np.linalg.inv(s1)))
    H_full = H_full / H_full[2, 2]
    return H_full

def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    model = RAFT()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    model.eval()
    return model

def predict_homography(model, image_1, image_2, mask):
    # check two image have the same size
    assert image_1.shape == image_2.shape

    # check image size is 128*128
    assert image_1.shape == (240, 320, 3)

    image_1 = torch.from_numpy(image_1).float().permute(2, 0, 1).unsqueeze(0)
    image_2 = torch.from_numpy(image_2).float().permute(2, 0, 1).unsqueeze(0)

    # send to device
    image_1 = image_1.cuda()
    image_2 = image_2.cuda()

    flow_pred12, homo_pred, residual, weight = model(image_1, image_2, None, iters=10)
    final_flow12 = flow_pred12[-1]
    h_pred12 = compute_homography(final_flow12, mask)

    return final_flow12, h_pred12[0]