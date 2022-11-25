"""Includes the Utility class full of static functions that are used all over the code."""

import h5py
import hdbscan
import numpy as np
import pyransac
import torch

from skimage import measure
from pyransac.base import Model
from skimage.segmentation import expand_labels


class Utility:

    @staticmethod
    def calculate_hypersim_focal_length():
        """This functions returns the focal length of the Hypersim dataset.

        Returns:
            (float): Focal length of the Hypersim dataset.
        """
        fov_x = np.pi / 3
        return 1024 / (2 * np.tan(fov_x / 2))

    @staticmethod
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

    @staticmethod
    def equalise_labelling(label_a, label_b):
        """This functions equalises the numbering of the second segmentation to the first one. Batches are supported.

        Code from: https://github.com/svip-lab/PlanarReconstruction

        Parameters:
            label_a (torch.Tensor): First segmentation. (1, h, w) or (b, 1, h, w)
            label_b (torch.Tensor): Second segmentation which is adapted to fit the first. (1, h, w) or (b, 1, h, w)

        Returns:
            torch.Tensor: Adapted second segmentation (b, 1, h, w)
        """
        assert label_a.dim() in (3, 4)
        # assert label_a.dim() == label_b.dim()

        if label_a.dim() == 3:
            label_a = label_a.unsqueeze(0)
        if label_b.dim() == 3:
            label_b = label_b.unsqueeze(0)

        print(label_a.shape)
        print(label_b.shape)
        # Extract parameters
        batch_size, _, height, width = label_a.size()
        device = label_a.device

        # result list
        all_results = list()

        # Batch loop
        for batch_id in range(batch_size):
            # Create label stack with one plane per layer.
            label_a_stack = (torch.unsqueeze(label_a[batch_id].squeeze(), -1) == torch.arange(1, label_a[batch_id].max() + 1).to(device)).type(torch.float)
            label_b_stack = (torch.unsqueeze(label_b[batch_id].squeeze(), -1) == torch.arange(1, label_b[batch_id].max() + 1).to(device)).type(torch.float)

            # Using best iou as metric for same plane
            intersection_mask = label_b_stack.unsqueeze(-1) * label_a_stack.unsqueeze(2) > 0.5
            intersection = intersection_mask.float().sum(0).sum(0)
            union = torch.max(label_b_stack.unsqueeze(-1), label_a_stack.unsqueeze(2)).sum(0).sum(0).float()
            # keep separate planes that don't have any match
            if 0 in intersection.size():
                resulting_labels = label_a[batch_id]
            else:
                best_iou = (intersection / torch.clamp(union, min=1e-4)).max(dim=0)[1]+1

                resulting_labels = (label_a_stack * best_iou.unsqueeze(0).unsqueeze(0)).sum(dim=2)
            all_results.append(resulting_labels)

        # Create batch tensor from result list
        return torch.stack(all_results)

    @staticmethod
    def split_and_remove_labels(labels, area_size):
        """This function splits groups that are not connected and removes to small groups.
        Parameters:
            labels (np.ndarray, torch.Tensor): label array to start from.
            area_size (float): Area size a group should have at least.
        Returns:
            np.ndarray: Final labelling without small groups and non-connected groups are split into two.
        """
        # Transform Tensor to numpy array
        if type(labels) == torch.Tensor:
            labels = labels.cpu().detach().numpy()

        # Split non-connected groups
        labels = measure.label(np.squeeze(labels), background=0, connectivity=1)
        for label_prop in measure.regionprops(labels):
            if (
                    # Remove groups that are smaller than area_size
                    label_prop.area < area_size or
                    # Remove groups that are very thin and long
                    label_prop.axis_minor_length / (label_prop.axis_major_length+1e-3) < 10/100
            ):
                labels[labels == label_prop.label] = 0

        return np.expand_dims(measure.label(labels, background=0), 0)

    @staticmethod
    def save_hdf5_file(data, file_path):
        """This functions saves data in the hdf5 file format.

        Parameters:
            data (torch.Tensor, np.ndarray): Data matrix
            file_path (str): File name and path to which the data is saved.
        """
        if type(data) != torch.Tensor:
            data = torch.Tensor(data)
        with h5py.File(file_path, "w") as f:
            f.create_dataset("dataset", data.shape, dtype='float16', data=data.type(torch.float16))
            f.close()

    @staticmethod
    def aggregate_parameters_ransac(labels, parameters, ransac_parameter):
        """This function aggregates the parameters over the segmentation with the help of RANSAC.

        Parameters:
            labels (torch.Tensor): Plane segmentation. (b, 1, h, w)
            parameters (torch.Tensor): Plane parameters which will be aggregated. (b, d, h, w)
            ransac_parameter (pyransac.RansacParams): Pyransacs parameters to run RANSAC.

        Returns:
            torch.Tensor: New parameter matrix with aggregated values (b, d, h, w)

        """
        batch_size = labels.size(0)

        new_params = torch.zeros_like(parameters)
        # Batch loop
        for batch_id in range(batch_size):
            # Aggregation per Label
            for label in range(1, labels[batch_id].max().int() + 1):
                # Plane mask
                agg_mask = labels[batch_id] == label
                # Parameter values as tensor "list" (n, d)
                values = torch.masked_select(parameters[batch_id], agg_mask).view(3, -1).moveaxis(0, 1)
                # Initialise model to fit
                ransac_model = PlaneParamModel()
                # Find inliers
                inliers = pyransac.find_inliers(list(values.cpu().numpy()), ransac_model, ransac_parameter)
                if len(inliers) == 0:
                    # If ransac fails no aggregation is performed
                    new_params[batch_id] += parameters[batch_id] * agg_mask
                    continue
                # Mean of inliers as new aggregated value
                mean_value = torch.Tensor(np.stack(inliers).mean(axis=0)).cuda()
                new_params[batch_id] += agg_mask.expand((3, agg_mask.size(1), agg_mask.size(2))) * mean_value.unsqueeze(
                    1).unsqueeze(1)

            new_params[batch_id] += parameters[batch_id] * (labels[batch_id] == 0)

        return new_params

    @staticmethod
    def aggregate_parameters_mean(labels, parameters):
        """This function aggregates the parameters over the segmentation by taking the mean.

        Parameters:
            labels (torch.Tensor): Plane segmentation. (b, 1, h, w)
            parameters (torch.Tensor): Plane parameters which will be aggregated. (b, d, h, w)

        Returns:
            torch.Tensor: New parameter matrix with aggregated values (b, d, h, w)
        """
        # parameter extraction
        batch_size = labels.size(0)
        n, height, width = parameters[0].shape

        # loop over batches and labels and take the mean for each plane as new value
        new_params = torch.zeros_like(parameters)
        for batch_id in range(batch_size):
            for label in range(1, labels[batch_id].max().int() + 1):
                agg_mask = labels[batch_id] == label
                values = torch.masked_select(parameters[batch_id], agg_mask).view(n, -1)
                new_params[batch_id] += agg_mask.expand((n, agg_mask.size(1), agg_mask.size(2))) * values.mean(dim=1).unsqueeze(
                    1).unsqueeze(1)

            new_params[batch_id] += parameters[batch_id] * (labels[batch_id] == 0)

        return new_params

    @staticmethod
    def embedding_segmentation(embedding):
        """This function cluster the embedding vector to receive the plane segmentation.

        Parameters:
            embedding (torch.Tensor): Embedding vector outputted by the network (b, 8, h, w) or (8, h, w)
        """
        labels = list()
        if len(embedding.shape) == 3:
            embedding = embedding.unsqueeze(0)

        # Batch loop
        for batch_id in range(embedding.size(0)):
            # transform tensor to numpy array
            emb = embedding[batch_id].detach().cpu().numpy()
            e, height, width = emb.shape
            emb = np.moveaxis(np.reshape(emb, (e, height * width)), 0, -1)

            # HDBSCAN as segmentation method
            hdb = hdbscan.HDBSCAN(cluster_selection_epsilon=0.1, min_samples=10)  # SCANNet
            cluster_result = hdb.fit(emb)
            label_image = torch.Tensor(np.reshape(cluster_result.labels_, (1, height, width))).cuda() + 1

            # #### Two Alternatives that turned out worse #### #

            # Meanshift
            # meanshift = MeanShift(bandwidth=0.4, bin_seeding=True, n_jobs=-1, cluster_all=False)
            # cluster_result = meanshift.fit(emb)
            # label_image = torch.Tensor(np.reshape(cluster_result.labels_, (1, height, width))).cuda() + 1

            # MiniSOM
            # som = MiniSom(16, 16, 8, sigma=0.3, learning_rate=0.5)  # initialization of 6x6 SOM
            # som.train_batch(emb, 500)
            # winner_coordinates = np.array([som.winner(x) for x in emb]).T
            # label_image = torch.Tensor(np.ravel_multi_index(winner_coordinates, (16, 16)).reshape((1, height, width))).cuda() + 1

            # Split and remove resulting labels
            label_image = torch.Tensor(Utility.split_and_remove_labels(label_image, 200)).cuda()
            label_image = label_image.reshape(1, int(height), int(width))
            labels.append(label_image)

        return torch.stack(labels, dim=0)


class PlaneParamModel(Model):
    """This class incorporates the RANSAC model to determine inliers for surface normal aggregation"""
    def __init__(self):
        self.normal = np.zeros(3)

    def make_model(self, points):
        n = len(points)
        self.normal = (sum(points) / n)[0:3]

    def calc_error(self, point):
        error = np.abs(np.dot(point[0:3], self.normal) / (np.linalg.norm(point[0:3])*np.linalg.norm(self.normal)))
        error = np.arccos(error)
        return error
