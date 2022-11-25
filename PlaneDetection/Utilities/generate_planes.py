"""
This file contains the GeneratePlanes class which could include several ways to generate plane parameters and
segmentation. For this work only the generation from normals and depth and Meanshift segmentation is implemented.
"""

import torch
import numpy as np

from scipy.ndimage import gaussian_filter
from sklearn.cluster import MeanShift

from Utilities.distance_to_point_cloud import Distance2PointCloud
from Utilities.util import Utility


class GeneratePlanes:
    """Generates planes and their segmentation from normal and depth information."""

    @staticmethod
    def plane_from_normals_and_depth(normals, depth_map, dataset, bandwidth=0.25, cpu_cores=4):
        """Generates plans and their segmentation from surface normal and depth.

        Parameters:
            normals (torch.Tensor): Surface normal of the scene. (b, 3, h, w) or (3, h, w)
            depth_map (torch.Tensor): Depth map of the scene. (b, 1, h, w) or (1, h, w)
            dataset (DatasetEnum): Which dataset is being used.
            bandwidth (float): Bandwidth of the Meanshift algorithm.
            cpu_cores (int): Number of cpu cores to be used.

        Returns:
            tuple: Tuple consisting of the plane parameters (b, 4, h, w) and plane segmentation (b, 1, h ,w)
        """
        assert depth_map.dim() in (3, 4)
        assert normals.dim() in (3, 4)
        assert normals.dim() == depth_map.dim()

        # Making input to single batch input (b, _, h, w)
        if normals.dim() == 3:
            normals = normals.unsqueeze(0)
            depth_map = depth_map.unsqueeze(0)

        # parameter extraction
        device = normals.device
        batch_size, _, height, width = normals.size()

        # Calculates parameter from normals and depth information
        plane_images = GeneratePlanes.plane_params_from_normals_and_depth(normals, depth_map, dataset)

        # preparing result Tensor
        label_images = torch.zeros_like(depth_map, dtype=torch.int8).to(device)

        # setting up meanshift, bin seeding for better performance
        mean_shift = MeanShift(bandwidth=bandwidth, n_jobs=cpu_cores, cluster_all=False, bin_seeding=True)

        # Batch loop to create plane segmentation for each image
        for batch_idx in range(batch_size):
            # creating smooth planes and list of points, helps for more coherent planes on textured surfaces
            smoothed_planes = gaussian_filter(np.array(torch.clone(plane_images[batch_idx]).detach().cpu()), 1.2)
            # smoothed_planes = np.array(torch.clone(plane_images[batch_idx]).detach().cpu())
            smoothed_planes = np.moveaxis(np.reshape(smoothed_planes, (4, height*width)), 0, -1)

            # clustering
            cluster_result = mean_shift.fit(smoothed_planes)
            label_image = np.reshape(cluster_result.labels_ + 1, (1, height, width))
            label_image = Utility.split_and_remove_labels(label_image, height * width / 400, smoothed_planes)
            label_images[batch_idx] = torch.Tensor(label_image).to(device)

        return plane_images, label_images

    @staticmethod
    def plane_params_from_normals_and_depth(normals, depth_map, dataset):
        """Calculates the 4 plan parameters from surface normals and depth.

        Parameters:
            normals (torch.Tensor): Surface normal of the scene. (b, 3, h, w) or (3, h, w)
            depth_map (torch.Tensor): Depth map of the scene. (b, 1, h, w) or (1, h, w)
            dataset (DatasetEnum): Which dataset is being used.

        Returns:
            torch.Tensor: Plane parameters (b, 4, h, w)
        """
        assert normals.dim() == depth_map.dim()
        assert normals.dim() in (3, 4)
        assert depth_map.dim() in (3, 4)

        if normals.dim() == 3:
            depth_map = depth_map.unsqueeze(0)
            normals = normals.unsqueeze(0)

        # parameter extraction
        batch_size, _, height, width = normals.size()

        # calculate 3D coordinates to calculated fourth plane parameter
        distance_to_pc = Distance2PointCloud(dataset)
        point_clouds = distance_to_pc.depth_to_point_cloud(depth_map)

        # multiply 3D coordinates with the surface normal
        normal_list = torch.moveaxis(torch.reshape(normals, (batch_size, 3, height*width)), -2, -1)
        d = -torch.reshape(torch.sum(point_clouds*normal_list, 2), (batch_size, 1, height, width))

        # combine the fourth parameter with the surface normal to receive plane parameters
        return torch.cat((normals, d), 1)
