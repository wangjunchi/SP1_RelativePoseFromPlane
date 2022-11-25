"""This file includes the Distance2PointCloud class which calculates a 3D point cloud given the depth map."""

import numpy as np
import torch

from Utilities.DatasetEnum import DatasetEnum
from Utilities.util import Utility


class Distance2PointCloud:
    """This class calculates the 3D point cloud of a depth map. Batch input is supported."""
    def __init__(self, dataset=None):
        """Initialises the Distance2PointCloud. Some parameters can be preset by defining the dataset it is going to
        calculate on.

        Parameters:
            dataset (DatasetEnum): Dataset for which the parameters are preset.
        """
        if dataset == DatasetEnum.HYPERSIM:
            # coordinate center image frame in percent
            self.__center = (0.5, 0.5)
            # coordinate orientation (1, 1, 1) meaning x to the right, y up and z in image direction.
            self.__orientation = (1, -1, -1)
            # original image size for reference if a scaled down depth is transformed
            self.__original_image_size = (1024, 768)
            # focal length
            self.__focal_length = Utility.calculate_hypersim_focal_length()
            # or intrinsic matrix as alternative to focal length
            self.__intrinsic_matrix = None
        elif dataset == DatasetEnum.SCANNET:
            self.__center = (0.5, 0.5)
            self.__orientation = (-1, -1, 1)
            self.__original_image_size = (640, 480)
            self.__focal_length = 517.97
            self.__intrinsic_matrix = None
        else:
            self.__center = None
            self.__orientation = None
            self.__original_image_size = None
            self.__focal_length = None
            self.__intrinsic_matrix = None

    def __distance_to_point_cloud_im(self, depth, grid, intrinsic_matrix):
        """This function calculated the point cloud with the help of an intrinsic matrix.

        Parameters:
            depth (torch.Tensor): depth map  (b, 1, h, w)
            grid (torch.Tensor): x, y coordinates for further calculations (h, w, 2)
            intrinsic_matrix (torch.Tensor): intrinsic matrix of the camera used (3, 3)

        Returns:
             torch.Tensor: Point cloud as a list (b, h*w, 3)
        """
        batch_size, _, height_pixels, width_pixels = depth.size()

        point_cloud_batch = torch.zeros((batch_size, height_pixels * width_pixels, 3)).to(depth.device)
        grid[:, :, 0] = grid[:, :, 0] / grid[:, :, 0].max()
        grid[:, :, 1] = grid[:, :, 1] / grid[:, :, 1].max()

        for batch_id in range(batch_size):
            x = grid[:, :, 0] - intrinsic_matrix[0, 2] * depth / intrinsic_matrix[0, 0]
            y = grid[:, :, 1] - intrinsic_matrix[1, 2] * depth / intrinsic_matrix[1, 1]
            z = depth
            xyz = torch.stack((x, y, z)).squeeze().reshape(3, -1).moveaxis(0, -1)
            point_cloud_batch[batch_id] = xyz

        return point_cloud_batch

    def __depth_to_point_cloud_fl(self, depth, grid, focal_length):
        """This function calculated the point cloud with the help of the focal length.

        Parameters:
            depth (torch.Tensor): depth map  (b, 1, h, w)
            grid (torch.Tensor): x, y coordinates for further calculations (h, w, 2)
            focal_length (float): focal length of the image

        Returns:
             torch.Tensor: Point cloud as a list (b, h*w, 1)
        """
        batch_size, _, height_pixels, width_pixels = depth.size()

        grid = torch.reshape(grid, (height_pixels * width_pixels, 2)).type(depth.type())
        point_cloud_batch = torch.zeros((batch_size, height_pixels * width_pixels, 3)).to(depth.device).type(depth.type())
        for batch_id in range(batch_size):
            result = grid / focal_length * torch.reshape(depth[batch_id], (height_pixels * width_pixels, 1))
            result = torch.cat((result, -torch.reshape(depth[batch_id], (height_pixels * width_pixels, 1)),
                                torch.ones_like(result[:, 0:1])), 1)
            point_cloud_batch[batch_id] = result[:, 0:3]

        return point_cloud_batch

    def depth_to_point_cloud(self, depth, coord_center=None, original_pict_size=None, coord_orientation=None,
                             focal_length=None, intrinsic_matrix=None):
        """This function transforms the depth map input into a point cloud. The input can be batched.

        Depth can be given as a pytorch Tensor or numpy array and can be a single image or a batch of images.

        Parameters:
            depth (torch.Tensor, np.ndarray): Depth map to be transformed (b, 1, h, w) or (1, h, w)
            coord_center (tuple): None to use preset from initialisation. Otherwise, tuple of image frame coord. center in percent.
            original_pict_size (tuple): None to use preset from initialisation. Tuple as (width, height) of unscaled image.
            coord_orientation (tuple): None to use preset from initialisation. Coordinate orientation (1, 1, 1) meaning x to the right, y up and z in image direction.
            focal_length (float): None to use preset from initialisation or intrinsic matrix.
            intrinsic_matrix (torch.Tensor): None to use preset from initialisation or focal length. 3 by 3 Matrix.

        Returns:
            Point cloud batch from the input (b, h*w, 3)
        """
        # uses the preset if it is not given
        if coord_center is None:
            coord_center = self.__center
        if original_pict_size is None:
            original_pict_size = self.__original_image_size
        if coord_orientation is None:
            coord_orientation = self.__orientation
        if focal_length is None:
            focal_length = self.__focal_length
        if intrinsic_matrix is None:
            intrinsic_matrix = self.__intrinsic_matrix

        # checks that all necessary information is available
        if None in (coord_center, original_pict_size, coord_orientation):
            missing = ["", "coordinate center", "original pict size", "coordinates orientation"]
            index = [(coord_center is None)*1, (original_pict_size is None)*2, (coord_orientation is None)*3]
            raise Exception(f"The parameter {[missing[i] for i in index]} is missing")

        # transforms numpy to pytorch Tensor
        if type(depth) == np.ndarray:
            depth = torch.as_tensor(depth)
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)

        # checks dimensionality and makes it (b, 1, h, w)
        assert depth.dim() in (3, 4)
        if depth.dim() == 3:
            depth = depth.unsqueeze(0)

        # parameter extraction
        batch_size, _, height_pixels, width_pixels = depth.size()
        device = depth.device

        # screen coordinates
        u, v = np.meshgrid(np.linspace(0, original_pict_size[0], width_pixels),
                           np.linspace(0, original_pict_size[1], height_pixels))

        u = coord_orientation[0] * (u - u.max()*coord_center[0])
        v = coord_orientation[1] * (v - v.max()*coord_center[0])
        depth = -coord_orientation[2] * depth
        grid = np.dstack((u, v))
        grid = torch.Tensor(grid).to(device)

        # calculate point cloud depending on given information.
        if intrinsic_matrix is not None:
            intrinsic_matrix = torch.Tensor(intrinsic_matrix).to(device)
            return self.__distance_to_point_cloud_im(depth, grid, intrinsic_matrix)
        elif focal_length is not None:
            return self.__depth_to_point_cloud_fl(depth, grid, focal_length)
        else:
            raise Exception("Focal length or intrinsic matrix is required!")
