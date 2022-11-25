import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib import cm
from torchvision.utils import save_image
from torchvision import transforms
from PlaneDetection.Utilities.util import Utility


class Visualisation:

    @staticmethod
    def image_show(data, title="", subtitle=""):
        """This function uses matplotlib to visualize one image.

        It can take numpy arrays and tensors and the order of the dimension does not matter.

        Parameters:
            data (np.ndarray, torch.Tensor): Image to be displayed. any of and order unimportant: (h, w), (1, h, w), (3, h, w)
            title (str): Title of the image
            subtitle (str): Subtitle of the image
        """
        # mpl.use("tkagg")
        if type(data) == torch.Tensor:
            data = data.cpu().detach().squeeze().numpy()
        dimension = data.shape
        # Check dimensionality
        if len(dimension) not in (2, 3):
            raise AttributeError("Images must have 2 or 3 dimensions!")

        # Change dimension order to fit matplotlib requirements
        if len(dimension) == 3:
            if dimension[0] < dimension[1]:
                if dimension[0] < dimension[2]:
                    data = np.moveaxis(data, 0, -1)
            else:
                if dimension[1] < dimension[2]:
                    data = np.moveaxis(data, 1, -1)

        # Normalise data if necessary
        if data.min() < 0:
            data = data - data.min()
            data = data/data.max()

        fig = plt.figure()
        img = fig.add_subplot(111)
        imgplot = plt.imshow(data)
        plt.suptitle(subtitle)
        plt.title(title)
        plt.show()

    @staticmethod
    def point_cloud(points, title="", subtitle=""):
        """This function can be used to visualise a 3D point cloud with matplotlib.

        Parameters:
            points (np.ndarray): Point matrix (n, 3)
            title (str): Title of the image
            subtitle (str): Subtitle of the image
        """
        mpl.use("tkagg")
        fig = plt.figure()
        cloud = fig.add_subplot(111, projection='3d')
        cloud.scatter(points[:, 0], points[:, 1], points[:, 2], c="k", marker='.', s=5)
        plt.title(title)
        plt.suptitle(subtitle)
        plt.show()

    @staticmethod
    def prepare_depth_comparison(depth, gt_depth):
        """This function creates two images with matching colors for the same depth.

        Parameters:
            depth (torch.Tensor, np.ndarray): Depth map 1, normally generated depth estimation.
            gt_depth (torch.Tensor, np.ndarray): Depth map 2, normally ground truth depth.

        Returns:
            tuple: Two images of the two input depth maps with matching colors for equal depth.
        """
        if type(depth) != torch.Tensor:
            depth = torch.from_numpy(depth)
        if type(gt_depth) != torch.Tensor:
            gt_depth = torch.from_numpy(gt_depth)

        # find minimum and maximum depth
        depth_min = torch.min(depth.min(), gt_depth[gt_depth >= 0].min())
        depth_max = torch.max(depth.max(), gt_depth.max())
        # normalise both depth maps in regard with the overall maximum and minimum
        depth_image = (depth - depth_min) / depth_max
        gt_depth_image = (gt_depth - depth_min) / depth_max

        # Use a matplotlib color map to colorise the normalised depth maps.
        color_map = cm.get_cmap("jet")
        depth_image = torch.from_numpy(color_map(depth_image.squeeze().cpu().numpy()))[:, :, 0:3]
        gt_depth_image = torch.from_numpy(color_map(gt_depth_image.squeeze().cpu().numpy()))[:, :, 0:3]

        return depth_image, gt_depth_image

    @staticmethod
    def prepare_label_comparison(labels, gt_labels):
        """This functions equalises the labelling of two segmentation and colorizes them equally.

        Parameters:
            labels (torch.Tensor, np.ndarray): Labels 1, normally generated plane segmentation.
            gt_labels (torch.Tensor, np.ndarray): Labels 2, normally ground truth plane segmentation.

        Returns:
            tuple: Two images of the two input plane segmentation with matching colors for equal planes.
        """
        if type(labels) != torch.Tensor:
            labels = torch.from_numpy(labels)
        if type(gt_labels) != torch.Tensor:
            gt_labels = torch.from_numpy(gt_labels)

        # resize the prediction
        labels = transforms.Resize(gt_labels.shape[1:3], interpolation=transforms.InterpolationMode.NEAREST)(labels)
        # labels = labels.permute((1,2,0))
        # print(labels.shape)
        # print(gt_labels.shape)
        assert labels.shape == gt_labels.shape

        # Label equalising
        labels = Utility.equalise_labelling(labels, gt_labels).squeeze()

        # Create colormap with maximum label
        max_label = torch.max(labels.max(), gt_labels.max())
        color_map = cm.get_cmap("viridis", int(max_label.item()))

        # Colorize images
        plane_label_image = torch.from_numpy(color_map(labels.type(torch.int8).cpu().numpy()))[:, :, 0:3]
        gt_plane_label_image = torch.from_numpy(color_map(gt_labels.squeeze().type(torch.int8).cpu().numpy()))[:, :, 0:3]

        return plane_label_image, gt_plane_label_image

    @staticmethod
    def prepare_normal_comparison(normals, gt_normals):
        """This functions normalises both surface normal matrices.

        Parameters:
            normals (torch.Tensor, np.ndarray): Surface normals 1, normally generated surface normals.
            gt_normals (torch.Tensor, np.ndarray): Surface normals 2, normally ground truth surface normals.

        Returns:
            tuple: Two images of the two input surface normals.
        """
        if type(normals) != torch.Tensor:
            normals = torch.from_numpy(normals)
        if type(gt_normals) != torch.Tensor:
            gt_normals = torch.from_numpy(gt_normals)

        normals = normals / torch.norm(normals, dim=0, keepdim=True)
        gt_normals = gt_normals / torch.norm(gt_normals, dim=0, keepdim=True)

        normals = (normals + 1) / 2
        gt_normals = (gt_normals + 1) / 2

        return normals.squeeze().moveaxis(0, -1), gt_normals.squeeze().moveaxis(0, -1)

    @staticmethod
    def save_tensor_as_image(output, file_name):
        """This functions saves a tensor as an image on the hard drive.

        Parameters:
            output (torch.Tensor): Tensor image which will be saved.
            file_name (str): Path and file name in which the image is saved.
        """
        assert(type(output) == torch.Tensor)
        save_image(output, file_name)
