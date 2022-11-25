"""Includes the class to log the training and evaluation data to Tensorboard."""

import torch

from torch.utils.tensorboard import SummaryWriter

from Utilities.AverageMeters import AverageMeter
from Utilities.Visualize import Visualisation as vs


class Logger:
    """Logs the training data to Tensorboard.

    Main function being for training: update_losses
    Log an epoch over: epoch_log

    For not mainstream logger the SummaryWriter is accessible over .writer
    """

    def __init__(self, folder_name=None, name="", log_frequency=1):
        """Initialises the Tensorboard writer and the necessary structure to log with any frequency

        Parameters:
            folder_name (str): Where to save the Tensorboard save file
            name (str): Name of the Tensorboard save file
            log_frequency (int): After how many batches the data is actually logged.
        """
        self.writer = SummaryWriter(log_dir=folder_name, comment=name)

        # training data
        self.loss_push_pull = AverageMeter()
        self.loss_normals_angle_avg = AverageMeter()
        self.loss_planes_depth_avg = AverageMeter()
        self.loss_avg = AverageMeter()
        self.overall_loss = AverageMeter()

        self.log_frequency = log_frequency
        self.counter = 0

    def update_losses(self, loss_push_pull, loss_depth, loss_normals, loss, n=1):
        """Saves the losses of the current batch. If the log frequency requires it, it automatically logs the losses.

        Parameters:
            loss_push_pull (float): Value of the Push-Pull-Loss
            loss_depth (float): Value of the Depth estimation loss.
            loss_normals (float): Value of the Normal estimation loss.
            loss (float): Overall loss value.
            n (int): How many images this batch included
        """
        # Update all values
        self.loss_avg.update(loss, n)
        self.loss_push_pull.update(loss_push_pull, n)
        self.loss_normals_angle_avg.update(loss_normals, n)
        self.loss_planes_depth_avg.update(loss_depth, n)

        self.overall_loss.update(loss, n)

        self.counter += 1
        if self.counter % self.log_frequency == 0:
            self.log_scalar_info()

    def log_scalar_info(self):
        """This functions logs the saved values to Tensorboard."""

        # Logging the values
        self.writer.add_scalar("loss_push_pull", self.loss_push_pull.avg, self.counter)
        self.writer.add_scalar("loss_normals_angle", self.loss_normals_angle_avg.avg, self.counter)
        self.writer.add_scalar("loss_planes_depth", self.loss_planes_depth_avg.avg, self.counter)
        self.writer.add_scalar("loss", self.loss_avg.avg, self.counter)
        self.writer.add_scalar(f"average Loss", self.overall_loss.avg, self.counter)

        # Resetting the tracker
        self.loss_avg.reset()
        self.loss_planes_depth_avg.reset()
        self.loss_normals_angle_avg.reset()
        self.loss_push_pull.reset()

    def epoch_log(self, batch_time, image, planes_para, gt_planes_para, labels, gt_labels, depth, gt_depth, epoch):
        """This functions logs all the metrics regarding a complete epoch.

        Parameters:
            batch_time (AverageMeter): Average meter with the average batch time and total training time of that epoch.
            image (torch.Tensor): Random original image of the last batch as an example. (3, h, w)
            planes_para (torch.Tensor): Generated plane parameters of example image. (4, h, w)
            gt_planes_para (torch.Tensor): Ground truth plane parameters of example image. (4, h, w)
            labels (torch.Tensor): Generated plane segmentation of example image. (1, h, w)
            gt_labels (torch.Tensor): Ground truth plane segmentation of example image. (1, h, w)
            depth (torch.Tensor): Generated depth estimation of example image. (1, h, w)
            gt_depth (torch.Tensor): Ground truth depth of example image. (1, h, w)
            epoch (int): Number of epoch that have been run.
        """
        # Scalar data logging
        self.writer.add_scalar("average loss epochs", self.overall_loss.avg, epoch)
        self.writer.add_scalar("average time epochs", batch_time.val, epoch)
        self.writer.add_scalar("total time epoch", batch_time.sum, epoch)

        # Reset all counters
        self.overall_loss.reset()

        # Log original image
        if image.size(0) == 1:
            self.writer.add_image(f"original", image, epoch, dataformats='CHW')
        else:
            self.writer.add_image(f"original", image, epoch)

        # depth comparison
        depth_image, gt_depth_image = vs.prepare_depth_comparison(depth, gt_depth)
        depth_comparison = torch.stack((depth_image, gt_depth_image), 0)
        self.writer.add_images(f"Depth", depth_comparison, epoch, dataformats='NHWC')

        # normal comparison
        normal_image, gt_normal_image = vs.prepare_normal_comparison(planes_para[:3], gt_planes_para[:3])
        normals_comparison = torch.stack((normal_image, gt_normal_image), 0)
        self.writer.add_images(f"Normals", normals_comparison, epoch, dataformats='NHWC')

        # planes rms
        diff = torch.sqrt(((planes_para - gt_planes_para) ** 2).sum(dim=0)).squeeze()
        self.writer.add_image("Plane rms", diff, epoch, dataformats='HW')

        # plane label comparison
        plane_label_image, gt_plane_label_image = vs.prepare_label_comparison(labels, gt_labels)
        plane_label_comparison = torch.stack((plane_label_image, gt_plane_label_image), 0)
        self.writer.add_images(f"Labels", plane_label_comparison, epoch, dataformats='NHWC')

    def print_info(self, epoch, i, dataloader_size, batch_time):
        """Prints some information into the console.

        Parameters:
            epoch (int): Current epoch
            i (int): Current batch number
            dataloader_size (int): total amount of batches
            batch_time (float): Time it took to finish this batch
        """
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})'
              .format(epoch, i, dataloader_size, batch_time=batch_time, loss=self.overall_loss))
