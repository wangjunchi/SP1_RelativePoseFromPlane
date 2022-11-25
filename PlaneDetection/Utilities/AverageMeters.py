"""This files contains the AverageMeter class and all of its variations."""
import numpy as np
import torch


class AverageMeter(object):
    """This class keeps track of a metric. It saves the avg, last value and total value."""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if np.isnan(val):
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterTensor(object):
    """This class is an extension of the AverageMeter class saving tensor instead of floats.
    It saves the avg, last value and total value.
    """
    def __init__(self):
        self.val = torch.Tensor([0])
        self.avg = torch.Tensor([0])
        self.sum = torch.Tensor([0])
        self.count = torch.Tensor([0])

    def reset(self):
        self.val = torch.Tensor([0])
        self.avg = torch.Tensor([0])
        self.sum = torch.Tensor([0])
        self.count = torch.Tensor([0])

    def update(self, val, n=1):
        if torch.isnan(val):
            return
        self.val = val
        self.sum = self.sum.to(val.device) + val * n
        self.count += n
        self.avg = self.sum / self.count.to(val.device)


class RecallMeter:
    """This class is another extension of the AverageMeter Class.
    It extends the concept by saving dictionaries of results. Technically an infinite amount of keys can be stored.
    Values belonging to a specific key are still summed up and averaged.
    """
    def __init__(self):
        self.val = dict()
        self.avg = dict()
        self.sum = dict()
        self.count = dict()

    def update(self, val_dict, n=1):
        for key in val_dict:
            self.val[key] = val_dict[key]
            if key in self.sum.keys():
                self.sum[key] += val_dict[key]
                self.count[key] += n
            else:
                self.sum[key] = val_dict[key]
                self.count[key] = n
            self.avg[key] = self.sum[key] / self.count[key]

    def reset(self):
        self.val = dict()
        self.avg = dict()
        self.sum = dict()
        self.count = dict()
