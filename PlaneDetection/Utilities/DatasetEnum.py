"""This file contains the DatasetEnum class, which can be used to declare what dataset is being used in that instance"""
from enum import Enum


class DatasetEnum(Enum):
    HYPERSIM = 1
    SCANNET = 2
