###################################

# IT - PreEpiSeizures

# Package: ml
# File: LinearModel
# Description:

# Contributors: João Saraiva
# Created: 31/05/2022

###################################
from abc import ABC, abstractmethod
from typing import List

from sklearn.linear_model import *
from sklearn.linear_model._base import LinearClassifierMixin
from numpy import array

from src.biosignals.Biosignal import Biosignal
from src.ml.MLModel import MLModel


class LinearModel(MLModel):

    def __init__(self, design, name: str = None, version: int = None):
        if type(design) is not LinearModel:
            raise TypeError("The design should be a linear design from Scikit Learn.")
        super().__init__(design, name, version)

    def train(self, object:List[Biosignal], target:array):
        for biosignal in object:
            for channel_name in object.channel_names:
                channel = object[channel_name]
                for segment in channel.segments:
                    self.model.fit()









