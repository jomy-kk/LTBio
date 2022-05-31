###################################

# IT - PreEpiSeizures

# Package: ml
# File: MLModel
# Description: Abstract Class representing a generic machine learning model.

# Contributors: João Saraiva
# Created: 31/05/2022

###################################

from abc import ABC, abstractmethod

from src.biosignals.Biosignal import Biosignal


class MLModel(ABC):

    def __init__(self, representation, name:str=None, version:int=None):
        self.model = representation
        self.name = name
        self.version = version

    @abstractmethod
    def train(self, object:Biosignal, target:MLTarget):
        pass

    @abstractmethod
    def test(self, object:Biosignal, target:MLTarget):
        pass


class NeuralNetwork(MLModel):

    def __init__(self):





class ConvolutionalAutoencoder(NeuralNetwork, nn.Module):

    def __init__(self):




