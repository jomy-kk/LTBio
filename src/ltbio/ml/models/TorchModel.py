# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: TorchModel
# Description: Class TorchModel, that encapsulates the API of PyTorch supervised models.

# Contributors: João Saraiva and code from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial
# Created: 24/07/2022

# ===================================
from os import makedirs
from os.path import exists, join
from typing import Collection

import numpy as np
import torch
from numpy import ndarray
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from ltbio.ml.datasets.TimeseriesToTimeseriesDataset import TimeseriesToTimeseriesDataset

from ltbio.biosignals import Timeseries
from ltbio.ml.models.SupervisedModel import SupervisedModel
from ltbio.ml.trainers import SupervisedTrainConditions


class TorchModel(SupervisedModel):

    def __init__(self, design: torch.nn.Module, name: str = None, version: int = None):
        if not isinstance(design, torch.nn.Module):
            raise ValueError("The design given is not a valid PyTorch module."
                             "Give a torch.nn.Module instance.")

        super().__init__(design, name, version)


    def setup(self, train_conditions: SupervisedTrainConditions, **kwargs):
        """
        Goal here is simply to validate the objects stored in 'loss' and 'optimizer' in train_conditions.
        """

        # Check loss function
        self.__loss = train_conditions.loss
        if not isinstance(self.__loss, _Loss):
            raise ValueError("The loss function given in 'train_conditions' is not a valid PyTorch loss function."
                             " Give an instance of one of the listed here: https://pytorch.org/docs/stable/nn.html#loss-functions")

        # Check optimizer algorithms
        self.__optimizer = train_conditions.optimizer
        if not isinstance(self.__optimizer, Optimizer):
            raise ValueError("The optimizer algorithm given in 'train_conditions' is not a valid PyTorch optimizer."
                             " Give an instance of one of the listed here: https://pytorch.org/docs/stable/optim.html#algorithms")

        # Get hyperparameters
        self.__epochs = train_conditions.epochs
        self.__batch_size = train_conditions.batch_size
        self.__learning_rate = train_conditions.learning_rate
        self.__validation_ratio = train_conditions.validation_ratio

        # Learning rate is a property of the optimizer
        self.__optimizer.lr = self.__learning_rate

        # Save train conditions
        self.last_train_conditions = train_conditions


    def train(self, object, target):

        def __dataloaders(object: Collection[Timeseries], target: Collection[Timeseries]) -> tuple[DataLoader, DataLoader]:
            # Create dataset
            dataset = TimeseriesToTimeseriesDataset(object, target)

            # Divide dataset into 2 smaller train and validation datasets
            validation_size = int(len(dataset) * self.__validation_ratio)
            train_size = len(dataset) - validation_size
            self.__train_dataset , self.__validation_dataset = random_split(dataset, (train_size, validation_size), generator=torch.Generator().manual_seed(42))  # Docs recommend to fix the generator seed for reproducible results

            # Decide on shuffling between epochs
            epoch_shuffle = False
            if self.last_train_conditions.epoch_shuffle is True:  # Shuffle in every epoch
                epoch_shuffle = True

            # Create DataLoaders
            train_dataloader = DataLoader(dataset=self.__train_dataset, batch_size=self.__batch_size, shuffle=epoch_shuffle)
            validation_dataloader = DataLoader(dataset=self.__validation_dataset, batch_size=self.__batch_size, shuffle=epoch_shuffle)

            return train_dataloader, validation_dataloader

        def __train(dataloader) -> float:

            size = len(dataloader.dataset)
            self.design.train()  # Sets the module in training mode
            for batch, (X, y) in enumerate(dataloader):
                # X, y = X.to(device), y.to(device)  # TODO: pass to cuda if available

                # Compute prediction and loss
                pred = self.design(X)
                loss = self.__loss(pred, y)

                # Backpropagation
                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            return loss.data.item()  # returns the last loss

        def __validate(dataloader: DataLoader) -> float:
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            self.design.eval()  # Sets the module in evaluation mode
            test_loss, correct = 0., 0
            with torch.no_grad():
                for X, y in dataloader:
                    # X, y = X.to(device), y.to(device)  # TODO: pass to cuda if available
                    pred = self.design(X)
                    test_loss += self.__loss(pred, y).data.item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size

            print(f"Validation Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            return test_loss

        # Prepare dataloaders
        train_dataloader, validation_dataloader = __dataloaders(object, target)

        # Repeat the train-validate process for N epochs
        self.__train_losses, self.__validation_losses = [], []
        try:
            for t in range(self.__epochs):
                print(f"Epoch {t + 1}\n-------------------------------")

                # Train and validate
                train_loss = __train(train_dataloader)
                validation_loss = __validate(validation_dataloader)
                self.__train_losses.append(train_loss)
                self.__validation_losses.append(validation_loss)

                # Remember the smaller loss and save checkpoint
                if t == 0:
                    best_loss = validation_loss  # defines the first
                elif validation_loss < best_loss:
                    best_loss = validation_loss
                    self.save(self.name + '_Best', epoch=t)

            print("Training finished")

        except KeyboardInterrupt:
            print("Training Interrupted")
            while True:
                answer = input("Save Parameters? (y/n): ").lower()
                if answer == 'y':
                    save_name = input("Save as: ") + '_Interrupted'
                    self.save(save_name, t)
                    print("Model and parameters saved.")
                    break
                elif answer == 'n':
                    print("Session Terminated. Parameters not saved.")
                    break
                else:
                    continue # asking


    def test(self, object, target=None):
        # Create dataset and dataloader
        self.__test_dataset = TimeseriesToTimeseriesDataset(object, target)
        dataloader = DataLoader(dataset=self.__test_dataset, batch_size=self.__batch_size)

        # Test by batch
        size = len(self.__test_dataset)
        num_batches = len(dataloader)
        self.design.eval()  # Sets the module in evaluation mode
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:  # for each batch
                # X, y = X.to(device), y.to(device)  # TODO: pass to cuda if available
                pred = self.design(X)
                test_loss += self.__loss(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size

        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        self.__test_loss = test_loss


    def report(self, reporter, show, save_to):
        pass

    @property
    def trained_parameters(self):
        pass

    @property
    def non_trainable_parameters(self):
        return {}

    def save(self, path:str = '.', epoch:int = None):
        if not exists(path):
            makedirs(path)

        #self.design.cpu()
        torch.save({'train_dataset': self.__train_dataset,
                    'validation_dataset': self.__validation_dataset,
                    'state_dict': self.design.state_dict(),
                    'epoch': epoch,
                    'optimizer': self.__optimizer,
                    'loss_function': self.__loss,
                    'learning_rate': self.__learning_rate
                    }, join(path, 'model.pth'))

        np.save(join(path, 'trainloss.npy'), self.__train_losses)
        np.save(join(path, 'valloss.npy'), self.__validation_losses)
