# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: TorchModel
# Description: Class TorchModel, that encapsulates the API of PyTorch supervised models.

# Contributors: João Saraiva and code from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial
# Created: 24/07/2022
# Last Updated: 07/08/2022

# ===================================

import torch
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

from ltbio.ml.models.SupervisedModel import SupervisedModel
from ltbio.ml.trainers.PredictionResults import PredictionResults
from ltbio.ml.trainers.SupervisedTrainResults import SupervisedTrainResults


class TorchModel(SupervisedModel):

    def __init__(self, design: torch.nn.Module, name: str = None):
        if not isinstance(design, torch.nn.Module):
            raise ValueError("The design given is not a valid PyTorch module. "
                             "Give a torch.nn.Module instance.")

        super().__init__(design, name)

    def train(self, dataset, conditions):

        def __train(dataloader) -> float:

            size = len(dataloader.dataset)
            self.design.train()  # Sets the module in training mode
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.float(), y.float()
                # X, y = X.to(device), y.to(device)  # TODO: pass to cuda if available

                # Compute prediction and loss
                pred = self.design(X)
                loss = conditions.loss(pred, y)

                # Backpropagation
                conditions.optimizer.zero_grad()
                loss.backward()
                conditions.optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    if self.verbose:
                        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            return loss.data.item()  # returns the last loss

        def __validate(dataloader: DataLoader) -> float:
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            self.design.eval()  # Sets the module in evaluation mode
            test_loss, correct = 0., 0
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.float(), y.float()
                    # X, y = X.to(device), y.to(device)  # TODO: pass to cuda if available
                    pred = self.design(X)
                    test_loss += conditions.loss(pred, y).data.item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size

            if self.verbose:
                print(f"Validation Error: \n Avg loss: {test_loss:>8f} \n")

            return test_loss

        # Call super for version control
        super().train(dataset, conditions)

        # Check it these optional conditions are defined
        conditions.check_it_has(('optimizer', 'learning_rate', 'validation_ratio', 'batch_size', 'epochs'))

        # Check loss function
        if not isinstance(conditions.loss, _Loss):
            raise ValueError("The loss function given in 'conditions' is not a valid PyTorch loss function."
                             " Give an instance of one of the listed here: https://pytorch.org/docs/stable/nn.html#loss-functions")

        # Check optimizer algorithm
        if not isinstance(conditions.optimizer, Optimizer):
            raise ValueError("The optimizer algorithm given in 'conditions' is not a valid PyTorch optimizer."
                             " Give an instance of one of the listed here: https://pytorch.org/docs/stable/optim.html#algorithms")

        # Learning rate is a property of the optimizer
        conditions.optimizer.lr = conditions.learning_rate

        # Divide dataset into 2 smaller train and validation datasets
        validation_size = int(len(dataset) * conditions.validation_ratio)
        train_size = len(dataset) - validation_size
        train_dataset, validation_dataset = dataset.split(train_size, validation_size, conditions.shuffle is True)

        # Decide on shuffling between epochs
        epoch_shuffle = False
        if conditions.epoch_shuffle is True:  # Shuffle in every epoch
            epoch_shuffle = True

        # Create DataLoaders
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=conditions.batch_size, shuffle=epoch_shuffle)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=conditions.batch_size,
                                           shuffle=epoch_shuffle)

        # Repeat the train-validate process for N epochs
        train_losses, validation_losses = [], []
        try:
            for t in range(conditions.epochs):
                if self.verbose:
                    print(f"Epoch {t + 1}\n-------------------------------")

                # Train and validate
                train_loss = __train(train_dataloader)
                validation_loss = __validate(validation_dataloader)
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)

                # Remember the smaller loss and save checkpoint
                if t == 0:
                    best_loss = validation_loss  # defines the first
                elif validation_loss < best_loss:
                    best_loss = validation_loss
                    self._SupervisedModel__update_current_version_state(epoch_concluded=t+1)

            print("Training finished")

        except KeyboardInterrupt:
            print("Training Interrupted")
            while True:
                answer = input("Save Parameters? (y/n): ").lower()
                if answer == 'y':
                    self._SupervisedModel__update_current_version_state(epoch_concluded=t+1)
                    print("Model and parameters saved.")
                    break
                elif answer == 'n':
                    print("Session Terminated. Parameters not saved.")
                    break
                else:
                    continue # asking

        return SupervisedTrainResults(train_losses, validation_losses)


    def test(self, dataset, evaluation_metrics = None, version = None):
        # Call super for version control
        super().test(dataset, evaluation_metrics, version)

        # Create dataset and dataloader
        dataloader = DataLoader(dataset=dataset, batch_size=self._SupervisedModel__current_version.conditions.batch_size)

        # Test by batch
        size = len(dataset)
        num_batches = len(dataloader)
        self.design.eval()  # Sets the module in evaluation mode
        test_loss, correct = 0, 0
        predictions = []
        with torch.no_grad():
            for X, y in dataloader:  # for each batch
                X, y = X.float(), y.float()
                # X, y = X.to(device), y.to(device)  # TODO: pass to cuda if available
                pred = self.design(X)
                predictions.append(pred.cpu().detach().numpy().squeeze())
                test_loss += self._SupervisedModel__current_version.conditions.loss(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size

        if self.verbose:
            print(f"Test Error: Avg loss: {test_loss:>8f} \n")

        results = PredictionResults(test_loss, dataset, tuple(predictions), evaluation_metrics)
        self._SupervisedModel__update_current_version_best_test_results(results)
        return results

    @property
    def trained_parameters(self):
        if not self.is_trained:
            raise ReferenceError("This model was not yet trained.")
        return self.design.state_dict()

    @property
    def non_trainable_parameters(self):
        if not self.is_trained:
            return {}
        else:
            return self._SupervisedModel__current_version.conditions.hyperparameters

    def _SupervisedModel__set_state(self, state):
        self.design.load_state_dict(state)

    def _SupervisedModel__get_state(self):
        return self.design.state_dict()
        # Optimizer state_dict is inside conditions.optimizer, hence also saved in Version
