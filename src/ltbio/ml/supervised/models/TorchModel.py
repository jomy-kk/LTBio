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
import gc
from pickle import dump

import torch
import torchmetrics
from torch import float32
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torchsummary import summary

from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset
from ltbio.ml.supervised.models.SupervisedModel import SupervisedModel
from ltbio.ml.supervised.results import PredictionResults
from ltbio.ml.supervised.results import SupervisedTrainResults


class TorchModel(SupervisedModel):

    DEVICE = torch.device('cpu')

    def __init__(self, design: torch.nn.Module, name: str = None):
        if not isinstance(design, torch.nn.Module):
            raise ValueError("The design given is not a valid PyTorch module. "
                             "Give a torch.nn.Module instance.")

        super().__init__(design, name)


        # Check for CUDA (NVidea GPU) or MPS (Apple Sillicon) acceleration
        try:
            if torch.backends.mps.is_built():
                self.DEVICE = torch.device('mps')
                self._SupervisedModel__design.to(self.DEVICE)
                self._SupervisedModel__design.to(float32)
        except:
            pass
        try:
            if torch.cuda.is_available():
                self.DEVICE = torch.device('cuda')
                self._SupervisedModel__design.to(self.DEVICE)
        except:
            pass

    def shapes_summary(self, dataset: BiosignalDataset):
        example_shape = dataset[0][0].shape
        self._SupervisedModel__design.to('cpu')
        try:
            summary(self._SupervisedModel__design, example_shape, device='cpu')
            self._SupervisedModel__design.to(self.DEVICE)
        finally:
            self._SupervisedModel__design.to(self.DEVICE)

    def train(self, dataset, conditions, n_subprocesses: int = 0, track_memory: bool = False):

        def __train(dataloader) -> float:
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            self._SupervisedModel__design.train()  # Sets the module in training mode
            sum_losses = 0.
            for i, (batch_objects, batch_targets) in enumerate(dataloader):
                #if track_memory:
                #    print_resident_set_size(f'before batch {i} processing')
                #print('!!! batch_objects.shape =', batch_objects.shape)
                #print('!!! batch_targets.shape =', batch_targets.shape)
                conditions.optimizer.zero_grad()  # Zero gradients for every batch
                pred = self._SupervisedModel__design(batch_objects)  # Make predictions for this batch
                loss = conditions.loss(pred, batch_targets)  # Compute loss
                loss.backward()  # Compute its gradients
                conditions.optimizer.step()  # Adjust learning weights

                if i % 10 == 0:
                    loss_value, current = loss.item(), i * len(batch_objects)
                    sum_losses += loss_value
                    if self.verbose:
                        print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

                del batch_objects, batch_targets, loss, pred
                gc.collect()

            #if self.verbose:
            #    print(f"Avg Train Loss: {sum_losses/(num_batches/10):>8f} \n")

            #if track_memory:
            #    print_resident_set_size('after epoch')
            return loss_value  # returns the last loss

        def __validate(dataloader: DataLoader) -> float:
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            self._SupervisedModel__design.eval()  # Sets the module in evaluation mode
            loss_value, correct = 0., 0
            with torch.no_grad():
                for batch_objects, batch_targets in dataloader:
                    pred = self._SupervisedModel__design(batch_objects)
                    loss = conditions.loss(pred, batch_targets)
                    loss_value += loss.data.item()
                    correct += (pred.argmax(1) == batch_targets).type(torch.float).sum().item()

                    del batch_objects, batch_targets, loss, pred
                    gc.collect()

            loss_value /= num_batches
            correct /= size

            if self.verbose:
                print(f"Avg Validation Loss: {loss_value:>8f} \n")

            return loss_value

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
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=conditions.batch_size, shuffle=epoch_shuffle,
                                      #pin_memory=True, #pin_memory_device=TorchModel.DEVICE.type,
                                      num_workers=n_subprocesses, prefetch_factor=2,
                                      drop_last=True)

        validation_dataloader = DataLoader(dataset=validation_dataset,
                                           batch_size=conditions.batch_size, shuffle=epoch_shuffle,
                                           #pin_memory=True, #pin_memory_device=TorchModel.DEVICE.type,
                                           num_workers=n_subprocesses, prefetch_factor=2,
                                           drop_last=True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(conditions.optimizer, mode='min', factor=0.1, patience=5)

        # Repeat the train-validate process for N epochs
        train_losses, validation_losses = [], []
        try:
            for t in range(conditions.epochs):
                if self.verbose:
                    print(f"Epoch {t + 1}\n-------------------------------")

                # Train and validate
                train_loss = __train(train_dataloader)
                validation_loss = __validate(validation_dataloader)
                scheduler.step(validation_loss)
                train_losses.append(train_loss)
                validation_losses.append(validation_loss)

                # Remember the smaller loss and save checkpoint
                if t == 0:
                    best_loss = validation_loss  # defines the first
                    count_loss_has_not_decreased = 0
                    self._SupervisedModel__update_current_version_state(epoch_concluded=t + 1)
                elif validation_loss < best_loss:
                    best_loss = validation_loss
                    self._SupervisedModel__update_current_version_state(epoch_concluded=t+1)
                else:
                    count_loss_has_not_decreased +=1

                if conditions.patience != None and count_loss_has_not_decreased == conditions.patience:
                    print(f'Early stopping at epoch {t}')
                    break

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

        # FIXME: This should be a PlotMetric (?)
        """
        finally:
            fig = plt.figure(figsize=(10, 5))
            plt.subplot(1, 1, 1)
            plt.title("Loss over the Epochs")
            plt.plot(range(1, len(train_losses) + 1), train_losses, "b-", label="Train Loss")
            if validation_losses is not None:
                plt.plot(range(1, len(validation_losses) + 1), validation_losses, "r-", label="Train Loss")
            plt.legend(loc="upper right")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            fig.tight_layout()
            plt.show()
            plt.close()
        """

        return SupervisedTrainResults(train_losses, validation_losses)


    def test(self, dataset, evaluation_metrics = (), version = None):
        # Call super for version control
        super().test(dataset, evaluation_metrics, version)

        # Get current conditions
        conditions = self._SupervisedModel__current_version.conditions

        # Create dataset and dataloader
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                                #pin_memory=True,
                                #pin_memory_device=TorchModel.DEVICE.type
                                )

        f1 = torchmetrics.F1Score(average='weighted', num_classes=2)
        # auc = torchmetrics.AUROC(average='weighted', num_classes=2)

        # Test by example
        size = len(dataset)
        num_batches = len(dataloader)
        self._SupervisedModel__design.eval()  # Sets the module in evaluation mode
        test_loss = 0
        predictions = []
        with torch.no_grad():
            for batch_objects, batch_targets in dataloader:  # for each batch
                pred = self._SupervisedModel__design(batch_objects)
                predictions.append(pred.cpu().detach().numpy().squeeze())
                test_loss += conditions.loss(pred, batch_targets).item()
                # compute metrics
                pred, batch_targets = pred.to('cpu'), batch_targets.to('cpu')
                # FIXME: these shoud be ValueMetric(s)
                #f1(pred, batch_targets)
                #auc(pred, batch_targets)

        test_loss /= num_batches

        if self.verbose:
            print(f"Test Error: Avg loss: {test_loss:>8f}")
            #print(f"Test F1-Score: {f1.compute()}")
            #print(f"Test AUC: {auc.compute()}")

        # FIXME: Remove these two lines below
        #dataset.redimension_to(1)
        #dataset.transfer_to_device('cpu')

        results = PredictionResults(test_loss, dataset, tuple(predictions), evaluation_metrics)
        self._SupervisedModel__update_current_version_best_test_results(results)
        return results

    @property
    def trained_parameters(self):
        if not self.is_trained:
            raise ReferenceError("This model was not yet trained.")
        return self._SupervisedModel__design.state_dict()

    @property
    def non_trainable_parameters(self):
        if not self.is_trained:
            return {}
        else:
            return self._SupervisedModel__current_version.conditions.hyperparameters

    def _SupervisedModel__set_state(self, state):
        self._SupervisedModel__design.load_state_dict(state)

    def _SupervisedModel__get_state(self):
        return self._SupervisedModel__design.state_dict()
        # Optimizer state_dict is inside conditions.optimizer, hence also saved in Version

    def save_design(self, path:str):
        self._SupervisedModel__design.to('cpu')
        with open(path, 'wb') as f:
            dump(self._SupervisedModel__design, f)
        self._SupervisedModel__design.to(self.DEVICE)
