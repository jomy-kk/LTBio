# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SupervisedTrainConditions
# Description: Class SupervisedTrainConditions, that holds values of parameters to train a model in a specific manner.

# Contributors: João Saraiva
# Created: 04/06/2022
# Last Updated: 07/06/2022

# ===================================
from copy import deepcopy
from typing import Iterable, Collection


class SupervisedTrainConditions():
    def __init__(self, loss,
                 optimizer = None,
                 train_size:int = None, train_ratio:float = None, test_size:int = None, test_ratio:float = None,
                 validation_ratio:float = None,
                 epochs: int = None, learning_rate:float = None, batch_size:int = None,
                 shuffle:bool=False, epoch_shuffle:bool = False,
                 stop_at_deltaloss:float = None, patience:int = None,
                 **hyperparameters):

        # Mandatory conditions

        self.loss = loss
        self.optimizer = optimizer

        # Versatile-mandatory conditions

        if train_size is not None:
            if isinstance(train_size, int) and train_size >= 1:
                self.train_size = train_size
            else:
                raise ValueError("Condition 'train_size' must be an integer >= 1.")
        else:
            self.train_size = None

        if test_size is not None:
            if isinstance(test_size, int) and test_size >= 1:
                self.test_size = test_size
            else:
                raise ValueError("Condition 'test_size' must be an integer >= 1.")
        else:
            self.test_size = None

        if train_ratio is not None:
            if isinstance(train_ratio, float) and 0 < train_ratio < 1:
                self.train_ratio = train_ratio
            else:
                raise ValueError("Condition 'train_ratio' must be between 0 and 1.")
        else:
            self.train_ratio = None

        if test_ratio is not None:
            if isinstance(test_ratio, float) and 0 < test_ratio < 1:
                self.test_ratio = test_ratio
            else:
                raise ValueError("Condition 'test_ratio' must be between 0 and 1.")
        else:
            self.test_ratio = None

        if train_size is None and test_size is None and train_ratio is None and test_ratio is None:
            raise AssertionError("Specify at least 'train_size' or 'test_size' or 'train_ratio' or 'test_ratio'.")

        # Optional conditions

        if validation_ratio is not None:
            if isinstance(validation_ratio, float) and 0 < validation_ratio < 1:
                self.validation_ratio = validation_ratio
            else:
                raise ValueError("Condition 'validation_ratio' must be between 0 and 1.")
        else:
            self.validation_ratio = None

        if epochs is not None:
            if isinstance(epochs, int) and epochs > 0:
                self.epochs = epochs
            else:
                raise ValueError("Condition 'epochs' must be an integer >= 1.")
        else:
            self.epochs = None

        if batch_size is not None:
            if isinstance(batch_size, int) and batch_size > 0:
                self.batch_size = batch_size
            else:
                raise ValueError("Condition 'batch_size' must be an integer >= 1.")
        else:
            self.batch_size = None

        if learning_rate is not None:
            if isinstance(learning_rate, float) and 0 < learning_rate < 1:
                self.learning_rate = learning_rate
            else:
                raise ValueError("Condition 'learning_rate' must be between 0 and 1.")
        else:
            self.learning_rate = None

        if shuffle is not None:
            if isinstance(shuffle, bool):
                self.shuffle = shuffle
            else:
                raise TypeError("Condition 'shuffle' must be True or False.")
        else:
            self.shuffle = None

        if epoch_shuffle is not None:
            if isinstance(epoch_shuffle, bool):
                self.epoch_shuffle = epoch_shuffle
            else:
                raise TypeError("Condition 'epoch_shuffle' must be True or False.")
        else:
            self.epoch_shuffle = None

        if stop_at_deltaloss is not None:
            if isinstance(stop_at_deltaloss, float):
                self.stop_at_deltaloss = stop_at_deltaloss
            else:
                raise TypeError("Condition 'stop_at_deltaloss' must be a float.")
        else:
            self.stop_at_deltaloss = None

        if patience is not None:
            if isinstance(patience, int) and patience > 0:
                self.patience = patience
            else:
                raise TypeError("Condition 'patience' must be an integer > 0.")
        else:
            self.patience = None

        self.hyperparameters = hyperparameters

    @property
    def _slots(self):
        return {
            'optimizer': self.optimizer,
            'loss': self.loss,
            'train_size': self.train_size,
            'test_size': self.test_size,
            'train_ratio': self.train_ratio,
            'test_ratio': self.test_ratio,
            'validation_ratio': self.validation_ratio,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'epoch_shuffle': self.epoch_shuffle,
            'learning_rate': self.learning_rate,
            'stop_at_deltaloss': self.stop_at_deltaloss,
            'patience': self.patience,
        }

    @staticmethod
    def differences_between(sets_of_conditions: Collection) -> tuple[dict, ...]:
        if not isinstance(sets_of_conditions, (list, tuple, set)) or not all(isinstance(_set, SupervisedTrainConditions) for _set in sets_of_conditions):
            raise TypeError("Sets of conditions must be a collection of SupervisedTrainConditions")

        slot_keys = sets_of_conditions[0]._slots.keys()
        differences = [{} for i in range(len(sets_of_conditions))]

        for key in slot_keys:
            all_values = [_set._slots[key] for _set in sets_of_conditions]
            for value in all_values:
                if (key == 'optimizer' or key == 'loss') and not isinstance(value, str):
                    if hasattr(all_values[0], '__getstate__') and hasattr(value, '__getstate__'):
                        if all_values[0].__getstate__() != value.__getstate__():
                            for i, x in enumerate(all_values):
                                differences[i][key] = x
                    else:
                        if all_values[0].__repr__() != value.__repr__():
                            for i, x in enumerate(all_values):
                                differences[i][key] = x
                else:
                    if value != all_values[0]:
                        for i, x in enumerate(all_values):
                            differences[i][key] = x

        # Find differences in hyperparameters
        all_hyperparameters = [_set.hyperparameters for _set in sets_of_conditions]
        if len(all_hyperparameters) > 1:
            different_keys_found = []
            for i in range(len(all_hyperparameters)):
                x = all_hyperparameters[i]
                for j in range(i+1, len(all_hyperparameters)):
                    y = all_hyperparameters[j]
                    diff = set(x.items()) - set(y.items())
                    if len(diff) > 0:
                        for d in diff:
                            different_keys_found.append(d[0])
            for key in different_keys_found:
                for i, x in enumerate(all_hyperparameters):
                    differences[i][key] = x[key] if key in x else None

        return tuple(differences)

    def check_it_has(self, attributes:Iterable[str]):
        slots = self._slots
        for a in attributes:
            if slots[a] == None:
                raise ValueError(f"This type of model requires '{a}' to be defined in the given conditions.")

    def __str__(self):
        res = f'Optimizer: {self.optimizer} | Loss Function: {self.loss}\n'

        if self.train_size is not None and self.test_size is not None:
            res += f'Train Size = {self.train_size} | Test Size = {self.test_size}'
        elif self.train_size is not None:
            res += f'Train Size = {self.train_size}'
        elif self.test_size is not None:
            res += f'Test Size = {self.test_size}'

        if self.train_ratio is not None and self.test_ratio is not None:
            res += f'Train Ratio = {self.train_ratio} | Test Ratio = {self.test_ratio}'
        elif self.train_ratio is not None:
            res += f'Train Ratio = {self.train_ratio}'
        elif self.test_ratio is not None:
            res += f'Test Ratio = {self.test_ratio}'

        if self.validation_ratio is not None:
            res += f' | Validation Ratio = {self.validation_ratio}'

        res += '\n'

        other_optionals = []
        if self.epochs is not None:
            other_optionals.append(f'Epochs = {self.epochs}')
        if self.batch_size is not None:
            other_optionals.append(f'Batch size = {self.batch_size}')
        if self.shuffle is not None:
            other_optionals.append(f'Shuffle: {self.shuffle}')
        if self.epoch_shuffle is not None:
            other_optionals.append(f'Shuffle in-Epoch: {self.epoch_shuffle}')
        if self.learning_rate is not None:
            other_optionals.append(f'Learning Rate = {self.learning_rate}')

        res += ' | '.join(other_optionals)

        res += '\nHyperparameters:\n'
        res += ' | '.join([key + ' = ' + value for key, value in self.hyperparameters.items()])

        return res

    def __copy__(self):
        return self.__class__(**deepcopy(self._slots), **deepcopy(self.hyperparameters))

    def __eq__(self, other):
        if isinstance(other, SupervisedTrainConditions):
            x_slots, y_slots = self._slots, other._slots

            optimizer, loss = True, True
            if not isinstance(x_slots['optimizer'], str):
                optimizer = x_slots['optimizer'].__repr__() == y_slots['optimizer'].__repr__()
                del x_slots['optimizer'], y_slots['optimizer']
            if not isinstance(x_slots['loss'], str):
                loss = x_slots['loss'].__repr__() == y_slots['loss'].__repr__()
                del x_slots['loss'], y_slots['loss']

            primitive_values = all([x_slots[label] == y_slots[label] for label in x_slots.keys()])

            return primitive_values and optimizer and loss
