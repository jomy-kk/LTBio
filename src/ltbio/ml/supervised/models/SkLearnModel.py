# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SkLearnModel
# Description: Class SkLearnModel, that encapsulates the API of SKLearn supervised models.

# Contributors: João Saraiva and code from https://scikit-learn.org/
# Created: 05/06/2022
# Last Updated: 25/06/2022

# ===================================

from warnings import warn

from matplotlib import pyplot as plt
from numpy import arange, argsort, array
from sklearn.base import is_classifier, is_regressor

import ltbio.ml.supervised.models.SupervisedModel as _SupervisedModel
from ltbio.ml.supervised.results import PredictionResults
from ltbio.ml.supervised.results import SupervisedTrainResults


class SkLearnModel(_SupervisedModel.SupervisedModel):

    def __init__(self, design, name: str = None):
        # Check design
        if not (is_classifier(design) or is_regressor(design)):
            raise ValueError("The design given is not a valid SkLearn classifier or regressor.")

        super().__init__(design, name)

    def __set_parameter_from_condition(self, parameter_label:str, conditions_label:str, value):
        if parameter_label in self.__required_parameters:
            if value is not None:
                self._SupervisedModel__design.set_params(**{parameter_label: value})
            else:
                warn(f"Omitted train condition '{conditions_label}' = {self._SupervisedModel__design.get_params()[parameter_label]} being used.")
        else:
            if value is not None:
                warn(f"Train condition '{conditions_label}' given is not required for this model. Ignoring it.")
            else:
                pass

    def train(self, dataset, conditions):
        # Call super for version control
        super().train(dataset, conditions)

        # Set whichever model hyperparameters were defined
        self._SupervisedModel__design.set_params(**conditions.hyperparameters)

        # Map some train conditions to model parameters
        self.__required_parameters = self._SupervisedModel__design.get_params().keys()
        self.__set_parameter_from_condition('max_iter', 'epochs', conditions.epochs)
        self.__set_parameter_from_condition('loss', 'loss', conditions.loss)
        self.__set_parameter_from_condition('tol', 'stop_at_deltaloss', conditions.stop_at_deltaloss)
        self.__set_parameter_from_condition('n_iter_no_change', 'patience', conditions.patience)
        self.__set_parameter_from_condition('solver', 'optimizer', conditions.optimizer)
        self.__set_parameter_from_condition('?', 'shuffle', conditions.shuffle)
        self.__set_parameter_from_condition('shuffle', 'epoch_shuffle', conditions.epoch_shuffle)
        self.__set_parameter_from_condition('batch_size', 'batch_size', conditions.batch_size)
        self.__set_parameter_from_condition('learning_rate_init', 'learning_rate', conditions.learning_rate)
        self.__set_parameter_from_condition('validation_fraction', 'validation_ratio', conditions.validation_ratio)
        self.__set_parameter_from_condition('?', 'test_ratio', conditions.test_ratio)
        self.__set_parameter_from_condition('?', 'train_ratio', conditions.train_ratio)
        self.__set_parameter_from_condition('?', 'test_size', conditions.test_size)
        self.__set_parameter_from_condition('?', 'train_size', conditions.train_size)

        # Fits the model
        self._SupervisedModel__design.fit(dataset.all_objects, dataset.all_targets)

        # Update version
        self._SupervisedModel__update_current_version_state(self)#, epoch_concluded=int(self._SupervisedModel__design.n_iter_))

        # Create results object
        return SupervisedTrainResults(self._SupervisedModel__design.loss_, None, None)

    def test(self, dataset, evaluation_metrics = None, version = None):
        # Call super for version control
        super().test(dataset, evaluation_metrics, version)
        # Make predictions about the objects
        predictions = self._SupervisedModel__design.predict(dataset.all_objects)
        # Create results object
        return PredictionResults(self._SupervisedModel__design.loss_, dataset, predictions, evaluation_metrics)

    @property
    def trained_parameters(self):
        try:
            return self._SupervisedModel__design.coef_, self._SupervisedModel__design.intercepts_
        except:
            raise ReferenceError("Unfortunately cannot find the trained parameters, but the design internal state is functional.")

    @property
    def non_trainable_parameters(self):
        return self._SupervisedModel__design.get_params()

    def _SupervisedModel__set_state(self, state):
        self._SupervisedModel__design.__setstate__(state)

    def _SupervisedModel__get_state(self):
        return self._SupervisedModel__design.__getstate__()



    def __plot_timeseries_importance(self, show:bool=True, save_to:str=None):
        """
        All code was adapted from https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py
        """
        # TImeseries importance
        timeseries_labels = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j')
        feature_importance = self._SupervisedModel__design.feature_importances_
        sorted_idx = argsort(feature_importance)
        pos = arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(6, 6))
        plt.subplot(1, 1, 1)
        plt.barh(pos, feature_importance[sorted_idx], align="center")
        plt.yticks(pos, array(timeseries_labels)[sorted_idx])
        plt.title("Timeseries Importance (MDI)")

        fig.tight_layout()
        if save_to is not None:
            fig.savefig(save_to)
        if show:
            plt.show()
            print("Timeseries Importance plot was shown.")
        else:
            plt.close()

    def __plot_timeseries_permutation_importance(self, show:bool=True, save_to:str=None):
        from sklearn.inspection import permutation_importance
        result = permutation_importance(self._SupervisedModel__design, self.__last_results.object, self.__last_results.target,
                                        n_repeats=10, random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()
        timeseries_labels = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j')
        fig = plt.figure(figsize=(6, 6))
        plt.subplot(1, 1, 1)
        plt.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=array(timeseries_labels)[sorted_idx],
        )
        plt.title("Timeseries Permutation Importance (test set)")

        fig.tight_layout()
        if save_to is not None:
            fig.savefig(save_to)
        if show:
            plt.show()
            print("Timeseries Permutation Importance plot was shown.")
        else:
            plt.close()
