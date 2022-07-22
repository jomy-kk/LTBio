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

from matplotlib import pyplot as plt
from numpy import zeros, arange, argsort, array
from sklearn.metrics import mean_squared_error

from ml.models.SupervisedModel import SupervisedModel
from ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions
from ml.trainers.SupervisedTrainResults import SupervisedTrainResults


class SkLearnModel(SupervisedModel):

    def __init__(self, design, name: str = None, version: int = None):
        super().__init__(design, name, version)

    def setup(self, train_conditions:SupervisedTrainConditions, **kwargs):
        params = {key: value for key, value in train_conditions.parameters.items() if key not in ('train_size', 'test_size', 'shuffle')}
        self.design.set_params(**params)

    def train(self, object, target):
        self.design.fit(object, target)

    def test(self, object, target=None):
        if target is None:
            return self.design.predict(object)
        else:
            y_pred = self.design.predict(object)
            self.__last_results = SupervisedTrainResults(object, target, y_pred)
            return y_pred

    def report(self, reporter, show:bool=True, save_to:str=None):
        mse = mean_squared_error(self.__last_results.target, self.__last_results.predicted)
        reporter.print_textual_results(mse=mse)
        if save_to is not None:
            file_names = (save_to + '_loss.png', save_to + '_importance.png', save_to + '_permutation.png')
            self.__plot_train_and_test_loss(show=show, save_to=file_names[0])
            self.__plot_timeseries_importance(show=show, save_to=file_names[1])
            self.__plot_timeseries_permutation_importance(show=show, save_to=file_names[2])

            reporter.print_loss_plot(file_names[0])
            reporter.print_small_plots(file_names[1:])

        return mse


    @property
    def trained_parameters(self):
        return None  # TODO

    @property
    def non_trainable_parameters(self):
        return self.design.get_params()


    def __plot_train_and_test_loss(self, show:bool=True, save_to:str=None):
        """
        All code was adapted from https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py
        """
        test_score = zeros((self.design.get_params()["n_estimators"],))
        for i, y_pred in enumerate(self.design.staged_predict(self.__last_results.object)):
            test_score[i] = self.design.loss_(self.__last_results.target, y_pred)
        fig = plt.figure(figsize=(8, 4))
        plt.subplot(1, 1, 1)
        plt.title("Train/Test Loss")
        plt.plot(
            arange(self.design.get_params()["n_estimators"]) + 1, self.design.train_score_, "b-", label="Training Set Deviance",
        )
        plt.plot(
            arange(self.design.get_params()["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
        )
        plt.legend(loc="upper right")
        plt.xlabel("Boosting Iterations")
        plt.ylabel("Deviance")

        fig.tight_layout()
        if save_to is not None:
            fig.savefig(save_to)
        if show:
            plt.show()
            print("Train/Test Loss plot was shown.")
        else:
            plt.close()

    def __plot_timeseries_importance(self, show:bool=True, save_to:str=None):
        """
        All code was adapted from https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py
        """
        # TImeseries importance
        timeseries_labels = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j')
        feature_importance = self.design.feature_importances_
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
        result = permutation_importance(self.design, self.__last_results.object, self.__last_results.target,
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
