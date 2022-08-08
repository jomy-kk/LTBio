# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SupervisedTrainReport
# Description: Class SupervisedTrainReport, produces a PDF report for a SupervisingTrainer.

# Contributors: João Saraiva
# Created: 06/05/2022
# Last Updated: 06/08/2022

# ===================================
import os

from matplotlib import pyplot as plt

from ltbio.ml.metrics import ValueMetric
from ltbio.ml.supervised.models import SupervisedModel
from ltbio.ml.supervised import SupervisedTrainConditions
from ltbio.ml.supervised.results import PredictionResults
from ltbio.ml.supervised.results import SupervisedTrainResults
from ltbio.pipeline.reports import Reporter


class SupervisingTrainerReporter(Reporter):

    def __init__(self, writer=None):
        super().__init__(writer)
        self.model: SupervisedModel = None
        self.model_descriptors: dict = {}
        self.training_conditions: list[SupervisedTrainConditions] = []
        self.train_results: list[SupervisedTrainResults] = []
        self.test_results: list[PredictionResults] = []

    def body(self):
        # Model Description
        self.begin_subsection('MODEL DESCRIPTION')
        self.add_text_block('Name: {0}'.format(self.model.name))
        self.add_text_block('Design class: {0}'.format(type(self.model.design).__name__))
        self.add_text_block("\t".join(['{0}={1}'.format(label, self.model_descriptors[label]) for label in self.model_descriptors]))

        # Experiments
        for i, (conditions, train_results, test_results) in enumerate(zip(self.training_conditions, self.train_results, self.test_results)):
            self.begin_subsection("EXPERIMENT {}".format(str(i+1)))
            # Conditions
            self.add_text_block(str(conditions))
            # Avg. Losses
            if train_results.train_losses is not None:
                self.add_text_block("Train Loss: {:.5f}".format(train_results.train_losses[-1]))
            if train_results.validation_losses is not None:
                self.add_text_block("Validation Loss: {:.5f}".format(train_results.validation_losses[-1]))
            if test_results.loss is not None:
                self.add_text_block("Avg. Test Loss: {:.5f}".format(test_results.loss))
            # Losses plot
            self.__plot_train_and_test_loss(train_results.train_losses, train_results.validation_losses, './losses.png')
            self.add_image_fullwidth('./losses.png')
            os.remove('./losses.png')
            # Other metrics
            grid_filepaths: list[str] = []
            for metric in test_results.metrics:
                if isinstance(metric, ValueMetric):
                    self.add_text_block(str(metric))
                else:  #elif isinstance(metric, PlotMetric):
                    grid_filepaths.append(metric.filepath)
            self.add_image_grid(tuple(grid_filepaths))

    def declare_model_description(self, model: SupervisedModel, **descriptors):
        self.model = model
        self.model_descriptors = descriptors

    def declare_training_session(self, train_conditions:SupervisedTrainConditions, train_results: SupervisedTrainResults, test_results: PredictionResults):
        self.training_conditions.append(train_conditions)
        self.train_results.append(train_results)
        self.test_results.append(test_results)

    def print_loss_plot(self, image_path: str):
        self.writer.__break_line()
        self.writer.image(image_path, w=self.writer.FULL_PIC_WIDTH, h=self.writer.FULL_PIC_HEIGHT)

    def print_small_plots(self, image_paths: str):
        """
        Prints a grid of n lines and 2 columns.
        """
        self.writer.__break_line()
        for i, image_path in enumerate(image_paths):
            if i % 2 == 0:
                self.writer.image(image_path, w=self.writer.SMALL_PIC_WIDTH, h=self.writer.SMALL_PIC_HEIGHT)
            else:
                self.writer.image(image_path, w=self.writer.SMALL_PIC_WIDTH, h=self.writer.SMALL_PIC_HEIGHT,
                           x=self.writer.x + self.writer.SMALL_PIC_WIDTH + self.writer.SMALL_PIC_SEP, y=self.writer.y - self.writer.SMALL_PIC_HEIGHT)

    def __plot_train_and_test_loss(self, train_losses: list[float], validation_losses: list[float], save_to: str):
        fig = plt.figure(figsize=(10, 5))
        plt.subplot(1, 1, 1)
        plt.title("Loss over the Epochs")
        plt.plot(range(1, len(train_losses)+1), train_losses, "b-", label="Train Loss")
        if validation_losses is not None:
            plt.plot(range(1, len(validation_losses)+1), validation_losses, "r-", label="Train Loss")
        plt.legend(loc="upper right")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        fig.tight_layout()
        fig.savefig(save_to)
        plt.close()
