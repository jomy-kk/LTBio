# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SupervisingTrainer
# Description: Class SupervisingTrainer, a type of PipelineUnit that trains supervised machine learning models.

# Contributors: João Saraiva
# Created: 04/06/2022
# Last Updated: 07/08/2022

# ===================================

from typing import Collection

from ltbio.biosignals import Timeseries
from ltbio.ml.datasets import SegmentToSegmentDataset
from ltbio.ml.datasets.BiosignalDataset import BiosignalDataset
from ltbio.ml.supervised.models import SupervisedModel as _SupervisedModel
from ltbio.ml.supervised.SupervisedTrainConditions import SupervisedTrainConditions
from ltbio.ml.supervised.SupervisingTrainerReporter import SupervisingTrainerReporter
from ltbio.pipeline.PipelineUnit import SinglePipelineUnit


class SupervisingTrainer(SinglePipelineUnit):
    PIPELINE_INPUT_LABELS = {'dataset': ('timeseries', 'target')}
    PIPELINE_OUTPUT_LABELS = {'results': 'results'}
    ART_PATH = 'resources/pipeline_media/ml.png'

    def __init__(self, model: _SupervisedModel.SupervisedModel,
                 train_conditions: Collection[SupervisedTrainConditions],
                 evaluation_metrics: Collection = None,
                 name: str = None, save_report_to: str = None):

        super().__init__(name)

        if not isinstance(model, _SupervisedModel.SupervisedModel):
            raise TypeError("Parameter 'model' must be an instance of SupervisedModel.")
        self.__model = model

        if len(train_conditions) == 0:
            raise AttributeError("Give at least one SupervisedTrainConditions to 'train_conditions'.")
        if not isinstance(train_conditions, (tuple, list, set)) or not all(
                isinstance(x, SupervisedTrainConditions) for x in train_conditions):
            raise TypeError("Parameter 'train_conditions' must be a collection of SupervisedTrainConditions objects.")
        self.train_conditions = train_conditions

        self.evaluation_metrics = evaluation_metrics
        self.save_report_to = save_report_to

        self.reporter = SupervisingTrainerReporter()
        self.reporter.declare_model_description(self.__model, **self.__model.non_trainable_parameters)

    def apply(self, dataset: BiosignalDataset, test_dataset: BiosignalDataset = None):

        if not isinstance(dataset, BiosignalDataset):
            raise TypeError(f"A BiosignalDataset is expected. Instead a {type(dataset)} was given.")

        # Infer what is different between all sets of the train conditions
        differences_in_conditions = SupervisedTrainConditions.differences_between(self.train_conditions)

        for i, set_of_conditions in enumerate(self.train_conditions):
            if test_dataset is None:
                # Train subdatset size
                if set_of_conditions.train_size != None:
                    train_subsize = set_of_conditions.train_size
                elif set_of_conditions.train_ratio != None:
                    train_subsize = int(set_of_conditions.train_ratio * len(dataset))
                else:
                    train_subsize = None
                # Test subdatset size
                if set_of_conditions.test_size != None:
                    test_subsize = set_of_conditions.test_size
                elif set_of_conditions.test_ratio != None:
                    test_subsize = int(set_of_conditions.test_ratio * len(dataset))
                else:
                    test_subsize = None
                # By inference
                if train_subsize is None:
                    train_subsize = len(dataset) - test_subsize
                if test_subsize is None:
                    test_subsize = len(dataset) - train_subsize
                # SupervisedTrainConditions garantees that at least one of these four conditions is defined to make these computations.

                # Prepare the train and test datasets
                train_dataset, test_dataset = dataset.split(train_subsize, test_subsize, set_of_conditions.shuffle is True)
            else:
                train_dataset = dataset

            # Train the model
            train_results = self.__model.train(train_dataset, set_of_conditions)

            # Test the model
            test_results = self.__model.test(test_dataset, self.evaluation_metrics)

            # Name each test result with what version number and differences in train conditions.
            test_results.name = f"[V{self.__model.current_version}: " + ', '.join([f'{key} = {value}' for key, value in differences_in_conditions[i].items()]) + ']'

            # Report results
            self.reporter.declare_training_session(set_of_conditions, train_results, test_results)

        if self.save_report_to is not None:
            self.reporter.output_report('Supervising Trainer Report', self.save_report_to)

        return self.__model.best_version_results

    def _transform_input(self, object:tuple[Timeseries], target:tuple[Timeseries]) -> BiosignalDataset:
        if len(target) == 1 and target[0].is_contiguous:
            # dataset = SegmentToValueDataset()
            pass  # TODO
        else:
            dataset = SegmentToSegmentDataset(object=object, target=target)

        return dataset
