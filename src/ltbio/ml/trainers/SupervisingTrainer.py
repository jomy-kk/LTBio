# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SupervisingTrainer
# Description: Class SupervisingTrainer, a type of PipelineUnit that trains supervised machine learning models.

# Contributors: João Saraiva
# Created: 04/06/2022
# Last Updated: 07/07/2022

# ===================================

from typing import Collection

from numpy import array
from sklearn.model_selection import train_test_split

from ltbio.biosignals import Timeseries
from ltbio.ml.models import SupervisedModel
from ltbio.ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions
from ltbio.ml.trainers.SupervisedTrainReport import SupervisedTrainReport
from ltbio.pipeline.PipelineUnit import SinglePipelineUnit


class SupervisingTrainer(SinglePipelineUnit):

    PIPELINE_INPUT_LABELS = {'object': 'timeseries', 'target': 'target'}
    PIPELINE_OUTPUT_LABELS = {'results': 'results'}
    ART_PATH = 'resources/pipeline_media/ml.png'

    def __init__(self, model:SupervisedModel, train_conditions:Collection[SupervisedTrainConditions], name:str=None):
        super().__init__(name)
        self.__model = model
        self.train_conditions = train_conditions

        if len(train_conditions) == 0:
            raise AttributeError("Give at least one SupervisedTrainConditions to 'train_conditions'.")


    def apply(self, object:Collection[Timeseries], target:Timeseries):
        self.reporter = SupervisedTrainReport()
        self.reporter.print_successful_instantiation()
        self.reporter.print_model_description(self.__model, **self.__model.non_trainable_parameters)

        # Convert object and target to arrays
        X = array([ts.to_array() for ts in (object.values() if isinstance(object, dict) else object)]).T # Assertion that every Timeseries only contains one Segment is guaranteed by 'to_array' conversion.
        y = target.to_array()

        results = []
        for i, set_of_conditions in enumerate(self.train_conditions):
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

            # Train the model
            self.__model.train(X_train, y_train)

            # Test the model
            self.__model.test(X_test, y_test)

            # Name each test result with what version number and differences in train conditions.
            test_results.name = f"[V{self.__model.current_version}: " + ', '.join([f'{key} = {value}' for key, value in differences_in_conditions[i].items()]) + ']'

        self.reporter.print_end_of_trains(len(self.train_conditions))
        self.reporter.output('Report.pdf')

        return results
