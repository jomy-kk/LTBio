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

from biosignals.timeseries.Timeseries import Timeseries
from ml.models.SupervisedModel import SupervisedModel
from ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions
from ml.trainers.SupervisedTrainReport import SupervisedTrainReport
from pipeline.PipelineUnit import SinglePipelineUnit


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
            # Prepare train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=set_of_conditions['test_size'],
                                                                #train_size=set_of_conditions['train_size'],
                                                                shuffle=set_of_conditions['shuffle'],
                                                                )
            self.reporter.print_start_of_train(i+1, len(self.train_conditions), set_of_conditions)

            # Do setup, if any
            self.__model.setup(set_of_conditions)

            # Train the model
            self.__model.train(X_train, y_train)

            # Test the model
            self.__model.test(X_test, y_test)

            # Produce report
            #self.__model.report(self.reporter, show=False, save_to='resources/reports_tests/my_test'+str(i+1))
            result = self.__model.report(self.reporter, show=False, save_to='resources/reports_tests/my_test'+str(i+1))
            results.append(result)

        self.reporter.print_end_of_trains(len(self.train_conditions))
        self.reporter.output('Report.pdf')

        return results
