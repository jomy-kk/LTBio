###################################

# IT - PreEpiSeizures

# Package: ml
# File: ModelTrainer
# Description: Class to train machine learning models

# Contributors: João Saraiva
# Created: 04/06/2022

###################################

from typing import Tuple

from sklearn.model_selection import train_test_split

from src.ml.models.SupervisedModel import SupervisedModel
from src.ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions
from src.pipeline.PipelineUnit import PipelineUnit
from src.biosignals.Timeseries import Timeseries

from numpy import array




class SupervisingTrainer(PipelineUnit):

    def __init__(self, model:SupervisedModel, train_conditions:Tuple[SupervisedTrainConditions], name:str=None):
        super().__init__(name)
        self.__model = model
        self.train_conditions = train_conditions

        if len(train_conditions) == 0:
            raise AttributeError("Give at least one SupervisedTrainConditions to 'train_conditions'.")


    def apply(self, object:Tuple[Timeseries], target:Timeseries) -> None:
        # Convert to object and target to arrays
        X = array([ts.to_array() for ts in object]).T # Assertion that every Timeseries only contains one Segment is guaranteed by 'to_array' conversion.
        y = target.to_array()

        for set_of_conditions in self.train_conditions:
            # Prepare train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=set_of_conditions['test_size'],
                                                                #train_size=set_of_conditions['train_size'],
                                                                shuffle=set_of_conditions['shuffle'],
                                                                )
            # Do setup, if any
            self.__model.setup(set_of_conditions)

            # Train the model
            self.__model.train(X_train, y_train)

            # Test the model
            self.__model.test(X_test, y_test)

            # Produce report
