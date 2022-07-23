# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: ml
# Module: SupervisedTrainResults
# Description: Class SupervisedTrainResults, that holds of supervised trains performed by a SupervisingTrainer.

# Contributors: João Saraiva
# Created: 04/05/2022

# ===================================

from typing import Dict

from numpy import shape


class SupervisedTrainResults():

    def __init__(self, object, target, predicted):
        self.predicted = predicted
        self.target = target
        self.object = object

        self.__n_timeseries = shape(object)[0]
        self.__n_samples = shape(object)[1]

        self.__metrics:Dict[str:float] = {}

    def __getitem__(self, metric):
        return self.__metrics[metric]

    def __setitem__(self, metric, value):
        self.__metrics[metric] = value
