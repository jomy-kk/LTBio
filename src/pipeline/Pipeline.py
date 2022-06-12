# -*- encoding: utf-8 -*-

# ===================================

# IT - PreEpiSeizures

# Package: pipeline
# File: Pipeline
# Description: Class representing a pipeline of steps that works on biosignals.

# Contributors: João Saraiva
# Created: 11/06/2022

# ===================================

from typing import List, Collection

from src.pipeline.Packet import Packet
from src.biosignals.Biosignal import Biosignal
from src.pipeline.PipelineUnit import PipelineUnit


class Pipeline():

    # Attributes
    __steps: List[PipelineUnit]
    __current_step: int
    __biosignals: Collection[Biosignal]
    __current_packet: Packet

    def __init__(self, name:str=None):
        self.name = name
        self.__current_step = 0
        self.__steps = []

    @property
    def current_step(self) -> int:
         if self.__current_step > 0:
             return self.__current_step
         else:
             raise AttributeError('Pipeline has not started yet.')

    def __len__(self):
        return len(self.__steps)

    def add(self, unit:PipelineUnit):
        if self.__check_completeness(unit):
            self.__steps.append(unit)
        else:
            raise ValueError('The input required by the unit trying to add does not match the output of the previous unit.')


    def next(self):
        if self.__current_step == 0:  # if starting
            self.__create_first_packet()

        # Do next step
        self.__current_packet = self.__steps[self.__current_step]._apply(self.__current_packet)
        self.__current_step += 1

        if self.__current_step == len(self) - 1:  # if ending
            return self.__current_packet

    def applyAll(self):
        N_STEPS = len(self)
        while self.__current_step < N_STEPS:
            self.next()

    def __create_first_packet(self):
        pass

    def __check_completeness(self, new_unit:PipelineUnit) -> bool:
        return True  # TODO




