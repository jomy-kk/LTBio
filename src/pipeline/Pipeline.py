# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: pipeline
# Module: Pipeline
# Description: Class Pipeline, representing a pipeline of steps to process Biosignals.

# Contributors: João Saraiva
# Created: 11/06/2022
# Last Updated: 07/07/2022

# ===================================

from inspect import signature
from typing import List, Collection

from biosignals.modalities.Biosignal import Biosignal
from pipeline.Input import Input
from pipeline.Packet import Packet
from pipeline.PipelineUnit import PipelineUnit, SinglePipelineUnit, PipelineUnitsUnion


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

    @property
    def current_packet(self) -> Packet:
        return self.__current_packet

    def __len__(self):
        return len(self.__steps)

    def __str__(self):
        res = 'Pipeline' + (' ' + self.name if self.name is not None else '')
        for i in range(len(self)):
            res += f'\nStep {i+1}: ' + str(self.__steps[i])
        return res

    def add(self, unit:PipelineUnit):
        #if len(self) > 0:
        #    self.__check_completeness(unit)
        self.__steps.append(unit)

    def __rshift__(self, other):
        '''
        Defines the >> operator, the fastest shortcut to create a Pipeline
        '''
        if isinstance(other, PipelineUnit):  # concatenate self.Pipeline + other.Unit = res.Pipeline
            self.add(other)
            return self
        elif isinstance(other, Pipeline):  # concatenate another self.Pipeline + other.Pipeline = res.Pipeline
            pass
        else:
            raise TypeError(f'Cannot join a PipelineUnit with a {type(other)}.')

    def load(self, biosignals: Biosignal | Collection[Biosignal]):
        if isinstance(biosignals, Biosignal):
            self.__biosignals = (biosignals, )
        else:
            self.__biosignals = biosignals

    def next(self):
        if self.__current_step == 0:  # if starting
            self.__create_first_packet()

        # Do next step
        self.__current_packet = self.__steps[self.__current_step]._apply(self.__current_packet)
        self.__current_step += 1

        return self.__current_packet

    def applyAll(self, biosignals: Biosignal | Collection[Biosignal]):
        self.load(biosignals)
        N_STEPS = len(self)
        while self.__current_step < N_STEPS:
            self.next()
        return self.__unpack_last_packet()

    def __create_first_packet(self):
        assert self.__biosignals is not None  # Check if Biosignals were loaded
        all_timeseries = {}
        for biosignal in self.__biosignals:
            timeseries = biosignal._to_dict()
            assert tuple(timeseries.keys()) not in all_timeseries  # Ensure there are no repeated keys
            all_timeseries.update(timeseries)

        self.__current_packet = Packet(timeseries=all_timeseries)

    def __unpack_last_packet(self) -> Biosignal | Collection[Biosignal]:
        return self.__current_packet._to_dict()

    def __check_completeness(self, new_unit:PipelineUnit):
        # Know what will be available up to this point
        load_that_will_be_available = {}
        for unit in self.__steps:
            # Get output label and type
            if isinstance(unit, SinglePipelineUnit):
                output_label = tuple(unit.PIPELINE_OUTPUT_LABELS.values())[0]
                output_type = signature(unit.apply).return_annotation
                load_that_will_be_available[output_label] = output_type  # If it's the case, it replaces type of same labels, as it should
            elif isinstance(unit, PipelineUnitsUnion):
                output_labels = tuple(unit.PIPELINE_OUTPUT_LABELS.values())


        # Know what the new unit needs
        if isinstance(new_unit, SinglePipelineUnit):
            new_unit_parameters = tuple(signature(new_unit.apply).parameters.values())
        elif isinstance(new_unit, PipelineUnitsUnion):
            new_unit_parameters = new_unit.all_input_parameters

        # Check if it matches
        for parameter in new_unit_parameters:
            parameter_name = parameter.name
            parameter_type = parameter.annotation
            input_label = new_unit.PIPELINE_INPUT_LABELS[parameter_name]  # Map to the label in Packet

            if input_label in load_that_will_be_available:
                if isinstance(new_unit, SinglePipelineUnit):  # TODO: Currently, we're jumpting verification of Union input and output types
                    if isinstance(parameter_type, type(load_that_will_be_available[input_label])):
                        continue
                    else:
                        raise AssertionError('Input type, {}, of the new unit does not match the output type, {}, of the last unit.'.format(
                                parameter_type, load_that_will_be_available[input_label]))
            else:
                raise AssertionError('{} input label of the new unit does not match to any output label of the last unit.'.format(
                        input_label))

    def plot_diagram(self, show:bool=True, save_to:str=None):
        from diagrams import Diagram
        from diagrams.custom import Custom

        with Diagram(name="Pipeline" + ((" " + self.name) if self.name is not None else ""), direction='LR', show=show, filename=save_to):
            blocks = []
            input_unit = False
            for unit in self.__steps:
                blocks.append(Custom(str(unit), unit.ART_PATH))
                if len(blocks) > 1:
                    if isinstance(unit, Input):
                        input_unit = True
                    elif input_unit:
                        blocks[-3] >> blocks[-1]
                        blocks[-2] >> blocks[-1]
                    else:
                        blocks[-2] >> blocks[-1]

