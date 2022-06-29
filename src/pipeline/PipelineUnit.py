###################################

# IT - PreEpiSeizures

# Package: pipeline
# File: PipelineUnit
# Description: Abstract class representing units of a processing pipeline.

# Contributors: João Saraiva
# Created: 02/06/2022

###################################

from abc import ABC, abstractmethod
from typing import Collection, Dict
from inspect import signature

import src.pipeline
from src.features.Features import Features
from src.pipeline.Packet import Packet
from src.biosignals.Timeseries import Timeseries

class PipelineUnit(ABC):
    """
    A Pipeline Unit is any agent that can act (process and make changes) to a collection of inputs, usually Timeseries.
    It is a concrete 'command' in the Command Design Pattern, where '_apply' serves as the 'execute' method.

    The inputs to apply a Pipeline Unit must be given within a Packet.
    When the Pipeline Unit is done, it should also create and populate Packet with its outputs, so the next Pipeline Unit can make use of them.

    The method 'apply' implements the concrete behaviour of a Pipeline Unit.
    So, that should be implemented in every subclass.
    To map the parameters' names of 'apply' to the labels inside any arriving Packet, PIPELINE_INPUT_LABELS and PIPELINE_OUTPUT_LABELS should be filled.

    PIPELINE_INPUT_LABELS maps every label of a needed input inside a Packet to the name of the corresponding apply' parameter.
    PIPELINE_OUTPUT_LABELS maps every output name of apply to a label to be saved inside a Packet.
    """

    PIPELINE_INPUT_LABELS: Dict[str, str]  # { apply parameter : packet label }
    PIPELINE_OUTPUT_LABELS: Dict[str, str]  # { apply output : packet label }
    ART_PATH = 'resources/pipeline_media/nd.png'

    def __init__(self, name:str):
        self.name = name

    @abstractmethod
    def apply(self, **kwargs):
        pass

    def _apply(self, packet:Packet, APPLY_FOR_ALL=False) -> Packet:
        # Flag APPLY_FOR_ALL
        # True = To apply once per each element in a Collection (apply-for-all)
        # False = To apply to all at elements at once (apply-to-all)

        # Get what this unit needs from the Packet
        what_this_unit_needs = tuple(signature(self.apply).parameters.values())

        # Unpack from the Packet what is needed
        input = {}
        for parameter in what_this_unit_needs:
            parameter_name = parameter.name
            parameter_type = parameter.annotation
            packet_label = self.PIPELINE_INPUT_LABELS[parameter_name]  # Map to the label in Packet

            #if isinstance(packet[packet_label], parameter_type):  # Check the types are correct
            #    input[parameter_name] = packet[packet_label]
            #else:
                # If there's a collection instead of a single element for this label, then apply once for each collection's element
            if isinstance(APPLY_FOR_ALL, bool) and APPLY_FOR_ALL and isinstance(packet[packet_label], Collection) and not isinstance(packet[packet_label], Timeseries):
                APPLY_FOR_ALL = (packet_label, parameter_name)
            else:
                input[parameter_name] = packet[packet_label]

        # Apply the unit processing to all at once
        if not APPLY_FOR_ALL:
            output = self.apply(**input)

        # Apply the unit processing for all, once at a time
        else:
            if isinstance(packet[APPLY_FOR_ALL[0]], dict):
                output = {}
                for label in packet[APPLY_FOR_ALL[0]]:
                    input[APPLY_FOR_ALL[1]] = packet[APPLY_FOR_ALL[0]][label]
                    output[label] = self.apply(**input)
            else:
                output = []
                for element in packet[APPLY_FOR_ALL[0]]:
                    input[APPLY_FOR_ALL[1]] = element
                    output.append(self.apply(**input))

        # Prepare load for Packet
        load = packet._to_dict()  # start with the contents already in the previous packet
        packet_label = tuple(self.PIPELINE_OUTPUT_LABELS.values())[0]
        load[packet_label] = output  # replace or add

        return Packet(**load)

    def __str__(self):
        res = self.__class__.__name__
        res += ' ' + self.name if self.name is not None else ''
        return res

    def __rshift__(self, other):
        '''
        Defines the >> operator, the fastest shortcut to create a Pipeline
        '''
        if isinstance(other, PipelineUnit):  # concatenate self.Unit + other.Unit = res.Pipeline
            res = src.pipeline.Pipeline.Pipeline()
            res.add(self)
            res.add(other)
            return res
        elif isinstance(other, src.pipeline.Pipeline.Pipeline):  # concatenate another self.Unit + other.Pipeline = res.Pipeline
            pass
        else:
            raise TypeError(f'Cannot join a PipelineUnit with a {type(other)}.')

