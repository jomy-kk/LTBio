###################################

# IT - PreEpiSeizures

# Package: pipeline
# File: PipelineUnit
# Description: Abstract class representing units of a processing pipeline.

# Contributors: João Saraiva
# Created: 02/06/2022

###################################

from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import Collection, Dict, Iterable, List
from inspect import signature

from biosignals.Timeseries import Timeseries
from src.pipeline.Packet import Packet


@unique
class Apply(Enum):
    SEPARATE = 0
    TOGETHER = 1


class PipelineUnit(ABC):
    """
    Pipeline Units are the building blocks of Pipelines.
    Following the Composite design pattern, a PipelineUnit is the abstract 'Component', so that Pipeline can deal with
    SingleUnit and Union in the same way.

    Subclasses
    ------------
    - SingleUnit: A single pipeline unit, that actually acts on Timeseries. It's the 'Leaf' in the design pattern.
    - Union: A collection of single units where the Pipeline branches to each of them. It's the 'Composite' in the
    design pattern.

    Abstract Method '_apply'
    ------------
    Acts as the 'operation' method in the design pattern, and it's implemented in each subclass.
    It receives a Packet with the necessary inputs to apply the unit and returns a Packet with the relevant outputs.
    """

    def __init__(self, name:str):
        self.name = name

    @abstractmethod
    def _apply(self, packet:Packet, how_to:Apply) -> Packet:
        """
        Receives a Packet with the necessary inputs to apply the unit and returns a Packet with the relevant outputs.
        It also receives a way of applying, 'how_to', in case a collection of Timeseries come with the packet.
        Acts as the 'operation' method in the composite design pattern.
        Acts as the 'template' method in the template method design pattern.
        """
        pass


class SinglePipelineUnit(PipelineUnit, ABC):
    """
    A Single Pipeline Unit is any agent that can act (use, process or make changes) to a collection (usually of Timeseries).
    Following the Command design pattern, a SingleUnit is the abstract 'Command', so that Pipeline can execute various
    kinds of processing by calling the 'apply' method of each concrete unit.

    Subclasses
    ------------
    E.g. Filter, Segmenter, FeatureExtractor, FeatureSelector, SupervisingTrainer, DecisionMaker
    Any subclass that implements 'apply'.

    Abstract Method 'apply'
    ------------
    Every subclass must define 'apply' and implement a concrete behaviour.
    To map the parameters' names of 'apply' to the labels inside any arriving Packet, PIPELINE_INPUT_LABELS should be
    defined. To map the outputs to the labels of the resulting Packet, PIPELINE_OUTPUT_LABELS should be defined.

    Labels
    ------------
    PIPELINE_INPUT_LABELS
    Maps every label of a needed input inside a Packet to the name of the corresponding 'apply' parameter.
    PIPELINE_OUTPUT_LABELS
    Maps every output name of 'apply' to a label to be saved inside a Packet.
    """

    # ===============================================================
    # Subclass-specific -- Define:

    PIPELINE_INPUT_LABELS: Dict[str, str]  # { apply parameter : packet label }
    PIPELINE_OUTPUT_LABELS: Dict[str, str]  # { apply output : packet label }

    def __init__(self, name:str=None):
        super(SinglePipelineUnit, self).__init__(name)

    @abstractmethod
    def apply(self, **kwargs):
        pass

    # ===============================================================
    # Framework below -- Do not alter:

    ART_PATH = 'resources/pipeline_media/nd.png'

    def __str__(self):
        res = self.__class__.__name__
        res += ' ' + self.name if self.name is not None else ''
        return res

    def __rshift__(self, other):
        '''
        Defines the >> operator, the fastest shortcut to create a Pipeline
        '''
        from src.pipeline.Pipeline import Pipeline
        if isinstance(other, PipelineUnit):  # concatenate self.Unit + other.Unit = res.Pipeline
            res = Pipeline()
            res.add(self)
            res.add(other)
            return res
        elif isinstance(other, Pipeline):  # concatenate another self.Unit + other.Pipeline = res.Pipeline
            pass
        else:
            raise TypeError(f'Cannot join a PipelineUnit with a {type(other)}.')


    def __unpack_all_timeseries(self, packet:Packet) -> dict:
        """
        Receives a Packet and returns a dictionary with the inputs needed to apply this unit.
        """

        # Get what this unit needs from the Packet
        what_this_unit_needs = tuple(signature(self.apply).parameters.values())

        # Unpack from the Packet what is needed
        input = {}
        for parameter in what_this_unit_needs:
            parameter_name = parameter.name
            parameter_type = parameter.annotation
            packet_label = self.PIPELINE_INPUT_LABELS[parameter_name]  # Map to the label in Packet

            input[parameter_name] = packet[packet_label]

        return input

    def __unpack_iterable_timeseries(self, packet:Packet) -> Iterable:
        """
        Receives a Packet and returns an iterable of input dictionaries, each with one Timeseries.
        """

        # Get what this unit needs from the Packet
        what_this_unit_needs = tuple(signature(self.apply).parameters.values())

        # Unpack from the Packet what is needed
        general_input = {}
        timeseries = None
        for parameter in what_this_unit_needs:
            parameter_name = parameter.name
            parameter_type = parameter.annotation
            packet_label = self.PIPELINE_INPUT_LABELS[parameter_name]  # Map to the label in Packet

            if parameter_name == 'timeseries' and parameter_type is dict:
                timeseries = packet[packet_label]
            else:
                general_input[parameter_name] = packet[packet_label]

        inputs = []
        for ts in timeseries:
            inputs.append({'timeseries': ts}.update({label:x for label, x in general_input if label != 'timeseries'}))

        return iter(inputs)

    def __pack(self, previous_packet:Packet, current_output) -> Packet:
        """
        Receives the received Packet and the output dictionary of 'apply' and returns a new Packet with the union of all
        contents. If some new content has the same label of a previous content, it will be replaced.
        """
        load = previous_packet._to_dict()  # start with the contents already in the previous packet
        packet_label = tuple(self.PIPELINE_OUTPUT_LABELS.values())[0]
        load[packet_label] = current_output  # replace or add
        return Packet(**load)

    def _apply(self, packet:Packet, how_to:Apply=None) -> Packet:

        if how_to is None or how_to is Apply.TOGETHER:
            # Step 1 - Unpack the packet contents, with all Timeseries in one element
            input = self.__unpack_all_timeseries(packet)
            # Step 2 - Apply to all inputs together
            outputs = self.apply(**input)
            # Step 3 - Create a Packet with outputs
            return self.__pack(packet, outputs)

        elif how_to is Apply.SEPARATE:
            # Step 1 - Unpack the packet contents, returning an iterable in which each element holds one Timeseries.
            inputs = self.__unpack_iterable_timeseries(packet)
            # Step 2 - Apply to each input separately
            outputs = {}
            for input in inputs:  # If there was only 1 input (i.e. 1 Timeseries), this cycle runs only once
                output = self.apply(**input)
                outputs += output  # join together
            # Step 3 - Create a Packet with outputs
            return self.__pack(packet, outputs)


class PipelineUnitsUnion(PipelineUnit, ABC):
    """
    A Union is a collection of single units where the Pipeline branches to each of them.
    Following the Template Method design pattern, a Union is the abstract class, where '_apply' is the 'template' method.

    Subclasses
    ------------
    - ApplyTogether: Runs all Timeseries together in a unique structure over each SingleUnit.
    - ApplySeparately: Runs each Timeseries separately over each SingleUnit.

    Template Method '_apply'
    ------------
    1. Unpacks, 2. Delegates and 3. Packs.
    Unpacking and packing is similar and independent of how application is delegated.
    So, Step 2, '__delegate' should be defined in each subclass.

    Abstract Method '__delegate'
    ------------
    This method should handle how each SingleUnit is applied to the Timeseries (when there are many) -- if together or
    separately.
    """

    def __init__(self, units: SinglePipelineUnit | Collection[SinglePipelineUnit], name:str=None):
        super(PipelineUnitsUnion, self).__init__(name)

        self.__current_unit = None
        self.__units = []
        if isinstance(units, SinglePipelineUnit):
            self.__units.append(units)
        elif isinstance(units, Collection) and not isinstance(units, dict):
            for unit in units:
                if isinstance(unit, SinglePipelineUnit):
                    self.__units.append(unit)
                else:
                    raise TypeError(f"{unit.__class__} is not a unitary PipelineUnit.")
        else:
            raise TypeError(f"{units.__class__} is not one or multiple PipelineUnits.")

    @property
    def current_unit(self):
        return self.__current_unit

    def _apply(self, packet:Packet) -> Packet:
        for unit in self.__units:
            self.__current_unit = unit
            input = self.__unpack(packet)
            output = self.__delegate(input)
            return self.__pack(packet, output)

    def __unpack(self, packet: Packet) -> dict:
        """
        Receives a Packet and returns a dictionary with the inputs needed to apply this unit.
        """

        # Get what this unit needs from the Packet
        what_this_unit_needs = tuple(signature(self.__current_unit.apply).parameters.values())

        # Unpack from the Packet what is needed
        input = {}
        for parameter in what_this_unit_needs:
            parameter_name = parameter.name
            parameter_type = parameter.annotation
            packet_label = self.__current_unit.PIPELINE_INPUT_LABELS[parameter_name]  # Map to the label in Packet

            input[parameter_name] = packet[packet_label]

        return input

    @abstractmethod
    def __delegate(self, input: dict):
        pass

    def __pack(self, previous_packet:Packet, current_output) -> Packet:
        """
        Receives the received Packet and the output dictionary of 'apply' and returns a new Packet with the union of all
        contents. If some new content has the same label of a previous content, it will be replaced.
        """
        load = previous_packet._to_dict()  # start with the contents already in the previous packet
        packet_label = tuple(self.__current_unit.PIPELINE_OUTPUT_LABELS.values())[0]
        load[packet_label] = current_output  # replace or add
        return Packet(**load)


class ApplyTogether(PipelineUnitsUnion):
    """
    An ApplyTogether is a collection of single units, to which each will be applied to all Timeseries at once.
    Following the Template Method design pattern, this is a concrete class, where '__delegate' is implemented.
    """

    def __init__(self, units: SinglePipelineUnit | Collection[SinglePipelineUnit], name: str = None):
        super(ApplyTogether, self).__init__(units, name)

    def _PipelineUnitsUnion__delegate(self, input: dict):
        return self.current_unit.apply(**input)  # Apply to all Timeseries together


class ApplySeparately(PipelineUnitsUnion):
    """
    An ApplySeparately is a collection of single units, to which each will be applied to one Timeseries at a time.
    Following the Template Method design pattern, this is a concrete class, where '__delegate' is implemented.
    """

    def __init__(self, units: SinglePipelineUnit | Collection[SinglePipelineUnit], name: str = None):
        super(ApplySeparately, self).__init__(units, name)

    def _PipelineUnitsUnion__delegate(self, input: dict):

        common_input = {}
        for label, content in input.items():
            if label != 'timeseries':
                common_input[label] = content

        separate_inputs = []
        for ts in input['timeseries'].values():
            this_input = {label: content for label, content in common_input.items()}  # Create copy of common content
            this_input['timeseries'] = ts  # Add 1 Timeseries
            separate_inputs.append(this_input)  # Save separate input

        separate_outputs = []
        for x in separate_inputs:  # If there was only 1 input (i.e. 1 Timeseries), this cycle runs only once
            output = self.current_unit.apply(**x)
            separate_outputs.append(output)

        # Currently, Pipeline Units only output 1 object
        return separate_outputs
