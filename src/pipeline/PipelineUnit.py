# -*- encoding: utf-8 -*-

# ===================================

# IT - LongTermBiosignals

# Package: pipeline
# Module: PipelineUnit
# Description: Classes PipelineUnit, SinglePipelineUnit, PipelineUnitsUnion, ApplyTogether, and ApplySeparately.

# Contributors: João Saraiva
# Created: 02/06/2022
# Last Updated: 07/07/2022

# ===================================

from abc import ABC, abstractmethod
from inspect import signature, Parameter
from typing import Collection, Dict, Iterable, Tuple

from biosignals.timeseries.Timeseries import Timeseries
from pipeline.Packet import Packet


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
    def _apply(self, packet:Packet) -> Packet:
        """
        Receives a Packet with the necessary inputs to apply the unit and returns a Packet with the relevant outputs.
        Acts as the 'operation' method in the composite design pattern.
        """
        pass

    def __rshift__(self, other):
        '''
        Defines the >> operator, the fastest shortcut to create a Pipeline
        '''
        from pipeline.Pipeline import Pipeline
        if isinstance(other, PipelineUnit):  # concatenate self.Unit + other.Unit = res.Pipeline
            res = Pipeline()
            res.add(self)
            res.add(other)
            return res
        elif isinstance(other, Pipeline):  # concatenate another self.Unit + other.Pipeline = res.Pipeline
            pass
        else:
            raise TypeError(f'Cannot join a PipelineUnit with a {type(other)}.')

    @staticmethod
    def _unpack_separately(packet:Packet, unit) -> Tuple[Iterable[str], Iterable[Dict]]:
        """
        Auxiliary class procedures.
        Receives a Packet and returns a Tuple of Iterables, (x, y), where:
        - y are dictionaries with the necessary inputs, each with one Timeseries.
        - x are the original labels of each Timeseries in the receiving Packet.
        """

        # Get what this unit needs from the Packet
        what_this_unit_needs = tuple(signature(unit.apply).parameters.values())

        # Unpack from the Packet what is needed
        common_input = {}
        for parameter in what_this_unit_needs:
            parameter_name = parameter.name
            parameter_type = parameter.annotation
            packet_label = unit.PIPELINE_INPUT_LABELS[parameter_name]  # Map to the label in Packet

            if packet_label == Packet.TIMESERIES_LABEL :  # Timeseries
                separate_inputs = []
                original_ts_labels = []
                if packet.has_timeseries_collection:  # Meaning there were discovered 1 or more Timeseries in a collection
                    for original_ts_label, ts in packet[packet_label].items():
                        this_input = {label: content for label, content in common_input.items()}  # Create copy of common content
                        if parameter_type is Timeseries:  # if apply only requires 1 Timeseries, rather than collection
                            this_input[parameter_name] = ts  # Add the element of 1 Timeseries
                        else:
                            this_input[parameter_name] = {original_ts_label: ts}  # Add only 1 the collection of 1 Timeseries
                        separate_inputs.append(this_input)  # Save separate input
                        original_ts_labels.append(original_ts_label)
                elif packet.has_single_timeseries:  # Meaning just 1 Timeseries was found outside a collection
                    this_input = {label: content for label, content in common_input.items()}  # Create copy of common content
                    this_input[parameter_name] = packet[packet_label]  # Add the only Timeseries
                    separate_inputs.append(this_input)  # Save separate input
                else:
                    pass  # There are no Timeseries

                return iter(original_ts_labels), iter(separate_inputs)

            else:  # Others
                common_input[parameter_name] = packet[packet_label]
                return iter((packet_label, )), iter((common_input, ))

    @staticmethod
    def _unpack_as_is(packet: Packet, unit) -> Dict:
        """
        Auxiliary class procedures.
        Receives a Packet and returns an input dictionaries with all necessary parameters
        """

        # Get what this unit needs from the Packet
        what_this_unit_needs = tuple(signature(unit.apply).parameters.values())

        # Unpack from the Packet what is needed
        input = {}
        for parameter in what_this_unit_needs:
            parameter_name = parameter.name
            parameter_type = parameter.annotation
            packet_label = unit.PIPELINE_INPUT_LABELS[parameter_name]  # Map to the label in Packet

            content = packet[packet_label]
            if isinstance(content, dict) and parameter_type is Timeseries:
                assert len(content) == 1
                input[parameter_name] = tuple(content.values())[0]  # arity match
            elif not isinstance(content, dict) and parameter_type is not Timeseries:
                input[parameter_name] = {'_': content}  # arity match
            else:
                input[parameter_name] = content  # arity already matches

        return input

    @staticmethod
    def _pack_as_is(previous_packet:Packet, current_output, unit) -> Packet:
        """
        Receives the received Packet and the output dictionary of 'apply' and returns a new Packet with the union of all
        contents. If some new content has the same label of a previous content, it will be replaced.
        """
        load = previous_packet._to_dict()  # start with the contents already in the previous packet
        packet_label = tuple(unit.PIPELINE_OUTPUT_LABELS.values())[0]
        load[packet_label] = current_output  # replace or add
        return Packet(**load)

    @staticmethod
    def _pack_with_original_ts_labels(previous_packet:Packet, current_output:list, unit, original_ts_labels:list) -> Packet:
        """
        Receives the received Packet, its original Timeseries labels, and the output dictionary of 'apply'.
        It returns a new Packet with the union of all contents.
        If some new content has the same label of a previous content, it will be replaced.
        """
        load = previous_packet._to_dict()  # start with the contents already in the previous packet
        packet_label = tuple(unit.PIPELINE_OUTPUT_LABELS.values())[0]

        # Timeseries
        timeseries = {}
        if packet_label == Packet.TIMESERIES_LABEL:
            for original_ts_label, ts in zip(original_ts_labels, current_output):
                assert isinstance(ts, Timeseries)  # Assuming only 1 Timeseries was outputted in each application
                timeseries[original_ts_label] = ts
            load[Packet.TIMESERIES_LABEL] = timeseries

        # Others
        else:
            load[packet_label] = current_output  # replace or add

        return Packet(**load)

    @staticmethod
    def _pack_separate_outputs(previous_packet:Packet, separate_outputs:list, unit, original_ts_labels:list) -> Packet:
        """
        Receives the received Packet, its original Timeseries labels, and a list of outputs, one per each time 'apply' was called.
        It returns a new Packet with the union of all contents.
        If some new content has the same label of a previous content, it will be replaced.
        """
        load = previous_packet._to_dict()  # start with the contents already in the previous packet
        packet_label = tuple(unit.PIPELINE_OUTPUT_LABELS.values())[0]

        res = {}
        for original_ts_label, output in zip(original_ts_labels, separate_outputs):
            if isinstance(output, dict):
                if len(separate_outputs) > 1:
                    for content_label, content in output.items():
                        res[original_ts_label+':'+content_label] = content
                else:  # no need to associate to each original label, because there is just 1 output, it means there was just 1 input
                    for content_label, content in output.items():
                        res[content_label] = content
            else:
                res[original_ts_label] = output

        load[packet_label] = res

        return Packet(**load)


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

    def _apply(self, packet:Packet) -> Packet:
        if self.__requires_one_timeseries() and packet.has_timeseries_collection:
            return ApplySeparately(self)._apply(packet)
        else:
            input = self.__unpack(packet)
            output = self.__apply(input)
            return self.__pack(packet, output)

    def __unpack(self, packet:Packet):
        return PipelineUnit._unpack_as_is(packet, self)

    def __apply(self, input: Iterable):
        return self.apply(**input)

    def __pack(self, previous_packet:Packet, current_output) -> Packet:
        return PipelineUnit._pack_as_is(previous_packet, current_output, self)

    def __requires_one_timeseries(self) -> bool:
        if Packet.TIMESERIES_LABEL in self.PIPELINE_INPUT_LABELS.values():
            what_this_unit_needs = tuple(signature(self.apply).parameters.values())
            for parameter in what_this_unit_needs:
                if self.PIPELINE_INPUT_LABELS[parameter.name] == Packet.TIMESERIES_LABEL:
                    if parameter.annotation is Timeseries:
                        return True
        return False

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

    Labels
    ------------
    PIPELINE_INPUT_LABELS
    Maps every label of a needed input inside a Packet to the parameter names of the corresponding 'apply' methods.
    PIPELINE_OUTPUT_LABELS
    Maps every output name of the 'apply' methods to a label to be saved inside a Packet.
    """

    PIPELINE_INPUT_LABELS: Dict[str, str]  # { apply parameter : packet label }
    PIPELINE_OUTPUT_LABELS: Dict[str, str]  # { apply output : packet label }

    def __init__(self, units: SinglePipelineUnit | Collection[SinglePipelineUnit], name:str=None):
        super(PipelineUnitsUnion, self).__init__(name)

        self.__units = []
        self.__current_unit = None

        if isinstance(units, SinglePipelineUnit):
            self.__units.append(units)
            self.PIPELINE_INPUT_LABELS = units.PIPELINE_INPUT_LABELS
            self.PIPELINE_OUTPUT_LABELS = units.PIPELINE_OUTPUT_LABELS
        elif isinstance(units, Collection) and not isinstance(units, dict):
            self.PIPELINE_INPUT_LABELS = {}
            self.PIPELINE_OUTPUT_LABELS = {}
            for unit in units:
                if isinstance(unit, SinglePipelineUnit):
                    if unit.name is not None:
                        self.__units.append(unit)
                        self.PIPELINE_INPUT_LABELS.update(unit.PIPELINE_INPUT_LABELS)
                        self.PIPELINE_OUTPUT_LABELS.update(unit.PIPELINE_OUTPUT_LABELS)
                    else:
                        raise AssertionError(f"Pipeline Unit of type {type(unit).__name__} must have a name if inside a Union, in order to resolve eventual conflicting labels.")
                else:
                    raise TypeError(f"{unit.__class__} is not a unitary PipelineUnit.")
        else:
            raise TypeError(f"{units.__class__} is not one or multiple PipelineUnits.")

    @property
    def current_unit(self):
        return self.__current_unit

    @property
    def all_input_parameters(self) -> Tuple[Parameter]:
        res = []  # shouldn't this be a Set
        for unit in self.__units:
            res += list(signature(unit.apply).parameters.values())
        return tuple(res)

    def __str__(self):
        return 'Union' + (': ' + self.name) if self.name is not None else ''

    def _apply(self, packet:Packet) -> Packet:
        """
        Acts as the 'template' method in the template method design pattern.
        """

        # Assert that there is not a single Timeseries and a single unit
        if len(self.__units) == 1 and packet.has_single_timeseries:
            raise AssertionError(f"There's only 1 Timeseries arriving to Union {self.name} comprising only 1 PipelineUnit. There's no use case for this. Instead, try inserting the PipelineUnit directly to the Pipeline, without using Unions.")

        output_packets = []
        for unit in self.__units:
            self.__current_unit = unit
            input = self.__unpack(packet)
            output = self.__delegate(input)
            output_packets.append(self.__pack(packet, output))

        return self.__return_packet(output_packets)

    @abstractmethod
    def __unpack(self, packet: Packet):
        pass

    @abstractmethod
    def __delegate(self, input):
        pass

    @abstractmethod
    def __pack(self, previous_packet:Packet, current_output) -> Packet:
        pass

    def __return_packet(self, output_packets:list) -> Packet:
        if len(output_packets) == 1:
            return output_packets[0]
        else:
            # There might exist some conflicts here, such as contents with the same label.
            # To ensure resolution, units must have names, and previous labels will be prefixed by the unit name.
            return Packet.join_packets(**{unit.name: packet for unit, packet in zip(self.__units, output_packets)})


class ApplyTogether(PipelineUnitsUnion):
    """
    An ApplyTogether is a collection of single units, to which each will be applied to all Timeseries at once.
    Following the Template Method design pattern, this is a concrete class, where '__delegate' is implemented.
    """

    def __init__(self, units: SinglePipelineUnit | Collection[SinglePipelineUnit], name: str = None):
        super(ApplyTogether, self).__init__(units, name)

    def _PipelineUnitsUnion__unpack(self, packet: Packet) -> dict:
        unpacked = PipelineUnit._unpack_as_is(packet, self.current_unit)
        return unpacked

    def _PipelineUnitsUnion__delegate(self, input: dict):
        return self.current_unit.apply(**input)  # Apply to all Timeseries together

    def _PipelineUnitsUnion__pack(self, previous_packet: Packet, current_output) -> Packet:
        return PipelineUnit._pack_as_is(previous_packet, current_output, self.current_unit)


class ApplySeparately(PipelineUnitsUnion):
    """
    An ApplySeparately is a collection of single units, to which each will be applied to one Timeseries at a time.
    Following the Template Method design pattern, this is a concrete class, where '__delegate' is implemented.
    """

    def __init__(self, units: SinglePipelineUnit | Collection[SinglePipelineUnit], name: str = None):
        super(ApplySeparately, self).__init__(units, name)

    def _PipelineUnitsUnion__unpack(self, packet: Packet) -> Iterable:
        original_labels, separate_inputs = PipelineUnit._unpack_separately(packet, self.current_unit)
        self.__original_ts_labels = original_labels
        return separate_inputs

    def _PipelineUnitsUnion__delegate(self, separate_inputs: Iterable) -> list:
        separate_outputs = []
        for input in separate_inputs:  # If there was only 1 input (i.e. 1 Timeseries), this cycle runs only once, which is okay
            output = self.current_unit.apply(**input)
            separate_outputs.append(output) # Currently, Pipeline Units only output 1 object

        return separate_outputs

    def _PipelineUnitsUnion__pack(self, previous_packet: Packet, current_output) -> Packet:
        if Packet.TIMESERIES_LABEL in current_output and len(self.__original_ts_labels) == len(current_output[Packet.TIMESERIES_LABEL]):
            return PipelineUnit._pack_with_original_ts_labels(previous_packet, current_output, self.current_unit, self.__original_ts_labels)
        else:
            return PipelineUnit._pack_separate_outputs(previous_packet, current_output, self.current_unit, self.__original_ts_labels)
