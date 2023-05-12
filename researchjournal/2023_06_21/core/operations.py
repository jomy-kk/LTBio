# -- encoding: utf-8 --
# ===================================
# ScientISST LTBio | Long-Term Biosignals
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar

from .exceptions import UndoableOperationError


# Package:
# Module: 
# Description:

# Contributors: João Saraiva
# Created: 
# Last Updated: 
# ===================================


class Operator(ABC):
    NAME: str
    DESCRIPTION: str
    SHORT: str

    def __init__(self, **parameters):
        ...

    def __call__(self, *args, **kwargs) -> Any:
        result, _ = self._apply(*args, **kwargs)
        return

    @abstractmethod
    def _apply(self, *args, **kwargs) -> (Any, 'Operation'):
        pass

    @abstractmethod
    def _undo(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def is_reversible(self) -> bool:
        # Check if "_undo" method is implemented, not if it is callable
        try:
            x = self._undo
            return True
        except NotImplementedError:
            return False


class Operation:
    def __init__(self, operator: Operator, when: datetime, allow_undo: bool, *objects, **kwargs):
        self.__operator = operator
        self.__when = when
        self.__objects = objects
        self.__kwargs = kwargs
        self.__allow_undo = allow_undo

    # Getters
    @property
    def operator(self) -> Operator:
        return self.__operator

    @property
    def when(self) -> datetime:
        return self.__when

    @property
    def objects(self):
        return self.__objects

    @property
    def is_undoable(self) -> bool:
        return self.__allow_undo and hasattr(self.__operator, "undo")

    def undo(self):
        if not self.__allow_undo:
            raise UndoableOperationError(self, by_nature=False)
        if not hasattr(self, "undo"):
            raise UndoableOperationError(self, by_nature=True)
        return self.__operator.undo(*self.__objects, **self.__kwargs)

    def __str__(self):
        return str(self.__operator)

    def __repr__(self):
        return repr(self.__operator) + " performed in " + str(self.__when)


class ArithmeticOperator(Operator, ABC): ...


class BinaryOperator(Operator, ABC): ...


class UnaryOperator(Operator, ABC): ...


class Addition(ArithmeticOperator, BinaryOperator):
    NAME = "Add"
    DESCRIPTION = "Adds two Biosignals, Timeseries or Segments, sample by sample."
    SHORT = "+"

    def _apply(self, first, second):
        return first + second

    def _undo(self, first, second):
        return first - second
