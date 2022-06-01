from abc import ABC, abstractmethod
from typing import List, Iterable, Dict
import numpy as np

from src.biosignals.Timeseries import Timeseries


class FeatureExtractor(ABC):

    __extracted:Dict[str, Iterable[float]]

    def __init__(self):
        self.__extracted = dict()

    @property
    def extracted(self):
        return self.__extracted

    def __getitem__(self, item:str):
        return self.__extracted

    def __extract(self, segments:Iterable[Timeseries.Segment], operation, name):
        """
        Extracts a list of the result of applying 'operation' to each segment in 'segments'.
        Users can get it by doing x[name], where x is the concrete extractor.
        """
        self.__extracted[name] = [operation(segment.samples) for segment in segments]

    def extract_all(self, segments:Iterable[Timeseries.Segment]):
        """
        Extracts all features defined by the concrete extractor.
        Users can get them by doing x.extracted, where x is the concrete extractor.
        """
        for method in [method for method in dir(self) if method.startswith('extract_')]:
            getattr(self, method)(segments)


class TimeFeatureExtractor(FeatureExtractor):

    def extract_mean(self, segments):
        self._FeatureExtractor__extract(segments, np.mean, 'mean')

    def extract_variance(self, segments):
        self._FeatureExtractor__extract(segments, np.var, 'variance')

    def extract_deviation(self, segments):
        self._FeatureExtractor__extract(segments, np.std, 'standard deviation')

