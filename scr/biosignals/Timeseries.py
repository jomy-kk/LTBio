from datetime import datetime
from numpy import array

from scr.biosignals.Unit import Unit

class Timeseries():
    def __init__(self, samples:array, sampling_frequency:float, units:Unit, initial_datetime:datetime=None, name:str=None):
        self.__samples = samples
        self.__sampling_frequency = sampling_frequency,
        self.__units = units,
        self.__initial_datetime = initial_datetime
        self.__name = name
        self.__raw_samples = None  # if some filter is applied to a biosignal, the raw version of each timeseries should be saved here


    # Getters and Setters

    @property
    def n_samples(self):
        return len(self.__samples)

    @property
    def sampling_frequency(self):
        return self.__sampling_frequency

    @property
    def units(self):
        return self.__units

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name:str):
        self.__name = name

    def get_raw_samples(self) -> array:
        return self.__raw_samples


    # Operations to the samples

    def __iadd__(self, other):
        '''The built-in increment operation (+=) increments one Timeseries to the end of another.'''
        if type(other) is Timeseries:
            self.__samples.append(other.get_samples())
            return self
        if type(other) is array:
            self.__samples.append(other)
            return self
        raise TypeError("{0} is invalid. Only an np.array or a Timeseries can be incremented to another Timeseries.".format(other))

    def __add__(self, other):
        '''The built-in sum operation (+) adds two Timeseries.'''
        if type(other) is Timeseries:
            return self.__samples + other.get_samples()
        if type(other) is array:
            return self.__samples + other
        raise TypeError("{0} is invalid. Only an np.array or a Timeseries can be added to another Timeseries.".format(other))

    def __getitem__(self, item):
        '''The built-in slicing and indexing ([x:y]) operations.'''
        try:
            if item.stop != None:
                return self.__get_samples(item.start, item.stop)
        except AttributeError: # when attribute 'stop' does not exist
            if isinstance(item, str | datetime):
                return self.__samples[int(item.timestamp() * self.__sampling_frequency)]
            elif isinstance(item, int):
                return self.__samples[item]

    def trim(self, initial_datetime: datetime, final_datetime: datetime):
        '''Trims the samples of the Timeseries and deletes the remaining.'''
        new_array = self.__get_samples(initial_datetime, final_datetime)
        self.__samples = new_array
        # FIXME: is it strings or datetimes the user has to give; or will we allow both and convert


    # Get sample statistics

    def get_mean(self) -> float:
        return self.__samples.mean()

    def get_variance(self) -> float:
        return self.__samples.var()

    def get_standard_deviation(self) -> float:
        return self.__samples.std()


    # Private Auxiliary Methods

    def __get_samples(self, initial_datetime: datetime = None, final_datetime: datetime = None) -> array:
        '''Returns the samples between the given initial and final datetimes, inclusively.'''
        initial_sample = int(initial_datetime.timestamp() * self.__sampling_frequency)
        final_sample = int(final_datetime.timestamp() * self.__sampling_frequency)
        return self.__samples[initial_sample:final_sample+1]



