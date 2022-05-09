###################################

# IT - PreEpiSeizures

# Package: biosignals
# File: HEM
# Description: Procedures to read and write datafiles from Hospital Egas Moniz.

# Contributors: João Saraiva, Mariana Abreu
# Last update: 28/04/2022

###################################
from os import listdir, path
from numpy import array
from neo import MicromedIO

from src.biosignals.BiosignalSource import BiosignalSource
from src.biosignals.Timeseries import Timeseries
from src.biosignals import ECG

class HEM(BiosignalSource):
    '''This class represents the source of Hospital de Santa Maria (Lisboa, PT) and includes methods to read and write
    biosignal files provided by them. Usually they are in the European EDF/EDF+ format.'''

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Hospital Egas Moniz"

    @staticmethod
    def __read_trc(list, metadata=False):
        """
        Return trc file information, whether it is the values or the metadata, according to boolean metadata
        :param list
        :param metadata

        """
        dirfile = list[0]
        sensor = list[1]
        # get edf data
        seg_micromed = MicromedIO(dirfile)
        hem_data = seg_micromed.read_segment()
        hem_sig = hem_data.analogsignals[0]
        ch_list = seg_micromed.header['signal_channels']['name']
        # get channels that correspond to type (POL Ecg = type ecg)
        find_idx = [hch for hch in range(len(ch_list)) if sensor.lower() in ch_list[hch].lower()]
        # returns ch_list of interest, sampling frequency, initial datetime
        if metadata:
            return ch_list[find_idx], float(hem_sig.sampling_rate), hem_data.rec_datetime, hem_sig.units
        # returns initial date and samples
        return array(hem_sig[:, find_idx].T), hem_data.rec_datetime
        # return Timeseries.Segment(samples=np.array(hem_sig[:, find_idx]).T, initial_datetime=hem_data.rec_datetime,
        #                          sampling_frequency=float(hem_sig.sampling_rate))

    @staticmethod
    def _read(dir, type):
        '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.'''
        # first a list is created with all the filenames that end in .edf and are inside the chosen dir
        # this is a list of lists where the second column is the type of channel to extract
        if type is ECG:
            label = 'ecg'
        # if type is EEG:
        #    label = 'eeg'

        all_files = sorted([[path.join(dir, file), label] for file in listdir(dir) if file.lower().endswith('.trc')])
        # run the edf read function for all files in list all_files
        channels, sfreq, start_datetime, units = HEM.__read_trc(all_files[0], metadata=True)
        all_trc = list(map(HEM.__read_trc, all_files))
        # run the trc read function for all files in list all_files
        new_dict = {}
        # TODO ADD UNITS TO TIMESERIES
        for ch in range(len(channels)):
            segments = [Timeseries.Segment(trc_data[0][ch], initial_datetime=trc_data[1], sampling_frequency=sfreq)
                        for trc_data in all_trc]
            new_timeseries = Timeseries(segments=segments, sampling_frequency=sfreq, ordered=True)
            new_dict[channels[ch]] = new_timeseries
        return new_dict
