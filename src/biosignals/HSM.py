###################################

# IT - PreEpiSeizures

# Package: biosignals
# File: HSM
# Description: Procedures to read and write datafiles from Hospital de Santa Maria.

# Contributors: João Saraiva, Mariana Abreu
# Last update: 23/04/2022

###################################
from os import listdir
import numpy as np
from mne.io import read_raw_edf

from src.biosignals.BiosignalSource import BiosignalSource

class HSM(BiosignalSource):
    '''This class represents the source of Hospital de Santa Maria (Lisboa, PT) and includes methods to read and write
    biosignal files provided by them. Usually they are in the European EDF/EDF+ format.'''

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Hospital de Santa Maria"

    @staticmethod
    def _read(path, type):
        '''Reads multiple EDF/EDF+ files on the directory 'path' and returns a Biosignal associated with a Patient.'''
        # FIXME
        """
        def get_signal(filename):
        log("Collecting {}...".format(filename))
        signal = ((pd.read_hdf(path.join(self.ecg_path, filename)))['ECG']).to_numpy()
        sf = 1000  # in Hz
        n = len(signal)
        t = n / sf
        signal = resample(signal, int(360 * t))
        return signal
        
        filenames = [file for file in sorted(listdir(self.ecg_path)) if file.endswith('.edf')]
        if filenames == []:
        log("There are no EDF files in {}".format(self.ecg_path), 2)
        else:
        self.data = np.array([])
        for filename in filenames:
        get_ecg_from_edf(path.join(self.ecg_path, filename))
        self.data = np.append(self.data, get_signal())
        return self.data
        
        EdfReader.(path)
        """

