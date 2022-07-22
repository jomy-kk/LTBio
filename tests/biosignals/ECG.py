import unittest
from datetime import timedelta

import numpy as np

from src.biosignals.ECG import ECG
from src.biosignals.MITDB import MITDB
from src.processing.FrequencyDomainFilter import FrequencyDomainFilter, FrequencyResponse, BandType

class ECGTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ecg = ECG('resources/MITDB_DAT_tests', MITDB)

    def test_plotting_summary_biosppy_filtering_default(self):
        self.ecg['2000-01-01 00:00:15':'2000-01-01 00:00:45']['V2'].plot_summary(show=True)

    def test_plotting_summary_own_filtering(self):
        self.ecg.filter(FrequencyDomainFilter(FrequencyResponse.FIR, BandType.LOWPASS, cutoff=20, order=4))
        self.ecg['2000-01-01 00:00:15':'2000-01-01 00:00:45']['V2'].plot_summary(show=True)
        self.ecg.undo_filters()

    def test_r_peaks(self):
        rpeaks = self.ecg['V5'].r_timepoints()
        first_10_true_rpeaks = np.array([timedelta(microseconds=872222), timedelta(seconds=1, microseconds=702778), timedelta(seconds=2, microseconds=497222), timedelta(seconds=3, microseconds=294444), timedelta(seconds=4, microseconds=102778), timedelta(seconds=4, microseconds=855556), timedelta(seconds=5, microseconds=722222), timedelta(seconds=6, microseconds=552778), timedelta(seconds=7, microseconds=358333), timedelta(seconds=8, microseconds=166667)])
        first_10_true_rpeaks += self.ecg.initial_datetime
        self.assertTrue(np.all(rpeaks[:10] == first_10_true_rpeaks))

    def test_heartbeats(self):
        heartbeats = self.ecg.heartbeats()
        #print(heartbeats)
        #heartbeats['2000-01-01 00:00:15':'2000-01-01 00:00:45'].plot()

if __name__ == '__main__':
    unittest.main()
