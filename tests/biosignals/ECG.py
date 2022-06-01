import unittest

from src.biosignals.ECG import ECG
from src.biosignals.MITDB import MITDB
from src.processing.FrequencyDomainFilter import FrequencyDomainFilter, FrequencyResponse, BandType

class ECGTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ecg = ECG('resources/MITDB_DAT_tests', MITDB)

    def test_plotting_summary_biosppy_filtering_default(cls):
        cls.ecg['2000-01-01 00:00:15':'2000-01-01 00:00:45']['V2'].plot_summary(show=True)

    def test_plotting_summary_own_filtering(cls):
        cls.ecg.filter(FrequencyDomainFilter(FrequencyResponse.FIR, BandType.LOWPASS, cutoff=20, order=4))
        cls.ecg['2000-01-01 00:00:15':'2000-01-01 00:00:45']['V2'].plot_summary(show=True)
        cls.ecg.undo_filters()

if __name__ == '__main__':
    unittest.main()
