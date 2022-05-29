import unittest

from src.biosignals.ECG import ECG
from src.biosignals.HSM import HSM
from src.processing.FrequencyDomainFilter import FrequencyDomainFilter, FrequencyResponse, BandType

class ECGTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ecg = ECG('../test', HSM)

    def test_plotting_summary_biosppy_filtering_default(cls):
        cls.ecg['2019-02-26 03:23:04':'2019-02-26 03:23:14']['POL Ecg'].plot_summary(show=True)

    def test_plotting_summary_own_filtering(cls):
        cls.ecg.filter(FrequencyDomainFilter(FrequencyResponse.FIR, BandType.LOWPASS, cutoff=20, order=4))
        cls.ecg['2019-02-26 03:23:04':'2019-02-26 03:23:14']['POL Ecg'].plot_summary(show=True)
        cls.ecg['2019-02-26 03:23:04':'2019-02-26 03:23:14']['POL Ecg'].undo_filters()

if __name__ == '__main__':
    unittest.main()
