import unittest
from datetime import datetime

from ltbio.biosignals.modalities.Speech import Speech
from ltbio.biosignals.sources.ADReSSo21 import ADReSSo21
from ltbio.biosignals.timeseries import Timeseries


class SpeechTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.speech = Speech({'single_channel': Timeseries([1.0, 2.0, 3.0], # random floats that are the samples
                                                          initial_datetime=datetime(2020, 1, 1),
                                                          sampling_frequency= 1024), # Hz
                             })

    def test_print(self):
        print(self.speech)

    def test_instantiate_from_adresso21(self):
        filepath = "..."
        x = Speech(filepath, ADReSSo21())
        print(x)

if __name__ == '__main__':
    unittest.main()
