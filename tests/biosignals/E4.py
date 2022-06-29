import unittest
from datetime import datetime

from src.biosignals.Event import Event
from src.biosignals.Timeseries import Timeseries
from src.biosignals import (E4, EDA, PPG, ACC, TEMP)

class E4TestCase(unittest.TestCase):

    def setUp(self):
        self.E4 = E4.E4() # E4 needs to be instantiated only to test _read and _write methods, for they are protected.
        self.testpath = 'resources/E4_CSV_tests/' # This is a test directory with CSV files in the E4 structure,
        self.initial1 = datetime(2022, 6, 2, 16, 16, 56)
        self.initial2 = datetime(2022, 6, 11, 19, 8, 28)
        self.event_onset_1 = datetime.utcfromtimestamp(1654186627.86)
        self.event_onset_2 = datetime.utcfromtimestamp(1654186639.28)
        self.event_onset_3 = datetime.utcfromtimestamp(1654186644.95)
        self.event_onset_4 = datetime.utcfromtimestamp(1654186645.19)
        self.all_onsets = (self.event_onset_1, self.event_onset_2, self.event_onset_3, self.event_onset_4)

    def verify_data(self, x, label, sf, n_samples, unit, first_samples):
        self.assertTrue(isinstance(x, dict))
        self.assertEqual(len(x), len(label))
        self.assertTrue(isinstance(list(x.keys())[0], str))
        self.assertEqual(tuple(x.keys()), label)
        for i, l in enumerate(label):
            self.assertTrue(isinstance(x[l], Timeseries))
            # And all these properties should match:
            self.assertEqual(x[l].sampling_frequency, sf)
            self.assertEqual(len(x[l]), n_samples)
            self.assertEqual(x[l].units, unit)
            self.assertEqual(len(x[l].segments), 2)  # 2 segments, because there are 2 subdirectories
            # Also, checking the second sample
            self.assertEqual(float((x[l])[self.initial1]), float(first_samples[0][i]))
            self.assertEqual(float((x[l])[self.initial2]), float(first_samples[1][i]))

    def test_read_EDA(self):
        x = self.E4._read(self.testpath, EDA.EDA)
        self.verify_data(x, ('eda', ), 4.0, 6084, None, ((0.0, ), (0.0, )) )

    def test_read_PPG(self):
        x = self.E4._read(self.testpath, PPG.PPG)
        self.verify_data(x, ('bvp', ), 64.0, 97361, None, ((0.0, ), (0.0, )) )

    def test_read_TEMP(self):
        x = self.E4._read(self.testpath, TEMP.TEMP)
        self.verify_data(x, ('temp', ), 4.0, 6080, None, ((34.630001068115234, ), (36.77000045776367, )) )

    def test_read_ACC(self):
        x = self.E4._read(self.testpath, ACC.ACC)
        self.verify_data(x, ('x', 'y', 'z'), 32.0, 48684, None, ((26,-33,47), (-60,-26,-16)) )

    def test_get_events(self):
        events = self.E4._events(self.testpath)
        self.assertTrue(isinstance(events, list))
        self.assertTrue(len(events) > 0)
        for event in events:
            self.assertTrue(event.onset in self.all_onsets)

if __name__ == '__main__':
    unittest.main()
