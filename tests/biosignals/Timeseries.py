import unittest
from datetime import datetime

from src.biosignals.Timeseries import Timeseries
from src.biosignals.Unit import Unit


class TimeseriesTestCase(unittest.TestCase):

    def setUp(self):
        self.samples1, self.samples2 = [0.34, 2.12, 3.75], [1.34, 3.12, 4.75]
        self.sf = 128
        self.units = Unit.V
        self.initial1, self.initial2 = datetime(2022, 1, 1, 16, 0), datetime(2022, 1, 3, 9, 0) # 1/1/2022 4PM and 3/1/2022 9AM
        self.name = "Test Timeseries 1"

    def test_create_continuous_timeseries(self):
        '''A continuous Timeseries is one where all the samples were acquired at each sampling interval. There were no acquisition interruptions on purpose.'''
        ts = Timeseries(self.samples1, self.sf, self.units, self.initial1, self.name)

        # Verify samples
        self.assertEquals(ts.n_samples, len(self.samples1))
        self.assertEquals(ts[0], self.samples1[0])
        self.assertEquals(ts[-1], self.samples1[-1])

        # Verify metadata
        self.assertEquals(ts.sampling_frequency, self.sf)
        self.assertEquals(ts.units, self.units)
        self.assertEquals(ts.name, self.name)

    def test_create_discrete_timeseries(self):
        '''A discrete Timeseries is one with interruptions between some sequences of samples. The initial datetime of each timeframe should be specified.'''
        ts = Timeseries({self.initial1: self.samples1, self.initial2: self.samples2},
                        self.sf, self.units, name=self.name)

        # Verify samples
        self.assertEquals(ts.n_samples, len(self.samples1)+len(self.samples2))
        self.assertEquals(ts[0], self.samples1[0])
        self.assertEquals(ts[-1], self.samples2[-1])
        self.assertEquals(ts[0:(len(self.samples1)+len(self.samples2))], self.samples1 + self.samples2) # verifies continuity

        # Verify metadata
        self.assertEquals(ts.sampling_frequency, self.sf)
        self.assertEquals(ts.units, self.units)
        self.assertEquals(ts.name, self.name)

    def test_indexing_with_datetime(self):
        ts = Timeseries({self.initial1: self.samples1, self.initial2: self.samples2}, self.sf)

        # Time point indexing
        self.assertEquals(ts['2022-01-01 16:00'], ts[0]) # FIXME
        self.assertEquals(ts['2022-01-01 16:00'], ts[1]) # FIXME
        self.assertEquals(ts['2022-01-01 16:00'], ts[2]) # FIXME
        self.assertEquals(ts['2022-01-01 16:00'], ts[3]) # FIXME
        self.assertEquals(ts['2022-01-01 16:00'], ts[4]) # FIXME
        self.assertEquals(ts['2022-01-01 16:00'], ts[5]) # FIXME
        self.assertEquals(ts[datetime(2022,1,1,16)], ts[5]) # trying with datetime object # FIXME

        # Time interval indexing
        self.assertEquals(ts['2022-01-01 16:00':'2022-01-01 16:00'], ts[0:3]) # FIXME
        self.assertEquals(ts['2022-01-01 16:00':'2022-01-01 16:00'], ts[3:6]) # FIXME
        self.assertEquals(ts['2022-01-01 16:00':'2022-01-01 16:00'], ts[0:6]) # FIXME

    def test_set_name(self):
        ts = Timeseries(dict(), self.sf, name=self.name)
        self.assertEqual(ts.name, self.name)
        ts.name = "New Name"
        self.assertEqual(ts.name, "New Name")

    def test_concatenate_two_timeseries(self):
        # With the same sampling frequency and units
        ts1 = Timeseries(self.samples1, self.sf, self.units, self.initial1)
        ts2 = Timeseries(self.samples2, self.sf, self.units, self.initial1)
        self.assertEquals(ts1 + ts2, self.samples1+self.samples2)

        # With different sampling frequencies
        ts2 = Timeseries(self.samples2, self.sf + 1, self.units, self.initial1)
        with self.assertRaises(ArithmeticError):
            ts1 + ts2

        # With different units
        ts2 = Timeseries(self.samples2, self.sf, Unit.G, self.initial1)
        with self.assertRaises(ArithmeticError):
            ts1 + ts2

        # With initial datetime of the latter coming before the final datetime of the former
        ts2 = Timeseries(self.samples2, self.sf, self.units, datetime(2022, 1, 1, 15)) # FIXME
        with self.assertRaises(ArithmeticError):
            ts1 + ts2



if __name__ == '__main__':
    unittest.main()
