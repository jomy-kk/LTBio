import unittest
from datetime import datetime, timedelta
from numpy import array, allclose

from src.biosignals.ECG import ECG
from src.biosignals.HEM import HEM
from src.processing.TimeDomainFilter import TimeDomainFilter, ConvolutionOperation


class TimeDomainFilterTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.samplesx1 = array([440.234375, 356.73828125, 191.69921875, 44.62890625, -74.21875, -126.85546875, -116.30859375, -50.78125, 34.47265625, 119.62890625, 189.74609375, 230.37109375, 275.29296875, 301.66015625, 294.43359375, 277.05078125])
        cls.samplesy1 = array([582.03125, 629.98046875, 620.01953125, 595.8984375, 526.7578125, 402.44140625, 276.5625, 210.15625, 150.390625, 153.125, 167.08984375, 170.41015625, 209.08203125, 244.82421875, 237.01171875, 209.08203125])
        cls.samplesx2 = array([-90.52734375, -92.7734375, -61.62109375, 65.13671875, -8.30078125, 28.22265625, 241.69921875, 187.5, 52.83203125, -58.49609375, 0.390625, 84.86328125, -11.71875, -277.734375, -31.8359375, -15.91796875])
        cls.samplesy2 = array([154.6875, 105.17578125, 60.64453125, 94.82421875, 101.171875, 92.87109375, 119.140625, 127.83203125, 102.34375, 39.74609375, 69.04296875, 128.90625, 103.90625, 74.8046875, 15.91796875, 31.0546875])

        cls.initial1, cls.initial2 = datetime(2018, 12, 11, 11, 59, 5), datetime(2018, 12, 11, 19, 39, 17)  # 1/1/2022 4PM and 3/1/2022 9AM

        cls.sf = 256.0

        cls.n_samplesx = 56736
        cls.n_samplesy = 56736

        cls.channelx, cls.channely = "ecg", "ECG"

        cls.testpath = 'resources/HEM_TRC_tests'
        cls.biosignal = ECG(cls.testpath, HEM)

    def check_samples(cls, targetx1, targetx2, targety1, targety2):
        # Check sizes
        cls.assertEquals(len(cls.biosignal), 2)
        cls.assertEquals(len(cls.biosignal[cls.channelx][:]), cls.n_samplesx)
        cls.assertEquals(len(cls.biosignal[cls.channely][:]), cls.n_samplesy)
        # Check first 10 samples of each segment
        cls.assertTrue(allclose(cls.biosignal[cls.channelx][cls.initial1:cls.initial1 + timedelta(seconds=16 / cls.sf)].segments[0][:], targetx1))
        cls.assertTrue(allclose(cls.biosignal[cls.channelx][cls.initial2:cls.initial2 + timedelta(seconds=16 / cls.sf)].segments[0][:], targetx2))
        cls.assertTrue(allclose(cls.biosignal[cls.channely][cls.initial1:cls.initial1 + timedelta(seconds=16 / cls.sf)].segments[0][:], targety1))
        cls.assertTrue(allclose(cls.biosignal[cls.channely][cls.initial2:cls.initial2 + timedelta(seconds=16 / cls.sf)].segments[0][:], targety2))

    def test_create_filter(cls):
        filter = TimeDomainFilter(ConvolutionOperation.MEDIAN, window_length=timedelta(seconds=0.1), overlap_length=timedelta(seconds=0.05))
        cls.assertEquals(filter.operation, ConvolutionOperation.MEDIAN)
        cls.assertEquals(filter.window_length, timedelta(seconds=0.1))
        cls.assertEquals(filter.overlap_length, timedelta(seconds=0.05))

    def test_undo_filters(cls):
        design = TimeDomainFilter(ConvolutionOperation.MEDIAN, window_length=timedelta(seconds=0.09766))
        cls.biosignal.filter(design)
        cls.biosignal.undo_filters()
        cls.check_samples(cls.samplesx1, cls.samplesx2, cls.samplesy1, cls.samplesy2)

    def test_apply_median(cls):
        cls.design = TimeDomainFilter(ConvolutionOperation.MEDIAN, window_length=timedelta(seconds=0.09766))
        cls.biosignal.filter(cls.design)
        filtered_samplesx1 = array([0.0, 0.0, 0.0, 0.0, 34.47265625, 44.62890625, 119.62890625, 189.74609375, 191.69921875, 202.734375, 202.734375, 202.734375, 202.734375, 194.921875, 191.69921875, 190.91796875])
        filtered_samplesx2 = array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.390625, 0.390625, 0.390625, 0.390625])
        filtered_samplesy1 = array([150.390625, 153.125, 167.08984375, 170.41015625, 186.71875, 188.671875, 199.4140625, 199.4140625, 199.4140625, 199.4140625, 199.4140625, 199.4140625, 199.4140625, 194.921875, 188.671875, 186.71875])
        filtered_samplesy2 = array([39.74609375, 60.64453125, 60.64453125, 60.64453125, 69.04296875, 69.04296875, 69.04296875, 69.04296875, 69.04296875, 69.04296875, 69.04296875, 69.04296875, 69.04296875, 60.64453125, 56.4453125, 52.83203125])

        cls.check_samples(filtered_samplesx1, filtered_samplesx2, filtered_samplesy1, filtered_samplesy2)

        cls.biosignal.undo_filters()


if __name__ == '__main__':
    unittest.main()
