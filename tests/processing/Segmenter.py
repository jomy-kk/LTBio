import unittest
from datetime import datetime, timedelta

from src.biosignals.MITDB import MITDB
from src.biosignals.Timeseries import Timeseries
from src.biosignals.ECG import ECG
from src.processing.Segmenter import Segmeter

class SegmenterTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.testpath = 'resources/MITDB_DAT_tests/'  # This is a test directory with DAT files in the MIT-DB structure,

        cls.samplesy = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.215, 0.235, 0.24, 0.24, 0.245,
                                        0.265]
        cls.samplesx = [-0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.15, -0.145, -0.135,
                                                 -0.11, -0.08, -0.04, 0.0, 0.125]

        cls.initial = datetime(2000, 1, 1, 0, 0, 0)  # 1/1/2000 0 AM
        cls.sf = 360
        cls.segmentx, cls.segmenty = Timeseries.Segment(cls.samplesx, cls.initial, cls.sf), \
                                       Timeseries.Segment(cls.samplesy, cls.initial, cls.sf)

        cls.n_samplesx_trimmed = 649998
        cls.n_samplesy_trimmed = 649998

        cls.channelx, cls.channely = "V5", "V2"


    def test_segment_without_overlap(self):
        ecg = ECG(self.testpath, MITDB)
        segmenter = Segmeter(timedelta(milliseconds=10))

        x,y = ecg._Biosignal__timeseries[self.channelx], ecg._Biosignal__timeseries[self.channely]
        x_segmented = segmenter.apply(x)
        y_segmented = segmenter.apply(y)

        self.assertEqual(len(x_segmented), 649998)
        self.assertEqual(len(y_segmented), 649998)

        segment_length = int(0.01*self.sf)  # each segment should have 3 samples

        for i, segmentx, segmenty in zip(range(0, len(self.samplesx), segment_length), x_segmented, y_segmented):
            self.assertEqual(len(segmentx), segment_length)
            self.assertEqual(len(segmenty), segment_length)
            self.assertEqual(segmentx.samples.tolist(), self.samplesx[i:i+segment_length])
            self.assertEqual(segmenty.samples.tolist(), self.samplesy[i:i+segment_length])


    def test_segment_with_overlap(self):
        ecg = ECG(self.testpath, MITDB)
        segmenter = Segmeter(timedelta(milliseconds=10), timedelta(milliseconds=3))

        x, y = ecg._Biosignal__timeseries[self.channelx], ecg._Biosignal__timeseries[self.channely]
        x_segmented = segmenter.apply(x)
        y_segmented = segmenter.apply(y)

        self.assertEqual(len(x_segmented), 1949994)
        self.assertEqual(len(y_segmented), 1949994)

        segment_length = int(0.01*self.sf)  # each segment should have 3 samples
        n_samples_overlap = int(0.003*self.sf)  # 1 sample of overlap

        for i, segmentx, segmenty in zip(range(0, len(self.samplesx), segment_length-n_samples_overlap), x_segmented, y_segmented):
            self.assertEqual(len(segmentx), segment_length)
            self.assertEqual(len(segmenty), segment_length)
            self.assertEqual(segmentx.samples.tolist(), self.samplesx[i:i + segment_length])
            self.assertEqual(segmenty.samples.tolist(), self.samplesy[i:i + segment_length])


if __name__ == '__main__':
    unittest.main()
