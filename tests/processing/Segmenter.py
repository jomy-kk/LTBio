import unittest
from datetime import datetime, timedelta

from src.biosignals.HSM import HSM
from src.biosignals.Timeseries import Timeseries
from src.biosignals.ECG import ECG
from src.processing.Segmenter import Segmeter

class SegmenterTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.testpath = 'resources/MITDB_DAT_tests/'  # This is a test directory with EDF files in the HSM structure,
     
        cls.samplesx1, cls.samplesx2, cls.samplesy1, cls.samplesy2 = [0.00023582690935384015,
                                                                          0.00023582690935384015,
                                                                          0.00023582690935384015], \
                                                                         [0.0001709850882900646, 0.0001709850882900646,
                                                                          0.0001709850882900646], \
                                                                         [0.00023582690935384015,
                                                                          0.00023582690935384015,
                                                                          0.00023582690935384015], \
                                                                         [0.0001709850882900646, 0.0001709850882900646,
                                                                          0.0001709850882900646]
        cls.initial1, cls.initial2 = datetime(2019, 2, 28, 8, 7, 16), datetime(2019, 2, 28, 10, 7,
                                                                                 31)  # 1/1/2022 4PM and 3/1/2022 9AM
        cls.sf = 360
        cls.segmentx1, cls.segmentx2 = Timeseries.Segment(cls.samplesx1, cls.initial1, cls.sf), \
                                         Timeseries.Segment(cls.samplesx2, cls.initial2, cls.sf)
        cls.segmenty1, cls.segmenty2 = Timeseries.Segment(cls.samplesy1, cls.initial1, cls.sf), \
                                         Timeseries.Segment(cls.samplesy2, cls.initial2, cls.sf)

        cls.n_samplesx = 12000
        cls.n_samplesy = 12000

        cls.channelx, cls.channely = "POL Ecg", "POL  ECG-"
        cls.tsx = Timeseries([cls.segmentx1, cls.segmentx2], True, cls.sf)
        cls.tsy = Timeseries([cls.segmenty1, cls.segmenty2], True, cls.sf)
        
    def test_segment_without_overlap(self):
        ecg = ECG(self.testpath, HSM)
        segmenter = Segmeter(timedelta(seconds=2))

        ecg_segmented = segmenter.segment(ecg)
        x,y = ecg_segmented._Biosignal__timeseries[self.channelx], ecg_segmented._Biosignal__timeseries[self.channely]

        self.assertEquals(len(x), self.n_samplesx)
        self.assertEquals(len(y), self.n_samplesy)
        self.assertEquals(x.segments[0].samples.tolist(), self.samplesx1[0:2])
        self.assertEquals(x.segments[1].samples.tolist(), self.samplesx1[2:4])

if __name__ == '__main__':
    unittest.main()
