import unittest
from datetime import datetime, timedelta

from src.decision.NAryDecision import NAryDecision
from src.decision.DecisionMaker import DecisionMaker
from src.features.Features import TimeFeatures
from src.features.FeatureExtractor import FeatureExtractor
from src.features.FeatureSelector import FeatureSelector
from src.biosignals.Timeseries import Timeseries
from src.pipeline.Packet import Packet
from src.processing.Segmenter import Segmenter


class PipelineUnitTestCase(unittest.TestCase):
    '''
    PipelineUnit is an abstract class, so the concrete classes are tested on their own test cases.
    However, it should be tested the '_apply' method that shall be used by a Pipeline.
    '''

    @classmethod
    def setUpClass(cls) -> None:
        cls.datetime = datetime.now()
        cls.ts1 = Timeseries([Timeseries.Segment([0, 1, 2, 3, 4], cls.datetime, 1), ], True, 1)
        cls.ts2 = Timeseries([Timeseries.Segment([0, 1, 2, 3, 4, 5], datetime.now(), 1), ], True, 1)
        cls.ts3 = Timeseries([Timeseries.Segment([0, 1], cls.datetime, 1),
                              Timeseries.Segment([2, 3], cls.datetime, 1),
                              Timeseries.Segment([4, 5], cls.datetime, 1),
                              ], True, 1, equally_segmented=True)
        cls.packetA = Packet(timeseries=cls.ts1)
        cls.packetB = Packet(timeseries=(cls.ts1, cls.ts2))
        cls.packetC = Packet(timeseries=cls.ts3)

    def test_apply_with_packet_1_to_1_timeseries(self):
        # Segmeters are a good example of taking 1 Timeseries and outputting 1 Timeseries.
        # There is coherence between Packet and parameters needed.
        unit = Segmenter(timedelta(seconds=1))
        result_packet = unit._apply(self.packetA, APPLY_FOR_ALL=False)  # apply-to-all, but in this case there's just 1
        self.assertTrue(isinstance(result_packet.single_timeseries, Timeseries))
        self.assertEqual(len(result_packet), 1)
        self.assertTrue('timeseries' in result_packet)
        self.assertTrue(isinstance(result_packet['timeseries'], Timeseries))

    def test_apply_with_packet_many_to_many_timeseries(self):
        # FeatureSelectors are a good example of taking many Timeseries and also outputting many Timeseries.
        # There is coherence between Packet and parameters needed.
        unit = FeatureSelector(lambda x: True)
        result_packet = unit._apply(self.packetB, APPLY_FOR_ALL=False)  # apply-to-all
        self.assertTrue(isinstance(result_packet.all_timeseries, dict))
        self.assertTrue(len(result_packet), 1)
        self.assertTrue(len(result_packet.all_timeseries), 2)
        self.assertTrue(all(isinstance(x, Timeseries) for x in result_packet.all_timeseries.values()))

    def test_apply_with_packet_one_to_many_timeseries(self):
        # FeatureExtractors are a good example of taking 1 Timeseries and outputting many Timeseries.
        # Hence, there is NOT coherence between Packet and parameters needed.
        unit = FeatureExtractor((TimeFeatures.mean, TimeFeatures.variance))
        result_packet = unit._apply(self.packetC, APPLY_FOR_ALL=False)  # apply-to-all, but in this case there's just 1
        self.assertTrue(isinstance(result_packet.all_timeseries, dict))
        self.assertTrue(len(result_packet), 1)
        self.assertTrue(len(result_packet.all_timeseries), 2)

    def test_apply_with_packet_many_timeseries_to_one_number(self):
        # DecisionMakers are a good example of taking many Timeseries and outputting a value.
        unit = DecisionMaker(NAryDecision(lambda x: 5))
        result_packet = unit._apply(self.packetC)
        self.assertTrue(isinstance(result_packet['decision'], int))
        self.assertEqual(result_packet['decision'], 5)
        self.assertTrue(len(result_packet), 1)


if __name__ == '__main__':
    unittest.main()
