import unittest
from datetime import datetime, timedelta

from biosignals.timeseries.Timeseries import Timeseries
from decision.DecisionMaker import DecisionMaker
from decision.NAryDecision import NAryDecision
from features.FeatureExtractor import FeatureExtractor
from features.FeatureSelector import FeatureSelector
from features.Features import TimeFeatures
from pipeline.Packet import Packet
from processing.Segmenter import Segmenter


class SinglePipelineUnitTestCase(unittest.TestCase):
    """
    SinglePipelineUnit is an abstract class, so the concrete classes are tested on their own test cases.
    However, it should be tested the:
    - '_apply' method, that shall be used by a Pipeline.
    - '__str__' method
    """

    datetime: datetime
    ts1: Timeseries
    ts2: Timeseries
    ts3: Timeseries
    packetA: Packet
    packetB: Packet
    packetC: Packet

    @classmethod
    def setUpClass(cls) -> None:
        cls.datetime = datetime(2022, 1, 1, 1, 1, 1)
        cls.ts1 = Timeseries([0, 1, 2, 3, 4], cls.datetime, 1)
        cls.ts2 = Timeseries([0, 1, 2, 3, 4, 5], datetime.now(), 1)
        cls.ts3 = Timeseries.withDiscontiguousSegments({cls.datetime: [0, 1],
                              cls.datetime + timedelta(seconds=2): [2, 3],
                              cls.datetime + timedelta(seconds=4): [4, 5],
                                                        }, 1)
        cls.packetA = Packet(timeseries=cls.ts1)
        cls.packetB = Packet(timeseries=(cls.ts1, cls.ts2))
        cls.packetC = Packet(timeseries=cls.ts3)

    def test_apply_with_packet_1_to_1_timeseries(self):
        # Segmeters are a good example of taking 1 Timeseries and outputting 1 Timeseries.
        # There is coherence between Packet and parameters needed.
        unit = Segmenter(timedelta(seconds=1))
        result_packet = unit._apply(self.packetA)
        self.assertEqual(len(result_packet), 1)
        self.assertTrue(result_packet.has_timeseries)
        self.assertTrue(isinstance(result_packet.timeseries, Timeseries))

    def test_apply_with_packet_many_to_many_timeseries(self):
        # FeatureSelectors are a good example of taking many Timeseries and also outputting many Timeseries.
        # There is coherence between Packet and parameters needed.
        unit = FeatureSelector(lambda x: True)
        result_packet = unit._apply(self.packetB)
        self.assertTrue(len(result_packet), 1)
        self.assertTrue(isinstance(result_packet.timeseries, dict))
        self.assertTrue(len(result_packet.timeseries), 2)
        self.assertTrue(all(isinstance(x, Timeseries) for x in result_packet.timeseries.values()))

    def test_apply_with_packet_one_to_many_timeseries(self):
        # FeatureExtractors are a good example of taking 1 Timeseries and outputting many Timeseries.
        # Hence, there is NO coherence between Packet and parameters needed.
        unit = FeatureExtractor((TimeFeatures.mean, TimeFeatures.variance))
        result_packet = unit._apply(self.packetC)
        self.assertTrue(len(result_packet), 1)
        self.assertTrue(isinstance(result_packet.timeseries, dict))
        self.assertTrue(len(result_packet.timeseries), 2)

    def test_apply_with_packet_many_timeseries_to_one_number(self):
        # DecisionMakers are a good example of taking many Timeseries and outputting a value.
        unit = DecisionMaker(NAryDecision(lambda x: 5))
        result_packet = unit._apply(self.packetC)
        self.assertTrue(isinstance(result_packet['decision'], int))
        self.assertEqual(result_packet['decision'], 5)
        self.assertTrue(len(result_packet), 1)


if __name__ == '__main__':
    unittest.main()
