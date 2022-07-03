import unittest
from datetime import datetime, timedelta

from src.pipeline.PipelineUnit import Apply, ApplyTogether, ApplySeparately
from src.features.FeatureSelector import FeatureSelector
from src.biosignals.Timeseries import Timeseries
from src.pipeline.Packet import Packet
from src.processing.Segmenter import Segmenter


class PipelineUnitsUnionTestCase(unittest.TestCase):
    """
    PipelineUnitsUnion can be instantiated with ApplyTogether and ApplySeparately.
    It should be tested the:
    - '_apply' method, that shall be used by a Pipeline.
    - '__str__' method
    """

    datetime: datetime
    ts1: Timeseries
    ts2: Timeseries
    packetA: Packet

    @classmethod
    def setUpClass(cls) -> None:
        cls.datetime = datetime.now()
        cls.ts1 = Timeseries([Timeseries.Segment([0, 1, 2, 3, 4], cls.datetime, 1), ], True, 1)
        cls.ts2 = Timeseries([Timeseries.Segment([0, 1, 2, 3, 4, 5], cls.datetime, 1), ], True, 1)

        cls.packetA = Packet(timeseries=(cls.ts1, cls.ts2))

    def test_get_name(self):
        unit1 = FeatureSelector(lambda x: True)
        union = ApplyTogether(unit1, name='Test Name')
        self.assertEqual(union.name, 'Test Name')

    def test_apply_many_units_together(self):
        # FeatureSelectors are a good example of taking many Timeseries together
        unit1 = FeatureSelector(lambda x: True)
        unit2 = FeatureSelector(lambda x: True)
        union = ApplyTogether((unit1, unit2))

        result_packet = union._apply(self.packetA)

        self.assertEqual(len(result_packet), 1)
        self.assertTrue('timeseries' in result_packet)
        self.assertTrue(isinstance(result_packet.all_timeseries, dict))
        self.assertTrue(len(result_packet.all_timeseries), 6)  # there should be 3*2=6 Timeseries there
        self.assertTrue(all(isinstance(x, Timeseries) for x in result_packet.all_timeseries.values()))

    def test_apply_many_units_separately(self):
        # Segmenters are a good example of taking 1 Timeseries at a time
        unit1 = Segmenter(timedelta(seconds=1))
        unit2 = Segmenter(timedelta(seconds=2))
        union = ApplySeparately((unit1, unit2))

        result_packet = union._apply(self.packetA)

        self.assertEqual(len(result_packet), 1)
        self.assertTrue('timeseries' in result_packet)
        self.assertTrue(isinstance(result_packet.all_timeseries, dict))
        self.assertTrue(len(result_packet.all_timeseries), 6)  # there should be 3*2=6 Timeseries there
        self.assertTrue(all(isinstance(x, Timeseries) for x in result_packet.all_timeseries.values()))


    # Even though a Union should be instantiated with multiple PipelineUnits, it can be used with just one PipelineUnit.
    # This is very much the case when the input Packet contains many Timeseries and one wishes to apply the PipelineUnit
    # to all Timeseries at once. Since the default is Apply.SEPARATELY, there's no need to instantiate
    # ApplySeparately with just one PipelineUnit; although it's possible.

    def test_apply_one_unit_together(self):
        # FeatureSelectors are a good example of taking many Timeseries together
        unit1 = FeatureSelector(lambda x: True)
        union = ApplyTogether(unit1)

        result_packet = union._apply(self.packetA)

        self.assertEqual(len(result_packet), 1)
        self.assertTrue('timeseries' in result_packet)
        self.assertTrue(isinstance(result_packet.all_timeseries, dict))
        self.assertTrue(len(result_packet.all_timeseries), 3)  # there should be 3*1=3 Timeseries there
        self.assertTrue(all(isinstance(x, Timeseries) for x in result_packet.all_timeseries.values()))

    def test_apply_one_unit_separately(self):
        # Segmenters are a good example of taking 1 Timeseries at a time
        unit1 = Segmenter(timedelta(seconds=1))
        union = ApplySeparately(unit1)

        result_packet = union._apply(self.packetA)

        self.assertEqual(len(result_packet), 1)
        self.assertTrue('timeseries' in result_packet)
        self.assertTrue(isinstance(result_packet.all_timeseries, dict))
        self.assertTrue(len(result_packet.all_timeseries), 3)  # there should be 3*1=3 Timeseries there
        self.assertTrue(all(isinstance(x, Timeseries) for x in result_packet.all_timeseries.values()))


if __name__ == '__main__':
    unittest.main()
