import unittest
from datetime import datetime, timedelta

from features.FeatureExtractor import FeatureExtractor
from features.Features import TimeFeatures
from src.pipeline.PipelineUnit import ApplyTogether, ApplySeparately
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
    ts3: Timeseries
    ts3_segmented: Timeseries
    ts1_label: str
    ts2_label: str
    ts3_label: str
    packetA: Packet
    packetB: Packet
    packetC: Packet
    packetD: Packet
    packetE: Packet

    @classmethod
    def setUpClass(cls) -> None:
        cls.datetime = datetime.now()
        cls.ts1 = Timeseries([Timeseries.Segment([0, 1, 2, 3, 4], cls.datetime, 1), ], True, 1)
        cls.ts2 = Timeseries([Timeseries.Segment([0, 1, 2, 3, 4, 5], cls.datetime, 1), ], True, 1)
        cls.ts3 = Timeseries([Timeseries.Segment([0, 1, 2, 3, 4, 5, 6], cls.datetime, 1), ], True, 1)
        cls.ts3_segmented = Timeseries([
            Timeseries.Segment([0, 1], cls.datetime, 1),
            Timeseries.Segment([2, 3], cls.datetime, 1),
            Timeseries.Segment([4, 5], cls.datetime, 1),
                                        ], True, 1, equally_segmented=True)
        cls.ts1_label = 'ts1'
        cls.ts2_label = 'ts2'
        cls.ts3_label = 'ts3'

        cls.packetA = Packet(timeseries={cls.ts1_label: cls.ts1})
        cls.packetB = Packet(timeseries={cls.ts1_label: cls.ts1, cls.ts2_label: cls.ts2})
        cls.packetC = Packet(timeseries={cls.ts1_label: cls.ts1, cls.ts2_label: cls.ts2, cls.ts3_label: cls.ts3})
        cls.packetD = Packet(timeseries={cls.ts1_label: cls.ts3_segmented, cls.ts2_label: cls.ts3_segmented})
        cls.packetE = Packet(timeseries={cls.ts3_label: cls.ts3_segmented})

    def test_get_name(self):
        unit1 = FeatureSelector(lambda x: True)
        union = ApplyTogether(unit1, name='Test Name')
        self.assertEqual(union.name, 'Test Name')

    """
    Let:
    - K be the number of units in the Union
    - I be the number of Timeseries that enter a Union
    - O be the number of Timeseries that leave a Union
    - i(k) be the function that for each unit k, gives the number of input Timeseries it takes
    - o(k) be the function that for each unit k, gives the number of output Timeseries it produces
    
    where:
              | I , if applied together                   
    i (I) = --|                                     o (k) = Given by the apply method
              | 1 , if applied separately            
    
    Given I Timeseries at the beginning of a Union, the number of Timeseries that come at its end, O, is given by:
    
                    | sum_k^K [ o(k) ]         , if applied together
    O (k, K, I) = --|
                    | I * sum_k^K [ o(k) ]     , if applied separately
    """

    @staticmethod
    def O(o:tuple, I, together=False, separately=False):
        assert together or separately
        if together:
            return sum([x for x in o])
        if separately:
            return I * sum([x for x in o])

    def verify_n_out_ts(self, packet:Packet, expected:int):
        self.assertTrue('timeseries' in packet)
        if expected > 1:
            all_timeseries = packet.timeseries
            self.assertTrue(isinstance(all_timeseries, dict))
            self.assertEqual(len(all_timeseries), expected)
            self.assertTrue(all(isinstance(x, Timeseries) for x in all_timeseries.values()))
        else:
            self.assertTrue(packet.has_single_timeseries)


    def verify_out_ts_labels(self, packet:Packet, expected:tuple):
        if len(expected) > 1:
            all_timeseries = packet.timeseries
            self.assertTrue(set(all_timeseries.keys()) == set(expected))

    """
    The tests below cover all combinations of K, I, and o(k), assuming o is equal for all k in {1, ..., K}.
    """

    def test_K_1_I_1_o_1(self):
        """ K = 1, I = 1, o = 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((1, ), 1, together=True)
        unit = Segmenter(timedelta(seconds=1))
        union = ApplyTogether(unit)
        with self.assertRaises(AssertionError):
            union._apply(self.packetA)  # There's nothing to use case for this: Apply 1 Timeseries through 1 Unit.

    def test_K_1_I_1_o_gt1(self):
        """ K = 1, I = 1, o > 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((1,), 1, together=True)
        unit = Segmenter(timedelta(seconds=1))
        union = ApplyTogether(unit)
        with self.assertRaises(AssertionError):
            union._apply(self.packetA)  # There's nothing to use case for this: Apply 1 Timeseries through 1 Unit.

    def test_K_1_I_gt1_o_1_together(self):
        """ K = 1, I > 1, o = 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((1,), 3, together=True)
        unit = FeatureSelector(lambda x: len(x)>6)  # From the 3, only 1 Timeseries will hold True for this condition
        union = ApplyTogether(unit)
        resulting_packet = union._apply(self.packetC)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, (self.ts3_label, ))
        # labels should remain the same, since there's only 1 Unit

    def test_K_1_I_gt1_o_1_separate(self):
        """ K = 1, I > 1, o = 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((1,), 3, separately=True)
        unit = Segmenter(timedelta(seconds=1))  # Segments 1 Timeseries at a time
        union = ApplySeparately(unit)
        resulting_packet = union._apply(self.packetC)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, (self.ts1_label, self.ts2_label, self.ts3_label))
        # labels should remain the same, since there's only 1 Unit

    def test_K_1_I_gt1_o_gt1_together(self):
        """ K = 1, I > 1, o > 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((2,), 3, together=True)
        unit = FeatureSelector(lambda x: len(x)>5)  # From the 3, only 2 Timeseries will hold True for this condition
        union = ApplyTogether(unit)
        resulting_packet = union._apply(self.packetC)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, (self.ts2_label, self.ts3_label))
        # labels should remain the same, since there's only 1 Unit

    def test_K_1_I_gt1_o_gt1_separate(self):
        """ K = 1, I > 1, o > 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((3,), 2, separately=True)
        unit = FeatureExtractor((TimeFeatures.mean, TimeFeatures.deviation, TimeFeatures.variance))  # For each Timeseries, it will output 3 feature Timeseries
        union = ApplySeparately(unit)
        resulting_packet = union._apply(self.packetD)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, (self.ts1_label+':mean', self.ts1_label+':deviation', self.ts1_label+':variance', self.ts2_label+':mean', self.ts2_label+':deviation', self.ts2_label+':variance'))  # FIXME

    def test_K_gt1_I_1_o_1(self):
        """ K > 1, I = 1, o = 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((1, 1), 1, together=True)
        unit1 = Segmenter(timedelta(seconds=1), name='Seg')
        unit2 = FeatureSelector(lambda x: True, name='Sel')
        union = ApplyTogether((unit1, unit2))
        resulting_packet = union._apply(self.packetA)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, ('Seg', 'ts1'))

        # This should do the same
        union = ApplySeparately((unit1, unit2))
        resulting_packet = union._apply(self.packetA)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, ('Seg:ts1', 'Sel:ts1'))

    def test_K_gt1_I_1_o_gt1(self):
        """ K > 1, I = 1, o > 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((2, 2), 1, together=True)
        unit1 = FeatureExtractor((TimeFeatures.mean, TimeFeatures.variance), name='A')
        unit2 = FeatureExtractor((TimeFeatures.deviation, TimeFeatures.variance), name='B')
        union = ApplyTogether((unit1, unit2))
        resulting_packet = union._apply(self.packetE)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, ('mean', 'A:variance', 'deviation', 'B:variance'))

        # This should do the same
        union = ApplySeparately((unit1, unit2))
        resulting_packet = union._apply(self.packetE)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, ('mean', 'A:variance', 'deviation', 'B:variance'))

    def test_K_gt1_I_gt1_o_1_together(self):
        """ K > 1, I > 1, o = 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((1, 1), 3, together=True)
        unit1 = FeatureSelector(lambda x: len(x)>6, name='A')  # From 3, only 1 Timeseries will be selected.
        unit2 = FeatureSelector(lambda x: len(x)>6, name='B')
        union = ApplyTogether((unit1, unit2))
        resulting_packet = union._apply(self.packetC)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, ('A:'+self.ts3_label, 'B:'+self.ts3_label))

    def test_K_gt1_I_gt1_o_1_separate(self):
        """ K > 1, I > 1, o = 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((1, 1), 3, separately=True)
        unit1 = Segmenter(timedelta(seconds=1), name='A')
        unit2 = Segmenter(timedelta(seconds=1), name='B')
        union = ApplySeparately((unit1, unit2))
        resulting_packet = union._apply(self.packetC)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        print(resulting_packet.timeseries.keys())
        self.verify_out_ts_labels(resulting_packet, ('A:' + self.ts1_label, 'A:' + self.ts2_label, 'A:' + self.ts3_label, 'B:' + self.ts1_label, 'B:' + self.ts2_label, 'B:' + self.ts3_label))

    def test_K_gt1_I_gt1_o_gt1_together(self):
        """ K > 1, I > 1, o > 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((2, 2), 3, together=True)
        unit1 = FeatureSelector(lambda x: len(x) > 5, name='A')  # From 3, only 2 Timeseries will be selected.
        unit2 = FeatureSelector(lambda x: len(x) > 5, name='B')
        union = ApplyTogether((unit1, unit2))
        resulting_packet = union._apply(self.packetC)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, ('A:' + self.ts2_label, 'A:' + self.ts3_label, 'B:' + self.ts2_label, 'B:' + self.ts3_label))

    def test_K_gt1_I_gt1_o_gt1_separate(self):
        """ K > 1, I > 1, o > 1 """
        n_out_ts = PipelineUnitsUnionTestCase.O((2, 2), 2, separately=True)
        unit1 = FeatureExtractor((TimeFeatures.mean, TimeFeatures.variance), name='A')
        unit2 = FeatureExtractor((TimeFeatures.deviation, TimeFeatures.variance), name='B')
        union = ApplySeparately((unit1, unit2))
        resulting_packet = union._apply(self.packetD)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, (self.ts1_label+':mean',
                                                     'A:'+self.ts1_label+':variance',
                                                     self.ts2_label+':mean',
                                                     'A:' + self.ts2_label + ':variance',
                                                     self.ts1_label + ':deviation',
                                                     'B:' + self.ts1_label + ':variance',
                                                     self.ts2_label + ':deviation',
                                                     'B:' + self.ts2_label + ':variance',
                                                     ))

    """
    The tests below cover all combinations of K, I, and o(k) varying for all k in {1, ..., K}.
    """

    def test_K_gt1_I_gt1_o_gt1_varying_together(self):
        """ K > 1, I > 1, o > 1 varying"""
        n_out_ts = PipelineUnitsUnionTestCase.O((3, 2), 3, together=True)
        unit1 = FeatureSelector(lambda x: len(x) > 4, name='A')  # 3 Timeseries will be selected.
        unit2 = FeatureSelector(lambda x: len(x) > 5, name='B')  # 2 Timeseries will be selected.
        union = ApplyTogether((unit1, unit2))
        resulting_packet = union._apply(self.packetC)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, (self.ts1_label, 'A:' + self.ts2_label, 'A:' + self.ts3_label, 'B:' + self.ts2_label, 'B:' + self.ts3_label))

    def test_K_gt1_I_gt1_o_ge1_varying_separate(self):
        """ K > 1, I > 1, o >= 1 varying"""
        n_out_ts = PipelineUnitsUnionTestCase.O((2, 1), 2, separately=True)
        unit1 = FeatureExtractor((TimeFeatures.mean, TimeFeatures.variance), name='A')
        unit2 = FeatureExtractor((TimeFeatures.deviation, ), name='B')
        union = ApplySeparately((unit1, unit2))
        resulting_packet = union._apply(self.packetD)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, (self.ts1_label + ':mean',
                                                     self.ts1_label + ':variance',
                                                     self.ts2_label + ':mean',
                                                     self.ts2_label + ':variance',
                                                     self.ts1_label + ':deviation',
                                                     self.ts2_label + ':deviation'
                                                     ))

    def test_K_gt1_I_1_o_gt1_varying(self):
        """ K > 1, I = 1, o > 1 varying"""
        n_out_ts = PipelineUnitsUnionTestCase.O((3, 2), 1, together=True)
        unit1 = FeatureExtractor((TimeFeatures.mean, TimeFeatures.variance, TimeFeatures.deviation), name='A')
        unit2 = FeatureExtractor((TimeFeatures.variance, TimeFeatures.deviation), name='B')
        union = ApplyTogether((unit1, unit2))
        resulting_packet = union._apply(self.packetE)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, ('mean', 'A:variance', 'A:deviation', 'B:variance', 'B:deviation'))

        # This should do the same
        union = ApplySeparately((unit1, unit2))
        resulting_packet = union._apply(self.packetE)
        self.verify_n_out_ts(resulting_packet, n_out_ts)
        self.verify_out_ts_labels(resulting_packet, ('mean', 'A:variance', 'A:deviation', 'B:variance', 'B:deviation'))


if __name__ == '__main__':
    unittest.main()
