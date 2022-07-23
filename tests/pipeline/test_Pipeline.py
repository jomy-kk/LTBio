import unittest
from datetime import timedelta, datetime

from ltbio.biosignals.modalities.ECG import ECG
from ltbio.biosignals.timeseries.Timeseries import Timeseries
from ltbio.decision.DecisionMaker import DecisionMaker
from ltbio.decision.NAryDecision import NAryDecision
from ltbio.features.FeatureExtractor import FeatureExtractor
from ltbio.features.Features import TimeFeatures
from ltbio.pipeline.Pipeline import Pipeline
from ltbio.pipeline.PipelineUnit import ApplySeparately
from ltbio.processing.formaters.Segmenter import Segmenter


class PipelineTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.unit1 = Segmenter(timedelta(seconds=1), name='A')
        cls.unit2 = Segmenter(timedelta(seconds=1), name='B')
        cls.unit3 = DecisionMaker(NAryDecision(lambda x: True), name='C')
        cls.unit4 = FeatureExtractor((TimeFeatures.mean, TimeFeatures.variance), name='D')

        cls.sf = 1
        cls.initial1 = datetime.now()
        cls.samples1 = [506.0, 501.0, 497.0, 374.5, 383.4, 294.2]
        cls.samples2 = [502.0, 505.0, 505.0, 924.3, 293.4, 383.5]
        cls.samples3 = [527.0, 525.0, 525.0, 849.2, 519.5, 103.4]
        cls.ts1 = Timeseries(cls.samples1, cls.initial1, cls.sf)
        cls.ts2 = Timeseries(cls.samples2, cls.initial1, cls.sf)
        cls.ecg1 = ECG({"a": cls.ts1, "b": cls.ts2})

    def test_create_pipeline(self):
        name = 'My first pipeline'
        pipeline = Pipeline(name='My first pipeline')

        self.assertEqual(pipeline.name, name)
        self.assertEqual(len(pipeline), 0)
        with self.assertRaises(AttributeError):
            x = pipeline.current_step

    def test_add_unit_to_pipeline(self):
        pipeline = Pipeline()
        self.assertEqual(len(pipeline), 0)
        pipeline.add(self.unit1)
        self.assertEqual(len(pipeline), 1)
        pipeline.add(self.unit2)
        self.assertEqual(len(pipeline), 2)

    """
    def test_add_non_consistent_unit_to_pipeline(self):
        pipeline = Pipeline()
        pipeline.add(self.unit3)  # Unit 3 returns an integer ...
        self.assertEqual(len(pipeline), 1)
        with self.assertRaises(AssertionError):
            pipeline.add(self.unit1)  # ... but Unit 1 requires a Timeseries
        self.assertEqual(len(pipeline), 1)
    """

    def test_add_union_and_unit_to_pipeline(self):
        pipeline = Pipeline()
        self.assertEqual(len(pipeline), 0)
        union = ApplySeparately((self.unit1, self.unit2))
        pipeline.add(union)
        self.assertEqual(len(pipeline), 1)
        pipeline.add(self.unit4)
        self.assertEqual(len(pipeline), 2)

    def test_add_unit_and_union_to_pipeline(self):
        pipeline = Pipeline()
        self.assertEqual(len(pipeline), 0)
        pipeline.add(self.unit4)
        self.assertEqual(len(pipeline), 1)
        union = ApplySeparately((self.unit1, self.unit2))
        pipeline.add(union)
        self.assertEqual(len(pipeline), 2)

    def test_fast_pipeline(self):
        pipeline = self.unit1 >> self.unit2
        self.assertEqual(len(pipeline), 2)

        pipeline = self.unit1 >> ApplySeparately((self.unit2, self.unit3)) >> self.unit4
        self.assertEqual(len(pipeline), 3)

        pipeline = ApplySeparately((self.unit2, self.unit3)) >> self.unit1 >> self.unit4
        self.assertEqual(len(pipeline), 3)

    def test_create_first_packet(self):
        # Given a pipeline with at least 1 unit, that hasn't started yet
        pipeline = Pipeline()
        pipeline.add(self.unit1)
        # Let's suppose we want to apply it to an ECG with 2 channels (= 2 Timeseries)
        pipeline._Pipeline__biosignals = (self.ecg1, )
        # Before starting, the first packet needs to be created
        pipeline._Pipeline__create_first_packet()

        # Check if it was created correctly
        first_packet = pipeline.current_packet
        self.assertEqual(len(first_packet), 1)
        self.assertEqual(first_packet.contents, {'timeseries': dict})
        self.assertEqual(first_packet.timeseries, {"a": self.ts1, "b": self.ts2})


if __name__ == '__main__':
    unittest.main()
