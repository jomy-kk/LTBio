import unittest
from datetime import timedelta, datetime

from src.features.FeatureExtractor import FeatureExtractor
from src.features.FeatureSelector import FeatureSelector
from src.features.Features import TimeFeatures
from src.ml.models.SkLearnModel import SkLearnModel
from src.ml.trainers.SupervisedTrainConditions import SupervisedTrainConditions
from src.ml.trainers.SupervisingTrainer import SupervisingTrainer
from src.pipeline.Input import Input
from src.pipeline.Packet import Packet
from src.biosignals.ECG import ECG
from src.biosignals.EDA import EDA
from src.biosignals.Timeseries import Timeseries
from src.pipeline.Pipeline import Pipeline
from src.processing.Segmenter import Segmenter
from src.biosignals.ACC import ACC


class PipelineIntegrationTests(unittest.TestCase):

    datetime: datetime
    ts1: Timeseries
    ts2: Timeseries
    ts3: Timeseries
    ts3_segmented: Timeseries
    ts1_label: str
    ts2_label: str
    ts3_label: str
    ecg_label: str
    ecg: ECG
    acc: ACC
    eda: EDA

    @classmethod
    def setUpClass(cls) -> None:
        cls.datetime = datetime.now()
        cls.ts1 = Timeseries([0, 1, 2, 3, 4, 7, 5], cls.datetime, 1)
        cls.ts2 = Timeseries([0, 1, 2, 3, 4, 5, 8], cls.datetime, 1)
        cls.ts3 = Timeseries([0, 1, 2, 3, 4, 5, 6], cls.datetime, 1)
        cls.ts3_segmented = Timeseries.withDiscontiguousSegments({
            cls.datetime: [0, 1],
            cls.datetime + timedelta(seconds=2): [2, 3],
            cls.datetime + timedelta(seconds=4): [4, 5],
        }, 1.0)
        cls.ecg_label = 'V5'
        cls.ts1_label = 'x'
        cls.ts2_label = 'y'
        cls.ts3_label = 'z'

        cls.ecg = ECG({cls.ecg_label: cls.ts1})
        cls.eda = EDA({cls.ts1_label: cls.ts1, cls.ts2_label: cls.ts2})
        cls.acc = ACC({cls.ts1_label: cls.ts1, cls.ts2_label: cls.ts2, cls.ts3_label: cls.ts3})

    def test_pipeline_with_one_single_unit(self):
        # 1. Create pipeline units
        unit1 = Segmenter(window_length = timedelta(seconds=2))

        # 2. Create pipeline and add units
        pipeline = Pipeline(name = 'My first pipeline')
        pipeline.add(unit1)

        # 3. Apply units
        segmented_output = pipeline.applyAll(self.acc)

        self.assertTrue(Packet.TIMESERIES_LABEL in segmented_output)
        timeseries = segmented_output[Packet.TIMESERIES_LABEL]
        self.assertTrue(self.ts1_label in timeseries)
        self.assertTrue(self.ts2_label in timeseries)
        self.assertTrue(self.ts3_label in timeseries)

    def test_pipeline_with_two_single_units(self):
        # 1. Create pipeline units
        unit1 = Segmenter(window_length = timedelta(seconds=2))
        unit2 = FeatureExtractor((TimeFeatures.mean, TimeFeatures.variance))

        # 2. Create pipeline and add units
        pipeline = Pipeline(name = 'My first pipeline')
        pipeline.add(unit1)
        pipeline.add(unit2)

        # 3. Apply units
        output = pipeline.applyAll(self.acc)

        self.assertTrue(Packet.TIMESERIES_LABEL in output)
        timeseries = output[Packet.TIMESERIES_LABEL]
        self.assertTrue(self.ts1_label + ':mean' in timeseries)
        self.assertTrue(self.ts1_label + ':variance' in timeseries)
        self.assertTrue(self.ts2_label + ':mean' in timeseries)
        self.assertTrue(self.ts2_label + ':variance' in timeseries)
        self.assertTrue(self.ts3_label + ':mean' in timeseries)
        self.assertTrue(self.ts3_label + ':variance' in timeseries)

    def test_pipeline_with_three_single_units(self):
        # 1. Create pipeline units
        unit1 = Segmenter(window_length = timedelta(seconds=2))
        unit2 = FeatureExtractor((TimeFeatures.mean, TimeFeatures.variance))
        unit3 = FeatureSelector(lambda x: x[0] > 0.4)  # random function; it will only select 'mean'

        # 2. Create pipeline and add units
        pipeline = Pipeline(name = 'My first pipeline')
        pipeline.add(unit1)
        pipeline.add(unit2)
        pipeline.add(unit3)

        # 3. Apply units
        output = pipeline.applyAll(self.acc)

        self.assertTrue(Packet.TIMESERIES_LABEL in output)
        timeseries = output[Packet.TIMESERIES_LABEL]
        self.assertTrue(self.ts1_label + ':mean' in timeseries)
        self.assertTrue(self.ts2_label + ':mean' in timeseries)
        self.assertTrue(self.ts3_label + ':mean' in timeseries)

    def test_pipeline_with_five_single_units(self):
        # 1. Create pipeline units
        unit1 = Segmenter(window_length = timedelta(seconds=2))
        unit2 = FeatureExtractor((TimeFeatures.mean, TimeFeatures.variance))
        unit3 = FeatureSelector(lambda x: x[0] > 0.4)  # random function; it will only select 'mean'

        target = Timeseries([0, 0, 1], self.datetime, 1.)
        unit4 = Input('target', target)
        from sklearn.ensemble import GradientBoostingRegressor
        model = SkLearnModel(GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=5))
        conditions1 = SupervisedTrainConditions(loss='squared_error', learning_rate=0.02, test_size=0.2, shuffle=False)
        unit5 = SupervisingTrainer(model, (conditions1, ), name='GBR Trainer')

        # 2. Create pipeline and add units
        pipeline = Pipeline(name = 'My first pipeline')
        pipeline.add(unit1)
        pipeline.add(unit2)
        pipeline.add(unit3)
        pipeline.add(unit4)
        pipeline.add(unit5)

        # 3. Apply units
        output = pipeline.applyAll(self.acc)
        print(output)

        self.assertTrue(Packet.TIMESERIES_LABEL in output)
        self.assertTrue('target' in output)
        self.assertTrue('results' in output)


if __name__ == '__main__':
    unittest.main()
