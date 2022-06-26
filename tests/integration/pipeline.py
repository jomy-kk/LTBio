import unittest
from datetime import timedelta

from src.biosignals.MITDB import MITDB
from src.clinical.Epilepsy import Epilepsy
from src.clinical.Patient import Patient
from src.pipeline.Pipeline import Pipeline
from src.processing.Segmenter import Segmenter
from src.biosignals.ECG import ECG


class PipelineIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.patient = Patient(101, 'Alice', 15, (Epilepsy(), ))

    def test_pipeline_with_one_unit(self):
        # 1. Load ECG
        ecg = ECG('resources/MITDB_DAT_tests', MITDB, self.patient)
        # 2. Create pipeline units
        unit1 = Segmenter(window_length = timedelta(seconds=1))
        # 3. Create pipeline and add units
        pipeline = Pipeline(name = 'My first pipeline')
        pipeline.add(unit1)
        # 4.
        segmented_output = pipeline.applyAll(ecg)

        self.assertEqual(tuple(segmented_output.all_timeseries.keys()), ('V5', 'V2'))
        self.assertEqual(segmented_output.all_timeseries['V5'].sampling_frequency, ecg['V5'].sampling_frequency)
        self.assertEqual(segmented_output.all_timeseries['V5'].initial_datetime, ecg['V5'].initial_datetime)
        #self.assertEqual(segmented_output.all_timeseries['V5'].final_datetime, ecg['V5'].final_datetime)
        self.assertEqual(segmented_output['V5'])


if __name__ == '__main__':
    unittest.main()
