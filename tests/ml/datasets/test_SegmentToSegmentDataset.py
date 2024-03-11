import unittest
from datetime import datetime, timedelta

from numpy import ndarray, array, allclose

from ltbio.biosignals.modalities import ACC, EDA, TEMP, PPG
from ltbio.biosignals.sources import E4
from ltbio.ml.datasets import SegmentToSegmentDataset
from ltbio.processing.formaters import Segmenter


class SegmentToSegmentDatasetTestCase(unittest.TestCase):
    """
    Any BiosignalDataset needs to be tested for:
    - Its name
    - Its length
    - Its objects and targets
    - All its examples
    - Indexing operation
    - References to the Biosignals that created it
    """

    @classmethod
    def setUpClass(cls):
        # 20 minutes of each
        cls.acc = ACC('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:30:00']
        cls.eda = EDA('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:30:00']
        cls.temp = TEMP('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:30:00']
        cls.ppg = PPG('resources/E4_CSV_tests', E4)['2022-06-11 19:10:00': '2022-06-11 19:30:00']

        segmenter = Segmenter(timedelta(seconds=2))
        for name, channel in cls.temp._Biosignal__timeseries.items():
            cls.temp._Biosignal__timeseries[name] = segmenter.apply(channel)
        for name, channel in cls.acc._Biosignal__timeseries.items():
            cls.acc._Biosignal__timeseries[name] = segmenter.apply(channel)
        for name, channel in cls.eda._Biosignal__timeseries.items():
            cls.eda._Biosignal__timeseries[name] = segmenter.apply(channel)

        cls.dataset_length = 20 * 60 / 2  # = 600 segments
        cls.object_segment_length = cls.temp.sampling_frequency * 2
        cls.target_segment_length = cls.acc.sampling_frequency * 2

        cls.dataset_name = 'My first dataset'

    def verify_data(self, dataset):
        self.assertEqual(dataset.name, self.dataset_name)
        self.assertEqual(len(dataset), self.dataset_length)
        objects, targets = dataset._BiosignalDataset__objects, dataset._BiosignalDataset__targets
        self.assertTrue(isinstance(objects, ndarray))
        self.assertEqual(objects.shape, (self.dataset_length, 2, self.object_segment_length))
        self.assertTrue(isinstance(targets, ndarray))
        self.assertEqual(targets.shape, (self.dataset_length, 3, self.target_segment_length))
        self.assertTrue(isinstance(dataset.all_examples, list))
        self.assertTrue(all([(isinstance(example, tuple) and len(example) == 2) for example in dataset.all_examples]))

        first_object = array([[36.77000046, 36.77000046, 36.77000046, 36.77000046, 36.79000092,
                               36.79000092, 36.79000092, 36.79000092],
                              [39.15777206, 39.21543503, 39.23337555, 39.25515747, 39.25131607,
                               39.20518494, 39.28207016, 39.29103851]])

        first_target = array([[-22., -23., -22., -21., -21., -22., -24., -25., -25., -25., -24.,
                               -24., -26., -28., -29., -29., -28., -28., -28., -28., -28., -28.,
                               -29., -29., -29., -29., -29., -29., -28., -29., -25., -25., -28.,
                               -28., -30., -30., -30., -29., -28., -26., -23., -19., -19., -21.,
                               -25., -25., -25., -24., -22., -23., -24., -26., -28., -31., -30.,
                               -28., -28., -28., -28., -27., -27., -26., -25., -24.],
                              [61., 60., 59., 56., 52., 50., 50., 51., 56., 59., 60.,
                               62., 65., 69., 69., 64., 60., 59., 58., 58., 57., 57.,
                               57., 58., 58., 60., 61., 61., 61., 62., 59., 58., 55.,
                               55., 58., 66., 68., 63., 58., 55., 55., 58., 61., 62.,
                               60., 58., 54., 52., 51., 51., 51., 52., 54., 59., 61.,
                               65., 69., 68., 61., 55., 56., 61., 60., 57.],
                              [14., 14., 14., 13., 12., 12., 11., 11., 12., 13., 11.,
                               9., 8., 7., 6., 4., 4., 4., 3., 3., 3., 3.,
                               2., 3., 5., 8., 11., 15., 17., 24., 23., 24., 26.,
                               25., 22., 21., 20., 18., 16., 12., 10., 7., 6., 5.,
                               3., 3., 5., 6., 8., 13., 14., 15., 16., 19., 19.,
                               20., 21., 20., 17., 14., 10., 9., 7., 4.]])

        self.assertTrue(allclose(dataset[0][0], first_object))
        self.assertTrue(allclose(dataset[0][1], first_target))

    def test_create_dataset_from_biosignals(self):
        # Useful to specify an order
        x, y, z = self.acc['x'], self.acc['y'], self.acc['z']
        # Given a dataset with 2 Timeseries as objects and 3 Timeseries as targets
        dataset = SegmentToSegmentDataset(object=(self.temp, self.eda), target=(x, y, z), name=self.dataset_name)
        # Assert
        self.verify_data(dataset)
        # And references to Biosignals
        self.assertTrue('object', 'target' in dataset.biosignals.keys())
        self.assertTrue(self.temp in dataset.biosignals['object'])
        self.assertTrue(self.eda in dataset.biosignals['object'])
        self.assertTrue(x in dataset.biosignals['target'])
        self.assertTrue(y in dataset.biosignals['target'])
        self.assertTrue(z in dataset.biosignals['target'])

    def test_create_dataset_from_timeseries(self):
        # Given a dataset with 2 Timeseries as objects and 3 Timeseries as targets
        self.temp._get_channel('temp').name = 'temp'
        self.eda._get_channel('eda').name = 'eda'
        self.acc._get_channel('x').name = 'x'
        self.acc._get_channel('y').name = 'y'
        self.acc._get_channel('z').name = 'z'
        dataset = SegmentToSegmentDataset(object=(self.temp._get_channel('temp'), self.eda._get_channel('eda')),
                                          target=(self.acc._get_channel('x'), self.acc._get_channel('y'), self.acc._get_channel('z')),
                                          name=self.dataset_name)
        # Assert
        self.verify_data(dataset)

    def test_create_empty_dataset_raises_error(self):
        with self.assertRaises(AssertionError):
            dataset = SegmentToSegmentDataset(object=(), target=(self.acc, ))
        with self.assertRaises(AssertionError):
            dataset = SegmentToSegmentDataset(object=(self.temp,), target=())

    def test_create_dataset_with_wrong_types_raises_error(self):
        with self.assertRaises(ValueError):
            dataset = SegmentToSegmentDataset(object=0, target=(self.acc, ))
        with self.assertRaises(ValueError):
            dataset = SegmentToSegmentDataset(object=(self.temp,), target='string')
        with self.assertRaises(ValueError):
            dataset = SegmentToSegmentDataset(object=(0, ), target=(self.acc, ))
        with self.assertRaises(ValueError):
            dataset = SegmentToSegmentDataset(object=(self.temp,), target=('string', ))

    def test_create_dataset_with_different_sampling_frequencies_raises_error(self):
        with self.assertRaises(AssertionError):
            dataset = SegmentToSegmentDataset(object=(self.acc, self.temp), target=(self.acc, ))
        with self.assertRaises(AssertionError):
            dataset = SegmentToSegmentDataset(object=(self.acc, ), target=(self.acc, self.temp))

    def test_create_dataset_with_different_domains_raises_error(self):
        """ allowed, for now # FIXME
        # PPG has the same initial and final datetimes, but wasn't segmented
        with self.assertRaises(AssertionError):
            dataset = SegmentToSegmentDataset(object=(self.acc, self.ppg), target=(self.acc, ))
        with self.assertRaises(AssertionError):
            dataset = SegmentToSegmentDataset(object=(self.acc, ), target=(self.acc, self.ppg))
        with self.assertRaises(AssertionError):
            dataset = SegmentToSegmentDataset(object=(self.acc, ), target=(self.ppg, ))
        """

if __name__ == '__main__':
    unittest.main()
