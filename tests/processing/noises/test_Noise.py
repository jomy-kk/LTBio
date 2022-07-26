import unittest
from datetime import timedelta

from numpy import ndarray

from ltbio.processing.noises.GaussianNoise import GaussianNoise

class NoiseTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.sf = 50.

    def test_get_properties(self):
        noise = GaussianNoise(0, 1, self.sf, 'My first noise')
        self.assertEqual(noise.sampling_frequency, self.sf)
        self.assertEqual(noise.name, 'My first noise')

    def test_set_name(self):
        noise = GaussianNoise(0, 1, self.sf, 'My first noise')
        self.assertEqual(noise.name, 'My first noise')
        noise.name = 'Second name'
        self.assertEqual(noise.name, 'Second name')

    def test_resample(self):
        noise = GaussianNoise(0, 1, self.sf)
        self.assertEqual(noise.sampling_frequency, self.sf)
        noise.resample(200.)
        self.assertEqual(noise.sampling_frequency, 200.)

    def test_get_samples_before_generating_gives_error(self):
        noise = GaussianNoise(0, 1, self.sf)
        with self.assertRaises(AttributeError):
            x = noise.samples

    def test_indexing(self):
        noise = GaussianNoise(0, 1, self.sf)

        # Must be equivalent methods
        x = noise[timedelta(minutes=2)]
        y = noise[2]
        self.assertEqual(len(x), len(y))

        # Must fail with other types
        with self.assertRaises(IndexError):
            z = noise["z"]

    def test_get_samples_after_generating(self):
        noise = GaussianNoise(0, 1, self.sf)
        x = noise[timedelta(minutes=2)]
        self.assertTrue(isinstance(noise.samples, ndarray))
        self.assertTrue(all(x == noise.samples))

    def test_plot_example(self):
        noise = GaussianNoise(0, 1, self.sf)
        noise.plot()

    def test_plot_last_samples(self):
        noise = GaussianNoise(0, 1, self.sf)
        x = noise[timedelta(minutes=2)]
        noise.plot()


if __name__ == '__main__':
    unittest.main()
