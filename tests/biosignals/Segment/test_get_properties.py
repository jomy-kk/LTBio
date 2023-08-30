import unittest

from resources.segments import get_segment, small_samples_1


class SegmentGetPropertiesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.small_segment = get_segment('small', 1)

    def test_get_samples(self):
        samples = self.small_segment.samples
        # Assert '.samples' is a view of the actual stored array in Segment
        self.assertFalse(samples.flags['OWNDATA'])
        self.assertTrue(samples.base is self.small_segment._Segment__samples)
        # Assert the content is the same
        self.assertTrue(all(samples == small_samples_1))

        
if __name__ == '__main__':
    unittest.main()
