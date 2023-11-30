import unittest

from ltbio.biosignals.timeseries.Unit import Volt, G, DegreeCelsius, Unitless, Multiplier


class UnitTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def setUp(self) -> None:
        pass

    def test_create_unit_default_multiplier(self):
        unit = Volt()
        self.assertEqual(unit.multiplier, Multiplier.m)
        self.assertEqual(unit.prefix, 'm')

        unit = G()
        self.assertEqual(unit.multiplier, Multiplier._)
        self.assertEqual(unit.prefix, '')

        unit = DegreeCelsius()
        self.assertEqual(unit.multiplier, Multiplier._)
        self.assertEqual(unit.prefix, '')

    def test_create_unit_with_multiplier(self):
        unit = Volt(Multiplier.k)
        self.assertEqual(unit.multiplier, Multiplier.k)
        self.assertEqual(unit.prefix, 'k')

    def test_get_short(self):
        unit = Volt()
        self.assertEqual(unit.SHORT, 'V')

    def test_get_str(self):
        unit = Volt()
        self.assertEqual(str(unit), 'mV')

        unit = Unitless()
        self.assertEqual(str(unit), 'n.d.')

if __name__ == '__main__':
    unittest.main()
