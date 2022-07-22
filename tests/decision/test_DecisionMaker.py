import unittest
from datetime import datetime

from biosignals.timeseries.Timeseries import Timeseries
from decision.BinaryDecision import BinaryDecision
from decision.DecisionMaker import DecisionMaker
from decision.NAryDecision import NAryDecision


class DecisionMakerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.ts1 = Timeseries([0, 1, 2, 3, 4], datetime.now(), 1)
        cls.ts2 = Timeseries([0, 1, 2, 3, 4, 5], datetime.now(), 1)

        def binary_decision_function(timeseries: Timeseries) -> bool:
            return len(timeseries) > 5

        cls.binary_decision = BinaryDecision(binary_decision_function)

        def nary_decision_function(timeseries: Timeseries) -> int:
            if len(timeseries) <= 5:
                return 0
            if 5 < len(timeseries) <= 15:
                return 1
            if len(timeseries) > 15:
                return 2

        cls.nary_decision = NAryDecision(nary_decision_function)


    def test_create_decision_maker(self):
        name = "My first decision maker"
        maker = DecisionMaker(self.binary_decision, name=name)
        self.assertEqual(maker.name, name)

    def test_with_binary_decision(self):
        maker = DecisionMaker(self.binary_decision)
        self.assertFalse(maker.apply(self.ts1))
        self.assertTrue(maker.apply(self.ts2))

    def test_with_nary_decision(self):
        maker = DecisionMaker(self.nary_decision)
        self.assertEqual(maker.apply(self.ts1), 0)
        self.assertEqual(maker.apply(self.ts2), 1)


if __name__ == '__main__':
    unittest.main()
