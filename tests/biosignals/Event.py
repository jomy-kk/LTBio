import unittest
from datetime import datetime, timedelta

from biosignals.ECG import ECG
from biosignals.Event import Event
from biosignals.Timeseries import Timeseries


class EventTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.datetime1 = datetime(2022, 6, 1, 16, 0, 2)
        cls.datetime2 = datetime(2022, 6, 1, 16, 0, 4)
        cls.datetime1_in_str = '2022-06-01 16:00:02'
        cls.name1 = 'My first Event'
        cls.name2 = 'My other Event'

    def setUp(self) -> None:
        self.timeseries = Timeseries([Timeseries.Segment([1., 2., 3.], self.datetime1, 1.), ], True, 1.)
        self.biosignal = ECG({'x': self.timeseries})

    def test_create_event(self):
        event = Event(self.name1, self.datetime1)  # with datetime object as onset
        self.assertEqual(event.onset, self.datetime1)
        self.assertEqual(event.name, self.name1)

        event = Event(self.name1, self.datetime1_in_str)  # with str
        self.assertEqual(event.onset, self.datetime1)
        self.assertEqual(event.name, self.name1)

        event = Event(self.name1, self.datetime1, self.datetime2)  # with onset and offset
        self.assertEqual(event.onset, self.datetime1)
        self.assertEqual(event.offset, self.datetime2)
        self.assertEqual(event.name, self.name1)

        event = Event(self.name1, offset=self.datetime2)  # with just offset
        self.assertEqual(event.offset, self.datetime2)
        self.assertEqual(event.name, self.name1)

    def test_not_giving_onset_nor_offset_raises_error(self):
        with self.assertRaises(AssertionError):
            event = Event(self.name1)

    def test_giving_offset_after_onset_raises_error(self):
        with self.assertRaises(AssertionError):
            event = Event(self.name1, self.datetime2, self.datetime1)

    def test_get_duration(self):
        event = Event(self.name1, self.datetime1, self.datetime2)
        self.assertEqual(event.duration, timedelta(seconds=2))

        # raises error if it's not a period
        event = Event(self.name1, self.datetime1)
        with self.assertRaises(AttributeError):
            x = event.duration

    def test_get_onset_and_offset(self):
        # onset
        event = Event(self.name1, onset=self.datetime1)
        self.assertEqual(event.onset, self.datetime1)
        with self.assertRaises(AttributeError):
            x = event.offset

        # offset
        event = Event(self.name1, offset=self.datetime1)
        self.assertEqual(event.offset, self.datetime1)
        with self.assertRaises(AttributeError):
            x = event.onset

    def test_two_events_at_same_datetime_are_not_equal(self):
        event1 = Event(self.name1, self.datetime1)
        event2 = Event(self.name2, self.datetime1)
        self.assertTrue(event1 == event1)
        self.assertTrue(event2 == event2)
        self.assertTrue(event1 != event2)

    def test_two_events_at_same_datetime_with_same_name_are_equal(self):
        event1 = Event(self.name1, self.datetime1)
        event2 = Event(self.name1, self.datetime1)
        self.assertTrue(event1 == event1)
        self.assertTrue(event2 == event2)
        self.assertTrue(event1 == event2)

    def test_one_event_comes_before_another(self):
        event1 = Event(self.name1, self.datetime1)
        event2 = Event(self.name2, self.datetime1 - timedelta(seconds=1))
        self.assertTrue(event2 < event1)
        self.assertTrue(event2 <= event1)
        self.assertTrue(event1 > event2)
        self.assertTrue(event1 >= event2)

    def test_associate_one_event_to_timeseries(self):
        event1 = Event(self.name1, self.datetime1)
        self.timeseries.associate(event1)
        self.assertTrue(self.name1 in self.timeseries)
        self.assertFalse(self.name2 in self.timeseries)

    def test_associate_multiple_events_to_timeseries(self):
        event1 = Event(self.name1, self.datetime1)
        event2 = Event(self.name2, self.datetime2)
        self.timeseries.associate((event1, event2))
        self.assertTrue(self.name1 in self.timeseries)
        self.assertTrue(self.name2 in self.timeseries)

    def test_associate_events_with_new_keys_to_timeseries(self):
        event1 = Event(self.name1, self.datetime1)
        event2 = Event(self.name2, self.datetime2)
        self.timeseries.associate({'a': event1, 'b': event2})
        self.assertTrue('a' in self.timeseries)
        self.assertTrue('b' in self.timeseries)
        self.assertFalse(self.name1 in self.timeseries)
        self.assertFalse(self.name2 in self.timeseries)

    def test_associate_events_with_same_name_raises_error(self):
        event1 = Event(self.name1, self.datetime1)
        event2 = Event(self.name1, self.datetime2)
        self.timeseries.associate(event1)
        with self.assertRaises(NameError):
            self.timeseries.associate(event2)
        self.assertTrue(self.name1 in self.timeseries)
        self.assertFalse(self.name2 in self.timeseries)

    def test_associate_event_out_of_timeseries_domain_raises_error(self):
        event1 = Event(self.name1, self.datetime1 + timedelta(seconds=4))
        with self.assertRaises(ValueError):
            self.timeseries.associate(event1)

    def test_get_ordered_events(self):
        event2 = Event(self.name2, self.datetime2)
        event1 = Event(self.name1, self.datetime1)
        self.assertTrue(event2 > event1)
        self.timeseries.associate((event2, event1))
        events = self.timeseries.events
        self.assertEqual(events, (event1, event2))

    def test_associate_one_event_to_biosignal(self):
        event1 = Event(self.name1, self.datetime1)
        self.biosignal.associate(event1)
        self.assertTrue(self.name1 in self.biosignal)
        self.assertFalse(self.name2 in self.biosignal)

    def test_associate_multiple_events_to_biosignal(self):
        event1 = Event(self.name1, self.datetime1)
        event2 = Event(self.name2, self.datetime2)
        self.biosignal.associate((event1, event2))
        self.assertTrue(self.name1 in self.biosignal)
        self.assertTrue(self.name2 in self.biosignal)

    def test_associate_events_with_new_keys_to_biosignal(self):
        event1 = Event(self.name1, self.datetime1)
        event2 = Event(self.name2, self.datetime2)
        self.biosignal.associate({'a': event1, 'b': event2})
        self.assertTrue('a' in self.biosignal)
        self.assertTrue('b' in self.biosignal)
        self.assertFalse(self.name1 in self.biosignal)
        self.assertFalse(self.name2 in self.biosignal)

if __name__ == '__main__':
    unittest.main()
