import unittest
from datetime import datetime

from src.biosignals.Timeseries import Timeseries
from src.pipeline.Packet import Packet


class PacketTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.ts1 = Timeseries([Timeseries.Segment([0, 1, 2, 3, 4], datetime.now(), 1), ], True, 1)
        cls.ts2 = Timeseries([Timeseries.Segment([0, 1, 2, 3, 4, 5], datetime.now(), 1), ], True, 1)

    def test_create_packet_with_single_timeseries(self):
        packet = Packet(timeseries=self.ts1, other=4)
        #self.assertEqual(packet.who_packed, self)

    def test_unpack_single_timeseries(self):
        packet = Packet(timeseries=self.ts1)
        unpacked = packet.timeseries
        self.assertEqual(unpacked, self.ts1)

    def test_unpack_multiple_timeseries(self):
        # As a tuple
        packet = Packet(timeseries=(self.ts1, self.ts2, ))
        unpacked = packet.timeseries
        self.assertEqual(unpacked, {'0': self.ts1, '1': self.ts2})  # it always returns a dictionary, even if labels were not provides, in which case they will be sequential numbers in string

        # As a dictionary
        timeseries = {'a': self.ts1, 'b': self.ts2}
        packet = Packet(timeseries=timeseries)
        unpacked = packet.timeseries
        self.assertEqual(unpacked, timeseries)

    def test_pack_another_content(self):
        packet = Packet(timeseries=self.ts1, id=294)
        unpacked = packet['id']
        self.assertEqual(unpacked, 294)

    def test_get_packet_contents(self):
        packet = Packet(timeseries=self.ts1, id=294)
        contents = packet.contents
        self.assertEqual(tuple(contents.keys()), ('timeseries', 'id'))
        self.assertEqual(tuple(contents.values()), (Timeseries, int))

    def test_print_packet(self):
        packet = Packet(timeseries=self.ts1, id=294)
        print(packet)

    def test_has_timeseries(self):
        packet = Packet(timeseries=self.ts1)
        self.assertTrue(packet.has_timeseries)
        packet = Packet(other = 4)
        self.assertFalse(packet.has_timeseries)

    def test_has_timeseries_collection(self):
        packet = Packet(timeseries={'a': self.ts1, 'b': self.ts2})
        self.assertTrue(packet.has_timeseries_collection)
        packet = Packet(timeseries={'a': self.ts1, })
        self.assertTrue(packet.has_timeseries_collection)
        packet = Packet(timeseries=self.ts1)
        self.assertFalse(packet.has_timeseries_collection)
        packet = Packet(other=4)
        self.assertFalse(packet.has_timeseries_collection)

    def test_has_multiple_timeseries(self):
        packet = Packet(timeseries={'a': self.ts1, 'b': self.ts2})
        self.assertTrue(packet.has_multiple_timeseries)
        packet = Packet(timeseries={'a': self.ts1, })
        self.assertFalse(packet.has_multiple_timeseries)
        packet = Packet(timeseries=self.ts1)
        self.assertFalse(packet.has_multiple_timeseries)
        packet = Packet(other=4)
        self.assertFalse(packet.has_multiple_timeseries)

    def test_has_single_timeseries(self):
        packet = Packet(timeseries={'a': self.ts1, })
        self.assertTrue(packet.has_single_timeseries)
        packet = Packet(timeseries=self.ts1)
        self.assertTrue(packet.has_single_timeseries)
        packet = Packet(timeseries={'a': self.ts1, 'b': self.ts2})
        self.assertFalse(packet.has_single_timeseries)
        packet = Packet(other=4)
        self.assertFalse(packet.has_single_timeseries)

    def test_contains(self):
        packet = Packet(timeseries=self.ts1, id=294)
        self.assertTrue('id' in packet)
        self.assertTrue('timeseries' in packet)
        self.assertFalse('other' in packet)

    def test_to_dict(self):
        packet = Packet(timeseries=self.ts1, id=294)
        d = packet._to_dict()
        self.assertTrue(id(d) != id(packet._Packet__load))
        self.assertTrue(d == packet._Packet__load)

    def test_join_packets(self):
        a = Packet(timeseries=self.ts1, id=294)
        b = Packet(timeseries=self.ts2, name='Alice')
        res = Packet.join_packets(a=a, b=b)
        load = res._to_dict()

        self.assertTrue('timeseries' in load)
        self.assertTrue('id' in load)
        self.assertTrue('name' in load)

        self.assertEqual(load['id'], 294)
        self.assertEqual(load['name'], 'Alice')

        timeseries = load['timeseries']
        self.assertEqual(timeseries['a'], self.ts1)
        self.assertEqual(timeseries['b'], self.ts2)

if __name__ == '__main__':
    unittest.main()
