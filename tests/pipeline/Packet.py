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
        packet = Packet(timeseries=self.ts1)
        #self.assertEqual(packet.who_packed, self)

    def test_unpack_single_timeseries(self):
        packet = Packet(timeseries=self.ts1)
        unpacked = packet.single_timeseries
        self.assertEqual(unpacked, self.ts1)

    def test_unpack_multiple_timeseries(self):
        # As a tuple
        packet = Packet(timeseries=(self.ts1, self.ts2, ))
        unpacked = packet.all_timeseries
        self.assertEqual(unpacked, {'0': self.ts1, '1': self.ts2})  # it always returns a dictionary, even if labels were not provides, in which case they will be sequential numbers in string

        # As a dictionary
        timeseries = {'a': self.ts1, 'b': self.ts2}
        packet = Packet(timeseries=timeseries)
        unpacked = packet.all_timeseries
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

if __name__ == '__main__':
    unittest.main()
