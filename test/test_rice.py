import unittest

import numpy as np

from straw import rice
from . import resources


class Rice(unittest.TestCase):
    """
    Rice parameter 4
    """
    m = 4
    r = rice.Ricer(m)

    def test_rice_ext_arange(self):
        """
        Tests the implementation with the reference
        Encode bitstream of numbers from 0 up to 20
        """
        frame = np.arange(20, dtype=np.int16)

        r_o = ""
        for i in frame:
            r_o += resources.rice_str(i, self.m)

        bits = self.r.frame_to_bitstream(frame)

        self.assertEqual(bits.to01(), r_o)

    def test_rice_ext_signal(self):
        frame = resources.get_signal()[0]

        r_o = ""
        for i in frame:
            r_o += resources.rice_str(i, self.m)

        bits = self.r.frame_to_bitstream(frame)

        self.assertEqual(bits.to01(), r_o)

    def test_rice_ext_arange_decode(self):
        frame = np.arange(20, dtype=np.int16)

        bits = self.r.frame_to_bitstream(frame)
        decoded = self.r.bitstream_to_frame(bits, len(frame))

        self.assertListEqual(frame.tolist(), decoded.tolist())

    def test_rice_ext_signal_decode(self):
        frame = resources.get_signal()[0]

        bits = self.r.frame_to_bitstream(frame)
        decoded = self.r.bitstream_to_frame(bits, len(frame))

        self.assertListEqual(frame.tolist(), decoded.tolist())

if __name__ == '__main__':
    unittest.main()
