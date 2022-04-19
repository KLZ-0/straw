import unittest

import numpy as np

from straw import rice
from . import resources


class Rice(unittest.TestCase):
    """
    Rice parameter 4
    """
    k = 2
    m = 1 << k
    r = rice.Ricer()

    def test_rice_ext_arange_decode(self):
        frame = np.arange(20, dtype=np.int16)

        bits = self.r.frame_to_bitstream(frame, self.k)
        decoded = self.r.bitstream_to_frame(memoryview(bits).tobytes(), len(frame), self.k)

        self.assertListEqual(frame.tolist(), decoded.tolist())

    def test_rice_ext_signal_encode_smallbuffer(self):
        # This frame would produce a residual larger than the given buffer, so it would be encoded raw
        frame = resources.get_signal()[0]

        max_allowed_bits = frame.nbytes * 8
        bits = self.r.frame_to_bitstream(frame, self.k)

        self.assertEqual(len(bits), max_allowed_bits)

    def test_problematic_residual1(self):
        frame = resources.get_problematic_residual1()

        bits = self.r.frame_to_bitstream(frame, self.k)
        decoded = self.r.bitstream_to_frame(memoryview(bits).tobytes(), len(frame), self.k)

        self.assertListEqual(frame.tolist(), decoded.tolist())

    def test_problematic_residual2(self):
        frame = resources.get_problematic_residual2()

        bits = self.r.frame_to_bitstream(frame, self.k)
        decoded = self.r.bitstream_to_frame(memoryview(bits).tobytes(), len(frame), self.k)

        self.assertListEqual(frame.tolist(), decoded.tolist())

if __name__ == '__main__':
    unittest.main()
