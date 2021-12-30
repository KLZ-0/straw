import unittest

import numpy as np

from straw import rice
from . import resources


class Rice(unittest.TestCase):
    r = rice.Ricer(4)

    def test_rice_ext_arange(self):
        """
        Rice parameter 4
        Encode bitstream of numbers from 0 up to 20
        :return:
        """
        frame = np.arange(20)

        r_o = ""
        for i in frame:
            r_o += resources.rice_str(i, 4)

        data = self.r.frame_to_bitstream(frame)

        self.assertEqual(data.to01(), r_o)

    def test_rice_ext_signal(self):
        frame = resources.get_signal()[0]

        r_o = ""
        for i in frame:
            r_o += resources.rice_str(i, 4)

        data = self.r.frame_to_bitstream(frame)

        self.assertEqual(data.to01(), r_o)


if __name__ == '__main__':
    unittest.main()
