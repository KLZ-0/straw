import unittest

from straw import rice


class Rice(unittest.TestCase):
    def test_encoding_integrity(self):
        """
        Rice parameter 4
        Encode bitstream of numbers from 0 up to 20
        :return:
        """
        r = rice.Ricer(4)
        r_o = ""

        for i in range(20):
            r.encode_single(i)
            r_o += rice.rice_str(i, 4)

        self.assertEqual(r.data.to01(), r_o)


if __name__ == '__main__':
    unittest.main()
