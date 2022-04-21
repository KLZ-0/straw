import unittest

import numpy as np
import pandas as pd

from straw import lpc
from straw.static import SubframeType
from . import resources


class LPCSignalIntegrity(unittest.TestCase):
    lpc_order = 8
    lpc_precision = 12  # bits

    signal, sr = resources.get_signal()
    signal_zeros = np.zeros(2 ** 12)

    def test_reconstruction(self):
        df = pd.DataFrame({"frame": [self.signal.copy()], "frame_type": SubframeType.LPC})
        lpc.compute_qlp(df, self.lpc_order, self.lpc_precision)

        lpc.compute_residual(df)
        df = lpc.compute_original(df, inplace=False)

        self.assertEqual((self.signal - df.loc[0, "restored"]).any(), False)

    def test_zeroframe_qlp(self):
        df = pd.DataFrame({"frame": [self.signal_zeros], "frame_type": SubframeType.LPC})
        lpc.compute_qlp(df, self.lpc_order, self.lpc_precision)
        self.assertEqual(df.loc[0, "shift"], 0)
        self.assertEqual(df.loc[0, "qlp_precision"], 0)

    def test_zeroframe_residual(self):
        df = pd.DataFrame({"frame": [self.signal_zeros], "frame_type": SubframeType.LPC})
        lpc.compute_qlp(df, self.lpc_order, self.lpc_precision)

        lpc.compute_residual(df)
        self.assertIsNone(df.loc[0, "qlp"])
        self.assertIsNone(df.loc[0, "residual"])
        self.assertEqual(df.loc[0, "frame_type"], SubframeType.CONSTANT)


if __name__ == '__main__':
    unittest.main()
