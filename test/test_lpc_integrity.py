import unittest

import numpy as np
import pandas as pd

from straw import lpc
from straw.lpc.steps import restore_signal
from straw.static import SubframeType
from . import resources


class LPCSignalIntegrity(unittest.TestCase):
    lpc_order = 8
    lpc_precision = 12  # bits

    signal, sr = resources.get_signal()
    signal_zeros = np.zeros(2 ** 12)

    def test_reconstruction(self):
        df = pd.DataFrame({"frame": [self.signal], "frame_type": SubframeType.LPC})
        qlp, precision, shift = lpc.compute_qlp(df.loc[0], self.lpc_order, self.lpc_precision)
        df["qlp"] = [qlp]
        df["shift"] = [shift]

        df = lpc.compute_residual(df.loc[0])
        restored = restore_signal(df["residual"], qlp, shift, self.signal[:self.lpc_order])

        self.assertEqual((self.signal - restored).any(), False)

    def test_zeroframe_qlp(self):
        df = pd.DataFrame({"frame": [self.signal_zeros], "frame_type": SubframeType.LPC})
        qlp, precision, shift = lpc.compute_qlp(df.loc[0], self.lpc_order, self.lpc_precision)
        self.assertEqual(shift, 0)
        self.assertEqual(precision, 0)

    def test_zeroframe_residual(self):
        df = pd.Series({
            "frame": self.signal_zeros,
            "frame_type": SubframeType.LPC,
            "qlp": None,
            "shift": 0,
        })

        df = lpc.compute_residual(pd.DataFrame(df).T)
        self.assertIsInstance(df.loc[0, "residual"], np.ndarray)
        self.assertEqual(df.loc[0, "residual"][0], 0)


if __name__ == '__main__':
    unittest.main()
