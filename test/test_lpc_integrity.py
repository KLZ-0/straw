import unittest

import pandas as pd

from straw import lpc
from test import get_signal


class LPCSignalIntegrity(unittest.TestCase):
    def test_reconstruction(self):
        signal, sr = get_signal()

        lpc_order = 8
        lpc_precision = 12  # bits

        df = pd.DataFrame({"frame": [signal]})
        qlp, shift = lpc.compute_qlp(df.loc[0], lpc_order, lpc_precision)
        df["qlp"] = [qlp]
        df["shift"] = [shift]

        residual = lpc.compute_residual(df.loc[0], lpc_order)
        restored = lpc.restore_signal(residual, qlp, lpc_order, shift, signal[:lpc_order])

        self.assertEqual((signal - restored).any(), False)


if __name__ == '__main__':
    unittest.main()
