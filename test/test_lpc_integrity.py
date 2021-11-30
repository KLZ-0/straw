import unittest

from src.straw import lpc
from test import get_signal_float, get_signal


class LPCSignalIntegrity(unittest.TestCase):
    def test_reconstruction(self):
        data, sr = get_signal_float()

        lpc_order = 8
        lpc_precision = 12  # bits

        qlp, quant_level = lpc.compute_qlp(data, lpc_order, lpc_precision)

        signal, sr = get_signal()

        residual = lpc.compute_residual(signal, qlp, lpc_order, quant_level)
        restored = lpc.restore_signal(residual, qlp, lpc_order, quant_level, signal[:lpc_order])

        self.assertEqual((signal - restored).any(), False)


if __name__ == '__main__':
    unittest.main()
