import unittest

import soundfile

from src.straw import lpc


class LPCSignalIntegrity(unittest.TestCase):
    def test_reconstruction(self):
        data, sr = soundfile.read("../inputs/maskoff_tone.wav")

        lpc_order = 8
        lpc_precision = 12  # bits

        bs = int(sr * 0.020)
        start = 400

        qlp, quant_level = lpc.compute_qlp(data[start:start + bs], lpc_order, lpc_precision)

        data, sr = soundfile.read("../inputs/maskoff_tone.wav", dtype="int16")
        signal = data[start:start + bs]

        residual = lpc.compute_residual(signal, qlp, lpc_order, quant_level)
        restored = lpc.restore_signal(residual, qlp, lpc_order, quant_level, signal[:lpc_order])

        self.assertEqual((signal - restored).any(), False)


if __name__ == '__main__':
    unittest.main()
