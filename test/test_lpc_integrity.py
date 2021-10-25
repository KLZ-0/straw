import unittest

import soundfile

from straw import lpc


class LPCSignalIntegrity(unittest.TestCase):
    def test_reconstruction(self):
        data, sr = soundfile.read("../inputs/maskoff_tone.wav")

        bs = int(sr * 0.020)
        start = 400

        lpc_c = lpc.lpc(data[start:start + bs], 8)

        data, sr = soundfile.read("../inputs/maskoff_tone.wav", dtype="int16")
        signal = data[start:start + bs]

        e = lpc.lpc_predict(signal, lpc_c)
        x = lpc.lpc_reconstruct(e, lpc_c)

        self.assertEqual((signal - x).any(), False)


if __name__ == '__main__':
    unittest.main()
