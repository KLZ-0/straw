import numpy as np

from straw.correctors.base import BaseCorrector
from straw.io.params import StreamParams
from straw.io.sizes import StrawSizes


class GainCorrector(BaseCorrector):
    def apply(self, samplebuffer: np.ndarray, params: StreamParams) -> (np.ndarray, np.ndarray):
        factors = self.find_factors(samplebuffer, StrawSizes.metadata_block_streaminfo.gain)
        for i in range(samplebuffer.shape[0]):
            # scaling by 1.0 is just a waste of time
            if factors[i] == 1.0:
                continue

            self.equalize(samplebuffer[i], factors[i])
        params.gain, params.gain_shift = self.quantize_factors(factors, StrawSizes.metadata_block_streaminfo.gain)

    def apply_revert(self, samplebuffer: np.ndarray, params: StreamParams) -> (np.ndarray, np.ndarray):
        if params.gain_shift is None:
            return

        factors = self.dequantize_factors(params.gain, params.gain_shift)
        for i in range(samplebuffer.shape[0]):
            # scaling by 1.0 is just a waste of time
            if factors[i] == 1.0:
                continue

            self.deequalize(samplebuffer[i], factors[i])

    @staticmethod
    def quantize_factors(factors: np.ndarray, precision: int):
        factors -= 1.0
        qmax = 1 << precision
        qmin = -qmax
        qmax -= 1

        for shift in reversed(range(0, precision + 1)):
            vals = (factors * (1 << shift)).astype(np.int64)
            if vals.max() <= qmax and vals.min() >= qmin:
                return vals, shift

    @staticmethod
    def dequantize_factors(factors: np.ndarray, shift: int):
        return (factors / (1 << shift)) + 1.0

    @staticmethod
    def energy(frame: np.ndarray):
        return frame.std()

    @staticmethod
    def equalize(frame: np.ndarray, factor: float):
        frame[:] = (frame * factor).round().astype(frame.dtype)

    @staticmethod
    def deequalize(frame: np.ndarray, factor: float):
        frame[:] = (frame / factor).round().astype(frame.dtype)

    def find_factors(self, samplebuffer: np.ndarray, precision: int):
        energies = np.asarray([self.energy(channel_data) for channel_data in samplebuffer])
        strongest_idx = energies.argmax()
        factors = energies[strongest_idx] / energies

        # Apply and de-apply quantization
        factors, shift = self.quantize_factors(factors, precision)
        factors = self.dequantize_factors(factors, shift)
        return factors
