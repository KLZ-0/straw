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
        """
        Quantize the factors to integer values
        :param factors: numpy array of factors in floating point format
        :param precision: desired quantization precision in bits
        :return: numpy array of factors in integer format and shift
        """
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
        """
        Dequantizes the factors to a floating point format
        :param factors: numpy array of factors in integer format
        :param shift: shift performed at quantization
        :return: numpy array of factors in floating point format
        """
        return (factors / (1 << shift)) + 1.0

    @staticmethod
    def energy(frame: np.ndarray):
        """
        Return the normalized energy of a frame
        :param frame: input frame
        :return: normalized energy
        """
        return frame.std()

    @staticmethod
    def equalize(frame: np.ndarray, factor: float):
        """
        Scale a channel with the given factor
        NOTE: this is performed inplace
        :param frame: input frame
        :param factor: the scaling factor
        :return: None
        """
        frame[:] = (frame * factor).round().astype(frame.dtype)

    @staticmethod
    def deequalize(frame: np.ndarray, factor: float):
        """
        Reverse the scaling of a channel with the given factor
        NOTE: this is performed inplace
        :param frame: input frame
        :param factor: the scaling factor
        :return: None
        """
        frame[:] = (frame / factor).round().astype(frame.dtype)

    def find_factors(self, samplebuffer: np.ndarray, precision: int):
        """
        Find the factors with which the channels should be multiplied
        :param samplebuffer: array to apply the corrections to - ndarray with dimensions (channels, samples)
        :param precision: desired quantization precision in bits
        :return:
        """
        energies = np.asarray([self.energy(channel_data) for channel_data in samplebuffer])
        strongest_idx = energies.argmax()
        factors = energies[strongest_idx] / energies

        # Apply and de-apply quantization
        factors, shift = self.quantize_factors(factors, precision)
        factors = self.dequantize_factors(factors, shift)
        return factors
