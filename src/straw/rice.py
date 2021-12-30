import numpy as np
import pyximport
from bitarray import bitarray

pyximport.install()
from .ext import rice_encode


class Ricer:
    """
    Rice encoder/decoder
    Currently only supports memory for for memory efficiency comparisons and benchmarks
    """

    def __init__(self, m):
        self.m = m
        self.k = int(np.log2(self.m))

    def frame_to_bitstream(self, frame: np.array) -> bitarray:
        """
        Rice encode a numpy frame to a bitsream
        :param frame: numpy array of samples to encode
        :return: encoded bitarray
        """
        data = bitarray()
        rice_encode.encode_frame(data, frame, self.m, self.k)
        return data
