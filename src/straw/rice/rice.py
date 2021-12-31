import numpy as np
import pandas as pd
import pyximport
from bitarray import bitarray

from ..compute import ParallelCompute

pyximport.install()
from . import rice_encode


class Ricer:
    """
    Rice encoder/decoder
    Currently only supports memory for for memory efficiency comparisons and benchmarks
    """

    def __init__(self, m):
        self.m = m
        self.k = int(np.log2(self.m))
        self.parallel = ParallelCompute()

    def frame_to_bitstream(self, frame: np.array) -> bitarray:
        """
        Rice encode a numpy frame to a bitsream
        :param frame: numpy array of samples to encode
        :return: encoded bitarray
        """
        data = bitarray()

        if frame is not None:
            rice_encode.encode_frame(data, frame, self.m, self.k)

        return data

    def frames_to_bitstream(self, frames: pd.Series, parallel: bool = True) -> pd.Series:
        """
        Rice encode a series of frames to a bitsream
        :param frames: series of frames
        :param parallel: if True then use multithreading
        :return: encoded bitarray
        """

        if not parallel:
            return frames.apply(self.frame_to_bitstream)

        return self.parallel.apply(frames, self.frame_to_bitstream)
