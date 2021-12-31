from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import pyximport
from bitarray import bitarray

from ..compute.df_parallel import parallelize_on_rows

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

    def frames_to_bitstream(self, frames: pd.Series, parallel: bool = True, cpus=None) -> pd.Series:
        """
        Rice encode a series of frames to a bitsream
        :param frames: series of frames
        :param parallel: if True then use multithreading
        :param cpus: number of CPUs to use when parallel is True, None means use all
        :return: encoded bitarray
        """

        if not parallel:
            return frames.apply(self.frame_to_bitstream)

        if cpus is None or not isinstance(cpus, int):
            cpus = cpu_count()

        return parallelize_on_rows(frames, self.frame_to_bitstream, cpus)
