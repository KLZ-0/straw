import numpy as np
import pandas as pd
import pyximport
from bitarray import bitarray

from ..compute import ParallelCompute

pyximport.install()
from . import ext


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
        Encode a numpy frame to a bitsream
        :param frame: numpy array of samples to encode
        :return: encoded bitarray
        """
        data = bitarray()

        if frame is not None:
            ext.encode_frame(data, frame, self.m, self.k)

        return data

    def frames_to_bitstreams(self, frames: pd.Series, parallel: bool = True) -> pd.Series:
        """
        Encode a series of frames to a series of bitsreams
        :param frames: series of frames
        :param parallel: if True then use multithreading
        :return: encoded bitarrays
        """

        if not parallel:
            return frames.apply(self.frame_to_bitstream)

        return self.parallel.apply(frames, self.frame_to_bitstream)

    def bitstream_to_frame(self, bitstream: bitarray, frame_size: int) -> np.array:
        """
        Decode a single frame from a given bitstream
        WARNING: The given bitstream is destroyed to prevent unnecessary memory duplication
        :param bitstream: rice encoded stream
        :param frame_size: frame size
        :return: decoded frame
        """
        frame = np.zeros(frame_size, dtype=np.short)

        if len(bitstream) > 0:
            ext.decode_frame(frame, bitstream, self.m, self.k)

        return frame

    def _bitstream_to_frame_df_expander(self, df: pd.DataFrame) -> np.array:
        return self.bitstream_to_frame(df["stream"], df["size"])

    def bitstreams_to_frames(self, bitstreams: pd.Series, frame_sizes: pd.Series, parallel: bool = True) -> pd.Series:
        """
        Encode a series of bitstreams to a series of frames
        :param bitstreams: series of bitstreams
        :param frame_sizes: series of frame sizes
        :param parallel: if True then use multithreading
        :return: series of decoded frames
        """

        comp = pd.DataFrame({"stream": bitstreams, "size": frame_sizes})

        if not parallel:
            return comp.apply(self._df_expander, axis=1, result_type="reduce")

        return self.parallel.apply(comp, self._bitstream_to_frame_df_expander, axis=1, result_type="reduce")
