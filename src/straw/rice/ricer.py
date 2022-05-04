import numpy as np
import pandas as pd
from bitarray import bitarray

from straw.compute import ParallelCompute
from straw.io.sizes import StrawSizes
from straw.static import SubframeType, Default
from . import ext_rice


class Ricer:
    """
    Rice encoder/decoder
    Currently only supports memory for for memory efficiency comparisons and benchmarks
    """

    def __init__(self, adaptive=True, responsiveness: int = Default.framing_resolution):
        self.adaptive = adaptive
        self.parallel = ParallelCompute.get_instance()
        self.responsiveness = responsiveness

    ##########################
    # Optimal order guessing #
    ##########################

    def guess_parameter(self, frame_residual: np.array) -> np.int8:
        if frame_residual is None:
            return np.int8(0)
        smallframe = frame_residual[:self.responsiveness].astype(np.int64)
        ext_rice.interleave_frame(smallframe)
        param = np.clip(np.log2(smallframe.mean()), 0, (1 << StrawSizes.residual.param) - 1)
        return np.round(param).astype(np.int8)

    ############
    # Encoding #
    ############

    def frame_to_bitstream(self, frame: np.array, bps: int) -> bitarray:
        """
        Encode a numpy frame to a bitsream
        :param frame: numpy array of samples to encode
        :param bps: expected bits per sample
        :return: encoded bitarray
        """
        data = np.zeros(frame.nbytes, dtype=np.uint8)
        bits = ext_rice.encode_frame(data, frame, bps, self.responsiveness, adaptive=self.adaptive)
        if bits == -1:
            return bitarray(buffer=frame)  # This will later fail in Encoder._ensure_compression
        else:
            return bitarray(buffer=data)[:bits]

    def _frame_to_bitstream_df_expander(self, df: pd.DataFrame) -> np.array:
        if df["frame_type"] not in (SubframeType.LPC, SubframeType.LPC_COMMON):
            return bitarray()

        return self.frame_to_bitstream(df["residual"], df["bps"])

    def frames_to_bitstreams(self, df: pd.DataFrame, parallel: bool = True) -> pd.Series:
        """
        Encode frames to a series of bitsreams
        :param df: DataFrame with columns ["residual", "bps", "frame_type"]
        :param parallel: if True then use multithreading
        :return: encoded bitarrays
        """

        if not parallel:
            return df.apply(self._frame_to_bitstream_df_expander, axis=1, result_type="reduce")

        return self.parallel.map(df, self._frame_to_bitstream_df_expander, axis=1, result_type="reduce")

    ############
    # Decoding #
    ############

    def bitstream_to_frame(self, bitstream_memoryview: bytes,
                           frame_size: int, bps: int,
                           own_frame: np.array = None,
                           bitarray_pos=0):
        """
        Decode a single frame from a given bitstream
        WARNING: The given bitstream is destroyed to prevent unnecessary memory duplication
        :param bitstream_memoryview: rice encoded stream
        :param frame_size: frame size
        :param bps: expected bits per sample
        :param own_frame: if not None, this frame will be filled
        :param bitarray_pos: starting position within the bitarray - use this to avoid slicing
        :return: decoded frame or number of bits read if own_frame is not None
        """
        bits_read = 0
        if own_frame is None:
            frame = np.zeros(frame_size, dtype=np.int64)
        else:
            frame = own_frame[:frame_size]

        if len(bitstream_memoryview) > 0:
            bits_read = ext_rice.decode_frame(frame, bitstream_memoryview, bps, self.responsiveness,
                                              adaptive=self.adaptive, starting_i=bitarray_pos)

        if own_frame is None:
            return frame
        else:
            return bits_read

    def _bitstream_to_frame_df_expander(self, df: pd.DataFrame) -> np.array:
        return self.bitstream_to_frame(df["stream"], df["size"], df["bps"])

    def bitstreams_to_frames(self, bitstreams: pd.Series, frame_sizes: pd.Series, bps: pd.Series,
                             parallel: bool = True) -> pd.Series:
        """
        Encode a series of bitstreams to a series of frames
        :param bitstreams: series of bitstreams
        :param frame_sizes: series of frame sizes
        :param bps: expected bits per second for this frame
        :param parallel: if True then use multithreading
        :return: series of decoded frames
        """

        comp = pd.DataFrame({"stream": bitstreams, "size": frame_sizes, "bps": bps})

        if not parallel:
            return comp.apply(self._bitstream_to_frame_df_expander, axis=1, result_type="reduce")

        return self.parallel.map(comp, self._bitstream_to_frame_df_expander, axis=1, result_type="reduce")

    ###########
    # Utility #
    ###########

    @staticmethod
    def frame_to_kparams(frame: np.ndarray, k: int, responsiveness: int = 16):
        frame = np.copy(frame)
        ext_rice.kparams(frame, k, responsiveness)
        return frame

    @staticmethod
    def frame_to_interleaved(frame: np.ndarray):
        frame = np.copy(frame)
        ext_rice.interleave_frame(frame)
        return frame
