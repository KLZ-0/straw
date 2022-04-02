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

    def __init__(self, adaptive=False, responsiveness: int = 6):
        self.adaptive = adaptive
        self.parallel = ParallelCompute()
        self.responsiveness = responsiveness

    ##########################
    # Optimal order guessing #
    ##########################

    @staticmethod
    def _compute_expected_bits_per_sample(lpc_error, residual_samples):
        error_scale = 0.5 / residual_samples

        if lpc_error > 0.0:
            bps = 0.5 * np.log(error_scale * lpc_error) / np.log(2)
            if bps >= 0.0:
                return bps
            else:
                return 0.0
        elif lpc_error < 0.0:
            return 1e32
        else:
            return 0.0

    @staticmethod
    def guess_parameter(lpc_error, residual_samples):
        # TODO: this gives a shitty output
        lpc_residual_bits_per_sample = Ricer._compute_expected_bits_per_sample(lpc_error, residual_samples)
        rice_parameter = int(lpc_residual_bits_per_sample + 0.5) if (lpc_residual_bits_per_sample > 0.0) else 0
        rice_parameter += 1  # account for signed conversion
        return rice_parameter

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
        data = bitarray()

        if frame is not None:
            # k = frexp(intrl(frame[0]))[1]
            ext.encode_frame(data, frame, bps, self.responsiveness, adaptive=self.adaptive)

        return data

    def _frame_to_bitstream_df_expander(self, df: pd.DataFrame) -> np.array:
        return self.frame_to_bitstream(df["frame"], df["bps"])

    def frames_to_bitstreams(self, frames: pd.Series, bps: pd.Series, parallel: bool = True) -> pd.Series:
        """
        Encode a series of frames to a series of bitsreams
        :param frames: series of frames
        :param bps: expected bits per second for this frame
        :param parallel: if True then use multithreading
        :return: encoded bitarrays
        """

        comp = pd.DataFrame({"frame": frames, "bps": bps})

        if not parallel:
            return comp.apply(self._frame_to_bitstream_df_expander, axis=1, result_type="reduce")

        return self.parallel.apply(comp, self._frame_to_bitstream_df_expander, axis=1, result_type="reduce")

    ############
    # Decoding #
    ############

    def bitstream_to_frame(self, bitstream: bitarray, frame_size: int, bps: int, want_bits: bool = False,
                           own_frame=None) -> np.array:
        """
        Decode a single frame from a given bitstream
        WARNING: The given bitstream is destroyed to prevent unnecessary memory duplication
        :param bitstream: rice encoded stream
        :param frame_size: frame size
        :param bps: expected bits per sample
        :param want_bits: if True returns the number of bits read
        :param own_frame: if not None, this frame will be filled
        :return: decoded frame
        """
        bits_read = 0
        if own_frame is None:
            frame = np.zeros(frame_size, dtype=np.short)
        else:
            frame = own_frame[:frame_size]

        if len(bitstream) > 0:
            bits_read = ext.decode_frame(frame, bitstream, bps, self.responsiveness, adaptive=self.adaptive)

        if want_bits:
            return frame, bits_read
        else:
            return frame

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

        return self.parallel.apply(comp, self._bitstream_to_frame_df_expander, axis=1, result_type="reduce")
