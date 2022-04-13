import sys
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import soundfile

from straw import lpc
from straw.codec.base import BaseCoder
from straw.correctors import BiasCorrector, ShiftCorrector, deconvolve, localized_deconvolve
from straw.io import Formatter
from straw.io.params import StreamParams
from straw.rice import Ricer


class Encoder(BaseCoder):
    # Values which should be parametrized
    # TODO: find the best values for these
    _lpc_order = 10  # can be sourced from len(df["qlp"]) once per group
    _lpc_precision = 12  # bits, stored in df["qlp_precision"] once per group
    _params = StreamParams()

    # TODO: make the coders accept other sizes
    _bits_per_sample = 16  # stored in StreamParams once per group

    ##########
    # Public #
    ##########

    def __init__(self, flac_mode=False, do_corrections=True):
        super(Encoder, self).__init__(flac_mode)
        self._ricer = Ricer(adaptive=True if not flac_mode else False)
        self._do_corrections = do_corrections

    def load_file(self, file):
        """
        Load the specified file into memory
        :param file: str or int or file-like object - anything that soundfile accepts
        :return: None
        """
        data, sr = soundfile.read(file, dtype=f"int{self._bits_per_sample}", always_2d=True)
        self.load_data(data, sr)

    def load_data(self, data: np.array, samplerate: int):
        if len(data.shape) == 1:
            self._samplebuffer = data.reshape((1, -1))
        elif data.shape[0] > data.shape[1]:
            self._samplebuffer = data.swapaxes(1, 0)
        else:
            self._samplebuffer = data

        self._source_size = self._samplebuffer.nbytes
        self._params.sample_rate = samplerate
        self._params.md5 = self.get_md5()
        self._apply_corrections()
        self._create_dataframe()

    def encode(self):
        """
        Encode the signal
        :return: None
        """
        self._parametrize()
        lpc_frames = self._set_frame_types()
        # self._deconvolve_signals()  # 1
        tmp = self._data[lpc_frames].groupby("seq").apply(lpc.compute_qlp, self._lpc_order, self._lpc_precision)
        self._data[["qlp", "qlp_precision", "shift"]] = tmp
        # self._deconvolve_signals()  # 2
        self._data = self._data.groupby("seq").apply(lpc.compute_residual)
        self._data["bps"] = np.full(len(self._data["residual"]), 4, dtype="B")
        # self._deconvolve_signals("residual")  # 3, 4
        self._deconvolve_signals("residual", localized=True)  # 3*
        self._data["stream"] = self._ricer.frames_to_bitstreams(self._data, parallel=True)
        self._data["stream_len"] = self._data["stream"].apply(len)
        self._ensure_compression()

    def save_file(self, output_file: Path):
        """
        Save the encoded signal
        :param output_file: target file
        :return: None
        """
        # self._data.groupby("seq").apply(lambda df: df["frame"].apply(cross_similarity, data_ref=df["frame"][df.index[0]]))
        # self._print_var(seq=4)
        # exp = 4
        # show_frame(self._data[self._data["seq"] == 4], terminate=False, limit=(1750, 1810))
        # show_frame(self._data[self._data["seq"] == 4], terminate=False, col_name="residual", limit=(1740, 1800))
        # show_frame(self._data[self._data["seq"] == 4], terminate=False, file_name="gain_shift_correction_after.png")
        # show_frame(self._data[self._data["seq"] == 4], col_name="residual")
        Formatter().save(self._data, self._params, output_file, self._flac_mode)

    ###########
    # Private #
    ###########

    def _create_dataframe(self):
        """
        Create a dataframe from the raw signal, this includes slicing the signal into specific views
        NOTE: the underlying memory stays as a contiguous memory chunk
        :return:
        """
        ds = {"seq": [], "frame": [], "channel": []}
        total_size = self._samplebuffer.shape[1] - np.max(self._params.lags)
        for channel, channel_data in enumerate(self._samplebuffer):
            lag = self._params.lags[channel]
            sliced = self._slice_channel_data_into_frames(channel_data[lag:total_size + lag])
            ds["seq"] += [i for i in range(len(sliced))]
            ds["frame"] += sliced
            ds["channel"] += [channel for _ in range(len(sliced))]

        self._data = pd.DataFrame(ds)

    def _parametrize(self):
        """
        Parameter extraction to be used for encoding the whole stream
        :return: None
        """
        if self._flac_mode:
            self._params.max_block_size = int(self._data["frame"].apply(len).max())
            self._params.min_block_size = self._params.max_block_size
            self._params.min_frame_size = 0  # unknown
            self._params.max_frame_size = 0
        self._params.channels = len(np.unique(self._data["channel"]))
        self._params.bits_per_sample = self._bits_per_sample
        # self._params.total_samples = int(self._data[self._data["channel"] == 0]["frame"].apply(len).sum())
        self._params.total_samples = int(self._samplebuffer.shape[1])

    def _set_frame_types(self):
        # all frames are LPC frames by default
        self._data["frame_type"] = np.full(len(self._data["frame"]), 0b11, dtype="B")

        # Constant frames
        const_frames = self._data["frame"].apply(lambda x: not (x - x[0]).any())
        self._data.loc[const_frames, "frame_type"] = 0b00

        return ~const_frames

    def _ensure_compression(self):
        max_allowed_bits = self._data["residual"].apply(len) * self._params.bits_per_sample
        self._data.loc[self._data["stream_len"] > max_allowed_bits, "frame_type"] = 0b01

        if self._flac_mode:
            max_residual_bytes = (self._data["stream_len"].max() // 8) + 1
            self._params.max_frame_size = int(max_residual_bytes) + 1000

    def _apply_corrections(self):
        if self._do_corrections:
            # self._data = self._data.groupby("seq").apply(GainCorrector().apply, col_name=col_name)
            self._samplebuffer = BiasCorrector().global_apply(self._samplebuffer, self._params)
            self._samplebuffer = ShiftCorrector().global_apply(self._samplebuffer, self._params)

    def _deconvolve_signals(self, col_name="frame", localized=False):
        if self._do_corrections:
            if localized:
                self._data = self._data.groupby("seq").apply(localized_deconvolve, col_name=col_name)
            else:
                self._data = self._data.groupby("seq").apply(deconvolve, col_name=col_name)

    def _print_var(self, seq=0):
        old_stream_len = 214523
        stream_len = self._data[self._data["seq"] == seq]["stream_len"].sum()
        print("- stream_len:", stream_len)
        print("- stream_len diff:", stream_len - old_stream_len)
        old_maxabs = np.asarray([352, 373, 581, 516, 432, 349, 380, 391])
        nocorr_var = np.asarray([10997.481, 24395.01, 50948.516, 36896.603, 21682.603, 11630.761,
                                 14912.361, 18267.361])
        self._tmp(seq, np.var, "var", nocorr_var)
        self._tmp(seq, lambda x: np.max(np.abs(x)), "absmax", old_maxabs)

    def _tmp(self, seq, func, name, old_vals=None):
        residuals = self._data[self._data["seq"] == seq]["residual"]
        residuals = residuals.apply(lambda x: x[1740:1800])
        var = residuals.apply(func).to_numpy()
        print(f"- {name}:", np.array2string(var, separator=", ", precision=3, suppress_small=True))
        if old_vals is not None:
            print(f"- original {name}:", np.array2string(old_vals, precision=3, suppress_small=True))
            print(f"- {name} difference:", np.array2string(var - old_vals, precision=3, suppress_small=True))
            print(f"total {name} diff: {(var - old_vals).sum():.3f}")

    ###########
    # Utility #
    ###########

    def print_stats(self, output_file: Path, stream: TextIO = sys.stdout):
        """
        Print a bunch of stuff...
        :param output_file: output file (only the size is needed)
        :param stream: stream where the output should be written
        :return: None
        """
        print(f"Number of frames: {len(self._data)}", file=stream)
        print(f"Source size: {self._source_size} ({self._source_size / 2 ** 20:.2f} MiB)", file=stream)
        size = self._data["stream_len"].sum()
        print(f"md5: {self._params.md5.hex(' ')}", file=stream)
        print(f"Length of residual bitstream: {size} bits, "
              f"bytes: {np.ceil(size / 8):.0f} aligned ({np.ceil(size / 8) / 2 ** 20:.2f} MiB)", file=stream)
        lpc_bytes = np.ceil(len(self._data) * self._lpc_precision * self._lpc_order * 1 / 8)
        print(f"Bytes needed for coefficients: {lpc_bytes:.0f} B", file=stream)
        print(f"Output file size: {output_file.stat().st_size}", file=stream)
        print(f"Grand Ratio = {output_file.stat().st_size / self._source_size:.3f}", file=stream)

        # FIXME: this is misleading
        print(f"Size of the resulting dataframe: {self.usage_mib():.3f} MiB", file=stream)
