import sys
from hashlib import md5
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import soundfile

from . import lpc
from .io import Formatter
from .rice import Ricer


class Encoder:
    # Values which should be parametrized
    # TODO: find the best values for these
    _lpc_order = 10
    _lpc_precision = 12  # bits
    _frame_size = 4096  # bytes

    _bits_per_sample = 16
    _md5: md5

    # Data from source
    _source_size = 0
    _samplerate = None

    # Member utils
    _encoder = Ricer(4)

    # Member variables
    _raw = None
    _data = None

    def __init__(self, args=None):
        self._args = args

    def usage_mib(self):
        """
        Returns the deep memory usage of the given dataframe in mebibytes
        :return: deep memory usage of the given dataframe in mebibytes
        """
        return self._data.memory_usage(index=True, deep=True).sum() / (2 ** 20)

    def load_files(self, filenames: list):
        """
        Load all specified files as separate channels of the same recording
        :param filenames: list of files to load
        :return: True on success, False on error
        """
        self._raw = []

        # TODO: verify if the files are from the same recording
        for filename in filenames:
            data, sr = soundfile.read(filename, dtype=f"int{self._bits_per_sample}")
            self._md5 = md5(data)
            self._source_size += data.nbytes
            if self._samplerate is None:
                self._samplerate = sr

            if self._samplerate != sr:
                self._clean()
                return False

            if len(data.shape) > 1:
                self._source_size = data.nbytes
                self._raw = data.swapaxes(1, 0)
                return True

            self._raw.append(data.flatten("F"))

        return True

    def _slice_data_into_frames(self, data):
        return [data[i:i + self._frame_size] for i in range(0, len(data), self._frame_size)]

    def create_frames(self):
        ds = {"seq": [], "frame": [], "channel": []}
        for channel, channel_data in enumerate(self._raw):
            sliced = self._slice_data_into_frames(channel_data)
            ds["seq"] += [i for i in range(len(sliced))]
            ds["frame"] += sliced
            ds["channel"] += [channel for _ in range(len(sliced))]

        self._raw = None  # free this reference, we don't need it anymore
        self._data = pd.DataFrame(ds)

    def load_stream(self, stream, samplerate):
        pass

    def encode(self):
        tmp = self._data.groupby("seq").apply(lpc.compute_qlp, self._lpc_order, self._lpc_precision)
        self._data[["qlp", "qlp_precision", "shift"]] = pd.DataFrame(tmp.to_list())

        self._data = self._data.groupby("seq").apply(lpc.compute_residual)

    def save_file(self, output_file: Path):
        self._data["bps"] = np.full(len(self._data["residual"]), 4, dtype="B")
        self._data["stream"] = self._encoder.frames_to_bitstreams(self._data["residual"], self._data["bps"])
        self._data["stream_len"] = self._data["stream"].apply(len)
        self._data.block_size = self._frame_size
        self._data.sample_rate = self._samplerate
        self._data.bits_per_sample = self._bits_per_sample
        self._data.md5 = self._md5
        Formatter(self._data).save(output_file)
        # TODO: actually save bitstreams

    def restore(self):
        self._data["residual_len"] = self._data["residual"].apply(len)
        self._data["residual"] = self._encoder.bitstreams_to_frames(
            self._data["stream"], self._data["residual_len"], self._data["bps"])

        self._data = self._data.groupby("seq").apply(lpc.compute_original)

        if self._data.apply(lpc.compare_restored, axis=1).all():
            print("Lossless :)")
        else:
            print("Not lossless :|")

    def print_stats(self, stream: TextIO = sys.stdout):
        print(f"Number of frames: {len(self._data)}", file=stream)
        print(f"Source size: {self._source_size} ({self._source_size / 2 ** 20:.2f} MiB)", file=stream)
        size = self._data["stream_len"].sum()
        print(f"Length of bitstream: {size} bits, "
              f"bytes: {np.ceil(size / 8):.0f} aligned ({np.ceil(size / 8) / 2 ** 20:.2f} MiB)", file=stream)
        lpc_bytes = np.ceil(len(self._data) * self._lpc_precision * self._lpc_order * 1 / 8)
        print(f"Bytes needed for coefficients: {lpc_bytes:.0f} B", file=stream)
        print(f"Ratio = {np.ceil(size / 8) / self._source_size:.3f}", file=stream)
        print(f"Ratio with LPC coeffs = {(np.ceil(size / 8) + lpc_bytes) / self._source_size:.3f}", file=stream)

        # FIXME: this is misleading
        print(f"Size of the resulting dataframe: {self.usage_mib():.3f} MiB", file=stream)

    def sample_frame(self) -> pd.Series:
        return self._data.loc[0]

    def sample_frame_multichannel(self) -> pd.DataFrame:
        return self._data[self._data["seq"] == 0]

    def get_data(self) -> pd.DataFrame:
        return self._data

    def _clean(self):
        self._raw = None
        self._data = None
        self._samplerate = None
        self._frame_size = None
