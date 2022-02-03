import sys
from typing import TextIO

import numpy as np
import pandas as pd
import soundfile

from . import lpc
from .compute import ParallelCompute
from .rice import Ricer


class Encoder:
    # Values which should be parametrized
    # TODO: find the best values for these
    _lpc_order = 8
    _lpc_precision = 12  # bits
    _frame_size = 4096  # bytes

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
            data, sr = soundfile.read(filename, dtype="int16")
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
        p = ParallelCompute(args=(self._lpc_order, self._lpc_precision),
                            apply_kwargs={"axis": 1, "result_type": "reduce"})
        tmp = p.apply(self._data[["frame"]], lpc.compute_qlp)

        self._data[["qlp", "shift"]] = pd.DataFrame(tmp.to_list())

        # Make sure shift is int
        self._data["shift"] = self._data["shift"].astype("i1")

        p.args = None
        self._data["residual"] = p.apply(self._data[["frame", "qlp", "shift"]], lpc.compute_residual)

    def save_file(self, filename):
        self._data["stream"] = self._encoder.frames_to_bitstream(self._data["residual"])
        self._data["stream_len"] = self._data["stream"].apply(len)
        # TODO: actually save bitstreams

    def print_stats(self, stream: TextIO = sys.stdout):
        print(f"Number of frames: {len(self._data)}", file=stream)
        print(f"Source size: {self._source_size} ({self._source_size / 2 ** 20:.2f} MiB)", file=stream)
        size = self._data["stream_len"].sum()
        print(f"Length of bitstream: {size} bits, "
              f"bytes: {np.ceil(size / 8):.0f} aligned ({np.ceil(size / 8) / 2 ** 20:.2f} MiB)", file=stream)
        print(f"Ratio = {np.ceil(size / 8) / self._source_size:.3f}", file=stream)

        # FIXME: this is misleading
        print(f"Size of the resulting dataframe: {self.usage_mib():.3f} MiB", file=stream)

    def sample_frame(self):
        return self._data.loc[0]

    def _clean(self):
        self._raw = None
        self._data = None
        self._samplerate = None
        self._frame_size = None
