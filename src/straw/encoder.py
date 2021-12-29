import numpy as np
import pandas as pd
import soundfile

from . import lpc
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
            self._source_size = len(data) * 2
            if self._samplerate is None:
                self._samplerate = sr

            if self._samplerate != sr:
                self._clean()
                return False

            self._raw.append(data)

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
        self._data[["qlp", "shift"]] = self._data[["frame"]].apply(
            lpc.compute_qlp,
            result_type="expand",
            axis=1,
            args=(self._lpc_order, self._lpc_precision))

        self._data["residual"] = self._data[["frame", "qlp", "shift"]].apply(
            lpc.compute_residual,
            axis=1,
            args=[self._lpc_order])

    def save_file(self, filename):
        print(f"Number of frames: {len(self._data)}")
        self._data["residual"].apply(self._encoder.encode_frame)

        size = self._encoder.get_size_bits_unaligned()
        print(f"Source size: {self._source_size}")
        print(f"Length of bitstream: {size} bits (bytes: {size / 8:.2f} unaligned, {np.ceil(size / 8):.0f} aligned)")
        print(f"Ratio = {np.ceil(size / 8) / self._source_size:.2f}")

    def _clean(self):
        self._raw = None
        self._data = None
        self._samplerate = None
        self._frame_size = None
