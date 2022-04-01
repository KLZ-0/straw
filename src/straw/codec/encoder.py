import sys
from hashlib import md5
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import soundfile

from straw import lpc
from straw.codec.base import BaseCoder
from straw.io import Formatter
from straw.io.params import StreamParams


class Encoder(BaseCoder):
    # Values which should be parametrized
    # TODO: find the best values for these
    _lpc_order = 10  # can be sourced from len(df["qlp"]) once per group
    _lpc_precision = 12  # bits, stored in df["qlp_precision"] once per group
    _frame_size = 4096  # bytes, can be sourced from len(df["frame"]) once per group
    _params = StreamParams()

    # TODO: make the coders accept other sizes
    _bits_per_sample = 16  # stored in StreamParams once per group

    def load_files(self, filenames: list):
        """
        Load all specified files as separate channels of the same recording
        :param filenames: list of files to load
        :return: True on success, False on error
        """
        self._raw = []

        # TODO: verify if the files are from the same recording
        self._source_size = 0
        self._samplerate = 0
        for filename in filenames:
            data, sr = soundfile.read(filename, dtype=f"int{self._bits_per_sample}")
            self._md5 = md5(data)
            self._source_size += data.nbytes
            self._samplerate = sr  # TODO: should be stored for each stream

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

        self._raw = []  # free this reference, we don't need it anymore
        self._data = pd.DataFrame(ds)

    def load_stream(self, stream, samplerate):
        pass

    def encode(self):
        tmp = self._data.groupby("seq").apply(lpc.compute_qlp, self._lpc_order, self._lpc_precision)
        self._data[["qlp", "qlp_precision", "shift"]] = pd.DataFrame(tmp.to_list())

        self._data = self._data.groupby("seq").apply(lpc.compute_residual)

    def save_file(self, output_file: Path):
        self._data["bps"] = np.full(len(self._data["residual"]), 4, dtype="B")
        self._data["stream"] = self._ricer.frames_to_bitstreams(self._data["residual"], self._data["bps"])
        self._data["stream_len"] = self._data["stream"].apply(len)
        self._data.block_size = self._frame_size
        self._data.sample_rate = self._samplerate
        self._data.bits_per_sample = self._bits_per_sample
        self._data.md5 = self._md5
        Formatter().save(self._data, output_file, self._flac_mode)
        # TODO: actually save bitstreams

    def restore(self):
        self._data["residual_len"] = self._data["residual"].apply(len)
        self._data["residual"] = self._ricer.bitstreams_to_frames(
            self._data["stream"], self._data["residual_len"], self._data["bps"])

        self._data = self._data.groupby("seq").apply(lpc.compute_original)

        if self._data.apply(lpc.compare_restored, axis=1).all():
            print("Lossless :)")
        else:
            print("Not lossless :|")

    def print_stats(self, output_file: Path, stream: TextIO = sys.stdout):
        print(f"Number of frames: {len(self._data)}", file=stream)
        print(f"Source size: {self._source_size} ({self._source_size / 2 ** 20:.2f} MiB)", file=stream)
        size = self._data["stream_len"].sum()
        print(f"Length of bitstream: {size} bits, "
              f"bytes: {np.ceil(size / 8):.0f} aligned ({np.ceil(size / 8) / 2 ** 20:.2f} MiB)", file=stream)
        lpc_bytes = np.ceil(len(self._data) * self._lpc_precision * self._lpc_order * 1 / 8)
        print(f"Bytes needed for coefficients: {lpc_bytes:.0f} B", file=stream)
        print(f"Output file size: {output_file.stat().st_size}", file=stream)
        print(f"Ratio = {np.ceil(size / 8) / self._source_size:.3f}", file=stream)
        print(f"Ratio with LPC coeffs = {(np.ceil(size / 8) + lpc_bytes) / self._source_size:.3f}", file=stream)
        lpc_saved = lpc_bytes * (len(np.unique(self._data["channel"])) - 1)
        print(f"Grand Ratio with common LPC = {(output_file.stat().st_size - lpc_saved) / self._source_size:.3f}",
              file=stream)
        print(f"Curent grand Ratio = {output_file.stat().st_size / self._source_size:.3f}", file=stream)

        # FIXME: this is misleading
        print(f"Size of the resulting dataframe: {self.usage_mib():.3f} MiB", file=stream)

