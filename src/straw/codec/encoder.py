import sys
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
    _params = StreamParams()

    # TODO: make the coders accept other sizes
    _bits_per_sample = 16  # stored in StreamParams once per group

    def load_file(self, filename: Path):
        """
        Load the specified file
        :param filename: file to load
        :return: True on success, False on error
        """
        # TODO: verify if the files are from the same recording
        self._source_size = 0

        self._samplebuffer, sr = soundfile.read(filename, dtype=f"int{self._bits_per_sample}")
        self._source_size = self._samplebuffer.nbytes
        self._params.sample_rate = sr

        if len(self._samplebuffer.shape) > 1:
            self._samplebuffer = self._samplebuffer.swapaxes(1, 0)

        self._params.md5 = self.get_md5()
        return True

    def create_frames(self):
        ds = {"seq": [], "frame": [], "channel": []}
        for channel, channel_data in enumerate(self._samplebuffer):
            sliced = self._slice_channel_data_into_frames(channel_data)
            ds["seq"] += [i for i in range(len(sliced))]
            ds["frame"] += sliced
            ds["channel"] += [channel for _ in range(len(sliced))]

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
        self._parametrize()
        Formatter().save(self._data, self._params, output_file, self._flac_mode)
        # TODO: actually save bitstreams

    def _parametrize(self):
        self._params.max_block_size = int(self._data["frame"].apply(len).max())
        self._params.min_block_size = self._params.max_block_size
        max_residual_bytes = (self._data["stream_len"].max() // 8) + 1
        self._params.min_frame_size = 0  # unknown
        self._params.max_frame_size = int(max_residual_bytes) + 1000
        self._params.channels = len(np.unique(self._data["channel"]))
        self._params.bits_per_sample = self._bits_per_sample
        self._params.total_samples = int(self._data[self._data["channel"] == 0]["frame"].apply(len).sum())

    def print_stats(self, output_file: Path, stream: TextIO = sys.stdout):
        print(f"Number of frames: {len(self._data)}", file=stream)
        print(f"Source size: {self._source_size} ({self._source_size / 2 ** 20:.2f} MiB)", file=stream)
        size = self._data["stream_len"].sum()
        print(f"md5: {self._params.md5.hex(' ')}", file=stream)
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

