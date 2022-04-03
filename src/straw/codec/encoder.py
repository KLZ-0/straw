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

    ##########
    # Public #
    ##########

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
        self._create_dataframe()

    def encode(self):
        """
        Encode the signal
        :return: None
        """
        tmp = self._data.groupby("seq").apply(lpc.compute_qlp, self._lpc_order, self._lpc_precision)
        self._data[["qlp", "qlp_precision", "shift"]] = pd.DataFrame(tmp.to_list())
        self._data = self._data.groupby("seq").apply(lpc.compute_residual)
        self._data["bps"] = np.full(len(self._data["residual"]), 4, dtype="B")
        self._data["stream"] = self._ricer.frames_to_bitstreams(self._data["residual"], self._data["bps"])
        self._data["stream_len"] = self._data["stream"].apply(len)

    def save_file(self, output_file: Path):
        """
        Save the encoded signal
        :param output_file: target file
        :return: None
        """
        self._parametrize()
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
        for channel, channel_data in enumerate(self._samplebuffer):
            sliced = self._slice_channel_data_into_frames(channel_data)
            ds["seq"] += [i for i in range(len(sliced))]
            ds["frame"] += sliced
            ds["channel"] += [channel for _ in range(len(sliced))]

        self._data = pd.DataFrame(ds)

    def _parametrize(self):
        """
        Parameter extraction to be used for encoding the whole stream
        :return: None
        """
        self._params.max_block_size = int(self._data["frame"].apply(len).max())
        self._params.min_block_size = self._params.max_block_size
        max_residual_bytes = (self._data["stream_len"].max() // 8) + 1
        self._params.min_frame_size = 0  # unknown
        self._params.max_frame_size = int(max_residual_bytes) + 1000
        self._params.channels = len(np.unique(self._data["channel"]))
        self._params.bits_per_sample = self._bits_per_sample
        self._params.total_samples = int(self._data[self._data["channel"] == 0]["frame"].apply(len).sum())

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

