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
    _data = None
    _residuals = None

    def load_files(self, filenames: list):
        """
        Load all specified files as separate channels of the same recording
        :param filenames: list of files to load
        :return: True on success, False on error
        """
        self._data = []

        # TODO: verify if the files are from the same recording
        for filename in filenames:
            data, sr = soundfile.read(filename, dtype="int16")
            self._source_size = len(data) * 2
            if self._samplerate is None:
                self._samplerate = sr

            if self._samplerate != sr:
                self._clean()
                return False

            self._data.append(data)

        return True

    def _slice_data_into_frames(self, data):
        return [data[i:i + self._frame_size] for i in range(0, len(data), self._frame_size)]

    def create_frames(self):
        self._data = pd.DataFrame({k: self._slice_data_into_frames(v) for k, v in enumerate(self._data)})

    def load_stream(self, stream, samplerate):
        pass

    def encode(self):
        self._residuals = []
        for channel, frames in enumerate(self._data):
            for frame_number, frame in enumerate(frames):
                qlp, quant_level = lpc.compute_qlp(self._data[channel][frame_number],
                                                   self._lpc_order, self._lpc_precision)

                self._residuals.append(lpc.compute_residual(frame, qlp, self._lpc_order, quant_level))

    def save_file(self, filename):
        print(f"Number of frames: {len(self._residuals)}")
        for res in self._residuals:
            self._encoder.encode_frame(res)

        size = self._encoder.get_size_bits_unaligned()
        print(f"Source size: {self._source_size}")
        print(f"Length of bitstream: {size} bits (bytes: {size / 8:.2f} unaligned, {np.ceil(size / 8):.0f} aligned)")
        print(f"Ratio = {np.ceil(size / 8) / self._source_size:.2f}")

    def _clean(self):
        self._data = None
        self._samplerate = None
        self._frame_size = None
        self._residuals = None
