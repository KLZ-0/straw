import mmap
from pathlib import Path

import numpy as np
import pandas as pd
from crcmod import mkCrcFun

from straw import static
from straw.io.bitarray import SlicedBitarray
from straw.io.params import StreamParams
from straw.rice import Ricer


class BaseIO:
    _data: pd.DataFrame
    _params: StreamParams
    _f = None

    class Crc:
        crc8 = mkCrcFun(0x107, initCrc=0, rev=False)
        crc16 = mkCrcFun(0x18005, initCrc=0, rev=False)

    def _format_specific_checks(self):
        """
        This should be overridden
        :return: None
        """
        pass

    def _stream(self):
        """
        This should be overridden
        :return: None
        """
        pass

    def get_data(self):
        return self._data

    def get_params(self):
        return self._params


class BaseWriter(BaseIO):
    def __init__(self, data: pd.DataFrame, params: StreamParams):
        self._data = data
        self._params = params

    def save(self, output_file: Path):
        """
        Saves the dataframe into a FLAC formatted binary file
        :param output_file: target file
        :return: None
        """
        self._format_specific_checks()
        with output_file.open("wb") as f:
            self._f = f
            self._stream()

    @staticmethod
    def encode_int_utf8(val: int) -> bytes:
        return chr(val).encode("utf-8")


class BaseReader(BaseIO):
    _raw: list
    _ricer = Ricer()

    _sec: SlicedBitarray()
    _samplebuffer_ptr: int = 0
    _samplebuffer: np.array

    def __init__(self):
        self._params = StreamParams()
        self._raw = []

    def load(self, input_file: Path) -> (pd.DataFrame, StreamParams):
        """
        Saves the dataframe into a FLAC formatted binary file
        :param input_file: source file
        :return: dataframe and params
        """
        with input_file.open("rb") as f:
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._sec = SlicedBitarray(buffer=m)
            self._stream()
        self._format_specific_checks()
        self._raw.sort(key=lambda x: x["idx"])
        self._data = pd.DataFrame(self._raw, columns=static.columns)

    def _allocate_buffer(self):
        channels = self._params.channels
        total_samples = self._params.total_samples
        dtype_bits = static.soundfile_dtype[self._params.bits_per_sample]
        self._samplebuffer = np.zeros((total_samples, channels), dtype=f"int{dtype_bits}")
        self._samplebuffer = self._samplebuffer.swapaxes(1, 0)

    def get_buffer(self):
        return self._samplebuffer
