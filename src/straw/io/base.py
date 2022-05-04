import mmap
from pathlib import Path
from typing import BinaryIO

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
    def __init__(self, initial_params: StreamParams, output_stream: BinaryIO):
        self._params = initial_params
        self._f = output_stream
        self._stream()

    def init_stream(self, params):
        self._params = params
        self._stream()

    def write(self, data):
        secs = data.groupby("seq").apply(self._frame)
        for sec in secs:
            sec.tofile(self._f)

    def close_stream(self, params):
        self._f.seek(0)
        self._params = params
        self._stream()

    def _frame(self, df: pd.DataFrame):
        """
        This should be overridden
        :return: None
        """
        pass

    @staticmethod
    def encode_int_utf8(val: int) -> bytes:
        return chr(val).encode("utf-8")


class BaseReader(BaseIO):
    _raw: list
    _ricer = Ricer()

    _sec: SlicedBitarray()
    _memview: bytes
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
            self._memview = memoryview(self._sec).tobytes()
            self._stream()
        self._format_specific_checks()
        self._raw.sort(key=lambda x: x["idx"])
        self._data = pd.DataFrame(self._raw, columns=static.columns, copy=False)

    def _allocate_buffer(self):
        channels = self._params.channels
        total_samples = self._params.total_samples
        dtype_bits = static.soundfile_dtype[self._params.bits_per_sample]
        self._samplebuffer = np.zeros((total_samples, channels), dtype=f"int{dtype_bits}")
        self._samplebuffer = self._samplebuffer.swapaxes(1, 0)

    def get_buffer(self):
        return self._samplebuffer
