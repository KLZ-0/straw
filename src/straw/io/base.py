from pathlib import Path

import pandas as pd
from crcmod import mkCrcFun

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


class BaseReader(BaseIO):
    _raw: dict
    _ricer: Ricer()

    def __init__(self):
        self._params = StreamParams()
        self._raw = {}

    def load(self, input_file: Path) -> (pd.DataFrame, StreamParams):
        """
        Saves the dataframe into a FLAC formatted binary file
        :param input_file: source file
        :return: dataframe and params
        """
        with input_file.open("rb") as f:
            self._f = f
            self._stream()
        self._format_specific_checks()
        self._data = pd.DataFrame(self._raw)
        return self._data, self._params
