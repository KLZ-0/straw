from pathlib import Path

import pandas as pd
from crcmod import mkCrcFun

from straw.io.params import StreamParams


class BaseWriter:
    _f = None
    _params: StreamParams

    class Crc:
        crc8 = mkCrcFun(0x107, initCrc=0, rev=False)
        crc16 = mkCrcFun(0x18005, initCrc=0, rev=False)

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
