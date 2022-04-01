from hashlib import md5

import pandas as pd

from straw.io.params import StreamParams
from straw.rice import Ricer


class BaseCoder:
    # Member utils
    _flac_mode: bool
    _ricer: Ricer

    # Member variables
    _raw: list
    _data: pd.DataFrame
    _params: StreamParams
    _md5: md5

    # TODO: these things should be in params
    _source_size: int
    _samplerate: int

    def __init__(self, flac_mode=False):
        self._flac_mode = flac_mode
        self._ricer = Ricer(adaptive=True if not flac_mode else False)
        self._params = StreamParams()

    def usage_mib(self):
        """
        Returns the deep memory usage of the given dataframe in mebibytes
        :return: deep memory usage of the given dataframe in mebibytes
        """
        return self._data.memory_usage(index=True, deep=True).sum() / (2 ** 20)

    def sample_frame(self) -> pd.Series:
        return self._data.loc[0]

    def sample_frame_multichannel(self) -> pd.DataFrame:
        return self._data[self._data["seq"] == 0]

    def get_data(self) -> pd.DataFrame:
        return self._data
