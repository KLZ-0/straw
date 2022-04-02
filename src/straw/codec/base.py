from hashlib import md5

import numpy as np
import pandas as pd

from straw.io.params import StreamParams
from straw.rice import Ricer


class BaseCoder:
    # Member utils
    _flac_mode: bool
    _ricer: Ricer

    # Member variables
    _default_frame_size: int = 4096  # bytes
    _data: pd.DataFrame
    _params: StreamParams
    _samplebuffer: np.array

    # TODO: these things should be in params
    _source_size: int

    def __init__(self, flac_mode=False):
        self._flac_mode = flac_mode
        self._ricer = Ricer(adaptive=True if not flac_mode else False)

    def _slice_channel_data_into_frames(self, data: np.array):
        return [data[i:i + self._default_frame_size] for i in range(0, len(data), self._default_frame_size)]

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

    def get_md5(self):
        return md5(self._samplebuffer.swapaxes(1, 0)).digest()
