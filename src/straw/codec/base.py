from hashlib import md5

import numpy as np
import pandas as pd

from straw.io.params import StreamParams


class BaseCoder:
    # Member utils
    _flac_mode: bool

    # Member variables
    _default_frame_size: int = 1 << 12  # bytes
    _data: pd.DataFrame
    _params: StreamParams
    _samplebuffer: np.ndarray

    # TODO: these things should be in params
    _source_size: int

    def __init__(self, flac_mode=False):
        self._flac_mode = flac_mode

    def _slice_channel_data_into_frames(self, data: np.array, limits: np.array = None):
        if limits is None:
            return [data[i:i + self._default_frame_size] for i in range(0, len(data), self._default_frame_size)]
        else:
            return [data[
                    limits[i]:
                    limits[i] + data.shape[0] if i + 1 >= limits.shape[0] else limits[i + 1]
                    ] for i in range(limits.shape[0])]

    def usage_mib(self):
        """
        Returns the deep memory usage of the given dataframe in mebibytes
        :return: deep memory usage of the given dataframe in mebibytes
        """
        return self._data.memory_usage(index=True, deep=True).sum() / (2 ** 20)

    def sample_frame(self, seq=0) -> pd.Series:
        return self._data.loc[seq]

    def sample_frame_multichannel(self, seq=0) -> pd.DataFrame:
        return self._data[self._data["seq"] == seq]

    def get_data(self) -> pd.DataFrame:
        return self._data

    def get_soundfile_compatible_array(self) -> np.array:
        return self._samplebuffer.swapaxes(1, 0)

    def get_params(self) -> StreamParams:
        return self._params

    def get_md5(self):
        try:
            return md5(self._samplebuffer.swapaxes(1, 0)).digest()
        except ValueError:
            return md5(np.ascontiguousarray(self._samplebuffer.swapaxes(1, 0))).digest()
