from hashlib import md5

import numpy as np
import pandas as pd

from straw.io.params import StreamParams


class BaseCoder:
    """
    Base class for the Encoder and Decoder
    """
    # Member utils
    _flac_mode: bool

    # Member variables
    _default_frame_size: int = 1 << 12  # bytes
    _data: pd.DataFrame
    _params: StreamParams
    _samplebuffer: np.ndarray

    _source_size: int

    _supported_subtypes = {
        "PCM_16": 16,
        # "PCM_24": 24,
        # "PCM_32": 32,
    }

    _subtype_pattern = "PCM_{}"

    def __init__(self, flac_mode=False, show_progress: bool = False):
        """
        Common constructor for Encoder and Decoder
        :param flac_mode: False - FLAC mode is not supported anymore
        :param show_progress: show encoding progress - currently has no effect for the encoder
        """
        if flac_mode:
            raise NotImplementedError("FLAC mode is not supported anymore")
        self._flac_mode = flac_mode
        self.show_progress = show_progress

    def _slice_channel_data_into_frames(self, data: np.array, limits: np.array = None):
        """
        Slice the given 1D data into a list of numpy views
        :param data: input samplebuffer
        :param limits: optional slicing points, if None then the default block size is used
        :return: list of numpy arrays (views)
        """
        if limits is None:
            return [data[i:i + self._default_frame_size] for i in range(0, len(data), self._default_frame_size)]
        else:
            return [data[limits[i]:limits[i + 1]] for i in range(limits.shape[0] - 1)]

    def usage_mib(self):
        """
        Returns the deep memory usage of the given dataframe in mebibytes
        :return: memory usage of the given dataframe
        """
        return self._data.memory_usage(index=True, deep=True).sum() / (2 ** 20)

    def sample_frame(self, seq=0) -> pd.Series:
        """
        Get a sample frame - one channel - series
        :param seq: the sequence number of the requested frame
        :return: sample frame
        """
        return self._data.loc[seq]

    def sample_frame_multichannel(self, seq=0) -> pd.DataFrame:
        """
        Get a sample frame - multiple channels - dataframe
        :param seq: the sequence number of the requested frame
        :return: sample frame
        """
        return self._data[self._data["seq"] == seq]

    def samplebuffer_frame_multichannel(self, seq=0, blocksize=4096) -> np.array:
        """
        Get a sample frame - multiple channels - array
        :param seq: the sequence number of the requested frame
        :param blocksize: presumed size of blocks when slicing
        :return: sample frame
        """
        return self._samplebuffer[:, seq * blocksize:seq * blocksize + blocksize].copy()

    def get_data(self) -> pd.DataFrame:
        """
        Getter for data
        :return: data
        """
        return self._data

    def get_soundfile_compatible_array(self) -> np.array:
        """
        Get a soundfile-compatible numpy view of the samplebuffer
        :return: soundfile-compatible numpy array
        """
        return self._samplebuffer.swapaxes(1, 0)

    def get_params(self) -> StreamParams:
        """
        Getter for params
        :return: params
        """
        return self._params

    def get_md5(self) -> bytes:
        """
        Calculates the md5sum of the current samplebuffer
        :return: md5 digests
        """
        try:
            return md5(self._samplebuffer.swapaxes(1, 0)).digest()
        except ValueError:
            return md5(np.ascontiguousarray(self._samplebuffer.swapaxes(1, 0))).digest()
