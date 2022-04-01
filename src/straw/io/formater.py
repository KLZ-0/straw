from pathlib import Path

import numpy as np
import pandas as pd

from straw.io.flac import FLACFormatWriter
from straw.io.params import StreamParams


class Formatter:
    """
    Base formatter class
    """
    formats = {
        ".flac": FLACFormatWriter,
        ".straw": FLACFormatWriter
    }

    def __init__(self, data: pd.DataFrame):
        self._data = data

    def validate_dataframe(self, params: StreamParams):
        """
        Validate the contents of the dataframe before opening the file stream
        Raise an error on invalid data
        :return: None
        """
        if len(self._data) == 0:
            raise ValueError("Empty dataframe")
        if params.channels == 0:
            raise ValueError(f"Invalid number of channels: {params.channels}")
        if params.bits_per_sample == 0:
            raise ValueError(f"Invalid bits per sample: {params.bits_per_sample}")

    def save(self, output_file: Path):
        """
        Saves the dataframe into a formatted binary file
        :param output_file: target file
        :return: None
        """
        params = self._parametrize()
        self.validate_dataframe(params)
        self.formats[output_file.suffix](self._data, params).save(output_file)

    def _parametrize(self) -> StreamParams:
        params = StreamParams()
        params.min_block_size = self._data.block_size
        params.max_block_size = self._data.block_size
        params.min_frame_size = 0  # unknown
        params.max_frame_size = 0  # unknown
        params.sample_rate = self._data.sample_rate
        params.channels = len(np.unique(self._data["channel"]))
        params.bits_per_sample = self._data.bits_per_sample
        params.total_samples = int(self._data[self._data["channel"] == 0]["frame"].apply(len).sum())
        params.md5 = self._data.md5.digest()
        return params
