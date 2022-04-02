from pathlib import Path

import numpy as np
import pandas as pd

from straw.io.flac import FLACFormatWriter, FLACFormatReader
from straw.io.params import StreamParams


class Formatter:
    """
    Base formatter class
    """
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, params: StreamParams):
        """
        Validate the contents of the dataframe before opening the file stream
        Raise an error on invalid data
        :param df: source dataframe
        :param params: stream params
        :return: None
        """
        # TODO: make normal checks
        return
        # noinspection PyUnreachableCode
        if len(df) == 0:
            raise ValueError("Empty dataframe")
        if params.channels == 0:
            raise ValueError(f"Invalid number of channels: {params.channels}")
        if params.bits_per_sample == 0:
            raise ValueError(f"Invalid bits per sample: {params.bits_per_sample}")

    def save(self, df: pd.DataFrame, output_file: Path, flac_mode: bool = False):
        """
        Saves the dataframe into a formatted binary file
        :param df: source dataframe
        :param output_file: target file
        :param flac_mode: if true the output file will be a FLAC decoder compatible file
        :return: None
        """
        params = self._parametrize(df)
        self.validate_dataframe(df, params)
        if flac_mode:
            FLACFormatWriter(df, params).save(output_file)
        else:
            # TODO
            pass

    def load(self, input_file: Path, flac_mode: bool = False) -> (pd.DataFrame, StreamParams):
        if flac_mode:
            df, params = FLACFormatReader().load(input_file)
            self.validate_dataframe(df, params)
            return df, params
        else:
            # TODO
            pass

    @staticmethod
    def _parametrize(df: pd.DataFrame) -> StreamParams:
        params = StreamParams()
        params.min_block_size = df.block_size
        params.max_block_size = df.block_size
        max_residual_bytes = (df["stream_len"].max() // 8) + 1
        params.min_frame_size = 0  # unknown
        params.max_frame_size = int(max_residual_bytes) + 1000
        params.sample_rate = df.sample_rate
        params.channels = len(np.unique(df["channel"]))
        params.bits_per_sample = df.bits_per_sample
        params.total_samples = int(df[df["channel"] == 0]["frame"].apply(len).sum())
        params.md5 = df.md5.digest()
        return params
