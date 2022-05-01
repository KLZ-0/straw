from pathlib import Path
from typing import BinaryIO

import pandas as pd

from straw.io.flac import FLACFormatReader
from straw.io.params import StreamParams
from straw.io.straw import StrawFormatWriter, StrawFormatReader


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

    def save(self, df: pd.DataFrame, params: StreamParams, output_stream: BinaryIO, flac_mode: bool = False):
        """
        Saves the dataframe into a formatted binary file
        :param df: source dataframe
        :param params: stream params
        :param output_stream: target file
        :param flac_mode: if true the output file will be a FLAC decoder compatible file
        :return: None
        """
        self.validate_dataframe(df, params)
        if flac_mode:
            raise NotImplementedError("FLAC mode is no longer supported")
        else:
            fw = StrawFormatWriter(params, output_stream)
            fw.write(df)
            fw.close_stream(params)

    def load(self, input_file: Path, flac_mode: bool = False) -> (pd.DataFrame, StreamParams):
        if flac_mode:
            reader = FLACFormatReader()
        else:
            reader = StrawFormatReader()

        reader.load(input_file)
        self.validate_dataframe(reader.get_data(), reader.get_params())
        return reader
