from pathlib import Path

import pandas as pd

from straw.io.flac import FLACFormat


class Formatter:
    """
    Base formatter class
    """
    formats = {
        ".flac": FLACFormat,
        ".straw": FLACFormat
    }

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def validate_dataframe(self):
        """
        Validate the contents of the dataframe before opening the file stream
        Raise an error on invalid data
        :return: None
        """
        if len(self.data) == 0:
            raise ValueError("Empty dataframe")

    def save(self, output_file: Path):
        """
        Saves the dataframe into a formatted binary file
        :param output_file: target file
        :return: None
        """
        self.validate_dataframe()
        self.formats[output_file.suffix](self.data).save(output_file)
