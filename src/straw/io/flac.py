from pathlib import Path

import pandas as pd


class FormatFLAC:
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

    def save(self, strawfile: Path):
        """
        Saves the dataframe into a straw formatted binary file
        :param strawfile: target file
        :return: None
        """
        self.validate_dataframe()
