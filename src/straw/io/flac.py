from pathlib import Path

import pandas as pd


class FLACFormat:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def save(self, output_file: Path):
        """
        Saves the dataframe into a FLAC formatted binary file
        :param output_file: target file
        :return: None
        """
        pass
