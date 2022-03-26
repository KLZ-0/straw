import numpy as np
import pandas as pd

from straw.correctors.base import BaseCorrector


class BiasCorrector(BaseCorrector):
    def apply(self, df: pd.DataFrame, col_name: str = "frame"):
        """
        Takes dataframe with 1-n channels
        TODO: deal with 1 channel
        :param col_name:
        :param df:
        :return:
        """
        super().apply(df, col_name)

        for i, row in df.iterrows():
            df[col_name][i] = self.remove_bias(row[col_name])

        return df

    @staticmethod
    def remove_bias(frame: np.ndarray):
        return frame - int(frame.mean())
