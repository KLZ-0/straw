import numpy as np
import pandas as pd


class BiasCorrector:
    def apply(self, df: pd.DataFrame, col_name: str = "frame"):
        """
        Takes dataframe with 1-n channels
        TODO: deal with 1 channel
        :param col_name:
        :param df:
        :return:
        """

        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not in dataframe")

        for i, row in df.iterrows():
            df[col_name][i] = self.remove_bias(row[col_name])

        return df

    @staticmethod
    def remove_bias(frame: np.ndarray):
        return frame - int(frame.mean())
