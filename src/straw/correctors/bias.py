import numpy as np
import pandas as pd


class BiasCorrector:
    def apply(self, df: pd.DataFrame):
        """
        Takes dataframe with 1-n channels
        TODO: deal with 1 channel
        :param df:
        :return:
        """

        for i, row in df.iterrows():
            df["frame"][i] = self.remove_bias(row["frame"])

        return df

    @staticmethod
    def remove_bias(frame: np.ndarray):
        return frame - int(frame.mean())
