from fractions import Fraction

import numpy as np
import pandas as pd


class GainCorrector:
    def apply(self, df: pd.DataFrame):
        """
        Takes dataframe with 1-n channels
        TODO: deal with 1 channel
        :param df:
        :return:
        """

        ref_idx = self.choose_idx(df["frame"])
        for i, row in df.iterrows():
            if i == ref_idx:
                continue
            df["frame"][i], factor = self.equalize(row["frame"], df["frame"][ref_idx])

        return df

    @staticmethod
    def energy(frame: np.ndarray):
        return frame.var()

    @staticmethod
    def equalize(frame: np.ndarray, reference: np.ndarray):
        """
        Equalizes frame to match the reference
        TODO: This is not lossless!!!
        :param frame:
        :param reference:
        :return:
        """
        # frame: 6
        # ref: 1
        # expected ratio > 1.0
        frame = frame.astype(np.float)
        factor = Fraction((GainCorrector.energy(reference) / GainCorrector.energy(frame))).limit_denominator(1 << 12)
        frame *= factor.numerator
        frame /= factor.denominator
        return frame.astype(np.int16), factor

    @staticmethod
    def deequalize(frame: np.ndarray, factor):
        frame = frame.astype(np.float)
        frame *= factor.denominator
        frame /= factor.numerator
        return frame.astype(np.int16)

    @staticmethod
    def choose_idx(frames: pd.Series):
        variances = frames.apply(np.var)
        # mid = variances.mean()
        mid = variances.min()
        return np.abs(variances - mid).idxmin()
