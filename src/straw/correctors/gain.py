from fractions import Fraction

import numpy as np
import pandas as pd

from straw.correctors.base import BaseCorrector
from straw.io.params import StreamParams


class GainCorrector(BaseCorrector):
    def global_apply(self, samplebuffer: np.ndarray, params: StreamParams) -> (np.ndarray, np.ndarray):
        for i in range(1, samplebuffer.shape[0]):
            factor = self.equalize(samplebuffer[i], reference=samplebuffer[0])

    @staticmethod
    def energy(frame: np.ndarray):
        return frame.std()

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
        tmp = frame.astype(np.double)
        factor = Fraction((GainCorrector.energy(reference) / GainCorrector.energy(frame))).limit_denominator(1 << 12)
        tmp *= factor.numerator
        tmp /= factor.denominator
        frame[:] = tmp.astype(np.int16)
        return factor

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
        # mid = variances.min()
        mid = variances.max()
        return np.abs(variances - mid).idxmin()
