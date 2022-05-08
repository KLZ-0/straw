import numpy as np
import pandas as pd

from straw.io.params import StreamParams


class BaseCorrector:
    def apply(self, samplebuffer: np.ndarray, params: StreamParams) -> (np.ndarray, np.ndarray):
        """
        Apply a correction
        :param samplebuffer: array to apply the corrections to - ndarray with dimensions (channels, samples)
        :param params: stream params where the correction params should be stored
        :return: None
        """
        return

    def apply_revert(self, samplebuffer: np.ndarray, params: StreamParams) -> (np.ndarray, np.ndarray):
        """
        Apply the reverse of a correction
        :param samplebuffer: array to apply the corrections to - ndarray with dimensions (channels, samples)
        :param params: stream params where the correction params are be stored
        :return: None
        """
        return

    def df_wrap_apply(self, frameset: pd.Series):
        """
        DataFrame wrapper
        Used mainly for testing and evaluating localized corrections
        :param frameset: Series of frames
        :return: None
        """
        ndarr = np.stack(frameset.tolist())
        self.apply(ndarr, StreamParams())
        for i, idx in enumerate(frameset.index):
            frameset[idx][:] = ndarr[i]
