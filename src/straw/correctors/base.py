import numpy as np
import pandas as pd

from straw.io.params import StreamParams


class BaseCorrector:
    def apply(self, samplebuffer: np.ndarray, params: StreamParams) -> (np.ndarray, np.ndarray):
        return

    def apply_revert(self, samplebuffer: np.ndarray, params: StreamParams) -> (np.ndarray, np.ndarray):
        return

    def df_wrap_apply(self, frameset: pd.Series):
        ndarr = np.stack(frameset.tolist())
        self.apply(ndarr, StreamParams())
        for i, idx in enumerate(frameset.index):
            frameset[idx][:] = ndarr[i]
