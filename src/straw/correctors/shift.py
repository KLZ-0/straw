import numpy as np
import pandas as pd
from scipy.signal import get_window

from straw.compute import ParallelCompute
from straw.correctors.base import BaseCorrector
from straw.io.params import StreamParams


class ShiftCorrector(BaseCorrector):
    _lags: np.array = None

    def apply(self, samplebuffer: np.ndarray, params: StreamParams, limit=10, parallel=True) -> (
    np.ndarray, np.ndarray):
        windowed = (samplebuffer * get_window("nuttall", samplebuffer.shape[1])).astype(np.int64)
        leading_channel = self._find_leading_channel(windowed, limit=limit, parallel=parallel)
        params.lags = self._find_lags(windowed, leading_channel, limit=limit)
        params.leading_channel = leading_channel
        total_size = samplebuffer.shape[1] - np.max(params.lags)
        for i in range(samplebuffer.shape[0]):
            lag = params.lags[i]
            params.removed_samples_start.append(samplebuffer[i][:lag])
            params.removed_samples_end.append(samplebuffer[i][total_size + lag:])
        self._lags = params.lags

    def apply_to_ndarray(self, data):
        """
        Performs shift inplace by rewriting each sample by a rolled array
        NOTE: this is very inefficient and this operation is normally performed at framing/blocking
         and this method is used only in figures where it needs to be done explicitly
        """
        for channel in range(data.shape[0]):
            lag = self._lags[channel]
            data[channel] = np.roll(data[channel], -lag)

    def df_wrap_apply(self, frameset: pd.Series):
        ndarr = np.stack(frameset.tolist())
        self.apply(ndarr, StreamParams(), parallel=False)
        self.apply_to_ndarray(ndarr)
        for i, idx in enumerate(frameset.index):
            frameset[idx][:] = ndarr[i]

    def _find_leading_channel(self, samplebuffer: np.ndarray, limit, parallel):
        lags = np.zeros(samplebuffer.shape[0], dtype=np.int8)
        reference = samplebuffer[0]
        if parallel:
            lags[:] = ParallelCompute.get_instance().map_ndarray(samplebuffer, self._double_sided_corr,
                                                                 reference=reference,
                                                                 limit=limit)
        else:
            for i in range(1, samplebuffer.shape[0]):
                lags[i] = self._double_sided_corr(samplebuffer[i], reference=reference, limit=limit)

        return np.argmin(lags)

    def _find_lags(self, samplebuffer: np.ndarray, leading_channel, limit) -> np.ndarray:
        lags = np.zeros(samplebuffer.shape[0], dtype=np.int8)
        reference = samplebuffer[leading_channel]
        # lags[:] = ParallelCompute.get_instance().map_ndarray(samplebuffer, self._corr, reference=reference,
        #                                                      end=limit-1)
        # lags[leading_channel] = 0
        for i in range(samplebuffer.shape[0]):
            if i == leading_channel:
                continue
            lags[i] = self._corr(frame=samplebuffer[i], reference=reference, end=limit - 1)
        return lags

    def _double_sided_corr(self, frame: np.array, reference: np.array, limit):
        lag_pos = self._corr(frame, reference, limit // 2 - 1)
        lag_neg = self._corr(reference, frame, limit // 2)
        return lag_pos if lag_pos >= lag_neg else -lag_neg

    @staticmethod
    def _corr(frame: np.array, reference: np.array, end: int, start: int = 0) -> np.array:
        return start + np.argmax([reference[:len(reference) - i].dot(frame[i:]) for i in range(start, end)])

    @staticmethod
    def choose_idx(frames: pd.Series):
        variances = frames.apply(np.var)
        # mid = variances.mean()
        mid = variances.min()
        return np.abs(variances - mid).idxmin()
