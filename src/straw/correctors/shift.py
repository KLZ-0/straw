import numpy as np
import pandas as pd
from scipy.signal import get_window

from straw.correctors.base import BaseCorrector
from straw.io.params import StreamParams


class ShiftCorrector(BaseCorrector):
    def apply(self, samplebuffer: np.ndarray, params: StreamParams, limit=10) -> (np.ndarray, np.ndarray):
        windowed = (samplebuffer * get_window("nuttall", samplebuffer.shape[1])).astype(np.int64)
        leading_channel = self._find_leading_channel(windowed, limit=limit)
        params.lags = self._find_lags(windowed, leading_channel, limit=limit)
        params.leading_channel = leading_channel
        total_size = samplebuffer.shape[1] - np.max(params.lags)
        for i in range(samplebuffer.shape[0]):
            lag = params.lags[i]
            params.removed_samples_start.append(samplebuffer[i][:lag])
            params.removed_samples_end.append(samplebuffer[i][total_size + lag:])

    def _find_leading_channel(self, samplebuffer: np.ndarray, limit):
        lags = np.zeros(samplebuffer.shape[0], dtype=np.int8)
        reference = samplebuffer[0]
        for i in range(1, samplebuffer.shape[0]):
            lags[i] = self._double_sided_corr(samplebuffer[i], reference=reference, limit=limit)

        return np.argmin(lags)

    def _find_lags(self, samplebuffer: np.ndarray, leading_channel, limit) -> np.ndarray:
        lags = np.zeros(samplebuffer.shape[0], dtype=np.int8)
        reference = samplebuffer[leading_channel]
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
