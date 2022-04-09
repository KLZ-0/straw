import numpy as np
import pandas as pd


class ShiftCorrector:
    def get_lags(self, samplebuffer: np.ndarray) -> (np.ndarray, np.ndarray):
        leading_channel = self._find_leading_channel(samplebuffer, limit=10)
        lags = self._find_lags(samplebuffer, leading_channel, limit=10)
        return lags

    def _find_leading_channel(self, samplebuffer: np.ndarray, limit=10):
        lags = np.zeros(samplebuffer.shape[0], dtype=np.int8)
        reference = samplebuffer[0].astype(np.int64)
        for i in range(1, samplebuffer.shape[0]):
            lags[i] = self._double_sided_corr(samplebuffer[i], reference=reference, limit=limit)

        return np.argmin(lags)

    def _find_lags(self, samplebuffer: np.ndarray, leading_channel, limit=10) -> np.ndarray:
        lags = np.zeros(samplebuffer.shape[0], dtype=np.int8)
        reference = samplebuffer[leading_channel].astype(np.int64)
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
        frame = frame.astype(np.int64)
        return start + np.argmax([reference[:len(reference) - i].dot(frame[i:]) for i in range(start, end)])

    @staticmethod
    def choose_idx(frames: pd.Series):
        variances = frames.apply(np.var)
        # mid = variances.mean()
        mid = variances.min()
        return np.abs(variances - mid).idxmin()
