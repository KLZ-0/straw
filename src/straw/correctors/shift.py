import numpy as np
import pandas as pd

from straw.correctors.base import BaseCorrector


class ShiftCorrector(BaseCorrector):
    def apply(self, df: pd.DataFrame, col_name: str = "frame"):
        """
        Takes dataframe with 1-n channels
        Strip samples from start
        TODO: deal with 1 channel
        :param col_name:
        :param df:
        :return:
        """
        super().apply(df, col_name)

        lags, ref_idx = self.get_lags(df[col_name])
        max_lag = np.max(lags)

        # show_frame(df, limit=60)

        for i, row in df.iterrows():
            lag = lags[i]
            new_frame_size = len(row[col_name]) - max_lag
            df[col_name][i] = row[col_name][lag:new_frame_size + lag]

        # show_frame(df, limit=60)

        # exit()
        return df

    def get_lags(self, frames, limit=10):
        """
        Return lags compared to the leading channel, and the index of the leading channel
        :param limit:
        :param frames:
        :return:
        """
        lags = frames.apply(self._double_sided_corr, reference=frames[frames.index.min()].astype(np.int64), limit=limit)
        idx_min = lags.idxmin()
        return frames.apply(self._corr, reference=frames[idx_min].astype(np.int64), end=limit - 1), idx_min

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
