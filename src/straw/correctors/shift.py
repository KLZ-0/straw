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

        ref_idx = self.choose_idx(df[col_name])

        lags = df[col_name].apply(self._corr, reference=df[col_name][ref_idx], end=10)
        max_lag = np.max(lags)

        # show_frame(df, limit=60)

        for i, row in df.iterrows():
            lag = lags[i]
            new_frame_size = len(row[col_name]) - max_lag
            df[col_name][i] = row[col_name][lag:new_frame_size + lag]

        # show_frame(df, limit=60)

        # exit()
        return df

    @staticmethod
    def _tmp(s1: np.array, s2: np.array, start: int, end: int) -> np.array:
        return np.asarray([s1[:len(s1) - i].dot(s2[i:]) for i in range(start, end)])

    @staticmethod
    def _corr(frame: np.array, reference: np.array, end: int, start: int = 0) -> np.array:
        return start + np.argmax([reference[:len(reference) - i].dot(frame[i:]) for i in range(start, end)])

    @staticmethod
    def choose_idx(frames: pd.Series):
        variances = frames.apply(np.var)
        # mid = variances.mean()
        mid = variances.min()
        return np.abs(variances - mid).idxmin()

    def align_frames(self, df: pd.DataFrame):
        pass

        # for i, row in df.iterrows():
        #     lag = self._tmp(row["frame"], df["frame"][mid_idx], 0, 30)
        #     print(self._corr(row["frame"], df["frame"][mid_idx], 0, 30), lag)
        #     self.show_frame(df.loc[[0, mid_idx]])
        #     df["frame"][i] = df["frame"][i][lag:]
        #     exit()

        # f1 = df["frame"][0]
        # f2 = df["frame"][235]
        # lag = self._corr(f1, f2, 0, 30)
        # print(self._tmp(f1, f2, 0, 30), lag)
        # # self.show_frame(df.loc[[235, 0]], "tmp1.png")
        # df["frame"][0] = f1[:len(f1) - lag]
        # df["frame"][235] = f2[lag:]
        # # self.show_frame(df.loc[[235, 0]], "tmp2.png")
        # f1 = df["frame"][0]
        # f2 = df["frame"][235]
        # lag = self._corr(f1, f2, 0, 30)
        # print(self._tmp(f1, f2, 0, 30), lag)
