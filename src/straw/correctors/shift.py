import numpy as np
import pandas as pd


class ShiftCorrector:
    def apply(self, df: pd.DataFrame):
        """
        Takes dataframe with 1-n channels
        TODO: deal with 1 channel
        :param df:
        :return:
        """
        self.align_frames(df)

    @staticmethod
    def _tmp(s1: np.array, s2: np.array, start: int, end: int) -> np.array:
        return np.asarray([s1[:len(s1) - i].dot(s2[i:]) for i in range(start, end)])

    @staticmethod
    def _corr(s1: np.array, s2: np.array, start: int, end: int) -> np.array:
        return start + np.argmax([s1[:len(s1) - i].dot(s2[i:]) for i in range(start, end)])

    def align_frames(self, df: pd.DataFrame):
        # TODO

        variances = df["frame"].apply(np.var)
        mid = variances.mean()
        mid_idx = np.abs(variances - mid).idxmin()
        # for i, row in df.iterrows():
        #     lag = self._tmp(row["frame"], df["frame"][mid_idx], 0, 30)
        #     print(self._corr(row["frame"], df["frame"][mid_idx], 0, 30), lag)
        #     self.show_frame(df.loc[[0, mid_idx]])
        #     df["frame"][i] = df["frame"][i][lag:]
        #     exit()

        f1 = df["frame"][0]
        f2 = df["frame"][235]
        lag = self._corr(f1, f2, 0, 30)
        print(self._tmp(f1, f2, 0, 30), lag)
        # self.show_frame(df.loc[[235, 0]], "tmp1.png")
        df["frame"][0] = f1[:len(f1) - lag]
        df["frame"][235] = f2[lag:]
        # self.show_frame(df.loc[[235, 0]], "tmp2.png")
        f1 = df["frame"][0]
        f2 = df["frame"][235]
        lag = self._corr(f1, f2, 0, 30)
        print(self._tmp(f1, f2, 0, 30), lag)
        exit()
