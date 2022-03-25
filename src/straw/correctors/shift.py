import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class ShiftCorrector:
    def show_frame(self, data, file_name="tmp.png"):
        df = {"x": [], "value": [], "Channel": []}
        for i, ds in enumerate(data["frame"]):
            # ds = ds[:160]
            df["x"] += [i for i in range(len(ds))]
            df["Channel"] += [i for _ in ds]
            df["value"] += list(ds)
        df = pd.DataFrame(df)

        s = sns.relplot(data=df, kind="line", x="x", y="value", hue="Channel", height=2.5, aspect=3)

        plt.title("Frame residuals with common LPC coefficients (averaged autoc)")
        s.set_xlabels("Sample")
        s.set_ylabels("Sample value (16-bit)")
        s.tight_layout()
        plt.savefig("outputs/" + file_name)
        plt.show()

    @staticmethod
    def _tmp(s1: np.array, s2: np.array, start: int, end: int) -> np.array:
        return np.asarray([s1[:len(s1) - i].dot(s2[i:]) for i in range(start, end)])

    @staticmethod
    def _corr(s1: np.array, s2: np.array, start: int, end: int) -> np.array:
        return start + np.argmax([s1[:len(s1) - i].dot(s2[i:]) for i in range(start, end)])

    def align_frames(self, df: pd.DataFrame):
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

    def apply(self, df: pd.DataFrame):
        self.align_frames(df)
