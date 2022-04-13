import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from figures.base import BasePlot


def sub(x1, x2):
    return x1 - x2


def func(df):
    """
    Finds the medium channels residual and subtracts it from all other channels
    :return: new residuals
    """
    variances = df["residual"].apply(np.var)
    mid = variances.mean()
    mid_idx = np.abs(variances - mid).idxmin()

    # NOTE: for some reason subtracting the weakest channel does the smallest harm
    # mid_idx = variances.idxmin()

    df["residual"] = df["residual"].apply(sub, x2=df["residual"][mid_idx])
    # df["residual"] = df["residual"].apply(np.subtract, x2=df["residual"][mid_idx])

    new_variances = df["residual"].apply(np.var)

    ov = sum(variances)
    nv = sum(new_variances) + variances[mid_idx]
    print(f"Old: {ov}\nNew: {nv}")
    return df["residual"]


class Experiments(BasePlot):
    def tmp(self):
        data = self._e.sample_frame_multichannel()

        new_residuals = func(data)
        data["residual"] = new_residuals

        df = {"x": [], "value": [], "Channel": []}
        for i, ds in enumerate(data["residual"]):
            ds = ds[:60]
            df["x"] += [i for i in range(len(ds))]
            df["Channel"] += [i for _ in ds]
            df["value"] += list(ds)
        df = pd.DataFrame(df)

        s = sns.relplot(data=df, kind="line", x="x", y="value", hue="Channel", height=2.5, aspect=3)

        plt.title("Frame residuals with common LPC coefficients (averaged autoc)")
        s.set_xlabels("Sample")
        s.set_ylabels("Sample value (16-bit)")
        s.tight_layout()

        self.save("tmp.png", True)
