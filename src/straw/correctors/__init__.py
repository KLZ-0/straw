import numpy as np

from .bias import BiasCorrector
from .gain import GainCorrector
from .shift import ShiftCorrector


def sub(x1, x2):
    diff = x1 - x2
    if diff.any():
        return diff
    else:
        return x1


def deconvolve(df, col_name: str = "frame"):
    """
    Finds the medium channels residual and subtracts it from all other channels
    :return: new residuals
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not in dataframe")

    variances = df[col_name].apply(np.var)
    mid = variances.mean()
    # mid_idx = np.abs(variances - mid).idxmin()

    # NOTE: for some reason subtracting the weakest channel does the smallest harm
    mid_idx = variances.idxmin()

    df[col_name] = df[col_name].apply(sub, x2=df[col_name][mid_idx])
    # df["residual"] = df["residual"].apply(np.subtract, x2=df["residual"][mid_idx])

    return df
