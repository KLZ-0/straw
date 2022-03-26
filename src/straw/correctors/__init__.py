import numpy as np

from .bias import BiasCorrector
from .gain import GainCorrector
from .shift import ShiftCorrector


def sub(x1, x2):
    return x1 - x2


def deconvolve(df):
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

    return df
