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


def deconvolve(df, col_name: str = "residual"):
    """
    Finds the medium channels residual and subtracts it from all other channels
    :return: new residuals
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not in dataframe")

    df[col_name] = df[col_name].apply(sub, x2=df[col_name][df.index[0]])
    # df["residual"] = df["residual"].apply(np.subtract, x2=df["residual"][mid_idx])

    return df


def localized_sub(x1, x2):
    diff = x1 - x2
    # too low will cause noise to be decorrelated
    # too high will cause audio that could be decorrelated to not be decorrelated
    limits = 150
    min_section_len = 2  # at least 2
    if diff.any():
        # return x1^x2
        nonzero = np.nonzero(np.abs(x2) > limits)[0]
        x1[nonzero] = diff[nonzero]
        return x1
    else:
        return x1


def localized_deconvolve(df, col_name: str = "residual"):
    """
    Finds the medium channels residual and subtracts it from all other channels
    :return: new residuals
    """
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not in dataframe")

    df[col_name] = df[col_name].apply(localized_sub, x2=df[col_name][df.index[0]])
    # df["residual"] = df["residual"].apply(np.subtract, x2=df["residual"][mid_idx])

    return df
