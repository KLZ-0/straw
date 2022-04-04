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
