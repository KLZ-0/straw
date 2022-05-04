import pandas as pd

from straw.lpc import steps
from straw.static import SubframeType

"""
Pandas-lever wrappers
"""


def compute_qlp(df: pd.DataFrame, order: int, qlp_coeff_precision: int) -> pd.DataFrame:
    """
    Compute LPC and quantize the LPC coefficients
    :param df: input dataframe with columns [frame]
    :param order: maximal LPC order
    :param qlp_coeff_precision: Bit precision for storing the quantized LPC coefficients
    :return: Series(qlp coefficients, qlp precision, qlp shift)
    """
    # frame["qlp"] = None
    # frame["qlp"] = frame["qlp"].astype(object)
    # frame["qlp_precision"] = 0
    # frame["shift"] = 0
    #
    # for idx, row in frame.iterrows():
    #     lpc = steps.compute_lpc(row["frame"], p=order)
    #     qlp, precision, shift = steps.quantize_lpc_cython(lpc, qlp_coeff_precision)
    #     frame["qlp_precision"][idx] = precision
    #     frame["shift"][idx] = shift
    #     frame["qlp"][idx] = qlp
    #
    # return frame

    df["qlp"] = None
    df["qlp_precision"] = 0
    df["shift"] = 0

    # Constant frames
    const_frames = df["frame"].apply(lambda x: not (x - x[0]).any())
    if const_frames.any():
        df.loc[const_frames, "frame_type"] = SubframeType.CONSTANT
        df.loc[~const_frames, "frame_type"] = SubframeType.RAW
        return df

    lpc = steps.compute_lpc(df["frame"], order)
    if lpc is None or not steps.lpc_is_stable(lpc):
        df["frame_type"] = SubframeType.RAW
        return df

    qlp, precision, shift = steps.quantize_lpc_cython(lpc, qlp_coeff_precision)
    df["frame_type"] = SubframeType.LPC_COMMON
    df["qlp"] = [qlp for _ in range(len(df["qlp"]))]
    df["qlp_precision"] = precision
    df["shift"] = shift
    return df


def compute_residual(df: pd.DataFrame):
    """
    Computes the residual from the given signal with quantized LPC coefficients
    :param df: input dfframe slice with columns [frame, qlp, shift]
    :return: the input dfframe slice with a [residual] column added
    """

    def wrap(dt: pd.DataFrame):
        if not (dt["frame_type"] in (SubframeType.LPC, SubframeType.LPC_COMMON)):
            return None
        return steps.predict_compute_residual(dt["frame"], dt["qlp"], dt["shift"])

    df["residual"] = df.apply(wrap, axis=1)
    df.loc[df["residual"].isna() & df["frame_type"].isin(
        (SubframeType.LPC, SubframeType.LPC_COMMON)), "frame_type"] = SubframeType.RAW


def compute_original(df: (pd.Series, pd.DataFrame), inplace=False):
    """
    Computes the original from the given residual signal with quantized LPC coefficients and warmup samples
    :param df: input dfframe with columns [frame, qlp, shift]
    :param inplace: whether the restoring should be done in place (faster)
    this ignores the residual column and expects the residual to be stored after the warmup samples in the frame column
    :return: residual as a numpy array
    """
    if isinstance(df, pd.DataFrame):
        return df.apply(compute_original, inplace=inplace, axis=1)

    if not (df["frame_type"] in (SubframeType.LPC, SubframeType.LPC_COMMON)):
        return df

    if inplace:
        frame = df["frame"]
    else:
        frame = df["frame"].copy()
        frame[len(df["qlp"]):] = df["residual"]
        df["restored"] = frame

    steps.restore_signal_cython(frame, df["qlp"], df["shift"])

    return df
