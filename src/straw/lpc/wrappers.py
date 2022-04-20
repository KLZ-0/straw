import numpy as np
import pandas as pd

from straw.lpc import steps
from straw.static import SubframeType

"""
Pandas-lever wrappers
"""


def compute_qlp(frame: pd.DataFrame, order: int, qlp_coeff_precision: int) -> pd.DataFrame:
    """
    Compute LPC and quantize the LPC coefficients
    :param frame: input dataframe with columns [frame]
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

    if not (frame["frame_type"] == SubframeType.LPC).all():
        return None

    df = pd.Series({
        "qlp": np.array([]),
        "qlp_precision": 0,
        "shift": 0,
    })

    lpc = steps.compute_lpc(frame["frame"], order)
    if lpc is None:
        return pd.DataFrame([df], index=[frame.index[0]], copy=False)

    qlp, precision, shift = steps.quantize_lpc_cython(lpc, qlp_coeff_precision)
    df["qlp"] = qlp
    df["qlp_precision"] = precision
    df["shift"] = shift
    return pd.DataFrame([df], index=[frame.index[0]], copy=False)


def compute_residual(data: pd.DataFrame):
    """
    Computes the residual from the given signal with quantized LPC coefficients
    :param data: input dataframe slice with columns [frame, qlp, shift]
    :return: the input dataframe slice with a [residual] column added
    """
    qlp_idx = data[["qlp"]].first_valid_index()
    shift_idx = data[["shift"]].first_valid_index()

    if qlp_idx is None:
        data["residual"] = data["frame"].apply(lambda x: x[[0]])
    elif isinstance(data["frame"], np.ndarray):
        data["residual"] = steps.predict_compute_residual(data["frame"], data["qlp"], data["shift"])
    else:
        data["residual"] = data["frame"].apply(steps.predict_compute_residual,
                                               qlp=data["qlp"][qlp_idx],
                                               shift=int(data["shift"][shift_idx]))
        if data["residual"].isna().any():
            data["frame_type"] = SubframeType.RAW


def _compute_original_df_expander(data: pd.DataFrame, qlp, shift, inplace):
    return steps.restore_signal_cython(data["frame"], qlp, shift, inplace)


def compute_original(data: pd.DataFrame, inplace=False):
    """
    Computes the original from the given residual signal with quantized LPC coefficients and warmup samples
    :param data: input dataframe with columns [frame, qlp, shift]
    :param inplace: whether the restoring should be done in place (faster)
    :return: residual as a numpy array
    """
    if not (data["frame_type"] == SubframeType.LPC).all():
        return data

    qlp = data["qlp"][data["qlp"].first_valid_index()]
    shift = int(data["shift"][data["shift"].first_valid_index()])

    tmp = data.apply(_compute_original_df_expander, qlp=qlp, shift=shift, axis=1, result_type="reduce", inplace=inplace)

    if not inplace:
        data["restored"] = tmp
        return data
