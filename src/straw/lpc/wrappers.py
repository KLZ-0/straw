import numpy as np
import pandas as pd

from straw.lpc import steps

"""
Pandas-lever wrappers
"""


def compute_qlp(frame, order: int, qlp_coeff_precision: int) -> (np.array, int):
    """
    Compute LPC and quantize the LPC coefficients
    :param frame: input dataframe with columns [frame]
    :param order: maximal LPC order
    :param qlp_coeff_precision: Bit precision for storing the quantized LPC coefficients
    :return: tuple(qlp coefficients, quantization level)
    """
    lpc = steps.compute_lpc(frame["frame"], order)
    if lpc is None:
        return None, 0, 0

    return steps.quantize_lpc_cython(lpc, qlp_coeff_precision)


def compute_residual(data: pd.DataFrame):
    """
    Computes the residual from the given signal with quantized LPC coefficients
    :param data: input dataframe slice with columns [frame, qlp, shift]
    :return: the input dataframe slice with a [residual] column added
    """
    qlp_idx = data[["qlp"]].first_valid_index()
    shift_idx = data[["shift"]].first_valid_index()

    if qlp_idx is None or shift_idx is None:
        data["residual"] = None
    elif isinstance(data["frame"], np.ndarray):
        data["residual"] = steps.predict_compute_residusal(data["frame"], data["qlp"], data["shift"])
    else:
        data["residual"] = data["frame"].apply(steps.predict_compute_residusal,
                                               qlp=data["qlp"][qlp_idx],
                                               shift=int(data["shift"][shift_idx]))

    return data


def _compute_original_df_expander(data: pd.DataFrame, qlp, shift, inplace):
    return steps.restore_signal_cython(data["frame"], qlp, shift, inplace)


def compute_original(data: pd.DataFrame, inplace=False):
    """
    Computes the original from the given residual signal with quantized LPC coefficients and warmup samples
    :param data: input dataframe with columns [frame, qlp, shift]
    :param inplace: whether the restoring should be done in place (faster)
    :return: residual as a numpy array
    """
    qlp = data["qlp"][data["qlp"].first_valid_index()]
    shift = int(data["shift"][data["shift"].first_valid_index()])

    tmp = data.apply(_compute_original_df_expander, qlp=qlp, shift=shift, axis=1, result_type="reduce", inplace=inplace)

    if not inplace:
        data["restored"] = tmp
        return data


def compare_restored(data: pd.DataFrame) -> bool:
    """
    Compares the restored signal to the original
    :param data: input dataframe with columns [frame, restored]
    :return: True if equal, False otherwise
    """
    return not (data["frame"] - data["original"]).any()
