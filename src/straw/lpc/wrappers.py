import numpy as np
import pandas as pd

from straw.lpc import steps


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
        return None, 0

    return steps.quantize_lpc_cython(lpc, qlp_coeff_precision)


def compute_residual(data: pd.DataFrame) -> np.array:
    """
    Computes the residual from the given signal with quantized LPC coefficients
    Pandas-lever wrapper
    :param data: input dataframe with columns [frame, qlp, shift]
    :return: residual as a numpy array
    """

    predicted = steps.predict_signal(data["frame"], data["qlp"], data["shift"])
    if predicted is None:
        return None

    return (data["frame"][len(data["qlp"]):] - predicted).astype(np.int16)
