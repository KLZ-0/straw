import math
import sys

import numpy as np
import pandas as pd
from scipy.linalg import solve_toeplitz

FLAC__SUBFRAME_LPC_QLP_SHIFT_LEN = 5


def _quantize_lpc(lpc_c, order, precision) -> (np.array, int):
    """
    Implementation: https://github.com/xiph/flac/blob/master/src/libFLAC/lpc.c
    TODO: can be heavily optimized
    :param lpc_c:
    :param order:
    :param precision:
    :return:
    """
    # reserve 1 bit for sign
    precision -= 1

    qmax = 1 << precision
    qmin = -qmax
    qmax -= 1

    cmax = 0.0
    for i in range(order):
        d = np.abs(lpc_c[i])
        if d > cmax:
            cmax = d

    if cmax <= 0:
        return None, 0

    max_shiftlimit = 1 << (1 << (FLAC__SUBFRAME_LPC_QLP_SHIFT_LEN-1)) - 1
    min_shiftlimit = -max_shiftlimit - 1

    _, log2cmax = math.frexp(cmax)
    log2cmax -= 1

    shift = precision - log2cmax - 1

    if shift > max_shiftlimit:
        shift = max_shiftlimit
    elif shift < min_shiftlimit:
        return None, 0

    # if shift >= 0
    # TODO: add way for negative shift
    if shift < 0:
        print("Negative shift not yet supported", file=sys.stderr)
        exit(1)

    error = 0.0
    q = 0
    qlp_c = np.zeros(len(lpc_c), dtype="i4")
    for i in range(order):
        error += lpc_c[i] * (1 << shift)
        q = round(error)

        # overflows
        if q > qmax + 1:
            print("Overflow1")
        if q < qmin:
            print("Overflow2")

        if q > qmax:
            q = qmax
        elif q < qmin:
            q = qmin

        error -= q
        qlp_c[i] = q

    return qlp_c, shift


def _quant_alt(lpc_c, order, precision):
    cmax = np.max(np.abs(lpc_c))
    shift = precision - math.frexp(cmax)[1] - 1
    return (lpc_c * 2 ** shift).round().astype(np.int32), shift


def _autocorr(signal: np.array, target_len: int) -> np.array:
    return np.asarray([signal[:len(signal) - i].dot(signal[i:]) for i in range(target_len)])


def _compute_lpc(signal: pd.DataFrame, p: int) -> np.array:
    """
    Calculates p LPC coefficients
    For fast Levinson-Durbin implementation see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_toeplitz.html
    :param signal: input dataframe with columns [frame]
    :param p: LPC order
    :return: array of length p containing LPC coefficients
    """

    # Extend to 64 bits to prevent overflows
    signal = signal["frame"].astype("i8")

    if not signal.any():
        return None

    r = _autocorr(signal, p + 1)

    return solve_toeplitz(r[:-1], r[1:], check_finite=False)


def compute_qlp(signal, order: int, qlp_coeff_precision: int) -> (np.array, int):
    """
    Compute LPC and quantize the LPC coefficients
    :param signal: input dataframe with columns [frame]
    :param order: maximal LPC order
    :param qlp_coeff_precision: Bit precision for storing the quantized LPC coefficients
    :return: tuple(qlp coefficients, quantization level)
    """
    lpc = _compute_lpc(signal, order)
    if lpc is None:
        return None, 0

    return _quantize_lpc(lpc, order, qlp_coeff_precision)


def compute_residual(data: pd.DataFrame, order: int) -> np.array:
    """
    Computes the residual from the given signal with quantized LPC coefficients
    :param data: input dataframe with columns [frame, qlp, shift]
    :param order: LPC order
    :return: residual as a numpy array
    """

    if order <= 0 or data["qlp"] is None:
        return None

    _sum = np.convolve(data["frame"], data["qlp"], mode="full")[order - 1:] >> data["shift"]
    return (data["frame"][order:] - _sum[:-order]).astype(np.int16)


def restore_signal(residual, qlp, order, lp_quantization, warmup_samples):
    """
    Restores the original signal given the residual with quantized LPC coefficients
    :param residual: residual signal
    :param qlp: quantized LPC coefficients
    :param order: LPC order
    :param lp_quantization: quantization level
    :param warmup_samples: warmup samples (the first order samples from the original signal)
    :return: reconstructed signal as a numpy array
    """
    if order <= 0:
        return None

    # the slower but clearer version...
    data = np.pad(warmup_samples, (0, len(residual)))

    for i in range(order, len(residual) + order):
        _sum = 0
        for j in range(order):
            _sum += qlp[j] * data[i - j - 1]
        data[i] = residual[i - order] + (_sum >> lp_quantization)

    return data
