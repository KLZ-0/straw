import math
import sys

import numpy as np
import pyximport
from scipy.linalg import solve_toeplitz

pyximport.install()
from . import ext


####################
# LPC coefficients #
####################


def _autocorr(signal: np.array, target_len: int) -> np.array:
    return np.asarray([signal[:len(signal) - i].dot(signal[i:]) for i in range(target_len)])


def compute_lpc(signal: np.array, p: int) -> np.array:
    """
    Calculates p LPC coefficients
    For fast Levinson-Durbin implementation see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_toeplitz.html
    :param signal: input signal
    :param p: LPC order
    :return: array of length p containing LPC coefficients
    """

    # Extend to 64 bits to prevent overflows
    signal = signal.astype("i8")

    if not signal.any():
        return None

    r = _autocorr(signal, p + 1)

    return solve_toeplitz(r[:-1], r[1:], check_finite=False)


################
# Quantization #
################

FLAC__SUBFRAME_LPC_QLP_SHIFT_LEN = 5


def quantize_lpc(lpc_c, precision) -> (np.array, int):
    """
    Implementation: https://github.com/xiph/flac/blob/master/src/libFLAC/lpc.c
    TODO: can be heavily optimized
    :param lpc_c:
    :param precision:
    :return: tuple(QLP, shift)
    """
    # reserve 1 bit for sign
    precision -= 1

    qmax = 1 << precision
    qmin = -qmax
    qmax -= 1

    cmax = 0.0
    for i in range(len(lpc_c)):
        d = np.abs(lpc_c[i])
        if d > cmax:
            cmax = d

    if cmax <= 0:
        return None, 0

    max_shiftlimit = 1 << (1 << (FLAC__SUBFRAME_LPC_QLP_SHIFT_LEN - 1)) - 1
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
    for i in range(len(lpc_c)):
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


def quant_alt(lpc_c, precision):
    # drop 1 bit for sign
    precision -= 1

    # set limits
    qmax = 1 << precision
    qmin = -qmax
    qmax -= 1

    # calculate qlp
    cmax = np.max(np.abs(lpc_c))
    shift = precision - math.frexp(cmax)[1]
    qlp = (lpc_c * (1 << shift)).round().astype(np.int32)

    # Limit the quantized values
    np.clip(qlp, qmin, qmax, out=qlp)

    return qlp.astype(np.int32), shift


def quantize_lpc_cython(lpc_c, precision) -> (np.array, int):
    """
    Wrapper around Cython extension for LPC coefficient quantization
    :param lpc_c: numpy array of LPC coefficients to be quantized
    :param precision: target precition in bits
    :return: tuple(QLP, shift)
    """
    shift = ext.quantize_lpc(lpc_c, precision)
    return lpc_c.astype(np.int32), shift


##############
# Prediction #
##############


def predict_signal(frame: np.array, qlp: np.array, shift: int):
    """
    Executes LPC prediction
    The resulting predicted signal starts with the order-th sample
    :param frame: signal frame
    :param qlp: quantized LPC coefficients
    :param shift: coefficient quantization shift
    :return: predicted frame with shape [order:]
    """
    if qlp is None or len(qlp) == 0:
        return None

    return np.convolve(frame, qlp, mode="full")[len(qlp) - 1:-len(qlp)] >> shift


###############
# Restoration #
###############


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
