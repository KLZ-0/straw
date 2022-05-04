import math
import sys

import numpy as np
import pandas as pd
from numpy.lib.polynomial import roots
from scipy.linalg import solve_toeplitz
from scipy.signal import get_window

from . import ext_lpc


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

    if isinstance(signal, pd.Series):
        # we are dealing with a multichannel LPC
        # Extend to 64 bits to prevent overflows
        if not signal.apply(np.any).all():
            return None

        window = get_window("tukey", signal[signal.index[0]].shape[0])
        r = np.asarray([_autocorr((s.astype(float) / (1 << 15)) * window, p + 1) for s in signal])
        r = np.mean(r, axis=0)
        # lpc_c = np.zeros(p)
        # for i in range(r.shape[0]):
        #     lpc_c += solve_toeplitz(r[i, :-1], r[i, 1:], check_finite=False)
        # return lpc_c / r.shape[0]
    else:
        if not signal.any():
            return None

        window = get_window("tukey", signal.shape[0])
        r = _autocorr((signal.astype(float) / (1 << 15)) * window, p + 1)

    return solve_toeplitz(r[:-1], r[1:], check_finite=False)


def lpc_is_stable(lpc_c) -> bool:
    poly = np.zeros(lpc_c.shape[0] + 1)
    poly[0] = 1
    poly[1:] = -lpc_c
    return np.max(np.abs(roots(poly))) < 1.0


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


def quant_alt(lpc_c, precision) -> (np.array, int, int):
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

    return qlp.astype(np.int32), precision, shift


def quantize_lpc_cython(lpc_c, precision) -> (np.array, int, int):
    """
    Wrapper around Cython extension for LPC coefficient quantization
    :param lpc_c: numpy array of LPC coefficients to be quantized
    :param precision: target precition in bits
    :return: tuple(QLP, precision, shift)
    """
    shift = ext_lpc.quantize_lpc(lpc_c, precision)
    return lpc_c.astype(np.int32), precision, shift


def simple_quantize(lpc_c, precision) -> (np.array, int, int):
    qmax = 1 << precision - 1
    qmin = -qmax
    qmax -= 1

    for shift in reversed(range(0, precision)):
        vals = (lpc_c * (1 << shift)).astype(np.int32)
        if vals.max() <= qmax and vals.min() >= qmin:
            return vals, precision, shift


##############
# Prediction #
##############


def predict_signal(frame: np.array, qlp: np.array, shift):
    """
    Executes LPC prediction
    The resulting predicted signal starts with the order-th sample
    :param frame: signal frame
    :param qlp: quantized LPC coefficients
    :param shift: coefficient quantization shift
    :return: predicted frame with shape [order:]
    """
    shift = int(shift)
    if qlp is None or len(qlp) == 0:
        return None

    return np.convolve(frame, qlp, mode="full")[len(qlp) - 1:-len(qlp)] >> shift


def predict_compute_residual(frame: np.array, qlp: np.array, shift: int):
    """
    Executes LPC prediction and returns the residual by subtracting the original frame from the predicted signal
    :param frame: signal frame
    :param qlp: quantized LPC coefficients
    :param shift: coefficient quantization shift
    :return: frame residual with shape [order:]
    """
    shift = int(shift)
    residual = frame.copy()
    ext_lpc.compute_residual(frame, residual, qlp, shift)
    tmp = residual[len(qlp):]
    # predicted = predict_signal(frame, qlp, shift)
    # tmp = (frame[len(qlp):] - predicted).astype(frame.dtype)
    if tmp.var() < frame.var():
        # TODO: we could use the same memory space but this would prevent us from using the raw signal after
        # frame[len(qlp):] = tmp
        # return frame[len(qlp):]
        return tmp
    else:
        return None


###############
# Restoration #
###############


def restore_signal(residual, qlp, lp_quantization, warmup_samples):
    """
    Restores the original signal given the residual with quantized LPC coefficients
    :param residual: residual signal
    :param qlp: quantized LPC coefficients
    :param lp_quantization: quantization level
    :param warmup_samples: warmup samples (the first order samples from the original signal)
    :return: reconstructed signal as a numpy array
    """
    order = qlp.shape[0]
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


def restore_signal_cython(frame: np.array, qlp: np.array, lp_quantization: int) -> np.array:
    """
    Restores the original signal given the residual with quantized LPC coefficients
    Wrapper around Cython extension for signal restoration
    :param frame: signal array initialized with the first samples from the original signal and the residual
    :param qlp: quantized LPC coefficients
    :param lp_quantization: quantization shift
    :return: reconstructed signal as a numpy array
    """
    # TODO: make this a proper wrapper without the need for duplicated lines
    ext_lpc.restore_signal(frame, qlp, lp_quantization)
