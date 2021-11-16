import math

import numpy as np
import soundfile

from straw import lpc
from .plotter import plot_list

FLAC__SUBFRAME_LPC_QLP_SHIFT_LEN = 5


def quantize_lpc(lpc_c, order, precision):
    """
    Source https://github.com/xiph/flac/blob/master/src/libFLAC/lpc.c
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
        return None

    max_shiftlimit = 1 << (1 << (FLAC__SUBFRAME_LPC_QLP_SHIFT_LEN-1)) - 1
    min_shiftlimit = -max_shiftlimit - 1

    _, log2cmax = math.frexp(cmax)
    log2cmax -= 1

    shift = precision - log2cmax - 1

    if shift > max_shiftlimit:
        shift = max_shiftlimit
    elif shift < min_shiftlimit:
        return None

    # if shift >= 0
    error = 0.0
    q = 0
    qlp_c = np.zeros(len(lpc_c), dtype="i4")
    for i in range(order):
        error += lpc_c[i] * (1 << shift)
        q = round(error)

        # overflows
        if q > qmax+1:
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


def compute_residual(data, qlp, order, lp_quantization):
    if order <= 0:
        return None

    # the slower but clearer version...
    residual = np.zeros(len(data), dtype="i4")

    for i in range(len(data)):
        _sum = 0
        for j in range(order):
            _sum += qlp[j] * data[i - j - 1]
        residual[i] = data[i] - (_sum >> lp_quantization)

    return residual


def restore_signal(residual, qlp, order, lp_quantization, data):
    # the slower but clearer version...
    # data = np.zeros(len(residual), dtype="i4")

    for i in range(order, len(residual)):
        _sum = 0
        for j in range(order):
            _sum += qlp[j] * data[i - j - 1]
        data[i] = residual[i] + (_sum >> lp_quantization)

def fig_lpc():
    data, sr = soundfile.read("inputs/maskoff_tone.wav")

    bs = int(sr * 0.020)
    start = 400
    signal = data[start:start + bs]

    lpc_c = lpc.lpc(signal, 8)

    data, sr = soundfile.read("inputs/maskoff_tone.wav", dtype="int16")
    signal = data[start:start + bs]

    # tmp
    warmup = signal[:8]
    qlp, shift = quantize_lpc(lpc_c, 8, 12)
    expected = np.asarray([1108, -803, 375, -435, 521, -545, 395, -114])
    print("warmup samples:", warmup)
    print("LPC:", lpc_c)
    print("QLP:", qlp)
    print("QLP shift:", shift)
    print("Expected:", expected)
    # tmp

    residual = compute_residual(signal, qlp, 8, shift)

    restored = np.pad(warmup, (0, len(residual) - 8))
    restore_signal(residual, qlp, 8, shift, restored)

    print(signal - restored)

    # e = lpc.lpc_predict(signal, lpc_c)
    # x = lpc.lpc_reconstruct(e, lpc_c)

    plot_list([signal, restored], "lpc_signals.png")
    plot_list([residual], "lpc_residual.png")
