import numpy as np
import soundfile

from straw import lpc
from .plotter import plot_list


def get_e(x, a):
    """
    Prediction implemented according to https://www.hpl.hp.com/techreports/1999/HPL-1999-144.pdf
    TODO: optimize this shit
    :param x: input signal
    :param a: LPC coefficients
    :return: residual
    """
    e = np.zeros(len(x))

    for n in range(len(x)):
        # sm = x[n]
        sm = 0
        for k in range(len(a)):
            if n - k - 1 < 0:
                continue

            sm += a[k] * x[n - k - 1]

        e[n] = x[n] - sm

    return e


def get_x(e, a):
    """
    Reconstruction implemented according to https://www.hpl.hp.com/techreports/1999/HPL-1999-144.pdf
    TODO: optimize this shit
    :param e: residual
    :param a: LPC coefficients
    :return: original signal
    """
    x = np.zeros(len(e))

    for n in range(len(e)):
        # sm = x[n]
        sm = 0
        for k in range(len(a)):
            if n - k - 1 < 0:
                continue

            sm += a[k] * x[n - k - 1]

        x[n] = e[n] + sm

    return x


def fig_lpc():
    data, sr = soundfile.read("inputs/maskoff_tone.wav")

    bs = int(sr * 0.020)
    start = 400
    signal = data[start:start + bs]

    lpc_c = lpc.lpc(signal, 8)

    data, sr = soundfile.read("inputs/maskoff_tone.wav", dtype="int16")
    signal = data[start:start + bs]

    # prediction
    e = get_e(signal, lpc_c)

    # reconstruction
    x = get_x(e, lpc_c)

    plot_list([signal, x], "lpc_signals.png")
    plot_list([e], "lpc_residual.png")
