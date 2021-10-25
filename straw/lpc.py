import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter


def _get_e(x, a):
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


def _get_x(e, a):
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


def lpc_predict(signal, lpc_c):
    """
    Prediction implemented according to https://www.hpl.hp.com/techreports/1999/HPL-1999-144.pdf
    TODO: optimize this shit
    :param signal: input signal
    :param lpc_c: LPC coefficients
    :return: residual
    """
    e = _get_e(signal, lpc_c)
    res = signal - lfilter(np.concatenate(([0], lpc_c)), 1, signal)
    # TODO: e != res

    return e


def lpc_reconstruct(residual, lpc_c):
    """
    Reconstruction implemented according to https://www.hpl.hp.com/techreports/1999/HPL-1999-144.pdf
    TODO: optimize this shit
    :param residual: residual
    :param lpc_c: LPC coefficients
    :return: original signal
    """
    x = _get_x(residual, lpc_c)
    rec = residual + lfilter([1], np.concatenate(([1], -lpc_c)), residual)
    # TODO: x != rec

    return x


def lpc(signal, p: int):
    """
    Calculates p LPC coefficients
    For fast Levinson-Durbin implementation see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_toeplitz.html
    :param signal: input signal
    :param p: LPC order
    :return: array of length p containing LPC coefficients
    """

    r = np.correlate(signal, signal, 'full')[len(signal) - 1:len(signal) + p]

    return solve_toeplitz(r[:-1], r[1:])
