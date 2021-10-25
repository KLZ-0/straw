import numpy as np
import scipy


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

    return scipy.linalg.solve_toeplitz(r[:-1], r[1:])
