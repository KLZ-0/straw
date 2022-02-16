# distutils: language = c
# cython: language_level=3
import numpy as np

################
# Quantization #
################

def quantize_lpc(double[:] lpc_c, int precision) -> int:
    """
    Quantizes LPC coefficients to a given precision
    Implemented from: https://github.com/xiph/flac/blob/master/src/libFLAC/lpc.c
    :param lpc_c: numpy array of LPC coefficients to be quantized
    :param precision: target precition in bits
    :return: shift in bits
    """
    # reserve 1 bit for sign
    precision -= 1

    cdef int qmax = 1 << precision
    cdef int qmin = -qmax
    qmax -= 1

    cdef double cmax = np.max(np.abs(lpc_c))

    cdef int shift = precision - np.frexp(cmax)[1]

    # TODO: this should probably be reimplemented from FLAC
    # if shift > max_shiftlimit:
    #     shift = max_shiftlimit
    # elif shift < min_shiftlimit:
    #     return None, 0

    cdef int order = lpc_c.shape[0]
    cdef double error = 0.0
    cdef double q
    for i in range(order):
        error += lpc_c[i] * (1 << shift)
        q = round(error)

        if q > qmax:
            q = qmax
        elif q < qmin:
            q = qmin

        error -= q
        lpc_c[i] = q

    return shift
