# distutils: language = c
# cython: language_level=3
import cython
import numpy as np

####################
# LPC coefficients #
####################

# @cython.cdivision(True)
# def compute_lp_coefficients(long[:] autoc, int order, double[:] lpc):
#     cdef int j
#     cdef double r, err
#     cdef double tmp
#
#     err = autoc[0]
#
#     r = -autoc[order + 1]
#     for j in range(order):
#         r -= lpc[j] * autoc[order - j]
#     r /= err
#
#     lpc[order] = r
#     for j in range(order >> 1):
#         tmp = lpc[j]
#         lpc[j] += r * lpc[order - 1 - j]
#         lpc[order - 1 - j] += r * tmp
#
#     if order & 1:
#         lpc[j] += lpc[j] * r
#
#     err *= (1.0 - r * r)
#     return err


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
    cdef int i
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

##############
# Prediction #
##############

def compute_residual(cython.integral[:] frame, cython.integral[:] residual, int[:] qlp, int lp_quantization):
    cdef Py_ssize_t data_len = frame.shape[0]
    cdef Py_ssize_t order = qlp.shape[0]
    cdef Py_ssize_t  i, j
    cdef long _sum

    for i in range(order, data_len):
        _sum = 0
        for j in range(order):
            _sum += qlp[j] * frame[i - j - 1]
        residual[i] = frame[i] - (_sum >> lp_quantization)

###############
# Restoration #
###############


def restore_signal(cython.integral[:] frame, int[:] qlp, int lp_quantization):
    """
    Restores the original signal given the residual with quantized LPC coefficients
    :param frame: signal array initialized with the first samples from the original signal and the residual
    :param qlp: quantized LPC coefficients
    :param lp_quantization: quantization shift
    :return: reconstructed signal as a numpy array
    # TODO: make this function take an array of blocksize where first order samples are warmup samples and the rest is residual
    """
    cdef Py_ssize_t data_len = frame.shape[0]
    cdef Py_ssize_t order = qlp.shape[0]
    cdef Py_ssize_t  i, j
    cdef long _sum

    if order <= 0:
        return None

    for i in range(order, data_len):
        _sum = 0
        for j in range(order):
            _sum += qlp[j] * frame[i - j - 1]
        frame[i] = frame[i] + (_sum >> lp_quantization)
