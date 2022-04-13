# distutils: language = c
# cython: language_level=3
import cython
from bitarray import bitarray

#########################
# Signedness correction #
#########################

cdef int _interleave(short x):
    """
    Implementation of the overlap and interleave scheme from https://en.wikipedia.org/wiki/Golomb_coding
    :param x: signed integer to be remaped
    :return: positive interleaved integer
    """
    if x == 0:
        return 0

    if x > 0:
        return 2 * x
    else:
        return -2 * x - 1

@cython.cdivision(True)
cdef short _deinterleave(int x):
    """
    Reverse of _interleave(short x)
    :param x: positive interleaved integer
    :return: original signed integer
    """
    if x == 0:
        return 0

    if x % 2 == 0:
        return x / 2
    else:
        return (x + 1) / -2

cdef void update_scale(int s, short m, short *scale):
    if s > m:
        scale[0] += 1
    elif s < m:
        scale[0] -= 1
    else:
        scale[0] = 0

############
# Encoding #
############

def _append_n_bits(bits: bitarray, short number, short n):
    """
    Appends the last n bits of number to the end of the current bitstream
    :param bits: bitaray to which the bits will be appended
    :param number: number to append
    :param n: number of bits to append
    :return: None
    """
    bits.extend([number >> (n - i - 1) & 1 for i in range(n)])

@cython.cdivision(True)
def encode_frame(bits: bitarray, short[:] frame, short k, short resp, short adaptive):
    """
    Encodes a whole residual frame and appends it to the end of the given bitstream
    :param bits: bitaray to which the bits will be appended
    :param frame: the frame to be encoded
    :param k: starting rice parameter
    :param resp: rice parameter responsiveness
    :param adaptive: if True do adaptive rice coding by varying the parameter
    :return: None
    """
    cdef short m, q
    cdef int s  # int because of interleaving
    cdef Py_ssize_t x_max, i
    x_max = frame.shape[0]
    m = 1 << k

    cdef short scale = 0

    for i in range(x_max):
        s = _interleave(frame[i])

        # Quotient code
        q = s / m

        for _ in range(q):
            bits.append(0)
        bits.append(1)

        _append_n_bits(bits, s, k)

        # TODO: feed-forward rice implementation
        if not adaptive:
            continue

        # Update rice param
        if scale > resp:
            scale = 0
            k += 1
            m = 1 << k
            # print("e switched up:\t\t ", i, s, m)
            continue
        if scale < -resp and k > 0:
            scale = 0
            k -= 1
            m = 1 << k
            # print("e switched down:\t ", i, s, m)
            continue

        update_scale(s, m, &scale)

############
# Decoding #
############

cdef char _get_bit(bits: bitarray, Py_ssize_t *bit_i):
    bit_i[0] += 1
    return bits[bit_i[0] - 1]

@cython.cdivision(True)
def decode_frame(short[:] frame, bits: bitarray, short k, short resp, short adaptive):
    """
    Decodes a whole residual frame from the given bitstream
    :param frame: numpy array where the decoded frame should be stored
    :param bits: bitaray from which the frame should be restored
    :param k: starting rice parameter
    :param resp: rice parameter responsiveness
    :param adaptive: if True do adaptive rice coding by varying the parameter
    :return: None
    """
    cdef short m, q, j
    cdef int s  # int because of interleaving
    cdef Py_ssize_t x_max, i
    cdef Py_ssize_t bit_i = 0
    x_max = frame.shape[0]
    m = 1 << k

    cdef short scale = 0

    for i in range(x_max):
        q = 0
        while not _get_bit(bits, &bit_i):
            q += 1

        s = m * q

        for j in range(k):
            s |= _get_bit(bits, &bit_i) << (k - j - 1)

        frame[i] = _deinterleave(s)

        if not adaptive:
            continue

        if scale > resp:
            scale = 0
            k += 1
            m = 1 << k
            # print("dec switched up:\t ", i, s, m)
            continue
        if scale < -resp and k > 0:
            scale = 0
            k -= 1
            m = 1 << k
            # print("dec switched down:\t ", i, s, m)
            continue

        update_scale(s, m, &scale)

    return bit_i

###########
# Utility #
###########

@cython.cdivision(True)
def kparams(short[:] frame, short k, short resp):
    """
    Encodes a whole residual frame and appends it to the end of the given bitstream
    :param frame: the frame to be encoded
    :param k: starting rice parameter
    :param resp: rice parameter responsiveness
    :return: None
    """
    cdef int s
    cdef Py_ssize_t x_max, i
    x_max = frame.shape[0]
    m = 1 << k

    cdef short scale = 0

    for i in range(x_max):
        s = _interleave(frame[i])
        frame[i] = k

        if scale > resp:
            scale = 0
            k += 1
            m = 1 << k
            continue
        if scale < -resp and k > 0:
            scale = 0
            k -= 1
            m = 1 << k
            continue

        update_scale(s, m, &scale)

def interleave_frame(int[:] frame):
    cdef Py_ssize_t x_max, i
    x_max = frame.shape[0]
    for i in range(x_max):
        frame[i] = _interleave(frame[i])
