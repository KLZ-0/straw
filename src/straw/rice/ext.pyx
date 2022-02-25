# distutils: language = c
# cython: language_level=3
import cython
from bitarray import bitarray
#########################
# Signedness correction #
#########################

cdef short _interleave(short x):
    """
    Implementation of the overlap and interleave scheme from https://en.wikipedia.org/wiki/Golomb_coding
    :param x: number to be remapped
    :return: positive integer which can be encoded
    """
    if x == 0:
        return 0

    if x > 0:
        return 2 * x
    else:
        return -2 * x - 1

@cython.cdivision(True)
cdef short _deinterleave(short x):
    if x == 0:
        return 0

    if x % 2 == 0:
        return x / 2
    else:
        return (x + 1) / -2

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
def encode_frame(bits: bitarray, short[:] frame, short m, short k):
    """
    Encodes a whole residual frame and appends it to the end of the current bitstream
    :param bits: bitaray to which the bits will be appended
    :param frame: the frame to be encoded
    :param m: rice m constant
    :param k: rice k constant
    :return: None
    """
    cdef short q, s
    cdef Py_ssize_t x_max, i
    x_max = frame.shape[0]

    for i in range(x_max):
        s = _interleave(frame[i])

        # Quotient code
        q = s / m

        for _ in range(q):
            bits.append(1)
        bits.append(0)

        _append_n_bits(bits, s, k)

############
# Decoding #
############

def decode_frame(short[:] frame, bits: bitarray, short m, short k):
    cdef short q, s, j, tmp
    cdef Py_ssize_t x_max, i
    cdef Py_ssize_t bit_i = 0
    x_max = frame.shape[0]

    for i in range(x_max):
        q = 0
        while bits[bit_i]:
            bit_i += 1
            q += 1
        bit_i += 1

        s = m * q

        for j in range(k):
            s |= bits[bit_i] << (k - j - 1)
            bit_i += 1

        frame[i] = _deinterleave(s)
