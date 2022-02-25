# distutils: language = c
# cython: language_level=3
import cython
from bitarray import bitarray

def _interleave(short x):
    """
    Implementation of the overlap and interleave scheme from https://en.wikipedia.org/wiki/Golomb_coding
    :param x: number to be remapped
    :return: positive integer which can be encoded
    """
    if x > 0:
        return 2 * x
    elif x < 0:
        return -2 * x - 1
    else:
        return 0

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
