# distutils: language = c
# cython: language_level=3
import cython
import numpy as np
from bitarray import bitarray

def append_n_bits(bits: bitarray, number: cython.int, n: cython.int):
    """
    Appends the n bits of number to the end of the current bitstream
    :param bits: bitaray to which the bits will be appended
    :param number: number to append
    :param n: number of bits to append
    :return: None
    """
    bits.extend([number >> (n - i - 1) & 1 for i in range(n)])

@cython.cdivision(True)
def encode_frame(bits: bitarray, frame: np.array, m: cython.int, k: cython.int):
    """
    Encodes a whole residual frame and appends it to the end of the current bitstream
    :param bits: bitaray to which the bits will be appended
    :param frame: the frame to be encoded
    :param m: rice m constant
    :param k: rice k constant
    :return: None
    """
    cdef int s, q, r, tmp

    for s in frame:
        # Quotient code
        q = s / m
        r = s % m

        for _ in range(q):
            bits.append(1)
        bits.append(0)

        # Remainder code
        tmp = (1 << k + 1) - m

        if r < tmp:
            append_n_bits(bits, r, k)
        else:
            append_n_bits(bits, r + tmp, k + 1)
