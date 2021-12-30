# distutils: language = c
# cython: language_level=3
import cython
from bitarray import bitarray

def append_n_bits(bits: bitarray, number: cython.int, n: cython.int):
    bits.extend([number >> (n - i - 1) & 1 for i in range(n)])

@cython.cdivision(True)
def encode(bits: bitarray, s: cython.int, m: cython.int, k: cython.int):
    cdef int q, r, b, tmp

    # Quotient code
    q = s / m
    r = s % m

    bits.extend([1 for _ in range(q)])
    bits.append(0)

    # Remainder code
    b = k
    tmp = (1 << b + 1) - m

    if r < tmp:
        append_n_bits(bits, r, b)
    else:
        append_n_bits(bits, r + tmp, b + 1)
