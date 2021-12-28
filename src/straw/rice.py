import numpy as np
from bitarray import bitarray


class Ricer:
    """
    Rice encoder/decoder
    Currently only supports memory for for memory efficiency comparisons and benchmarks
    """

    def __init__(self, m):
        self.m = m
        self.k = np.log2(self.m)
        self.data = bitarray()

    def _append_n_bits(self, number, n):
        """
        Appends the n bits of number to the end of the current bitstream
        :param number: number to append
        :param n: number of bits to append
        :return: None
        """
        for i in range(n):
            self.data.append(number >> (n - i - 1) & 1)

    def encode_single(self, s):
        """
        Encodes a single number s and appends it to the end of the current bitstream
        :param s: number to encode
        :return: None
        """
        # Quotient code
        q = s // self.m
        r = s % self.m

        for _ in range(q):
            self.data.append(1)
        self.data.append(0)

        # Remainder code
        b = int(self.k)
        tmp = np.power(2, b + 1) - self.m

        if r < tmp:
            self._append_n_bits(r, b)
        else:
            self._append_n_bits(r + tmp, b + 1)

    def get_size_bits_unaligned(self):
        """
        Size of the current bitstream
        can be used for benchmarks
        :return: number of raw bits in the current bitstream
        """
        return len(self.data)
