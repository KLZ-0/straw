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
        self.data.extend([number >> (n - i - 1) & 1 for i in range(n)])

    def encode_single(self, s: int):
        """
        Encodes a single number s and appends it to the end of the current bitstream
        :param s: number to encode
        :return: None
        """
        # Quotient code
        q = s // self.m
        r = s % self.m

        self.data.extend([1 for _ in range(q)])
        self.data.append(0)

        # Remainder code
        b = int(self.k)
        tmp = (1 << b + 1) - self.m

        if r < tmp:
            self._append_n_bits(r, b)
        else:
            self._append_n_bits(r + tmp, b + 1)

    def encode_frame(self, frame: np.array):
        """
        Encodes a whole frame and appends it to the end of the current bitstream
        NOTE: This is really fucking inneficient - it needs to be heavily optimized
        :param frame: frame to encode
        :return: None
        """
        for v in frame:
            self.encode_single(v)

    def get_size_bits_unaligned(self):
        """
        Size of the current bitstream
        can be used for benchmarks
        :return: number of raw bits in the current bitstream
        """
        return len(self.data)
