import numpy as np
from bitarray import bitarray


class Ricer:
    def __init__(self, m):
        self.m = m
        self.k = np.log2(self.m)
        self.data = bitarray()

    def _append_n_bits(self, number, n):
        for i in range(n):
            self.data.append(number >> (n - i - 1) & 1)

    def encode_single(self, s):
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
