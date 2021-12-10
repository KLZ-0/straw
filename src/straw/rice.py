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


def rice_str(num, m):
    """
    Return the rice encoded number with golomb parameter m
    TODO: probably rework to pure rice encoding with m -> k so that m = 2^k
    TODO: also create a version with bitstream output instead of ascii
    :param num: number to be encoded
    :param m: rice parameter
    :return: None
    """
    out = ""

    q = num // m
    r = num % m

    # Quotient code
    for _ in range(q):
        out += "1"
    out += "0"

    # separator
    # print(",", end="")

    # Remainder code
    b = int(np.log2(m))
    tmp = np.power(2, b + 1) - m

    if r < tmp:
        out += ("{0:0" + str(b) + "b}").format(r)
    else:
        out += ("{0:0" + str(b + 1) + "b}").format(r + tmp)

    return out
