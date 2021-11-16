import numpy as np


def rice_print(num, m):
    """
    Print the rice encoded number with golomb parameter m
    TODO: probably rework to pure rice encoding with m -> k so that m = 2^k
    TODO: also create a version with bitstream output instead of ascii
    :param num: number to be encoded
    :param m: rice parameter
    :return: None
    """
    q = num // m
    r = num % m

    # Quotient code
    for _ in range(q):
        print(1, end="")
    print(0, end="")

    # separator
    # print(",", end="")

    # Remainder code
    b = int(np.log2(m))
    tmp = np.power(2, b + 1) - m

    if r < tmp:
        print(("{0:0" + str(b) + "b}").format(r))
    else:
        print(("{0:0" + str(b + 1) + "b}").format(r + tmp))
