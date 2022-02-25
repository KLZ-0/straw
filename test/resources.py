import numpy as np


def _interleave(x):
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

    num = _interleave(num)

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


def get_signal() -> np.array:
    return np.asarray(
        [358, 623, 771, 912, 993, 1030, 1141, 1296, 1565, 1909, 2257, 2538, 3034, 3548, 4116, 4632, 5324, 5521, 5098,
         4343, 3095, 2164, 1078, -313, -1354, -1939, -2308, -2242, -2031, -1410, -645, -199, 174, 377, 503, 646, 630,
         737, 991, 973, 1069, 1230, 1291, 1386, 1167, 815, 422, -178, -636, -1120, -1411, -1546, -1774, -1817, -1749,
         -1697, -1493, -1268, -1092, -888, -908, -899, -851, -772, -655, -644, -656, -738, -1001, -1187, -1350, -1533,
         -1620, -1788, -1843, -1748, -1604, -1378, -1149, -940, -832, -880, -999, -1120, -1235, -1298, -1364, -1359,
         -1249, -1135, -946, -671, -365, -118, 117, 323, 488, 654, 730, 786, 848, 1078, 1268, 1483, 1674, 1993, 2321,
         2640, 3144, 3617, 4242, 4789, 5335, 5264, 4775, 3921, 2795, 1950, 709, -570, -1458, -1934, -2206, -2175, -1815,
         -1174, -574, -205, 157, 371, 577, 613, 624, 831, 962, 937, 1099, 1293, 1365, 1341, 1086, 746, 277, -297, -736,
         -1153, -1407, -1650, -1779, -1720, -1632, -1567, -1455, -1242, -1133, -1022, -1014, -957, -844, -751, -719,
         -672, -630, -659, -831, -1002, -1196, -1495, -1706, -1891, -1931, -1831, -1655, -1431, -1164, -960, -864, -862,
         -957, -1087, -1237, -1370, -1437, -1411, -1295, -1112, -896, -613, -351, -155, 39, 272, 523, 667, 795, 855,
         901, 1068, 1143, 1366, 1648, 1948, 2303, 2632, 3106, 3667, 4281, 4957, 5547, 5281, 4681, 3755, 2692, 1757, 362,
         -903, -1576, -2055, -2331, -2153, -1677, -865, -377, -82, 246, 327, 355, 384, 453, 788, 884, 839, 1156, 1353,
         1431, 1364, 1051, 686, 127, -497, -945, -1266, -1455, -1599, -1692, -1569, -1507, -1553, -1421, -1250, -1100,
         -1072, -1109, -1022, -894, -809, -695, -584, -498, -626, -948, -1082, -1293, -1584, -1776, -1906, -1870, -1753,
         -1622, -1358, -1056, -877, -858, -978, -1101, -1289, -1499, -1566, -1533, -1427, -1283, -1094, -824, -490,
         -277, -116, 83, 254, 457, 545, 688, 761, 805, 932, 1058, 1352, 1628, 1983, 2258, 2660, 3123, 3523, 4056, 4593,
         5215, 5214, 4639, 3835, 2780, 2002, 817, -378, -1221, -1696, -2097, -2131, -1810, -1258, -618, -289],
        dtype="i2"), 16000
