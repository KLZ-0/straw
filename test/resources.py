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


def get_signal() -> (np.array, int):
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


def get_problematic_residual1() -> np.array:
    return np.asarray(
        [7, 1, -7, 2, 0, -2, 2, -3, 1, 3, -5, -5, 3, 0, 2, -3, 1, 0, 3, -2, 4, 3, 9, -4, -2, -5, 0, -1, -3, 3, 1, 0, 1,
         -2, 8, -2, 1, 1, -3, -1, 3, -1, 4, -4, -2, 0, -1, -8, 3, 1, 1, 2, -2, 2, 6, -2, 1, 5, 1, -1, -1, 1, -2, 0, 2,
         0, 2, -6, -3, 6, 0, 0, 5, -2, 2, 1, 0, -2, 2, 2, -2, -3, 0, 6, -2, -1, 7, 1, 0, 5, 1, -1, -1, -2, 2, 2, 7, 1,
         1, 5, 0, 1, 5, 0, 4, -2, 3, -1, -3, -1, 5, -4, 4, 0, 5, 2, 1, 4, 6, 1, -3, -3, -4, -4, -2, 7, 1, -1, 1, 9, 6,
         1, 0, -4, -2, 4, -4, -7, 0, 2, -3, -2, 3, 3, 2, 0, -4, 4, 5, 1, 2, 0, 0, -1, 0, 5, 2, -1, -3, -1, 4, -4, -5, 5,
         -2, -5, 2, 3, 1, 5, -2, 1, 2, 6, 0, 4, -5, 2, -3, -4, 1, 3, -2, 1, -1, 1, 2, 1, 0, 0, 3, 7, 6, 0, 4, -2, -6,
         -2, 2, 3, 2, 2, 1, 4, 5, 2, 1, 2, 1, -3, -3, 5, 5, -3, -2, 4, 3, 0, 1, -1, 3, 0, -7, -1, 4, 6, 1, 1, 0, 0, 1,
         -2, 3, 0, -3, -2, 0, -2, 1, 2, -2, 1, 8, 3, -1, -1, -3, 3, 3, 0, -1, 0, 2, -1, 0, 3, -1, 4, 0, -1, 6, -1, 1, 0,
         3, -4, -2, 4, 5, -1, -1, 0, 1, 0, 5, -3, -5, 2, -6, 2, 6, 2, -3, -1, 3, -1, -8, 6, 6, 0, -2, -1, -4, -1, 2, 1,
         -1, -3, -2, 1, -1, 2, 4, 2, 1, -1, -8, -2, 3, -1, -1, 0, -2, 1, -6, 6, 0, -1, -1, -1, 0, 6, 0, -3, 7, 1, -1, 4,
         4, 1, -2, -2, 0, 0, -4, 1, 0, -2, 1, 2, 0, 5, 4, 1, 5, -5, 3, -3, -2, 1, -2, -5, 1, 0, -1, -1, 4, 0, 3, 4, 2,
         0, -4, 8, 0, -2, 7, -4, 0, 3, -3, -1, -1, -1, 6, 1, 2, -7, 3, 1, -3, 2, 1, -3, -4, 9, 1, 3, 1, -5, 0, -1, -1,
         -1, 4, 2, 2, 2, -4, 2, 6, 2, -2, -5, -1, -1, -1, 0, 0, 2, -1, -3, 0, -2, 1, 3, 2, 4, 2, 5, -4, 4, 1, -5, 0, 0,
         2, -2, -4, 4, -6, 0, 1, 3, 2, 2, 0, 3, 0, 6, -6, 3, -2, 4, -1, 1, -1, 10, -1, 4, -4, 4, -3, -1, -1, 4, 2, 1, 3,
         3, 5, 3, 0, -2, 4, -5, 3, 1, -3, 1, 4, 0, 0, 4, 0, 6, -4, 2, 4, 0, -2, 7, -1, 3, 1, 0, 1, 0, 1, -2, -3, 1, 2,
         -1, -2, 2, 8, 1, -5, -2, -2, 2, -1, 3, 2, 2, 0, -1, 1, 6, -2, -2, 4, 3, 4, 1, 2, -5, 2, -2, -2, -3, -5, 1, 3,
         -2, 7, 1, 3, 0, -3, 2, 1, 3, 1, -5, 5, -3, -3, 0, 0, -6, -2, 2, -1, -4, -1, -2, -1, -3, 1, 5, 3, -4, -1, 7, -2,
         -1, -6, 1, -4, 2, 0, -4, 1, 0, -2, 2, -4, 2, 4, -3, -4, -2, 1, 4, -3, 3, -1, 2, 1, 3, 0, -3, -6, 10, -4, 2, 2,
         3, 2, 3, 4, -4, 0, 3, -4, -2, 2, -2, 3, -1, -5, 3, -3, 2, 5, -1, 3, 3, -1, -4, 4, -1, -2, 1, 3, 4, 6, -2, -2,
         1, -4, 5, -3, 2, 1, 8, -3, 5, 3, -4, 1, 1, -2, 2, 1, 0, -1, 2, -3, 2, 0, 4, 2, -1, 0, 7, 0, 0, -8, 5, 3, -3,
         -7, 4, 1, -1, -1, 0, 3, 3, 2, 1, 0, 2, 0, -2, 5, 4, 0, -3, -3, -4, 2, 4, -2, -1, 4, -2, 0, 2, 6, 4, 8, -1, 0,
         1, 3, 1, 3, -2, 1, 1, -2, 2, -4, 1, 3, 1, 9, -1, -4, 3, 3, 2, 1, 7, -2, 2, -1, -6, 2, 0, 4, 0, 3, 2, -1, -1,
         -1, -1, 4, 4, 2, 2, -2, 5, 1, 1, 4, -1, -3, 1, 3, -1, 3, 0, -3, 0, -7, 4, 6, 2, 1, 2, 0, 2, 6, 3, 3, -1, -2,
         -2, 3, 0, 5, -2, 0, 1, 0, 0, -5, 2, 5, 0, -5, -1, 2, 6, -4, -9, -1, 0, -4, 5, 5, 6, -1, 0, 3, 2, 5, -1, -2, 2,
         1, 1, -2, 1, 5, -1, -6, 0, 1, 1, -3, 1, 0, -1, 0, 3, -3, 5, -4, 0, 1, 4, -2, -4, -2, 1, -1, -6, 0, -2, 1, 4,
         -7, 4, 1, -1, 1, 4, -3, 5, 0, 6, -5, -5, -3, -1, -2, -1, -1, 0, -1, 4, 5, -4, -2, 4, 2, 6, -3, 4, 0, 3, -1, -4,
         -3, -6, -4, 2, 2, 0, 5, 3, 2, -1, 3, 0, 1, 4, -5, -4, 2, 1, -4, 0, 1, 2, -4, 6, 1, 1, 2, -3, -1, 6, 1, 1, 0, 2,
         1, -3, 0, -6, -2, 3, -3, 5, 0, 1, 2, 1, 0, 1, -1, 3, 1, -2, 2, 0, 0, -3, -4, -3, 2, -1, 4, -3, 4, -3, 2, 5, 2,
         8, -1, 1, 2, 2, 0, -3, -5, 1, -3, 4, -4, -3, 0, -4, 5, 5, 5, -4, -3, -3, -2, 3, 6, 2, 3, 0, 3, 0, 4, 0, -2, 5,
         8, -2, 3, 2, -6, -2, 6, 4, 3, 5, 1, -3, -3, 5, 2, -4, 1, 0, 3, 3, 2, 0, 6, 2, 4, -4, -2, -5, 3, 1, -1, -6, 4,
         0, -2, 1, 5, 2, 4, -2, 7, 0, 1, 0, 2, 2, -1, 1, 3, -2, 2, -5, -1, 0, 7, -5, 2, -2, 2, 3, 2, 4, -4, 3, 2, 0, 6,
         -3, 4, 0, -7, -2, 0, -2, 3, -2, 3, -1, -2, -2, 2, 4, 2, -3, 1, 3, 6, -1, 0, 4, 0, -1, -1, -4, 0, 2, 2, -2, 1,
         2, 1, -1, 3, -6, 4, -4, -2, 0, -5, 2, -4, 3, -4, -1, 4, 3, 1, 4, -1, 2, 0, 5, -5, -1, -6, 4, -6, -3, -2, 1, 0,
         -4, -3, 3, 1, 0, 1, -1, 4, -6, 5, 1, 1, 0, -5, -1, 0, 0, -6, 3, -1, -4, 4, 0, 5, 3, -1, 2, 1, 1, -1, 1, -2, 3,
         -3, -2, -1, 3, -1, -5, 6, 7, -4, 3, 6, -1, 0, 2, 1, -2, -1, 5, 2, 2, 0, -1, -1, 4, -2, 1, 3, -6, 4, 3, 3, -1,
         4, 2, 3, -1, -2, 2, -2, -3, 0, -3, -1, 10, 3, 0, 5, 0, -5, 2, 4, -5, -1, 2, 2, 0, 3, 0, 0, -5, 2, 0, -1, 1, 0,
         2, 5, 2, -3, 0, -1, 6, 2, 4, -3, -4, 4, 1, -4, 0, -4, 0, 0, 0, 1, 1, 4, -1, 6, 6, 0, -1, -1, 3, 2, 3, -4, -3,
         -2, 7, 0, 3, 3, 2, -3, -1, 3, 3, 7, 3, 4, -5, 7, 0, 2, 1, -5, 0, 0, 5, 0, 0, 7, -1, 0, 3, 3, -2, -3, 0, 5, -1,
         0, 2, -2, -2, 1, 2, 6, 2, 2, 4, -3, -2, -3, 5, 4, 1, 0, 1, 1, 7, -2, 3, -1, 1, 1, -7, -3, 0, 4, 1, -1, -1, 3,
         0, 3, 4, 6, -1, 1, 1, -2, -2, -2, 0, -1, -2, 1, -4, 6, -1, 3, 2, 5, 2, -1, -3, -3, -2, -3, -2, -2, 0, 3, 6, -1,
         2, 4, -3, -2, 0, -4, 0, -5, -1, 2, -1, 0, 0, -1, 5, 3, -4, -4, 0, 2, -4, 1, -3, 1, 1, 7, 1, 1, -1, 3, -1, -2,
         1, -2, -2, 2, -1, 0, 1, 1, 2, -1, -1, 1, 1, -1, 2, 1, -1, -4, -3, 2, -4, 0, -2, 3, 1, 5, 3, -1, 3, 2, 0, 3, -3,
         1, 1, 0, 5, -3, -3, 2, -3, 1, -1, -2, 5, -2, -1, 2, 4, 7, 6, -3, 4, -2, -2, 3, 3, 0, -6, -7, 6, -3, 1, -3, -1,
         2, -1, 2, 4, 1, -1, -2, 4, 6, -2, 4, -5, 0, -3, -3, 0, 1, 1, -1, 0, 3, 4, 0, 3, 4, 1, 3, -1, 1, -1, 4, -5, 4,
         -6, -3, 2, 2, -3, 0, -1, 5, 2, 4, 2, 4, 0, -2, 1, 0, 3, 5, 2, -4, 0, 2, 2, -3, 1, 9, 2, 4, 0, -3, 3, 1, 2, -5,
         3, 0, 2, -2, 1, 0, 3, -1, 2, -1, -2, -1, 2, 1, 3, 0, 5, -1, 8, 4, -3, 1, 2, -2, 2, -1, 1, 2, 0, 4, 2, -2, 1, 0,
         4, -3, -6, 5, 0, -5, -5, 2, 4, 1, 5, 0, -2, -5, 2, 1, 0, 2, 2, 5, 0, 2, 4, 4, -4, 2, -4, 2, 0, -5, -3, 2, 3, 0,
         1, 1, 1, 6, 1, 4, 0, -4, -1, 0, -1, 3, 4, -4, 0, -2, -5, 1, -1, 2, -4, 5, 1, -6, 6, -2, 2, 0, -3, 4, 2, -1, 2,
         -3, -2, 0, -3, -5, -6, -2, 0, 0, 1, 1, 5, 0, -2, 4, -5, 1, -2, 2, 1, 0, -1, -6, 0, -3, 0, 4, -1, 0, -3, 4, 5,
         0, 3, 3, -4, 0, 0, 1, -2, -5, 5, -6, -1, -2, 1, 3, 2, 0, 3, 6, -1, -1, 4, 5, 3, 1, -2, 3, 1, 1, 1, -4, 2, -2,
         1, 2, 5, 0, 1, -2, 5, 8, -1, 2, -5, -2, -2, 3, -4, 2, 3, 4, 1, 2, 6, 4, 1, 1, 0, 3, 4, -1, -1, -2, -1, 0, -5,
         -5, -7, 4, -2, -1, -1, 3, 7, 4, 6, 3, 2, -1, -1, 0, -3, -1, 3, 0, -2, 3, -6, -1, 1, 2, 0, 2, -3, -5, 1, -1, 2,
         9, 6, 2, -3, -2, 2, -4, -1, 2, 3, 0, -2, 6, 5, 5, 2, 2, 0, 1, 2, 5, -2, 0, 2, 4, 0, 2, 2, 2, 1, 2, 4, 0, 2, 3,
         -1, -2, -2, 1, 1, 3, 1, 4, 3, -2, -1, 2, 5, 2, 1, 10, -1, -1, -3, 4, -3, 1, 1, 4, -3, -2, -2, 5, -2, 2, 2, 0,
         -1, -2, -3, 8, 3, -1, 3, -1, -1, 2, 1, -3, -1, -7, -3, 0, -1, 0, 1, -1, 6, -3, 4, 0, 1, 2, 3, 6, 4, -2, -1, -7,
         1, 2, 2, -4, 2, -1, -2, -2, 4, -1, 2, -1, -3, 1, 1, -2, 5, -3, -2, 3, -1, 0, -1, 1, 1, -1, -3, 3, -3, -4, -4,
         1, 6, 1, 2, 1, 3, -4, -3, 0, -1, -2, -2, -2, -1, -5, 0, 4, -2, 4, -1, -3, 0, -6, 4, -3, -8, -4, 1, 3, -3, 1,
         -1, -1, 2, 3, 1, 0, -6, 0, -1, -3, 4, 3, 1, -5, -1, 1, -2, 6, -5, -2, 3, 5, 6, 4, 0, 0, 1, -4, 4, 7, 4, 0, 1,
         -4, 4, 1, 0, 5, 0, 0, -5, -4, 9, 3, -1, 0, -2, 2, 5, 1, 0, 4, -4, 4, -6, -3, -2, 1, 4, 1, 2, 11, -3, -1, 1, 1,
         1, 3, -1, 0, -3, 2, 3, 1, -5, -3, -1, -3, -1, 1, 4, 4, 1, 1, 0, 0, 2, -2, 1, 1, 0, -2, -4, 6, 0, -2, 4, 1, 1,
         1, 3, 0, 1, 0, 8, 2, -1, 0, -1, -2, -1, -1, -3, -2, 2, 6, -7, 0, 4, 6, -2, 6, -1, 6, 2, 5, -1, 2, -3, 0, 2, -1,
         0, 0, 6, 5, -2, 2, 0, 2, 4, 0, -1, 0, 2, 6, 3, -2, 1, 2, -4, -4, -1, -1, 3, 3, 1, 5, 5, 3, 0, -2, 3, 1, 0, -4,
         3, 1, -2, 3, 0, -5, 6, -6, -5, 2, 2, 7, -4, 2, 1, -1, -2, 1, 2, 4, -2, 0, 0, 1, -2, -3, -3, 1, 0, 0, 4, 5, 0,
         4, 5, 2, -2, -2, -6, -2, 3, -6, 0, 5, -2, -3, 0, -2, 3, 3, 0, 3, 1, 0, -3, -3, -3, -3, 3, -2, -1, -2, -3, -1,
         -5, 2, -2, 5, -5, -4, -2, 4, 2, 6, -3, -5, -1, -3, 2, 1, 0, -3, -2, -2, 1, 2, -1, 6, 0, -1, 6, 3, -1, -2, -1,
         2, 1, -1, 2, -2, -1, -4, -1, -2, -3, 3, 7, 6, 1, 3, 0, 1, 1, 0, 3, 2, 5, 2, 4, -5, 1, -4, 6, -4, 0, 6, 2, 3,
         -4, 2, -5, 5, 0, -1, 1, -2, 4, 2, 3, 1, -4, 0, 1, -3, 3, 7, 5, -1, 2, 3, -2, -4, 2, -4, 0, 4, 2, -1, 5, 1, 4,
         -1, -2, -2, -1, 1, -1, 1, -1, 2, 3, 0, -5, 2, 2, -2, 0, -1, 4, 5, 0, 3, 3, 0, 1, 5, 1, 2, 3, -1, -1, 1, -2, -1,
         5, -2, 2, 5, 2, -2, 5, 2, -2, 2, 2, 5, 2, -3, 0, 5, 4, 3, -1, 0, 0, -2, -2, -2, 6, 0, 2, 4, -2, 5, 0, 3, 1, -1,
         3, 1, 1, 2, -2, -1, -1, 4, 5, -2, -3, 2, 2, 2, 5, 3, 3, 0, 2, -1, 6, -2, 0, 0, 4, 4, 3, 4, -2, 2, 4, -5, -2, 1,
         2, -1, 1, -1, 4, 2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 5, 3, -2, 2, 1, 6, -1, 3, 0, 0, -5, -5, 0, 6, -5, -5, -1,
         5, -1, 1, 1, -1, 5, -3, -8, 2, -1, -3, -2, 0, 1, 1, 1, -3, 2, -2, -1, -6, 2, -3, 8, 5, -7, -4, 0, -5, 1, -3,
         -2, -4, 6, -1, 0, 1, -1, 1, -3, 5, 1, -6, -2, 3, -2, -2, -7, 1, 4, -5, 1, 0, 0, 3, 4, -3, 1, -3, 1, 0, -4, 6,
         -4, 1, 1, 4, 2, 3, 1, 2, 0, 4, -2, -8, 0, 3, 0, 4, 7, 0, -3, -2, 1, 1, 2, 6, 5, 5, -1, -4, -1, -4, 1, -7, 1, 4,
         -3, 5, 4, 2, 0, 6, 2, 1, 3, 2, -7, -2, 1, 0, -2, 2, -4, 3, 3, 0, 6, 3, 3, 0, 0, 7, 6, -1, -2, 0, 4, -5, -5, 2,
         -3, -2, 0, 1, 3, 1, 2, 3, 1, 1, 3, -3, 4, 0, 3, -6, -6, -1, -2, 0, -3, 0, 3, 2, 5, -1, -2, 6, 1, 2, 3, -1, 1,
         -4, 1, 0, -7, -1, 3, -1, 8, 0, -1, 5, -1, -2, 2, 2, -4, 2, 3, -1, 5, -2, -1, 3, 3, -1, 6, 5, -1, 3, 6, 1, 2, 3,
         0, -1, -2, -3, -2, -2, 0, 1, -2, -1, 2, 3, 1, 1, 6, 0, 3, 1, 1, 3, 5, 2, 4, -2, 5, -3, -3, -1, -2, -8, 6, 2, 5,
         7, 5, 2, 0, -2, -5, 2, 1, -1, -1, -5, -1, 0, -3, -3, -2, -2, 4, 2, 3, 4, -3, 1, -1, 6, 4, -3, 2, 0, -4, 2, -3,
         0, -1, -6, 2, 1, 5, -2, 2, 2, 3, -3, -1, -5, 5, 1, -7, 1, 1, -3, -3, 4, 2, 5, -1, -3, -4, -4, -2, 2, -2, 2, 4,
         1, 1, 1, -3, 3, 0, 0, -4, 5, -7, 3, -2, 4, -3, 3, -2, 2, -1, 1, 1, 2, 2, 0, -3, 0, -1, 3, -1, -4, 1, 0, 5, -1,
         4, 8, 2, -3, 3, 0, 0, 3, 2, 4, -3, 2, 1, 4, -5, 2, -1, 5, -1, 0, -1, -5, -2, 1, -1, 7, 1, -1, 0, 0, 0, 5, 2,
         -1, 2, 1, -3, 2, 0, 2, 2, 3, 0, -1, -3, 1, -3, 5, -1, 7, -2, -1, -4, -3, 6, 3, -4, 0, 6, 3, 10, -3, -5, 1, -3,
         -2, 7, 0, -4, 2, 2, 2, 3, 0, 2, 3, 3, -1, -4, -9, -3, 6, 1, 0, 1, 11, 6, 1, -1, 1, 2, 2, -5, 0, -1, -2, 2, 2,
         3, 2, -2, 2, 5, 2, 0, 0, 0, -2, 0, 6, 5, 0, -8, 1, 3, 2, 2, 2, -2, 3, -4, 4, 4, -2, 5, 0, -1, -1, -4, 2, 7, 3,
         1, 2, 1, 4, 4, -2, 1, 4, 0, 1, 0, 2, 2, 3, 2, 6, 1, 4, 4, -2, -2, 3, 0, 4, 4, 1, 2, 4, -5, 6, -1, -4, -3, -1,
         -3, 0, -1, 5, -3, 1, 4, 5, 0, -3, 1, -5, 3, -2, -3, -3, -6, 2, 5, 1, -1, -6, -1, -5, -5, -5, 2, 2, -1, 1, 0, 2,
         1, 1, 3, 1, 0, -3, -5, 0, 0, 0, 2, 1, 1, 2, 0, 1, -1, 5, -2, 0, -4, 1, -4, -2, -6, -1, 0, 1, 1, 0, -2, 3, -3,
         0, 0, 5, -3, 3, 1, -3, -1, -2, 4, 0, 2, -6, -5, 0, 1, -1, -1, -5, 2, 7, 2, 1, 2, 0, -2, 0, 2, 2, 3, 0, -2, 0,
         -1, 4, 4, 4, 3, 5, 1, 7, 3, -2, -7, -4, -2, 0, -1, 5, -1, 8, 0, 2, 3, -1, 1, -1, 2, 2, 3, 4, 3, -2, 5, 0, -1,
         -4, -1, 0, 0, -1, 1, -3, 6, 1, 4, -3, 6, -1, 5, -1, -2, 3, -1, -1, 3, -3, 1, 3, -3, 0, -2, -2, -1, 3, -4, 4,
         -4, 7, 2, -2, -1, 2, 1, 2, -2, -1, 1, -1, 6, -6, -3, -4, 6, 4, 2, 1, 2, 5, 0, 0, -3, 0, 3, -5, -3, 2, 1, -1, 3,
         -1, 0, 2, 0, -1, 6, 3, 0, 7, 0, 2, 5, -3, 4, -2, 1, -2, -1, 1, 1, -3, 0, -3, -2, 3, 1, 1, 1, 4, 3, -2, 4, 1,
         -6, 2, -2, -1, -1, -2, 0, -1, 4, 6, 5, 1, 5, -1, 0, -2, 2, 0, -3, -6, 6, 0, -4, -5, 2, 1, 4, 1, 1, 3, -3, -3,
         2, -2, -1, 2, -1, 1, -1, -1, 1, -3, 1, -1, -3, 3, 0, 5, 1, 3, -2, 3, -3, 4, -5, -1, -6, 0, -1, 4, -8, -2, 1,
         -1, -3, 1, 3, -5, -2, -3, 2, 3, 1, 0, -1, -2, -5, 6, 3, -1, 0, 0, 3, -7, 3, -1, 5, 1, -2, -1, 3, 3, -3, 1, -5,
         -4, -3, -2, 1, 6, -5, -3, 5, -2, 3, 3, 1, 0, -2, 1, -1, 6, 0, 1, -3, 0, 0, 3, 3, 1, -1, 2, 1, 1, 2, 4, 1, 1, 3,
         -1, 3, 0, 0, 0, -3, 0, -1, 0, 2, 1, 4, 7, 4, 5, 7, 3, -2, 0, 4, 0, -4, 2, -2, 1, 1, -1, -3, -1, 3, 5, -1, 2, 3,
         0, 1, 3, 1, 0, -1, 0, 0, 1, 0, 0, -1, 2, 0, 6, -2, 2, 4, -1, 4, 0, 0, 1, -1, 0, 4, 0, 4, -1, 1, -4, 1, 2, -1,
         1, 0, -1, 6, 1, 3, -1, 2, 2, 3, 2, 7, 0, 0, -2, 2, -3, -3, 3, 2, 7, 3, 0, 3, 1, -4, 2, -2, 6, -5, 3, -5, 0, -6,
         2, -1, 2, 1, 4, 1, 6, 2, 2, -1, 0, 1, -3, 0, -1, 1, 3, -3, -3, 4, 0, 1, 0, 1, 0, 4, 0, -2, 1, 1, -3, 0, 0, 4,
         3, 2, -1, 2, 4, 3, -3, 6, 1, 0, 3, -3, -2, 2, 0, 2, 1, 0, 3, -5, -2, -2, 0, 6, -1, 6, -2, -6, 3, 1, -3, 2, 1,
         -2, -3, -1, 2, 3, 0, 4, -4, -3, -4, -3, -8, -1, 1, 4, -1, 1, 1, 1, -3, 1, -4, 0, -2, -3, 3, -1, 0, 0, -5, -1,
         0, 8, -6, -2, 0, 1, -1, -3, 5, -4, 0, -6, 3, 1, -1, -1, -3, -2, -3, -1, -5, 2, 2, 0, -4, 6, 2, 9, -1, 1, 3, 6,
         -1, 4, -4, -2, -3, 3, -5, 0, -2, 0, 6, 0, 5, 4, 2, 1, 4, 2, 3, 5, -5, -1, 1, -1, -5, -1, 1, -2, 3, 5, -1, 1,
         -3, 4, 1, 4, 1, 3, 6, 5, -2, 4, 6, 0, 4, -2, -4, 3, -1, 3, 5, -2, -1, -3, 1, 0, 1, 1, 3, -1, 2, -1, 1, 1, 4,
         -4, -3, -3, 6, 9, 4, 0, -3, 0, 2, -4, 1, 0, 2, 1, -2, -1, -2, -2, 2, 3, 2, 2, 4, 5, -4, 5, -3, 5, 2, 2, -10, 3,
         -2, 0, 6, -4, 0, 2, -1, 3, -1, -1, -1, -3, 11, 3, 0, 1, 4, 1, -2, 1, -5, -4, 2, 3, -2, 0, 1, 1, -1, -4, 1, -2,
         -3, 2, 5, 7, -1, -2, 2, 0, 5, -2, -1, -6, 2, -3, 3, 4, 3, 2, 0, -4, 5, -4, 1, 2, 5, -4, -2, -4, 2, -4, 5, 3, 2,
         3, 3, 4, -2, 5, -2, 0, -3, -2, -5, -1, -1, 0, 2, 1, -2, 0, 3, 4, 3, 2, -2, -2, 0, 3, 2, 7, -2, -4, 0, -3, 1, 1,
         -3, 1, 1, 2, -4, 4, -2, 1, 1, 2, 4, -2, 0, 1, 3, -7, 2, 1, -5, -4, -2, 1, -3, 3, -4, 3, -2, 1, 2, -4, -3, 1,
         -2, 3, -2, -3, -4, -2, 0, 0, 2, -3, 4, 1, 4, -4, -1, 4, 0, 0, 4, 1, 0, 2, -2, 1, 3, -3, -2, 0, 4, 1, -6, -1, 8,
         1, 2, 3, 0, 2, -1, 0, 8, 0, 2, 3, -2, -1, 5, 1, 0, -1, 1, -1, -4, -2, 0, 0, 1, -3, 4, -4, 7, -1, 2, -2, 8, -4,
         -1, 6, 1, 3, -1, 7, 2, 3, 4, -1, -1, 4, 1, -3, 2, -1, -1, -2, 0, 4, 3, 0, 5, 5, 0, 6, 3, -1, 0, 4, 0, -2, 0,
         -2, 4, 8, -2, 0, 0, 6, -1, 3, -3, 2, 5, 4, 1, 3, -1, -2, 5, 3, 4, 1, -2, 3, 6, 0, -1, 1, 3, -3, 1, 3, 4, 5, 4,
         3, 3, -3, -3, -2, 0, 0, 2, 0, 1, 1, 1, 3, 4, 1, 2, 1, 0, 0, 0, 3, 1, -1, 4, -1, -4, -4, 0, -1, 5, -2, -4, 3, 2,
         5, 4, 2, 4, 6, 2, -2, 5, -2, 0, 3, -1, -5, -5, 2, 1, -2, 0, 4, 6, 2, 6, 4, -2, 2, -2, 0, -2, 0, 0, -1, 4, -6,
         5, -1, 2, 0, 10, 4, -1, 0, 1, 1, 1, 1, -2, 6, 1, -5, -2, 0, -1, -2, -5, -5, 2, 4, 2, 2, -5, 2, -5, 1, 1, -5,
         -1, -2, 0, 4, 4, -1, -3, 0, 3, -2, -4, -1, 2, -3, -5, 0, -1, 0, 0, -5, 2, 0, -1, 1, -3, -2, -1, 5, -1, 1, 3, 1,
         0, -4, 1, -4, 3, 0, -2, 1, -2, -4, 2, -3, 5, -1, -1, 0, -2, 3, 3, -5, -5, 0, -1, -1, 1, 4, 0, 0, 1, -2, -1, 1,
         4, 0, 4, 2, -1, 0, 1, 3, 1, -7, -4, 1, 2, 7, -1, 1, -2, 2, 0, -3, 0, 2, 0, 5, 1, 2, -3, 1, 0, 0, 2, 0, 4, 1,
         -1, 6, 0, 4, 5, 1, -3, -5, 3, 4, 0, 4, 0, -1, -1, 1, 3, 1, 0, 0, 5, 5, 4, 1, 1, 3, 4, 5, 1, 3, -1, -7, 1, 5,
         -2, 4, -2, -1, 1, -1, 8, -2, 1, -3, 1, 1, 7, 1, 0, -3, 4, -1, 4, 1, 1, -2, 0, -4, 1, 0, -2, -3, 4, -2, 4, 6,
         -5, 1, 4, 1, 2, -2, 3, 4, 1, -3, 2, 5, -2, -1, 2, 1, -4, -2, 0, -1, 2, 0, 4, 2, 5, -5, 1, -5, -2, 3, 1, -1, 2,
         -4, 0, 4, 0, -6, 3, 6, 5, -1, 1, 0, 1, 0, 1, -2, 1, 0, -7, 2],
        dtype="i2")


def get_problematic_residual2() -> np.array:
    return np.asarray(
        [-1, -4, -4, 4, 4, 1, 1, -2, 0, -3, -7, -5, -3, 1, -2, -3, -1, 2, 1, 2, -3, -7, -4, -1, 3, -1, 0, 2, 5, -4, -3,
         -4, -6, -3, -1, 0, -1, 0, 1, 2, 0, 1, -2, -2, -2, -8, -6, -2, -2, 1, -1, -3, -3, 4, 2, -3, -2, -3, -2, -2, -3,
         -9, -4, 0, -7, -5, 3, 2, -2, 0, 0, 3, 4, -1, 0, 5, 5, -3, -4, -3, -4, -2, -2, 0, 1, -1, -3, 4, 1, -2, -1, 1, 4,
         1, -1, -3, 0, -2, -3, -1, 0, -1, 0, -5, -1, -2, 1, 4, 2, 2, 5, 4, 3, 1, 2, -3, 0, 2, 3, 3, -1, -3, 0, 1, 0, -4,
         -3, -3, 1, 7, 0, 1, 5, 3, 3, 5, 7, 3, 8, 4, -2, 5, 5, 0, -2, -2, 0, 1, 1, 2, 4, 2, 2, 6, -1, 2, 2, 2, 2, 2, 1,
         -4, -1, -2, 0, 7, 10, 4, 0, 4, 3, -1, 4, 6, -1, -3, 1, 3, 4, 3, 4, 0, 3, 3, -2, 4, 2, -6, -4, 6, 6, 0, 3, 4, 1,
         5, 5, 1, -1, 0, 0, -4, 1, 3, 1, 0, 7, 6, 0, -1, 1, 0, 2, 1, 1, -2, 0, 3, 0, 2, 3, 4, 1, 1, 4, 2, 5, 6, 1, -1,
         0, -2, -2, 2, 2, -1, -2, -2, 2, 0, 1, 1, -4, 4, 1, -3, -1, -5, 0, 6, -1, 3, 3, 4, -1, -3, 5, 0, -4, -1, -3, -5,
         2, -1, -6, 0, -1, 0, 7, 6, -2, -3, -2, -1, -2, -5, -4, -1, 0, 0, 1, -2, -2, -6, 0, 0, -2, 1, 3, -1, -1, 4, -1,
         -4, -1, -10, -7, -5, -6, -1, 0, -3, 0, 1, 1, 1, -2, 0, -1, -4, -2, -3, 2, 2, -5, 0, 1, -2, -2, -5, 1, -1, -2,
         1, -8, -4, -2, -2, 2, -3, -2, 7, 1, 2, 4, 1, -2, -3, 4, 7, 0, -5, -1, 1, -1, 0, -3, -1, 3, 1, 1, 1, -3, -1, 4,
         1, -2, 2, 2, 1, 4, 7, 1, 0, 3, 3, 2, 1, 0, -6, -3, 5, -1, 3, 5, -2, -2, 2, 2, 2, -2, 3, 5, -3, -2, 3, 3, 6, 9,
         7, 4, 1, -3, 3, 1, -6, -2, 4, 2, 1, -2, 2, 4, 1, -1, 5, 2, 1, 2, -2, -4, 3, 5, -1, -3, 2, -2, -3, 2, 3, 4, 3,
         4, 1, 1, 8, 7, 4, 1, 1, -3, 0, 7, 1, 1, 1, 1, -1, 0, 2, 1, -2, -1, -5, 2, 4, -2, -1, 3, 6, 8, 7, 2, 1, -2, 2,
         7, 2, 0, 5, 5, 3, 3, 7, 5, -2, 1, 3, -3, -5, 0, 1, 0, 4, 1, -1, -2, -1, 5, 5, 3, 2, -5, 1, 2, -6, -4, -5, -3,
         -1, 3, 4, 0, 7, 4, 0, 6, 3, -2, 1, 0, -3, -2, -2, -3, 2, -1, -4, 0, 3, 0, -2, 0, 5, 4, -1, -2, 0, 1, -2, -3, 8,
         2, -1, -3, -5, 0, -1, -3, 0, -1, -3, -2, 0, 5, 3, 1, -4, -3, -1, -2, -2, -3, -1, -2, -5, 0, 0, -4, 0, 0, -1, 0,
         5, 6, -5, -7, -2, -6, 2, 7, -4, -5, -1, -7, 0, 3, -3, 2, -3, -4, 1, 5, 2, -2, 2, -3, -2, 1, -2, 0, 6, -1, -1,
         6, 3, 3, 8, 2, 1, 5, 6, -2, 0, -4, -4, -6, 1, -3, -2, -3, 0, 2, 2, 6, 5, -1, -3, -3, 5, 4, 0, 5, 6, 0, 1, 4, 2,
         1, 2, -4, 0, 2, -2, -6, -1, 4, 3, 0, 5, -2, -1, 1, 3, 2, -2, 4, 6, -3, 1, 2, 4, 4, 1, -1, 0, -1, 2, -2, -1, 3,
         1, -3, 5, 7, -2, 2, -2, -4, 0, 1, 2, 1, 0, 4, 7, 6, 0, -3, -4, -3, 1, 5, 1, -1, 4, -1, -5, -1, -1, 0, -1, 1, 6,
         1, -9, 0, 0, 0, 11, 1, -4, -4, -2, 2, 1, -2, -5, -5, 1, -5, 0, 5, 0, 5, 2, -5, 5, 2, 2, 4, -4, -5, -2, 3, 5,
         -3, -2, -1, -3, 6, 7, 3, -6, -12, -3, 3, 5, 8, 3, -1, 3, 8, 5, -3, -2, -1, -4, -1, 2, -3, -5, 2, 1, 1, 7, 3, 2,
         -4, 0, 2, -3, -4, -2, -1, -1, -4, -2, -2, -1, 0, 0, 2, 2, 2, 2, 1, 5, 1, -5, -4, 5, 1, -3, -2, -3, -2, 0, -2,
         -1, -5, -2, 6, 1, 2, 1, 0, 3, -1, -1, 5, -2, -3, 5, 5, -2, 1, 5, 0, -4, -2, -2, -2, -5, 0, 5, 5, 0, -2, 4, 4,
         -3, -4, 2, 1, -2, 2, 4, 0, 4, 3, 4, 1, -2, 4, 4, 1, -1, 2, 0, -2, 0, 1, 4, 0, -1, 6, 3, 4, 3, -1, 6, 4, 1, 2,
         4, 4, 3, 2, -11, -5, 2, 0, 2, 1, 3, 3, 5, 5, 5, 0, -4, -1, 7, -5, 0, 6, -2, 2, 5, 7, 3, 0, 0, -2, 2, 0, 0, 4,
         1, -5, -3, 4, 4, -1, 1, 1, -4, -3, -5, -3, 6, 0, 3, 5, 2, 3, -4, -4, 5, -1, 1, 3, 3, 6, 0, 2, 2, -1, -3, 0, -1,
         -4, -2, 0, -1, -2, 1, 3, -3, 1, 0, -9, 2, 4, 1, -2, -3, -1, -5, -3, -3, 0, 8, 5, 10, 12, 0, -6, -3, 1, 2, 2, 2,
         -6, -5, 2, -7, -8, -1, -1, -1, -1, 5, 3, 1, 8, -1, -5, -1, 4, -2, 1, 5, -4, -3, 3, 2, -4, -2, 5, 2, 4, 8, 2,
         -1, -3, 0, 2, -4, -2, -8, -6, 4, 5, 2, -3, -4, 3, 7, 1, -6, -5, 3, 0, -2, 2, 3, 8, 5, -3, 1, 8, 0, -6, -5, -2,
         -2, -1, 2, 1, -3, -6, -2, 1, -1, 0, 1, 3, 5, 4, 4, 2, -4, -2, 0, -3, -1, 0, -4, -4, 3, 3, 5, -3, 1, 5, 0, -3,
         1, -1, -3, 1, 3, 2, -1, 3, 0, 2, 7, -3, -8, 3, 5, -6, 4, 11, -2, 0, 0, 4, 4, 2, 2, -1, -2, 0, 0, 3, 4, -1, 0,
         2, 0, -3, -1, -2, -1, -3, 2, 1, 4, 7, 3, 1, 3, -3, 2, 1, -3, 2, 1, -3, 1, 0, 4, 0, -3, 1, 1, 1, 3, 0, -2, -3,
         -2, 1, 2, 0, -1, 1, -3, 1, 5, 5, 4, 2, 0, 1, 1, 3, 4, 4, 2, -1, -1, 2, -4, -1, 1, -3, 1, -1, 0, 3, 3, 2, -2, 0,
         2, -1, 3, 5, 3, -3, -4, 1, 1, 0, 0, 0, 0, 2, 6, 0, 0, 1, -2, -1, 2, 4, 1, 1, 4, -1, -4, 0, 3, -4, 1, -1, 1, 5,
         1, -1, -1, 2, 0, 0, 2, 0, 2, 5, 0, -2, -6, -5, 3, 5, 2, 2, 7, 2, -4, -2, -4, -2, 1, 1, 2, -3, -2, -2, -3, 4, 2,
         -5, 0, 3, 2, 5, -1, -6, 2, 4, 4, 1, 4, 2, -3, 0, -2, -3, 0, 2, 4, 7, 2, -4, 1, -1, 0, 0, -3, -4, 0, 0, 0, 0, 2,
         7, 4, 0, 7, 4, -1, -4, -6, -4, 0, 3, 4, 3, -3, -1, 5, 4, 3, 0, -2, 0, 0, -2, 8, 8, 2, 0, -4, 0, -3, 2, 2, 0, 4,
         5, 0, 3, -1, -2, 5, -2, 3, 0, -4, 7, 0, -4, 5, -1, -4, -3, 0, -1, 1, -2, -4, 2, 1, -1, 4, 2, 2, -2, 0, 0, 0,
         -1, 3, 3, -2, -1, -1, 3, 1, 1, 4, 2, 6, 1, -4, 3, 4, 2, 0, 1, 1, -1, 2, 2, -2, -2, -2, -3, 1, 2, -2, 0, 2, 3,
         7, 4, 0, -3, 0, 1, 1, 8, 8, -2, 2, 2, -2, -1, 5, -2, -1, -4, -6, -4, 2, 2, -2, -1, -7, -6, 4, 2, 3, 1, -5, 0,
         -5, -4, 3, 5, 7, 0, -6, 2, -1, 3, 3, -6, 2, 0, -4, -2, -1, 2, -3, -6, -3, -3, 1, 0, -2, 1, -1, -2, 1, 0, 0, 3,
         1, 0, 0, 5, 0, -8, -1, 2, -2, -3, 4, 10, 2, -5, -3, -1, 6, 0, -1, 5, -1, -5, -2, 2, 2, -4, -3, 2, 2, 0, 4, 4,
         2, 0, 1, 5, 2, 2, 2, 0, -1, -2, 2, 0, 4, 2, -3, 4, 5, 3, 5, -4, -5, -5, -4, 2, 1, 4, 1, -1, 2, 3, 5, 5, -3, -2,
         -1, 0, 1, 5, 9, 3, -2, -5, 1, 10, 1, -3, 1, 1, 0, 2, 4, 5, 0, -2, 1, 8, 1, 3, 7, 3, -2, -2, -1, 5, 0, 3, 3, 4,
         5, 1, 5, 4, -1, 5, 5, 0, -3, 2, 6, 0, -2, -1, -2, -1, -2, -2, 0, 3, 9, 11, 5, 3, 0, 0, 2, 8, 6, -1, -1, -3, -6,
         6, 5, -2, 1, 3, 5, 3, -3, -2, 3, -2, -3, -4, -4, 2, 0, 1, 8, 5, 1, 1, 1, 2, 2, 3, 4, 0, -4, -3, 0, 3, 1, 4, 1,
         -3, -5, -2, 0, 8, 6, 3, 4, 4, 1, -1, -4, -4, -2, -3, -5, 0, 4, 2, 1, 2, -1, -1, 1, 1, -2, -1, -2, -6, -3, 1, 2,
         0, -6, -5, 4, 1, -2, -4, -3, 0, -4, -5, 0, -5, -6, -7, -4, 4, -2, -1, 0, 4, 5, 1, 5, -1, 2, -1, -6, -3, 1, -5,
         -4, -1, 0, -7, -7, -2, 2, 3, 0, -1, -1, 3, -2, -7, 1, 3, -1, 2, 4, -1, -1, 2, -4, -2, 3, 3, -3, 0, 2, 1, 3, 5,
         -3, 0, 5, 0, 2, 4, -4, -3, 3, 2, 2, -3, -2, 1, 2, 4, 2, -6, -5, 0, 0, -1, 0, 5, 0, -8, 0, -2, -6, -1, -1, 2, 5,
         1, 6, 7, 4, -4, -3, 7, 0, -5, -2, -2, 0, -1, -2, -1, 1, 1, 2, 1, 14, 4, -3, 8, 7, -4, -1, 1, 4, -5, -3, 3, 4,
         5, 10, 6, -4, 5, 7, -1, -6, 0, 4, 5, 2, 1, 2, 7, 4, -5, -2, 6, 2, 0, -3, -6, 1, 2, -2, 4, 6, 2, 5, 3, 1, 5, 1,
         1, 2, -2, -4, -1, -1, -3, -1, 8, 6, 3, -3, -3, 1, 1, 6, 5, -4, -3, 1, 5, 9, 0, 2, 4, -1, 1, -1, 4, 8, -5, 2, 5,
         2, 5, -2, -1, 0, 4, 3, 1, 2, 8, 2, -5, -6, -2, 3, 5, 1, -5, -3, 0, -3, 0, 0, 5, 6, 1, 2, 7, 0, -8, -6, -2, 2,
         6, -2, -6, 0, 0, -1, 4, 3, 4, 1, -3, 1, 0, -2, -4, -5, -1, -7, -3, 7, -2, -4, -3, -1, 3, 5, 6, -1, -4, 7, 0,
         -2, 2, -11, -12, -5, -3, -7, -6, -1, 2, 4, 9, 4, 1, -4, -1, 2, 0, -3, -2, 5, -1, -3, 2, -4, 0, 5, 0, -6, -6,
         -2, -1, 0, 3, -2, 2, 5, -5, 0, 4, 2, 0, -5, -2, 0, 4, 6, 2, -2, 3, 5, 1, -1, 3, 4, -1, -4, 3, 0, -5, -4, -4,
         -1, -2, -6, -1, 1, 3, 5, 2, 3, -3, -3, 2, 1, -2, 0, 4, -1, -1, 2, 1, -2, 0, 5, 4, 2, 4, 0, 0, 5, 7, 7, 6, 1, 2,
         4, 0, -4, -5, -2, -3, -1, -2, 1, 3, -1, 0, 7, 7, 3, 4, 1, -2, 6, -1, -3, 0, 4, 3, 3, 3, -1, -3, 0, -4, 0, 4, 2,
         1, -5, -5, 2, 3, 11, 8, -2, 0, 3, 3, 7, 0, -5, -7, -2, 2, 3, 4, 3, -1, 4, 3, 3, 1, 1, 6, 4, -3, -5, 0, -1, 1,
         3, 3, -3, 2, 7, 1, -2, 1, 0, 2, 1, 2, 5, 0, 6, 5, -1, 2, 1, -4, -3, -3, 0, 5, 2, -3, 2, 2, 1, 8, -1, -4, -1, 4,
         0, -3, 5, 1, -2, 2, 1, 1, 0, -3, 4, 2, -1, 0, -2, -3, -3, -4, 4, 3, -1, -1, -3, 0, 2, -4, -4, 1, -6, -3, 5, 1,
         -5, -4, -4, -7, -2, 0, -4, -5, -1, 0, 1, 2, -1, 0, 1, -2, 1, -3, -2, 1, 4, 6, 1, -2, -3, 1, -3, -6, -2, 0, -2,
         4, 1, 1, 4, 0, 1, 3, 3, 0, -3, 1, 1, 0, 2, 1, -3, -5, -4, -3, 1, 1, -6, 2, 7, 0, -3, 3, 3, -1, -4, 2, 2, 0, -1,
         0, -5, -1, 3, 4, 7, 4, -1, 4, -1, -16, -7, 3, 5, 4, -7, -3, -1, -1, 6, 5, 12, 7, -2, 2, -4, -1, -1, 2, 9, -6,
         -1, 3, 1, 2, 2, 3, 8, 1, 4, 6, -6, 0, 3, 0, -1, -2, 9, 3, 6, 9, 1, 5, 3, -4, 2, -1, 2, 1, -3, -2, 1, 4, 0, 0,
         9, 4, -2, 4, 7, 4, 4, 3, 1, -5, 2, -1, -4, 5, 3, 1, 4, 5, -3, -3, 4, 4, 3, 2, 4, -4, -1, -3, 1, 0, -6, 1, 5, 1,
         6, 1, -1, -6, -5, -2, 0, 1, 5, 10, 4, 1, 3, 4, 2, 7, 9, -3, -3, 3, -2, -5, -1, 1, 0, 1, 7, 4, -4, 2, 4, -2, -8,
         -2, -5, -6, 0, -4, 1, 3, -4, -1, 3, -4, -3, 6, 5, 4, 3, -2, -3, -1, -1, 0, 0, 3, 0, -5, -5, 0, 4, 1, 1, -3, -2,
         0, -1, 1, 4, 0, -5, -2, -6, -6, -6, -2, 1, -1, 0, -2, 3, 4, 7, 1, -5, 3, 4, -4, -8, -3, -2, -4, -4, -3, 0, -2,
         -2, -5, -2, 2, -1, -2, -2, -5, -1, 8, 4, 0, 4, 3, 2, 3, 0, 0, 0, 0, 0, -6, -1, 3, -6, -2, -1, -2, 4, 6, 4, -1,
         2, -1, -5, 3, -1, -2, 2, 5, 5, 3, 4, 9, -1, -5, -6, -4, 1, -3, 1, 1, -3, 0, -2, -6, -3, 2, 5, 5, 2, 5, 0, -3,
         -4, 6, 1, 0, 3, -1, -1, -4, 2, 9, 6, 2, 2, 5, 4, 0, -5, 3, 4, 3, 2, -3, 0, 4, 4, 4, -1, -4, 2, 3, -1, 3, 2, 4,
         4, -3, 2, 4, 5, 4, 7, 7, -2, 4, 4, -5, 1, 5, -5, -4, 6, 4, 6, 8, 0, -1, 4, 4, 0, -2, -2, 0, 0, 2, 3, -1, -1, 0,
         1, -6, 5, 9, 7, 5, 3, 8, 2, -5, -3, -3, 3, 8, 2, -3, -1, -3, -2, 12, 7, -1, 0, -1, -3, -1, 0, 1, 6, 2, -3, 0,
         10, 1, 1, 1, 0, 0, -1, 5, 5, -4, -6, -3, 5, 2, 0, -3, 3, 6, 1, 3, -1, -5, 1, -1, -3, 2, -2, 1, 2, 0, -1, -1, 2,
         -2, -4, 2, 3, -1, -1, -5, -3, 2, -2, -2, -5, -9, -3, 2, 0, 5, 4, -2, -2, 0, 2, -2, -5, 6, -2, -7, 5, 0, -3, 1,
         2, 1, 0, 8, 4, -3, 0, -5, -8, 4, 1, -1, 3, 0, 0, -2, 1, 4, 3, -1, 0, -5, -1, -2, -4, 2, 2, 0, 1, -3, -2, 4, 0,
         1, 7, -3, -5, 0, 0, -3, -1, 5, -1, -3, 1, -2, 2, 2, 0, -3, -1, 0, -4, -4, -1, 0, 1, -4, 5, 9, 4, -3, -2, 1, 3,
         2, 2, -2, 4, 6, -5, -2, 5, 3, -2, -1, 2, 3, 4, 3, -4, 1, 7, 2, -4, -5, -1, -1, 2, 4, -5, -2, 12, 11, 1, 7, 3,
         3, 6, -1, -4, 3, 5, -1, 2, 5, 8, 2, 0, 1, 1, -2, -1, 1, 0, 0, -2, 1, 3, 7, 5, -3, 0, -1, -1, 5, 6, -1, 0, 1,
         -3, -5, 5, 9, -3, -3, 2, 7, 4, 3, 5, 0, -2, 5, 0, 0, 0, -4, -1, -3, -5, 4, 3, -1, 2, 2, 0, -2, 3, 2, 2, 6, 5,
         4, 2, 0, -2, -1, 1, -4, 2, 3, 0, -2, -2, 2, 1, 3, 2, 0, -1, -2, 0, 2, 0, 0, -4, -5, 0, 2, -2, 3, -1, -2, -1, 3,
         3, 0, 5, 6, -4, -8, -7, -1, -2, -11, -5, 1, -2, 1, 5, 4, -5, -2, 4, 0, -2, -2, -5, -1, 3, 6, -6, -10, 2, 1, -1,
         4, -1, -3, -6, -6, -5, 0, 7, 1, 5, 4, 0, 2, 4, -2, 0, -1, -7, -6, 0, -2, -9, 3, 9, -5, -2, 2, 2, 6, 3, 2, -1,
         -5, 0, 3, 4, 3, -3, -4, -2, 1, 4, 0, -3, -5, 0, 3, 8, 4, -2, -8, -3, 3, 2, -1, 0, 5, 5, 4, 3, 5, 3, -2, -2, 4,
         -1, -1, 2, -2, 2, 1, 4, 5, -3, -3, -3, -2, 5, 2, 3, 3, 0, 6, 8, 2, -1, 6, 2, -2, 0, -3, -3, 5, 5, 4, 0, 6, 4,
         4, 4, 2, 2, 3, 2, 2, 1, 2, 3, 1, -2, 4, 2, 0, -2, 1, 5, 4, 2, 5, 3, 4, 7, 6, 0, 0, 9, 10, 0, -1, 4, 1, -12, -7,
         1, -1, 5, 7, 0, -4, 3, 5, 1, -6, -5, -1, 5, 2, 3, 0, 3, 0, 0, 2, 0, 3, 5, 1, -1, 2, 0, 1, 4, -1, -4, 5, 6, -4,
         1, 6, 5, -2, -5, 1, -3, -1, -5, -4, 0, -5, 0, 7, -2, 1, 11, 5, 2, 2, -2, -3, 1, 4, -1, -3, -3, -5, 3, 9, -3,
         -2, 2, 0, -4, -5, -5, -3, 1, 1, -2, -6, -3, 3, 1, -1, -3, -4, 1, -5, -4, -2, -3, -1, 0, -2, 0, 0, -2, -3, -3,
         -1, 0, -3, 1, 1, -2, -1, -3, 0, 2, -2, 1, -4, -4, 2, 4, 6, 3, 0, 4, 2, -2, 0, -3, -4, -1, 0, 0, 3, -2, 2, 2,
         -1, -2, -3, 1, 2, -2, -3, 2, 6, 0, 1, -3, -6, -1, 0, -2, 2, 4, 3, 4, -1, -5, 4, -3, -8, 0, 0, 3, -3, -4, 1, 6,
         3, 3, 6, 9, 0, -5, 0, 2, -2, 4, 5, -5, 2, 3, 1, 2, 0, 2, 9, 5, -4, -1, 3, 4, 7, 4, 2, -1, 0, 3, 6, 1, -1, 3, 0,
         1, 1, -1, -2, 1, 4, 5, 0, 2, 3, 2, 8, 3, 2, 6, -2, 0, 0, -2, 0, 2, 4, -1, 2, 8, 3, 6, 4, 4, 4, 3, 4, 0, 4, 2,
         -7, -7, -2, 0, 4, 5, 2, 2, 3, 2, 0, 1, 3, 4, 4, -2, -9, -4, 1, -2, -1, 3, 3, 0, 5, 6, 3, 7, 8, 1, 8, 6, -4, -2,
         0, -1, -1, 1, 5, 0, 3, 1, 1, 3, 2, 2, -2, -5, 1, -5, -3, 2, -1, -1, 4, 2, 5, -2, -2, 3, 2, 5, 3, 0, 2, -1, -1,
         -1, 3, 3, -8, -2, -3, -7, -1, 1, 0, 1, 0, -9, -1, 7, -2, -3, 2, 5, -3, -1, -3, -7, -2, -9, -7, -2, 1, 6, 6, 1,
         -3, -1, 3, 1, 1, 0, -8, -5, 2, -4, -6, 1, -2, -2, -1, -4, 3, 4, 0, -1, 6, 6, -4, -7, -3, -1, 2, 0, 1, 0, -1,
         -2, -1, 1, 2, 2, 1, -5, 1, 4, -1, -4, -6, 0, -2, -6, 7, 11, -2, -4, -5, 2, 6, -1, -6, -2, 2, 2, -3, 3, 6, 0,
         -10, -4, 3, 5, -2, -2, 5, 0, -2, -2, 1, -3, -1, -1, -3, 0, 8, 2, 0, 6, 2, 6, 7, 9, -1, -4, 2, 3, -4, -6, 1, -1,
         1, -1, 1, 4, 3, -2, -2, -2, 2, 5, 4, 0, 4, 8, 2, 3, 0, -1, -3, 2, 0, 4, 2, -4, 1, 2, 2, 3, -2, 0, 4, 6, 6, 4,
         0, 1, -2, -2, 3, 0, -1, 7, 3, -2, 1, -2, 0, -3, -1, 3, 6, 2, 1, 3, 0, 7, 1, -6, -3, -2, 4, -2, -9, -1, -3, 1,
         6, -1, 2, 8, 6, 1, 2, 1, -3, -2, -1, 1, 11, 6, 4, 0, -3, 2, 5, -2, -1, 5, 5, 2, -1, 0, -3, -2, -2, -3, -3, 6,
         5, -1, 1, 2, -2, -1, 4, 2, -2, 0, -5, -3, 4, -1, -1, 2, -2, -5, -4, -3, 3, 1, -2, 2, 2, 0, 0, 2, 1, -7, -5, 0,
         4, -1, -2, -2, 1, 1, 2, 5, 1, -7, -3, 6, 5, -5, -5, -1, -3, 9, 3, -3, -1, -2, -2, -6, -2, 5, 1, 1, 3, 0, -6, 4,
         4, -4, 0, 2, -1, 5, 5, 3, 7, 3, 5, 4, -3, -3, -6, -6, 1, -5, -9, -1, 4, 4, 3, -1, 0, 10, 0, -8, 2, -3, -4, 3,
         -2, 4, 6, 1, -9, -5, 1, 2, 5, 4, 7, 10, 5, 0, -1, 1, -6, -5, 2, 0, 6, 4, -2, 8, 1, -2, 4, 6, 3, 4, 3, -3, 1, 6,
         5, 2, -8, -10, 3, 7, 3, 4, 5, -1, -4, -1, 2, -4, 1, 5, 3, 8, 8, -1, -7, -5, -4, -8, -6, 3, 3, 6, 7, 8, 10, 1,
         -9, 3, 5, -3, -9, 1, 3, -7, -6, -4, 3, 4, -1, 4, 4, 1, -3, -1, 1, 3, 1, -5, 1, -2, -3, 5, 1, -7, -8, 3, 6, -1,
         1, 0, -1, 0, -7, 0, 4, -3, 2, 7, 4, 2, -3, -3, 3, -8, -7, 3, 2, -3, 5, 7, 3, 2, -2, -7, 2, 2, -5, -5, -7, -2,
         7, 1, 5, 0, -1, 11, 3, -9, 3, 0, 0, 2, -2, -11, 3, 10, 1, -1, -1, -9, -4, 1, -3, -6, -4, 0, -7, -1, 7, 0, -3,
         6, 2, 1, -2, -4, -3, 1, -1, -5, -2, -3, 1, -4, -2, 2, 6, 6, -2, 0, 3, 6, -3, -2, 0, -3, -7, -7, 0, 2, -2, -4,
         3, -4, 4, 2, -5, 4, 8, 4, 3, -2, 0, 1, 3, -4, -8, 0, 1, 0, 2, 0, 1, 0, 3, -2, -4, 9, 3, 0, 2, -1, 3, -2, 3, -5,
         2, 29, 4, -16, 2, -7, -10, 1, 8, 7, 2, 4, 1, -7, 3, -1, 1, 8, 7, 2, -2, -4, 2, 4, -5, -2, 9, 4, -7, -4, 4, 3,
         -3, 3, 10, 9, 3, 2, 0, -2, -1, 4, 4, 3, 1, -4, -4, 0, 0, 0, 4, 6, 6, 5, 0, 6, 3, 2, 5, 6, 3, 1, -4, -10, -3, 0,
         0, 3, 3, 6, -1, -4, 7, 3, 1, 1, -1, 5, 2, -2, 2, 1, -2, 2, 2, -3, -2, 7, -1, 0, 0, -5, -8, 4, 4, 3, 0, 3, -4,
         -7, 5, 8, 0, 7, 0, 0, 6, -6, -9, 7, -4, -3, 11, 0, -2, -1, -2, 5, 0, -6, 5, 2, -2, -2, -6, 3, 2, -5, 6, 9, 1,
         -2, -6, 5, -1, -6, 8, 4, -7, 3, 1, -8, -1, -2, 4, 5, -3, -1, -8, 1, 11, -12, -4, 12, -6, 1, -5, -10, 10, -2,
         -2, 13, -6, -1, 13, -7, 0, 8, -7, 5, 4, 0, 10, -5, -7, 18, 0, -20, -2, 5, 1, -2, 8, 3, -18, -1, 1, -8, 5, -5,
         -3, 2, -8, 3, 10, -3, 11, 8, -8, -9, 3, 5, -5, 3, -3, -10, 8, 6, 1, -2, -5, -5, -2, -2, -5, 12, 7, -2, 14, -3,
         -13, 2, -10, 1, 8, -4, 11, -2, -9, 14, -2, 1, 11, -3, 0, -2, -10, 7, 9, -4, -4, -2, 0, 8, -12, 1, 21, -9, -1,
         -1, -14, 8, 0, -4, 8, -11, 6, 15, -1, -6, -5, 13, -5, 4, 13, -15, 5, 10, -1, 2, -1, -1, 1, 7, 2, -5, 1, -16, 7,
         26, -8, 12, 16, -17, 5, 10, 5, -4, 1, 16, -3, 3, -7, 2, 10, -5, 4, 1, 2, 0, 1, 9, -5, 10, 5, 1, 10, 1],
        dtype="i2")
