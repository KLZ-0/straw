# distutils: language = c
# cython: language_level=3
import cython

#########################
# Signedness correction #
#########################

cdef long _interleave(long x):
    """
    Implementation of the overlap and interleave scheme from https://en.wikipedia.org/wiki/Golomb_coding
    :param x: signed integer to be remaped
    :return: positive interleaved integer
    """
    if x == 0:
        return 0

    if x > 0:
        return 2 * x
    else:
        return -2 * x - 1

@cython.cdivision(True)
cdef long _deinterleave(long x):
    """
    Reverse of _interleave(short x)
    :param x: positive interleaved integer
    :return: original signed integer
    """
    if x == 0:
        return 0

    if x % 2 == 0:
        return x / 2
    else:
        return (x + 1) / -2

cdef void update_scale(long s, short m, short *scale):
    if s > m:
        scale[0] += 1
    elif s < m:
        scale[0] -= 1
    else:
        scale[0] = 0

############
# Encoding #
############

@cython.cdivision(True)
cdef void _push_bit(unsigned char[:] bits, Py_ssize_t *bit_i, char val):
    if val == 0:
        bit_i[0] += 1
        return

    cdef Py_ssize_t byte_i = bit_i[0] / 8
    cdef char real_bit_i = 7 - (bit_i[0] % 8)
    bit_i[0] += 1
    bits[byte_i] |= 1 << real_bit_i

@cython.cdivision(True)
def encode_frame(unsigned char[:] bits, cython.integral[:] frame, short k, short resp, short adaptive):
    """
    Encodes a whole residual frame and appends it to the end of the given bitstream
    :param bits: bitaray to which the bits will be appended
    :param frame: the frame to be encoded
    :param k: starting rice parameter
    :param resp: rice parameter responsiveness
    :param adaptive: if True do adaptive rice coding by varying the parameter
    :return: None
    """
    cdef short j
    cdef long m, q, s
    cdef Py_ssize_t x_max, i
    cdef Py_ssize_t bit_i = 0
    cdef Py_ssize_t bit_i_max = bits.shape[0] * 8
    x_max = frame.shape[0]
    m = 1 << k

    cdef short scale = 0

    for i in range(x_max):
        s = _interleave(frame[i])

        # Quotient code
        q = s / m

        if bit_i + q + k + 1 >= bit_i_max:
            return -1

        bit_i += q
        _push_bit(bits, &bit_i, 1)

        for j in range(k):
            _push_bit(bits, &bit_i, s >> (k - j - 1) & 1)

        # TODO: feed-forward rice implementation
        if not adaptive:
            continue

        # Update rice param
        if scale > resp:
            scale = 0
            k += 1
            m = 1 << k
            # print("e switched up:\t\t ", i, s, m)
            continue
        if scale < -resp and k > 0:
            scale = 0
            k -= 1
            m = 1 << k
            # print("e switched down:\t ", i, s, m)
            continue
        elif scale < -resp:
            scale = -resp

        update_scale(s, m, &scale)

    return bit_i

############
# Decoding #
############

@cython.cdivision(True)
cdef char _get_bit(const unsigned char[:] bits, Py_ssize_t *bit_i):
    cdef Py_ssize_t byte_i = bit_i[0] / 8
    cdef char real_bit_i = 7 - (bit_i[0] % 8)
    bit_i[0] += 1
    return (bits[byte_i] >> real_bit_i) & 1

@cython.cdivision(True)
def decode_frame(cython.integral[:] frame, const unsigned char[:] bits, short k, short resp, short adaptive, long starting_i = 0):
    """
    Decodes a whole residual frame from the given bitstream
    :param frame: numpy array where the decoded frame should be stored
    :param bits: bitaray from which the frame should be restored
    :param k: starting rice parameter
    :param resp: rice parameter responsiveness
    :param adaptive: if True do adaptive rice coding by varying the parameter
    :return: None
    """
    cdef short j
    cdef long m, q, s
    cdef Py_ssize_t x_max, i
    cdef Py_ssize_t bit_i = starting_i
    x_max = frame.shape[0]
    m = 1 << k

    cdef short scale = 0

    for i in range(x_max):
        q = 0
        while not _get_bit(bits, &bit_i):
            q += 1

        s = m * q

        for j in range(k):
            s |= _get_bit(bits, &bit_i) << (k - j - 1)

        frame[i] = _deinterleave(s)

        if not adaptive:
            continue

        if scale > resp:
            scale = 0
            k += 1
            m = 1 << k
            # print("dec switched up:\t ", i, s, m)
            continue
        if scale < -resp and k > 0:
            scale = 0
            k -= 1
            m = 1 << k
            # print("dec switched down:\t ", i, s, m)
            continue
        elif scale < -resp:
            scale = -resp

        update_scale(s, m, &scale)

    return bit_i - starting_i

###########
# Utility #
###########

@cython.cdivision(True)
def kparams(cython.integral[:] frame, short k, short resp):
    """
    Encodes a whole residual frame and appends it to the end of the given bitstream
    :param frame: the frame to be encoded
    :param k: starting rice parameter
    :param resp: rice parameter responsiveness
    :return: None
    """
    cdef long s
    cdef Py_ssize_t x_max, i
    x_max = frame.shape[0]
    m = 1 << k

    cdef short scale = 0

    for i in range(x_max):
        s = _interleave(frame[i])
        frame[i] = k

        if scale > resp:
            scale = 0
            k += 1
            m = 1 << k
            continue
        if scale < -resp and k > 0:
            scale = 0
            k -= 1
            m = 1 << k
            continue

        update_scale(s, m, &scale)

def interleave_frame(cython.integral[:] frame):
    cdef Py_ssize_t x_max, i
    x_max = frame.shape[0]
    for i in range(x_max):
        frame[i] = _interleave(frame[i])
