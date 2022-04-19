# distutils: language = c
# cython: language_level=3
import cython

###########
# Reading #
###########

@cython.cdivision(True)
def read_frame(cython.integral[:] target, const unsigned char[:] source, Py_ssize_t bits_per_sample, char little_endian = 0):
    cdef Py_ssize_t source_size, i, byteshift
    cdef Py_ssize_t bytes_per_sample = bits_per_sample / 8
    source_size = source.shape[0]

    for i in range(source_size):
        byteshift = i % bytes_per_sample
        if little_endian:
            byteshift = bytes_per_sample - byteshift - 1
        target[i / bytes_per_sample] |= source[i] << (8 * byteshift)

@cython.cdivision(True)
def write_frame(unsigned char[:] target, cython.integral[:] source, Py_ssize_t bits_per_sample, char little_endian = 0):
    cdef Py_ssize_t source_size, i, byte_i, byteshift
    cdef Py_ssize_t bytes_per_sample = bits_per_sample / 8
    source_size = source.shape[0]

    for i in range(source_size):
        for byte_i in range(bytes_per_sample):
            byteshift = byte_i
            if not little_endian:
                byteshift = bytes_per_sample - byteshift - 1

            target[bytes_per_sample*i + byte_i] = (source[i] >> (8 * byteshift)) & 0xff
