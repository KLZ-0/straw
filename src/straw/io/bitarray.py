from bitarray import bitarray
from bitarray.util import ba2int


class SlicedBitarray(bitarray):
    """
    Sliced version of bitarray
    Holds a pointer to the current position within the bitstream
    Querying a value from the bitstream moves the pointer forward unless specified otherwise
    """
    _current_ptr: int = 0

    def get_int(self, length: int = 1, signed=False) -> int:
        """
        Get an integer from the bitstream
        :param length: length of the integer in bits
        :param signed: whether the integer is in a signed format
        :return: integer
        """
        part = self[self._current_ptr:self._current_ptr + length]
        if len(part) == 0:
            raise ValueError("Unexpected EOF")
        self._current_ptr += length
        return ba2int(part, signed=signed)

    def get_bytes(self, length: int = 8) -> bytes:
        """
        Get bytes from the bitstream
        :param length: number of bits to read
        :return: bytes
        """
        part = self[self._current_ptr:self._current_ptr + length]
        if len(part) == 0:
            raise ValueError("Unexpected EOF")
        self._current_ptr += length
        return part.tobytes()

    def get_int_utf8(self) -> int:
        """
        Get UTF-8 encoded integer from the stream
        NOTE: this function reads until it finds a valid UTF-8 encoded integer value
        :return: integer
        """
        c = self.get_bytes()
        while True:
            try:
                return ord(c.decode("utf-8"))
            except UnicodeDecodeError:
                c += self.get_bytes()

    def get_from(self, start: int) -> bitarray:
        """
        Get a slice of this bitarray from a given position until the current one
        :param start: starting position
        :return: bitarray slice
        """
        return self[start:self._current_ptr]

    def get_pos(self) -> int:
        """
        Get the current position within the bitarray
        :return: bitarray position
        """
        return self._current_ptr

    def advance(self, length: int = 8):
        """
        Advance the bitstream by a given amount of bits
        :param length: number of bits to advance
        :return: None
        """
        self._current_ptr += length

    def skip_padding(self):
        """
        Skip padding until the bitarray is byte-aligned
        :return: None
        """
        if self._current_ptr % 8 != 0:
            self._current_ptr = ((self._current_ptr // 8) + 1) * 8

    def is_eof(self) -> bool:
        """
        Check whether the bitarray is at its end and has no bits to read
        :return: True is EOF, False otherwise
        """
        return len(self[self._current_ptr:self._current_ptr + 1]) == 0
