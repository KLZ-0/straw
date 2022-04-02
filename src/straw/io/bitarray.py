from bitarray import bitarray
from bitarray.util import ba2int


class SlicedBitarray(bitarray):
    _current_ptr: int = 0

    def get_int(self, length: int = 1, signed=False) -> int:
        part = self[self._current_ptr:self._current_ptr + length]
        if len(part) == 0:
            raise ValueError("Unexpected EOF")
        self._current_ptr += length
        return ba2int(part, signed=signed)

    def get_bytes(self, length: int = 8) -> bytes:
        part = self[self._current_ptr:self._current_ptr + length]
        if len(part) == 0:
            raise ValueError("Unexpected EOF")
        self._current_ptr += length
        return part.tobytes()

    def get_int_utf8(self) -> int:
        c = self.get_bytes()
        while True:
            try:
                return ord(c.decode("utf-8"))
            except UnicodeDecodeError:
                c += self.get_bytes()

    def get_from(self, start: int) -> bitarray:
        return self[start:self._current_ptr]

    def get_pos(self) -> int:
        return self._current_ptr

    def advance(self, length: int = 8):
        self._current_ptr += length

    def skip_padding(self):
        if self._current_ptr % 8 != 0:
            self._current_ptr = ((self._current_ptr // 8) + 1) * 8

    def is_eof(self) -> bool:
        return len(self[self._current_ptr:self._current_ptr + 1]) == 0
