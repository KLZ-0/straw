import numpy as np
from bitarray import bitarray
from bitarray.util import int2ba

from straw.io.base import BaseFormat


class FLACFormat(BaseFormat):
    def _format_specific_checks(self):
        if len(np.unique(self._data["channel"])) > 8:
            raise ValueError("The FLAC format only supports up to 8 channels")

        # TODO: check if everything necessary in StreamParams is not None

    def _stream(self):
        self._f.write("fLaC".encode("utf-8"))
        self._metadata_block()

    def _metadata_block(self):
        sec = self._metadata_block_data()
        sec = self._metadata_block_header(len(sec.tobytes())) + sec
        sec.tofile(self._f)

    def _metadata_block_header(self, data_size) -> bitarray:
        sec = bitarray()
        sec.append(1)
        sec += int2ba(0, length=7)
        sec += int2ba(data_size, length=24)
        return sec

    def _metadata_block_data(self) -> bitarray:
        sec = bitarray()
        sec += int2ba(self._params.min_block_size, length=16)
        sec += int2ba(self._params.max_block_size, length=16)
        sec += int2ba(self._params.min_frame_size, length=24)
        sec += int2ba(self._params.max_frame_size, length=24)
        sec += int2ba(self._params.sample_rate, length=20)
        sec += int2ba(self._params.channels - 1, length=3)
        sec += int2ba(self._params.bits_per_sample - 1, length=5)
        sec += int2ba(self._params.total_samples, length=36)
        sec.frombytes(self._params.md5)
        return sec
