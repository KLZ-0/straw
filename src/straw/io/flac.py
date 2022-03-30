import numpy as np
import pandas as pd
from bitarray import bitarray
from bitarray.util import int2ba
from crc8 import crc8

from straw.io.base import BaseFormat


class FLACFormat(BaseFormat):
    def _format_specific_checks(self):
        if len(np.unique(self._data["channel"])) > 8:
            raise ValueError("The FLAC format only supports up to 8 channels")

        # TODO: check if everything necessary in StreamParams is not None

    def _stream(self):
        self._f.write("fLaC".encode("utf-8"))
        self._metadata_block()
        self._data.groupby("seq").apply(self._frame)

    def _metadata_block(self):
        sec = self._metadata_block_data()
        sec = self._metadata_block_header(len(sec.tobytes())) + sec
        sec.tofile(self._f)

    def _metadata_block_header(self, data_size) -> bitarray:
        sec = bitarray()
        sec.append(1)  # this block is the last metadata block before the audio blocks
        sec += int2ba(0, length=7)  # BLOCK_TYPE: STREAMINFO
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

    def _frame(self, df: pd.DataFrame):
        sec = self._frame_header(df)
        sec.tofile(self._f)

    def _frame_header(self, df: pd.DataFrame) -> bitarray:
        sec = bitarray()
        sec += int2ba(0b11111111111110, length=14)  # sync code
        sec.append(0)  # mandatory value
        sec.append(0)  # fixed-blocksize stream; frame header encodes the frame number

        # block size start
        blocksize = int(df["frame"].apply(len).max())
        if blocksize == 0:
            raise ValueError("Blocksize cannot be 0")
        tmp = np.log2(blocksize)
        if tmp % 1 == 0.0:
            sec += int2ba(int(tmp), length=4)
            tmp = 0
        else:
            if blocksize < 1 << 8:
                tmp = 8
                sec += int2ba(0b0110, length=4)
            else:
                tmp = 16
                sec += int2ba(0b0111, length=4)
        # block size end

        sec += int2ba(0, length=4)  # sample rate: get from STREAMINFO metadata block
        sec += int2ba(self._params.channels - 1, length=4)  # (number of independent channels)-1
        sec += int2ba(0, length=3)  # sample size in bits: get from STREAMINFO metadata block
        sec.append(0)  # mandatory value
        sec.frombytes(chr(df.index.min()).encode("utf-8"))
        if tmp:
            sec += int2ba(blocksize - 1, length=tmp)
        sec.frombytes(crc8(sec.tobytes()).digest())
        return sec
