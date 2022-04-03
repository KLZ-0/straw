import numpy as np
import pandas as pd
from bitarray import bitarray
from bitarray.util import int2ba

from straw.io.base import BaseWriter


class StrawFormatWriter(BaseWriter):
    def _format_specific_checks(self):
        if self._params.channels < 1:
            raise ValueError("No channels present!")

    def _stream(self):
        self._f.write("sTrW".encode("utf-8"))
        self._metadata_block()
        self._data.groupby("seq").apply(self._frame)

    def _metadata_block(self):
        sec = self._metadata_block_header()
        sec += self._metadata_block_data()
        sec.tofile(self._f)

    def _metadata_block_header(self) -> bitarray:
        sec = bitarray()
        sec.append(1)  # this block is the last metadata block before the audio blocks
        sec += int2ba(0, length=7)  # BLOCK_TYPE: STREAMINFO
        # ignore if streaminfo...
        # sec += int2ba(data_size, length=24)
        return sec

    def _metadata_block_data(self) -> bitarray:
        sec = bitarray()
        sec += int2ba(self._params.sample_rate, length=20)
        sec.frombytes(self.encode_int_utf8(self._params.channels - 1))
        sec += int2ba(self._params.bits_per_sample - 1, length=5)
        sec += int2ba(0, length=3)
        sec += int2ba(self._params.total_samples, length=36)
        sec.frombytes(self._params.md5)
        return sec

    def _frame(self, df: pd.DataFrame):
        sec = bitarray()
        qlp = df["qlp"][df["qlp"].first_valid_index()]
        tmp = df.apply(self._subframe, axis=1, order=len(qlp))
        for subframe_bitstream in tmp:
            sec += subframe_bitstream
        sec.fill()  # zero-padding to byte alignment

        # Header
        sec = self._frame_header(df, len(sec.tobytes()) + 2) + sec

        # Footer
        sec += int2ba(self.Crc.crc16(sec.tobytes()), length=16)
        sec.tofile(self._f)

    def _frame_header(self, df: pd.DataFrame, frame_data_size: int) -> bitarray:
        # TODO: in straw this should contain the size of the whole frame so we can partition and parallelize it
        sec = bitarray()
        sec += int2ba(0b10101010101010, length=14)  # sync code
        contains_lpc = df["qlp"].first_valid_index() is not None
        sec.append(contains_lpc)

        # block size start
        blocksize = int(df["frame"].apply(len).max())
        if blocksize == 0:
            raise ValueError("Blocksize cannot be 0")
        tmp = np.log2(blocksize)
        if tmp % 1 == 0.0:
            sec.append(0)  # get 8 bit exponent for (2^n) samples
            sec += int2ba(int(tmp), length=8)
        else:
            sec.append(1)  # get 16 bit (blocksize-1)
            sec += int2ba(blocksize - 1, length=16)
        # block size end

        # LPC
        if contains_lpc:
            qlp = df["qlp"][df["qlp"].first_valid_index()]
            qlp_precision = int(df["qlp_precision"][df["qlp_precision"].first_valid_index()])
            shift = int(df["shift"][df["shift"].first_valid_index()])

            sec += int2ba(len(qlp) - 1, length=5)
            sec += int2ba(qlp_precision - 1, length=4)
            sec += int2ba(shift, length=4)
            for coeff in qlp:
                sec += int2ba(int(coeff), length=qlp_precision, signed=True)
            sec.fill()

        sec.frombytes(self.encode_int_utf8(df.index.min()))
        sec += int2ba(frame_data_size + len(sec.tobytes()) + 4 + 1, length=32)
        sec += int2ba(self.Crc.crc8(sec.tobytes()), length=8)
        return sec

    def _subframe(self, df: pd.Series, order: int = 0) -> bitarray:
        sec = self._subframe_header(df)
        sec += self._subframe_data(df, order)
        return sec

    def _subframe_header(self, df: pd.Series) -> bitarray:
        sec = bitarray()
        sec += int2ba(df["frame_type"], length=2)
        return sec

    def _subframe_data(self, df: pd.Series, order: int = 0) -> bitarray:
        frame_type = df["frame_type"]
        if frame_type == 0b00:  # SUBFRAME_CONSTANT
            pass
        elif frame_type == 0b01:  # SUBFRAME_RAW
            pass
        elif frame_type == 0b11:  # SUBFRAME_LPC
            return self._subframe_lpc(df, order)
        else:
            raise ValueError(f"Invalid frame type: {frame_type}")

    def _subframe_lpc(self, df: pd.Series, order: int) -> bitarray:
        sec = bitarray()
        for warmup_sample in df["frame"][:order]:
            sec += int2ba(int(warmup_sample), length=self._params.bits_per_sample, signed=True)
        sec += self._residual(df)
        return sec

    def _residual(self, df: pd.Series) -> bitarray:
        sec = bitarray()
        sec += int2ba(int(df["bps"]), length=4)
        sec += df["stream"]
        return sec
