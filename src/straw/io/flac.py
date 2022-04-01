import numpy as np
import pandas as pd
from bitarray import bitarray
from bitarray.util import int2ba

from straw.io.base import BaseWriter, BaseReader


class FLACFormatWriter(BaseWriter):
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
        # Header
        sec = self._frame_header(df)

        # Process subframes
        # sec.tobytes().hex(" ")
        qlp = df["qlp"][df["qlp"].first_valid_index()]
        qlp_precision = int(df["qlp_precision"][df["qlp_precision"].first_valid_index()])
        shift = int(df["shift"][df["shift"].first_valid_index()])
        tmp = df.apply(self._subframe, axis=1, qlp=qlp, qlp_precision=qlp_precision, shift=shift)
        origsec = sec.copy()
        for subframe_bitstream in tmp:
            sec += subframe_bitstream
        sec.fill()  # zero-padding to byte alignment

        # Footer
        sec += int2ba(self.Crc.crc16(sec.tobytes()), length=16)
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
        sec += int2ba(self.Crc.crc8(sec.tobytes()), length=8)
        return sec

    def _subframe(self, df: pd.Series, qlp: np.array, qlp_precision: int, shift: int) -> bitarray:
        sec = self._subframe_header(df, qlp, qlp_precision, shift)
        sec += self._subframe_lpc(df, qlp, qlp_precision, shift)
        return sec

    def _subframe_header(self, df: pd.Series, qlp: np.array, qlp_precision: int, shift: int) -> bitarray:
        sec = bitarray()
        sec.append(0)  # zero bit padding, to prevent sync-fooling string of 1s
        sec.append(1)  # subframe type: 1xxxxx : SUBFRAME_LPC
        sec += int2ba(len(qlp) - 1, length=5)  # xxxxx=order-1
        sec.append(0)
        return sec

    def _subframe_lpc(self, df: pd.Series, qlp: np.array, qlp_precision: int, shift: int) -> bitarray:
        sec = bitarray()
        for warmup_sample in df["frame"][:len(qlp)]:
            sec += int2ba(int(warmup_sample), length=self._params.bits_per_sample, signed=True)
        sec += int2ba(qlp_precision - 1, length=4)
        sec += int2ba(shift, length=5, signed=True)  # NOTE: in our implementation this shuld not be signed
        for coeff in qlp:
            sec += int2ba(int(coeff), length=qlp_precision, signed=True)
        sec += self._residual(df)
        return sec

    def _residual(self, df: pd.Series) -> bitarray:
        sec = bitarray()
        sec += int2ba(0, length=2)  # partitioned Rice coding with 4-bit Rice parameter
        sec += int2ba(0, length=4)  # 2^0 partitions
        sec += int2ba(int(df["bps"]), length=4)
        sec += df["stream"]
        return sec


class FLACFormatReader(BaseReader):
    def _stream(self):
        marker = self._f.read(4)
        if marker.decode("utf-8") != "fLaC":
            raise ValueError("Not a valid FLAC file!")
        # self._metadata_block()
        # self._data.groupby("seq").apply(self._frame)
