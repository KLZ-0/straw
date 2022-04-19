import numpy as np
import pandas as pd
from bitarray import bitarray
from bitarray.util import int2ba
from tqdm import tqdm

from straw.io.base import BaseWriter, BaseReader
from straw.io.params import FLACStreamParams
from straw.rice import Ricer


class FLACFormatWriter(BaseWriter):
    _params: FLACStreamParams

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
        for subframe_bitstream in tmp:
            sec += subframe_bitstream
        sec.fill()  # zero-padding to byte alignment

        # Footer
        sec += int2ba(self.Crc.crc16(sec.tobytes()), length=16)
        sec.tofile(self._f)

    def _frame_header(self, df: pd.DataFrame) -> bitarray:
        # TODO: in straw this should contain the size of the whole frame so we can partition and parallelize it
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
        sec.append(0)  # no wasted bits-per-sample in source subblock, k=0
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
    # TODO: make an enum of sizes to not use magic numbers
    _ricer = Ricer(adaptive=False)

    def _stream(self):
        marker = self._f.read(4)
        if marker.decode("utf-8") != "fLaC":
            raise ValueError("Not a valid FLAC file!")
        self._sec.fromfile(self._f)
        expected_frames = self._metadata_block()
        pbar = tqdm(range(expected_frames))
        pbar.set_description(f"Loading frames")
        for i in pbar:
            if self._sec.is_eof():
                pbar.close()
                break

            self._frame(expected_frames)

    def _metadata_block(self) -> int:
        data_len = self._metadata_block_header()
        return self._metadata_block_data()

    def _metadata_block_header(self) -> int:
        last_metadata_block = self._sec.get_int()  # this block is the last metadata block before the audio blocks
        block_type = self._sec.get_int(length=7)  # BLOCK_TYPE: STREAMINFO
        data_len = self._sec.get_int(length=24)
        return data_len

    def _metadata_block_data(self) -> int:
        self._params.min_block_size = self._sec.get_int(length=16)
        self._params.max_block_size = self._sec.get_int(length=16)
        self._params.min_frame_size = self._sec.get_int(length=24)
        self._params.max_frame_size = self._sec.get_int(length=24)
        self._params.sample_rate = self._sec.get_int(length=20)
        self._params.channels = self._sec.get_int(length=3) + 1
        self._params.bits_per_sample = self._sec.get_int(length=5) + 1
        self._params.total_samples = self._sec.get_int(length=36)
        self._params.md5 = self._sec.get_bytes(length=128)

        # Allocate sample buffer
        self._allocate_buffer()
        return self._params.total_samples // self._params.max_block_size + 1

    def _frame(self, expected_frames: int):
        start = self._sec.get_pos()
        frame_num, blocksize = self._frame_header()
        for i in range(self._params.channels):
            row = self._subframe(blocksize, i)
            row["seq"] = frame_num
            row["idx"] = i * expected_frames + frame_num
            self._raw.append(row)
        self._samplebuffer_ptr += blocksize
        self._sec.skip_padding()

        # Footer
        expected_crc = self.Crc.crc16(self._sec.get_from(start).tobytes())
        checksum = self._sec.get_int(length=16)
        if expected_crc != checksum:
            raise RuntimeError(f"Inavalid frame checksum at frame {frame_num}")
        return frame_num

    def _frame_header(self) -> (int, int):
        start = self._sec.get_pos()
        if self._sec.get_int(length=14) != 0b11111111111110:  # sync code
            raise RuntimeError("Lost sync")
        self._sec.get_int()  # mandatory value
        self._sec.get_int()  # fixed-blocksize stream; frame header encodes the frame number
        tmp = self._sec.get_int(length=4)
        blocksize = 0
        if tmp == 0b0110:
            tmp = 8
        elif tmp == 0b0111:
            tmp = 16
        else:
            blocksize = 1 << tmp
            tmp = 0
        self._sec.get_int(length=4)  # sample rate: get from STREAMINFO metadata block
        self._sec.get_int(length=4)  # (number of independent channels)-1
        self._sec.get_int(length=3)  # sample size in bits: get from STREAMINFO metadata block
        self._sec.get_int()  # mandatory value
        frame_num = self._sec.get_int_utf8()
        if tmp:
            blocksize = self._sec.get_int(length=tmp) + 1
        expected_crc = self.Crc.crc8(self._sec.get_from(start).tobytes())
        checksum = self._sec.get_int(length=8)
        if expected_crc != checksum:
            raise RuntimeError(f"Inavalid frame header checksum at frame {frame_num}")
        return frame_num, blocksize

    def _subframe(self, blocksize: int, subframe_num: int) -> dict:
        order = self._subframe_header()
        return self._subframe_lpc(order, blocksize, subframe_num)

    def _subframe_header(self) -> int:
        self._sec.get_int()  # zero bit padding, to prevent sync-fooling string of 1s
        self._sec.get_int()  # subframe type: 1xxxxx : SUBFRAME_LPC
        order = self._sec.get_int(length=5) + 1  # xxxxx=order-1
        self._sec.get_int()  # no wasted bits-per-sample in source subblock, k=0
        return order

    def _subframe_lpc(self, order: int, blocksize: int, subframe_num: int) -> dict:
        row = {
            "channel": subframe_num,
            "frame": self._samplebuffer[subframe_num][self._samplebuffer_ptr:self._samplebuffer_ptr + blocksize],
            "residual": self._samplebuffer[subframe_num][
                        self._samplebuffer_ptr + order:self._samplebuffer_ptr + blocksize]
        }
        for i in range(order):
            row["frame"][i] = self._sec.get_int(length=self._params.bits_per_sample, signed=True)
        row["qlp_precision"] = self._sec.get_int(length=4) + 1
        row["shift"] = self._sec.get_int(length=5, signed=True)  # NOTE: in our implementation this shuld not be signed

        row["qlp"] = np.asarray([self._sec.get_int(length=row["qlp_precision"], signed=True) for _ in range(order)],
                                dtype=np.int32)
        row["residual"], row["bps"] = self._residual(blocksize - order, array=row["residual"])
        return row

    def _residual(self, samples: int, array: np.array) -> np.array:
        self._sec.get_int(length=2)  # partitioned Rice coding with 4-bit Rice parameter
        self._sec.get_int(length=4)  # 2^0 partitions
        bps = self._sec.get_int(length=4)
        bits_read = self._ricer.bitstream_to_frame(
            self._sec[self._sec.get_pos():self._sec.get_pos() + self._params.max_frame_size * 8],
            samples, bps, own_frame=array)
        self._sec.advance(bits_read)
        return array, bps
