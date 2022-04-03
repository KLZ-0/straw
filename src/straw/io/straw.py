import numpy as np
import pandas as pd
from bitarray import bitarray
from bitarray.util import int2ba
from tqdm import tqdm

from straw.io.base import BaseWriter, BaseReader


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
        sec += int2ba(int(len(self._data[self._data["channel"] == 0])), length=27)
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
        subframe_type = df["frame_type"]
        if subframe_type == 0b00:  # SUBFRAME_CONSTANT
            raise NotImplementedError("Encoding SUBFRAME_CONSTANT not yet implemented")
        elif subframe_type == 0b01:  # SUBFRAME_RAW
            return self._subframe_raw(df)
        elif subframe_type == 0b11:  # SUBFRAME_LPC
            return self._subframe_lpc(df, order)
        else:
            raise ValueError(f"Invalid frame type: {subframe_type}")

    def _subframe_raw(self, df: pd.Series) -> bitarray:
        sec = bitarray()
        for sample in df["frame"]:
            sec += int2ba(int(sample), length=self._params.bits_per_sample, signed=True)
        return sec

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


class StrawFormatReader(BaseReader):
    def _stream(self):
        marker = self._f.read(4)
        if marker.decode("utf-8") != "sTrW":
            raise ValueError("Not a valid Straw file!")
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
        if block_type == 0:
            return 0
        else:
            data_len = self._sec.get_int(length=24)
            return data_len

    def _metadata_block_data(self) -> int:
        self._params.sample_rate = self._sec.get_int(length=20)
        self._params.channels = self._sec.get_int_utf8() + 1
        self._params.bits_per_sample = self._sec.get_int(length=5) + 1
        expected_frames = self._sec.get_int(length=27)
        self._params.total_samples = self._sec.get_int(length=36)
        self._params.md5 = self._sec.get_bytes(length=128)

        # Allocate sample buffer
        self._allocate_buffer(channels=self._params.channels,
                              bits_per_sample=self._params.bits_per_sample,
                              total_samples=self._params.total_samples)
        return expected_frames

    def _frame(self, expected_frames: int):
        start = self._sec.get_pos()
        common, frame_size, blocksize = self._frame_header()
        for i in range(self._params.channels):
            row = self._subframe(len(common["qlp"]) if "qlp" in common else 0, blocksize, i) | common
            row["idx"] = i * expected_frames + common["seq"]
            self._raw.append(row)
        self._samplebuffer_ptr += blocksize
        self._sec.skip_padding()

        # Footer
        expected_crc = self.Crc.crc16(self._sec.get_from(start).tobytes())
        checksum = self._sec.get_int(length=16)
        if expected_crc != checksum:
            raise RuntimeError(f"Inavalid frame checksum at frame {common['seq']}")

    def _frame_header(self) -> (dict, int, int):
        common = {}
        start = self._sec.get_pos()
        if self._sec.get_int(length=14) != 0b10101010101010:  # sync code
            raise RuntimeError("Lost sync")
        contains_lpc = self._sec.get_int()
        if self._sec.get_int():
            blocksize = self._sec.get_int(length=16) + 1
        else:
            blocksize = 1 << self._sec.get_int(length=8)

        if contains_lpc:
            order = self._sec.get_int(length=5) + 1
            common["qlp_precision"] = self._sec.get_int(length=4) + 1
            common["shift"] = self._sec.get_int(length=4)
            common["qlp"] = np.asarray(
                [self._sec.get_int(length=common["qlp_precision"], signed=True) for _ in range(order)], dtype=np.int32)
            self._sec.skip_padding()

        common["seq"] = self._sec.get_int_utf8()
        frame_size = self._sec.get_int(length=32)

        expected_crc = self.Crc.crc8(self._sec.get_from(start).tobytes())
        checksum = self._sec.get_int(length=8)
        if expected_crc != checksum:
            raise RuntimeError(f"Inavalid frame header checksum at frame {common['seq']}")
        return common, frame_size, blocksize

    def _subframe(self, order: int, blocksize: int, subframe_num: int) -> dict:
        subframe_type = self._subframe_header()
        return self._subframe_data(subframe_type, order, blocksize, subframe_num)

    def _subframe_header(self) -> int:
        return self._sec.get_int(length=2)

    def _subframe_data(self, subframe_type: int, order: int, blocksize: int, subframe_num: int) -> dict:
        if subframe_type == 0b00:  # SUBFRAME_CONSTANT
            raise NotImplementedError("Decoding SUBFRAME_CONSTANT not yet implemented")
        elif subframe_type == 0b01:  # SUBFRAME_RAW
            return self._subframe_raw(blocksize, subframe_num)
        elif subframe_type == 0b11:  # SUBFRAME_LPC
            return self._subframe_lpc(order, blocksize, subframe_num)
        else:
            raise ValueError(f"Invalid frame type: {subframe_type}")

    def _subframe_raw(self, blocksize: int, subframe_num: int) -> dict:
        row = {
            "channel": subframe_num,
            "frame": self._samplebuffer[subframe_num][self._samplebuffer_ptr:self._samplebuffer_ptr + blocksize]
        }
        for i in range(blocksize):
            row["frame"][i] = self._sec.get_int(length=self._params.bits_per_sample, signed=True)
        return row

    def _subframe_lpc(self, order: int, blocksize: int, subframe_num: int) -> dict:
        row = {
            "channel": subframe_num,
            "frame": self._samplebuffer[subframe_num][self._samplebuffer_ptr:self._samplebuffer_ptr + blocksize],
            "residual": self._samplebuffer[subframe_num][
                        self._samplebuffer_ptr + order:self._samplebuffer_ptr + blocksize]
        }
        for i in range(order):
            row["frame"][i] = self._sec.get_int(length=self._params.bits_per_sample, signed=True)
        row["residual"], row["bps"] = self._residual(array=row["residual"])
        return row

    def _residual(self, array: np.array) -> np.array:
        bps = self._sec.get_int(length=4)
        bits_read = self._ricer.bitstream_to_frame(
            self._sec[self._sec.get_pos():self._sec.get_pos() + len(array) * self._params.bits_per_sample],
            len(array), bps, own_frame=array)
        self._sec.advance(bits_read)
        return array, bps
