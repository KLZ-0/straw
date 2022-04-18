import numpy as np
import pandas as pd
from bitarray import bitarray
from bitarray.util import int2ba
from tqdm import tqdm

from straw.io.base import BaseWriter, BaseReader
from straw.io.sizes import StrawSizes


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
        sec += self._metadata_block_streaminfo()
        sec.tofile(self._f)

    def _metadata_block_header(self) -> bitarray:
        sizes = StrawSizes.metadata_block_header
        sec = bitarray()
        sec.append(1)  # this block is the last metadata block before the audio blocks
        sec += int2ba(0, length=sizes.type)  # BLOCK_TYPE: STREAMINFO
        # ignore if streaminfo...
        # sec += int2ba(data_size, length=24)
        return sec

    def _metadata_block_streaminfo(self) -> bitarray:
        sizes = StrawSizes.metadata_block_streaminfo
        sec = bitarray()
        sec += int2ba(self._params.sample_rate, length=sizes.samplerate)
        sec.frombytes(self.encode_int_utf8(self._params.channels - 1))
        sec += int2ba(self._params.bits_per_sample - 1, length=sizes.bps)
        sec += int2ba(int(len(self._data[self._data["channel"] == 0])), length=sizes.frames)
        sec += int2ba(self._params.total_samples, length=sizes.samples)
        sec.frombytes(self._params.md5)

        # Shift
        if self._params.lags.any():
            sec.append(1)
            sec.frombytes(self.encode_int_utf8(self._params.leading_channel))
            for val in self._params.lags:
                sec += int2ba(int(val), length=sizes.shift)
            for c in range(self._params.channels):
                for val in self._params.removed_samples_start[c]:
                    sec += int2ba(int(val), length=self._params.bits_per_sample, signed=True)
                for val in self._params.removed_samples_end[c]:
                    sec += int2ba(int(val), length=self._params.bits_per_sample, signed=True)
        else:
            sec.append(0)

        # Bias
        if self._params.bias.any():
            sec.append(1)
            for val in self._params.bias:
                sec += int2ba(int(val), length=sizes.bias, signed=True)
        else:
            sec.append(0)

        # Gain
        if self._params.gain.any():
            sec.append(1)
            for val in self._params.gain:
                sec += int2ba(int(val), length=sizes.gain)
            sec += int2ba(self._params.gain_shift, length=sizes.gain_shift)
        else:
            sec.append(0)

        sec.fill()
        return sec

    def _frame(self, df: pd.DataFrame):
        footer_sizes = StrawSizes.frame_footer
        sec = bitarray()
        qlp_idx = df["qlp"].first_valid_index()
        qlp = None if qlp_idx is None else df["qlp"][qlp_idx]
        tmp = df.apply(self._subframe, axis=1, qlp=qlp)
        for subframe_bitstream in tmp:
            sec += subframe_bitstream
        sec.fill()  # zero-padding to byte alignment

        # Header
        sec = self._frame_header(df, len(sec.tobytes()) + footer_sizes.crc // 8) + sec

        # Footer
        sec += int2ba(self.Crc.crc16(sec.tobytes()), length=footer_sizes.crc)
        sec.tofile(self._f)

    def _frame_header(self, df: pd.DataFrame, frame_data_size: int) -> bitarray:
        sizes = StrawSizes.frame_header
        sec = bitarray()
        sec += int2ba(0b10101010101010, length=sizes.sync_code)  # sync code
        sec.append(0)

        # block size start
        blocksize = int(df["frame"].apply(len).max())
        if blocksize == 0:
            raise ValueError("Blocksize cannot be 0")
        tmp = np.log2(blocksize)
        if tmp % 1 == 0.0:
            sec.append(0)  # get 8 bit exponent for (2^n) samples
            sec += int2ba(int(tmp), length=sizes.block_size_log2)
        else:
            sec.append(1)  # get 16 bit (blocksize-1)
            sec += int2ba(blocksize - 1, length=sizes.block_size_exact)
        # block size end

        sec.frombytes(self.encode_int_utf8(df.index.min()))
        sec += int2ba(frame_data_size + len(sec.tobytes()) + sizes.frame_bytes // 8 + sizes.crc // 8,
                      length=sizes.frame_bytes)
        sec += int2ba(self.Crc.crc8(sec.tobytes()), length=sizes.crc)
        return sec

    def _subframe(self, df: pd.Series, qlp: np.array) -> bitarray:
        sec = self._subframe_header(df)
        sec += self._subframe_data(df, qlp)
        return sec

    def _subframe_header(self, df: pd.Series) -> bitarray:
        sizes = StrawSizes.subframe_header
        sec = bitarray()
        sec += int2ba(df["frame_type"], length=sizes.type)
        return sec

    def _subframe_data(self, df: pd.Series, qlp: np.array) -> bitarray:
        subframe_type = df["frame_type"]
        if subframe_type == 0b00:  # SUBFRAME_CONSTANT
            return self._subframe_constant(df)
        elif subframe_type == 0b01:  # SUBFRAME_RAW
            return self._subframe_raw(df)
        elif subframe_type == 0b11:  # SUBFRAME_LPC
            return self._subframe_lpc(df, len(qlp))
        else:
            raise ValueError(f"Invalid frame type: {subframe_type}")

    def _subframe_constant(self, df: pd.Series) -> bitarray:
        sec = bitarray()
        sec += int2ba(int(df["frame"][0]), length=self._params.bits_per_sample, signed=True)
        return sec

    def _subframe_raw(self, df: pd.Series) -> bitarray:
        sec = bitarray()
        for sample in df["frame"]:
            sec += int2ba(int(sample), length=self._params.bits_per_sample, signed=True)
        return sec

    def _subframe_lpc(self, df: pd.Series, order: int) -> bitarray:
        sizes = StrawSizes.subframe_lpc
        sec = bitarray()
        contains_lpc = isinstance(df["qlp"], np.ndarray)
        sec.append(contains_lpc)

        # LPC
        if contains_lpc:
            qlp = df["qlp"]
            order = len(qlp)
            qlp_precision = int(df["qlp_precision"])
            shift = int(df["shift"])

            sec += int2ba(order - 1, length=sizes.lpc_order)
            sec += int2ba(qlp_precision - 1, length=sizes.lpc_prec)
            sec += int2ba(shift, length=sizes.lpc_shift)
            for coeff in qlp:
                sec += int2ba(int(coeff), length=qlp_precision, signed=True)
        else:
            sec.append(df["was_coded"])

        for warmup_sample in df["frame"][:order]:
            sec += int2ba(int(warmup_sample), length=self._params.bits_per_sample, signed=True)
        sec += self._residual(df)
        return sec

    def _residual(self, df: pd.Series) -> bitarray:
        sizes = StrawSizes.residual
        sec = bitarray()
        sec += int2ba(int(df["bps"]), length=sizes.param)
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
        return self._metadata_block_streaminfo()

    def _metadata_block_header(self) -> int:
        sizes = StrawSizes.metadata_block_header
        last_metadata_block = self._sec.get_int()  # this block is the last metadata block before the audio blocks
        block_type = self._sec.get_int(length=sizes.type)  # BLOCK_TYPE: STREAMINFO
        if block_type == 0:
            return 0
        else:
            data_len = self._sec.get_int(length=sizes.size)
            return data_len

    def _metadata_block_streaminfo(self) -> int:
        sizes = StrawSizes.metadata_block_streaminfo
        self._params.sample_rate = self._sec.get_int(length=sizes.samplerate)
        self._params.channels = self._sec.get_int_utf8() + 1
        self._params.alloc_arrays()
        self._params.bits_per_sample = self._sec.get_int(length=sizes.bps) + 1
        expected_frames = self._sec.get_int(length=sizes.frames)
        self._params.total_samples = self._sec.get_int(length=sizes.samples)
        self._params.md5 = self._sec.get_bytes(length=sizes.md5)

        # Allocate sample buffer
        self._allocate_buffer()

        # Shift
        has_shift = self._sec.get_int()
        if has_shift:
            self._params.leading_channel = self._sec.get_int_utf8()
            for i in range(self._params.lags.shape[0]):
                self._params.lags[i] = self._sec.get_int(length=sizes.shift)
            total_size = self._samplebuffer.shape[1] - np.max(self._params.lags)
            for c in range(self._params.channels):
                lag = self._params.lags[c]
                for s in range(lag):
                    self._samplebuffer[c][s] = self._sec.get_int(length=self._params.bits_per_sample, signed=True)
                for e in range(total_size + lag, self._samplebuffer.shape[1]):
                    self._samplebuffer[c][e] = self._sec.get_int(length=self._params.bits_per_sample, signed=True)

        # Bias
        has_bias = self._sec.get_int()
        if has_bias:
            for i in range(self._params.bias.shape[0]):
                self._params.bias[i] = self._sec.get_int(length=sizes.bias, signed=True)

        # Gain
        has_gain = self._sec.get_int()
        if has_gain:
            for i in range(self._params.gain.shape[0]):
                self._params.gain[i] = self._sec.get_int(length=sizes.gain)
            self._params.gain_shift = self._sec.get_int(length=sizes.gain_shift)

        self._sec.skip_padding()
        return expected_frames

    def _frame(self, expected_frames: int):
        start = self._sec.get_pos()
        seq, frame_size, blocksize = self._frame_header()
        last_order = 0
        for i in range(self._params.channels):
            row = self._subframe(last_order, blocksize, i)
            row["seq"] = seq
            row["idx"] = i * expected_frames + seq
            # we expect that the first channel will have lpc coefficients in every case
            if isinstance(row["qlp"], np.ndarray):
                last_order = len(row["qlp"])
            self._raw.append(row)
        self._samplebuffer_ptr += blocksize
        self._sec.skip_padding()

        # Footer
        footer_sizes = StrawSizes.frame_footer
        expected_crc = self.Crc.crc16(self._sec.get_from(start).tobytes())
        checksum = self._sec.get_int(length=footer_sizes.crc)
        if expected_crc != checksum:
            raise RuntimeError(f"Inavalid frame checksum at frame {seq}")

    def _frame_header(self) -> (dict, int, int):
        sizes = StrawSizes.frame_header
        start = self._sec.get_pos()
        if self._sec.get_int(length=sizes.sync_code) != 0b10101010101010:  # sync code
            raise RuntimeError("Lost sync")
        self._sec.get_int()
        if self._sec.get_int():
            blocksize = self._sec.get_int(length=sizes.block_size_exact) + 1
        else:
            blocksize = 1 << self._sec.get_int(length=sizes.block_size_log2)

        seq = self._sec.get_int_utf8()
        frame_size = self._sec.get_int(length=sizes.frame_bytes)

        expected_crc = self.Crc.crc8(self._sec.get_from(start).tobytes())
        checksum = self._sec.get_int(length=sizes.crc)
        if expected_crc != checksum:
            raise RuntimeError(f"Inavalid frame header checksum at frame {seq}")
        return seq, frame_size, blocksize

    def _subframe(self, order: int, blocksize: int, subframe_num: int) -> dict:
        subframe_type = self._subframe_header()
        return self._subframe_data(subframe_type, order, blocksize, subframe_num)

    def _subframe_header(self) -> int:
        sizes = StrawSizes.subframe_header
        return self._sec.get_int(length=sizes.type)

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
            "frame": self._samplebuffer[subframe_num][
                     self._samplebuffer_ptr + self._params.lags[subframe_num]:self._samplebuffer_ptr +
                                                                              self._params.lags[
                                                                                  subframe_num] + blocksize]
        }
        for i in range(blocksize):
            row["frame"][i] = self._sec.get_int(length=self._params.bits_per_sample, signed=True)
        return row

    def _subframe_lpc(self, order: int, blocksize: int, subframe_num: int) -> dict:
        sizes = StrawSizes.subframe_lpc
        row = {
            "channel": subframe_num,
            "frame": self._samplebuffer[subframe_num][
                     self._samplebuffer_ptr + self._params.lags[subframe_num]:self._samplebuffer_ptr +
                                                                              self._params.lags[
                                                                                  subframe_num] + blocksize],
            "qlp_precision": 0,
            "shift": 0,
            "qlp": np.nan,
            "was_coded": 0,
        }
        contains_lpc = self._sec.get_int()
        if contains_lpc:
            order = self._sec.get_int(length=sizes.lpc_order) + 1
            row["qlp_precision"] = self._sec.get_int(length=sizes.lpc_prec) + 1
            row["shift"] = self._sec.get_int(length=sizes.lpc_shift)
            row["qlp"] = np.asarray(
                [self._sec.get_int(length=row["qlp_precision"], signed=True) for _ in range(order)], dtype=np.int32)
        else:
            row["was_coded"] = self._sec.get_int()

        for i in range(order):
            row["frame"][i] = self._sec.get_int(length=self._params.bits_per_sample, signed=True)

        # get the residual
        row["residual"] = self._samplebuffer[subframe_num][
                          self._samplebuffer_ptr + self._params.lags[subframe_num] + order:self._samplebuffer_ptr +
                                                                                           self._params.lags[
                                                                                               subframe_num] + blocksize]
        row["residual"], row["bps"] = self._residual(array=row["residual"])
        return row

    def _residual(self, array: np.array) -> np.array:
        sizes = StrawSizes.residual
        bps = self._sec.get_int(length=sizes.param)
        bits_read = self._ricer.bitstream_to_frame(
            self._sec[self._sec.get_pos():self._sec.get_pos() + len(array) * self._params.bits_per_sample],
            len(array), bps, own_frame=array)
        self._sec.advance(bits_read)
        return array, bps
