import sys
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import soundfile

from straw import lpc, static, correctors
from straw.codec.base import BaseCoder
from straw.compute import ParallelCompute
from straw.io import Formatter
from straw.io.params import StreamParams
from straw.rice import Ricer
from straw.static import SubframeType, Default
from straw.util import Signals


class EncoderStats:
    frames: int = 0
    file_size: int = 0
    ratio: int = 0


class Encoder(BaseCoder):
    _lpc_order = 20  # can be sourced from len(df["qlp"]) once per group
    _lpc_precision = 12  # bits, stored in df["qlp_precision"] once per group
    _params = StreamParams()

    ##########
    # Public #
    ##########

    def __init__(self,
                 flac_mode=False,
                 do_corrections=("shift", "bias"),
                 dynamic_blocksize=False,
                 min_block_size=Default.min_frame_size,
                 max_block_size=Default.max_frame_size,
                 framing_treshold=Default.framing_treshold,
                 framing_resolution=Default.framing_resolution,
                 responsiveness=Default.rice_responsiveness,
                 parallelize=True,
                 show_progress: bool = False):
        """
        Encoder constructor
        :param do_corrections: tuple of correction to be performed
        :param dynamic_blocksize: whether to perform dynamic block slicing
        :param min_block_size: minimal block size
        :param max_block_size: maximal block size
        :param framing_treshold: framing treshold
        :param framing_resolution: framing resolution
        :param responsiveness: Rice coding responsiveness
        :param parallelize: if True use parallelization while encoding
        """
        super(Encoder, self).__init__(flac_mode, show_progress=show_progress)
        self._ricer = Ricer(adaptive=True if not flac_mode else False, responsiveness=responsiveness)
        self._params.responsiveness = responsiveness
        self._do_corrections = do_corrections
        self._do_dynamic_blocking = dynamic_blocksize
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.framing_treshold = framing_treshold
        self.framing_resolution = framing_resolution
        self.parallelize = parallelize

    def set_rice_responsiveness(self, responsiveness):
        """
        Set a new Rice coding responsiveness
        :param responsiveness: new Rice responsiveness
        :return: None
        """
        self._ricer.responsiveness = responsiveness
        self._params.responsiveness = responsiveness

    def load_file(self, file):
        """
        Load the specified file into memory
        :param file: str or int or file-like object - anything that soundfile accepts
        :return: None
        """
        self._source_size = Path(file).stat().st_size
        with soundfile.SoundFile(file, "r") as wav:
            subtype = wav.subtype
            if subtype not in self._supported_subtypes:
                raise ValueError(f"Subtype '{subtype}' not supported, must be one of {self._supported_subtypes.keys()}")
            bits_per_sample = self._supported_subtypes[subtype]
            dtype_bits = static.soundfile_dtype[bits_per_sample]
            data = wav.read(dtype=f"int{dtype_bits}", always_2d=True)
            data >>= dtype_bits - bits_per_sample
            sr = wav.samplerate

        self.load_data(data, sr, bits_per_sample)

    def load_data(self, data: np.array, samplerate: int, bits_per_sample: int):
        """
        Load the given data into the internal DataFrame
        :param data: data to be loaded
        :param samplerate: samplerate of the given data
        :param bits_per_sample: bits per sample of the original data (can be lower than the bit width of data.dtype)
        :return: None
        """
        if len(data.shape) == 1:
            self._samplebuffer = data.reshape((1, -1))
        elif data.shape[0] > data.shape[1]:
            self._samplebuffer = data.swapaxes(1, 0)
        else:
            self._samplebuffer = data

        self._params.channels = int(self._samplebuffer.shape[0])
        self._params.total_samples = int(self._samplebuffer.shape[1])
        self._params.bits_per_sample = bits_per_sample
        self._params.alloc_arrays()

        # If file size not available, use raw sample size
        if not hasattr(self, "_source_size"):
            self._source_size = self._params.bits_per_sample * self._params.total_samples * self._params.channels

        self._params.sample_rate = samplerate
        self._params.md5 = self.get_md5()
        correctors.apply_corrections(self._samplebuffer, self._do_corrections, self._params)
        self._create_dataframe()

    def encode(self):
        """
        Encode the data in the internal DataFrame
        :return: None
        """
        # Initialize frame types
        self._init_frame_types()

        groups = self._data.groupby("seq")
        if self.parallelize and self._params.channels > 1:
            self._data = ParallelCompute.get_instance().map_group(groups, self._encode_frame)
        else:
            self._data = groups.apply(self._encode_frame)

    def _encode_frame(self, data_slice: pd.DataFrame):
        """
        Encode one frame - group of subframes
        :param data_slice: slice of a DataFrame containing a group of subframes
        :return: modified data_slice
        """
        # correctors.ShiftCorrector().df_wrap_apply(data_slice["frame"])
        lpc.compute_qlp(data_slice, order=self._lpc_order, qlp_coeff_precision=self._lpc_precision)
        lpc.compute_residual(data_slice)
        # correctors.GainCorrector().df_wrap_apply(data_slice["residual"])
        # correctors.ShiftCorrector().df_wrap_apply(data_slice["residual"])
        # correctors.BiasCorrector().df_wrap_apply(data_slice["residual"])
        data_slice = correctors.Decorrelator().midside_decorrelate(data_slice, "residual")
        data_slice["bps"] = data_slice["residual"].apply(self._ricer.guess_parameter)
        data_slice["stream"] = self._ricer.frames_to_bitstreams(data_slice, parallel=False)
        data_slice["stream_len"] = data_slice["stream"].apply(len)
        data_slice["frame_type"] = self._should_be_raw_maxbytes(data_slice)
        return data_slice

    def save_file(self, output_file):
        """
        Save the encoded signal
        :param output_file: target file
        :return: None
        """
        self._params.total_frames = int(len(self._data[self._data["channel"] == 0]))
        opened = False
        if not hasattr(output_file, "write"):
            output_file = open(output_file, "wb")
            opened = True

        Formatter().save(self._data, self._params, output_file, self._flac_mode)
        if opened:
            output_file.close()

    ###########
    # Private #
    ###########

    def _create_dataframe(self):
        """
        Create a dataframe from the raw signal, this includes slicing the signal into specific views
        NOTE: the underlying memory stays as a contiguous memory chunk
        :return: None
        """

        total_size = self._samplebuffer.shape[1] - np.max(self._params.lags)
        if self._do_dynamic_blocking:
            lag = self._params.lags[0]
            limits = Signals.get_frame_limits_by_energy(self._samplebuffer[0][lag:total_size + lag],
                                                        min_block_size=self.min_block_size,
                                                        max_block_size=self.max_block_size,
                                                        treshold=self.framing_treshold,
                                                        resolution=self.framing_resolution)
        else:
            limits = None

        ds = {"seq": [], "frame": [], "channel": []}
        for channel, channel_data in enumerate(self._samplebuffer):
            lag = self._params.lags[channel]
            sliced = self._slice_channel_data_into_frames(channel_data[lag:total_size + lag], limits=limits)
            ds["seq"] += [i for i in range(len(sliced))]
            ds["frame"] += sliced
            ds["channel"] += [channel for _ in range(len(sliced))]

        self._data = pd.DataFrame(ds, copy=False)

    def _init_frame_types(self):
        # all frames are LPC frames by default
        self._data["frame_type"] = np.full(len(self._data["frame"]), SubframeType.LPC, dtype="B")

    def _should_be_raw_maxbytes(self, df: pd.DataFrame):
        if not (df["frame_type"].isin((SubframeType.LPC, SubframeType.LPC_COMMON))).all():
            return df["frame_type"]

        max_allowed_bits = df.loc[df.index[0], "residual"].shape[0] * self._params.bits_per_sample
        if (df["stream_len"] >= max_allowed_bits).any():
            df["frame_type"] = SubframeType.RAW
        return df["frame_type"]

    @staticmethod
    def _decorrelate_signals(data_slice, col_name="residual"):
        # self._data = self._data.groupby("seq").apply(Decorrelator().localized_decorrelate, col_name=col_name)
        # self._data = ParallelCompute.get_instance().map_group(self._data.groupby("seq"),
        #                                                       Decorrelator().midside_decorrelate, col_name=col_name)
        data_slice = data_slice.groupby("seq").apply(correctors.Decorrelator().midside_decorrelate, col_name=col_name)
        data_slice["was_coded"] = 0

    ###########
    # Utility #
    ###########

    def get_stats(self, output_file: Path) -> EncoderStats:
        """
        Return the stats of the encoding
        :param output_file: the output file for comparison
        :return:
        """
        stats = EncoderStats()
        stats.file_size = output_file.stat().st_size
        stats.ratio = output_file.stat().st_size / self._source_size
        stats.frames = len(self._data.groupby('seq').groups)
        return stats

    def print_stats(self, output_file: Path, stream: TextIO = sys.stdout):
        """
        Verbose output - print a bunch of stuff...
        :param output_file: output file (only the size is needed)
        :param stream: stream where the output should be written
        :return: None
        """
        print(f"Number of frames: {len(self._data.groupby('seq').groups)}", file=stream)
        print(f"Number of subframes: {len(self._data)}", file=stream)
        print(f"Source size: {self._source_size} ({self._source_size / (2 ** 20):.2f} MiB)", file=stream)
        size = self._data["stream_len"].sum()
        print(f"md5: {self._params.md5.hex(' ')}", file=stream)
        print(f"Length of residual bitstream: {size} bits, "
              f"bytes: {np.ceil(size / 8):.0f} aligned ({np.ceil(size / 8) / (2 ** 20):.2f} MiB)", file=stream)
        lpc_bytes = np.ceil(len(self._data) * self._lpc_precision * self._lpc_order * 1 / 8)
        print(f"Bytes needed for coefficients: {lpc_bytes:.0f} B", file=stream)
        print(f"Output file size: {output_file.stat().st_size} ({output_file.stat().st_size / (2 ** 20):.2f} MiB)",
              file=stream)
        print(f"Grand Ratio = {output_file.stat().st_size / self._source_size:.4f}", file=stream)

        print(f"Size of the resulting dataframe: {self.usage_mib():.3f} MiB", file=stream)
