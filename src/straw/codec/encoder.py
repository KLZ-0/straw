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
    # Values which should be parametrized
    # TODO: find the best values for these
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
                 responsiveness=Default.rice_responsiveness):
        """

        :param flac_mode:
        :param do_corrections: an iterable containing the corrections to be done, can contain "gain", "bias" and "shift"
        :param dynamic_blocksize:
        """
        super(Encoder, self).__init__(flac_mode)
        self._ricer = Ricer(adaptive=True if not flac_mode else False, responsiveness=responsiveness)
        self._params.responsiveness = responsiveness
        self._do_corrections = do_corrections
        self._do_dynamic_blocking = dynamic_blocksize
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.framing_treshold = framing_treshold
        self.framing_resolution = framing_resolution

    def set_rice_responsiveness(self, responsiveness):
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
        Encode the signal
        :return: None
        """
        # Extract stream parameters & initialize frame types
        self._parametrize()
        self._init_frame_types()

        groups = self._data.groupby("seq")
        # self._data = groups.apply(self._encode_frame)
        self._data = ParallelCompute.get_instance().map_group(groups, self._encode_frame)

    def _encode_frame(self, data_slice: pd.DataFrame):
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
        self._tmp()
        # self._data.to_pickle("/tmp/old_streamlen.pkl.gz")
        # new_lens = self._data
        # old_lens = pd.read_pickle("/tmp/old_streamlen.pkl.gz")
        # diff = (new_lens - old_lens)["stream_len"]
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
        :return:
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

    def _parametrize(self):
        """
        Parameter extraction to be used for encoding the whole stream
        :return: None
        """
        if self._flac_mode:
            self._params.max_block_size = int(self._data["frame"].apply(len).max())
            self._params.min_block_size = self._params.max_block_size
            self._params.min_frame_size = 0  # unknown
            self._params.max_frame_size = 0
        # self._params.total_samples = int(self._data[self._data["channel"] == 0]["frame"].apply(len).sum())

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

    def _decorrelate_signals(self, data_slice, col_name="residual"):
        # TODO: do not decorrelate for frames with separate LPC
        # self._data = self._data.groupby("seq").apply(Decorrelator().localized_decorrelate, col_name=col_name)
        # self._data = ParallelCompute.get_instance().map_group(self._data.groupby("seq"),
        #                                                       Decorrelator().midside_decorrelate, col_name=col_name)
        data_slice = data_slice.groupby("seq").apply(correctors.Decorrelator().midside_decorrelate, col_name=col_name)
        data_slice["was_coded"] = 0

    #########
    # Other #
    #########

    def _print_var(self, seq=0):
        old_stream_len = 214523
        stream_len = self._data[self._data["seq"] == seq]["stream_len"].sum()
        print("- stream_len:", stream_len)
        print("- stream_len diff:", stream_len - old_stream_len)
        old_maxabs = np.asarray([352, 373, 581, 516, 432, 349, 380, 391])
        nocorr_var = np.asarray([10997.481, 24395.01, 50948.516, 36896.603, 21682.603, 11630.761,
                                 14912.361, 18267.361])
        self._print_var_details(seq, np.var, "var", nocorr_var)
        self._print_var_details(seq, lambda x: np.max(np.abs(x)), "absmax", old_maxabs)

    def _print_var_details(self, seq, func, name, old_vals=None):
        residuals = self._data[self._data["seq"] == seq]["residual"]
        residuals = residuals.apply(lambda x: x[1740:1800])
        var = residuals.apply(func).to_numpy()
        print(f"- {name}:", np.array2string(var, separator=", ", precision=3, suppress_small=True))
        if old_vals is not None:
            print(f"- original {name}:", np.array2string(old_vals, precision=3, suppress_small=True))
            print(f"- {name} difference:", np.array2string(var - old_vals, precision=3, suppress_small=True))
            print(f"total {name} diff: {(var - old_vals).sum():.3f}")

    def _tmp(self):
        """
        Temporary method for experiments and plots
        """
        # self._data.groupby("seq").apply(lambda df: df["frame"].apply(cross_similarity, data_ref=df["frame"][df.index[0]]))
        # self._print_var(seq=5)
        # from figures import show_frame
        # show_frame(self._data[self._data["seq"] == 4], terminate=False, limit=(1750, 60))
        # show_frame(self._data[self._data["seq"] == 4], terminate=False, col_name="residual", limit=(1730, 60))
        # show_frame(self._data[self._data["seq"] == 5], terminate=False, limit=(1750, 60))
        # show_frame(self._data[self._data["seq"] == 5], terminate=False, col_name="residual", limit=(1730, 60))
        # exit()
        # df = self._data[(self._data["seq"] == 66) & (self._data["channel"] == 0)]
        # show_frame(df, col_name="frame", terminate=False)
        # # df["zeros"] = df["frame"].apply(self._get_zerocrossing_rate)
        # # show_frame(df, col_name="zeros", terminate=False)
        # df["energy"] = df["frame"].apply(self._get_shorttime_energy)
        # show_frame(df, col_name="energy")
        pass

    ###########
    # Utility #
    ###########

    def get_stats(self, output_file: Path) -> EncoderStats:
        stats = EncoderStats()
        stats.file_size = output_file.stat().st_size
        stats.ratio = output_file.stat().st_size / self._source_size
        stats.frames = len(self._data.groupby('seq').groups)
        return stats

    def print_stats(self, output_file: Path, stream: TextIO = sys.stdout):
        """
        Print a bunch of stuff...
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

        # FIXME: this is misleading
        print(f"Size of the resulting dataframe: {self.usage_mib():.3f} MiB", file=stream)
