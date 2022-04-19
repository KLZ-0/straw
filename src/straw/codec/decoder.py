from pathlib import Path

import pandas as pd
import soundfile

from straw import lpc
from straw.codec.base import BaseCoder
from straw.correctors import Decorrelator, GainCorrector, BiasCorrector
from straw.io import Formatter
from straw.static import SubframeType


class Decoder(BaseCoder):
    ##########
    # Public #
    ##########

    def load_file(self, input_file: Path):
        """
        Load the specified file into memory
        :param input_file: file to load
        :return: None
        """
        reader = Formatter().load(input_file, self._flac_mode)
        self._data = reader.get_data()
        self._params = reader.get_params()
        self._samplebuffer = reader.get_buffer()
        self._add_biascorr_to_raw_frames()

    def decode(self):
        """
        Decode the signal and verify its integrity
        :return: None
        """
        self._revert_decorrelate("residual")
        self._data.groupby("seq").apply(lpc.compute_original, inplace=True)
        self._revert_corrections()
        if self.get_md5() != self._params.md5:
            print(f"md5 restored: {self.get_md5().hex(' ')}")
            print(f"md5 file:     {self._params.md5.hex(' ')}")
            raise ValueError("Non-matching md5 checksum")

    def save_file(self, output_file: Path):
        """
        Save the decoded signal
        :param output_file: target file
        :return: None
        """
        with soundfile.SoundFile(output_file, "w",
                                 subtype=self._subtype_pattern.format(self._params.bits_per_sample),
                                 samplerate=self._params.sample_rate,
                                 channels=self._params.channels) as wav:
            dtype_bits = self._samplebuffer.itemsize * 8
            shift = dtype_bits - self._params.bits_per_sample
            data = (self._samplebuffer.swapaxes(1, 0) << shift)
            wav.write(data)

    ###########
    # Private #
    ###########

    def _add_biascorr_to_raw_frames(self):
        """
        This is done because the bias correction can result in samples outside of the valid dtype range
        when saved to a straw file, thus it is safer to just remove the correction in this section
        :return:
        """

        def _correct_bias(df: pd.Series):
            df["frame"] += self._params.bias[df["channel"]]

        self._data[self._data["frame_type"] == SubframeType.RAW].apply(_correct_bias, axis=1)

    def _revert_corrections(self):
        BiasCorrector().apply_revert(self._samplebuffer, self._params)
        GainCorrector().apply_revert(self._samplebuffer, self._params)

    def _revert_decorrelate(self, col_name="residual"):
        self._data = self._data.groupby("seq").apply(Decorrelator().midside_decorrelate_revert, col_name=col_name)
        # self._data = self._data.groupby("seq").apply(Decorrelator().localized_decorrelate_revert, col_name=col_name)

    ###########
    # Utility #
    ###########

    def test(self, wav_file: Path = Path("inputs/1min.wav")):
        """
        Test the decoder with a sample file
        :return:
        """
        tmp, _ = soundfile.read(wav_file, dtype="int16")
        tmp = tmp.swapaxes(1, 0)
        if not (self._samplebuffer - tmp).any():
            print("Lossless :)")
        else:
            print("Not lossless :|")
