from pathlib import Path

import soundfile

from straw import lpc
from straw.codec.base import BaseCoder
from straw.correctors import Decorrelator, GainCorrector, BiasCorrector
from straw.io import Formatter


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

    def decode(self):
        """
        Decode the signal and verify its integrity
        :return: None
        """
        self._revert_decorrelate("residual")
        self._data.apply(lpc.compute_original, axis=1, inplace=True)
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
