from pathlib import Path

import soundfile

from straw import lpc
from straw.codec.base import BaseCoder
from straw.correctors import Decorrelator
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
        soundfile.write(output_file, self._samplebuffer.swapaxes(1, 0), samplerate=self._params.sample_rate)

    ###########
    # Private #
    ###########

    def _revert_corrections(self):
        for i in range(self._params.channels):
            self._samplebuffer[i] += self._params.bias[i]

    def _revert_decorrelate(self, col_name="residual"):
        self._data = self._data.groupby("seq").apply(Decorrelator().midside_decorrelate_revert, col_name=col_name)
        self._data = self._data.groupby("seq").apply(Decorrelator().localized_decorrelate_revert, col_name=col_name)

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
