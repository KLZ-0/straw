from pathlib import Path

import soundfile

from straw import lpc
from straw.codec.base import BaseCoder
from straw.io import Formatter


class Decoder(BaseCoder):
    def load_file(self, input_file: Path):
        reader = Formatter().load(input_file, self._flac_mode)
        self._data = reader.get_data()
        self._params = reader.get_params()
        self._samplebuffer = reader.get_buffer()
        self._data = self._data.groupby("seq").apply(lpc.compute_original)
        print(self._params.md5)
        print(self.get_md5())

        data, sr = soundfile.read("inputs/1min.wav", dtype="int16")
        # mine = self._data.loc[235, "frame"]
        mine = self._samplebuffer[1][:self._params.max_block_size]
        original = data.swapaxes(1, 0)[1][:self._params.max_block_size]
        exit()

    def save_wav(self, output_file: Path):
        soundfile.write(output_file, self._samplebuffer.swapaxes(1, 0), samplerate=self._params.sample_rate)
