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
        self._data.groupby("seq").apply(lpc.compute_original, inplace=True)
        if self.get_md5() != self._params.md5:
            raise ValueError("Non-matching md5 checksum")

        self.test()

    def test(self):
        tmp, _ = soundfile.read("inputs/1min.wav", dtype="int16")
        orig_frames = []
        for channel_data in tmp.swapaxes(1, 0):
            orig_frames += self._slice_channel_data_into_frames(channel_data)
        self._data["original"] = orig_frames
        if self._data.apply(lpc.compare_restored, axis=1).all():
            print("Lossless :)")
        else:
            print("Not lossless :|")

        print(f"restored: {self.get_md5().hex(' ')}")
        print(f"file:     {self._params.md5.hex(' ')}")

    def save_wav(self, output_file: Path):
        soundfile.write(output_file, self._samplebuffer.swapaxes(1, 0), samplerate=self._params.sample_rate)
