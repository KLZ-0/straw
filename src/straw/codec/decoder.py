from pathlib import Path

from straw.codec.base import BaseCoder
from straw.io import Formatter


class Decoder(BaseCoder):
    def load_file(self, input_file: Path):
        self._data, self._params = Formatter().load(input_file, self._flac_mode)
