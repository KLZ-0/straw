import soundfile


class Encoder:
    _data = None
    _data_fp = None
    _samplerate = None

    def __init__(self):
        pass

    def load_files(self, filenames: list):
        """
        Load all specified files as separate channels of the same recording
        :param filenames: list of files to load
        :return: True on success, False on error
        """
        self._data = []
        self._data_fp = []

        # TODO: verify if the files are from the same recording
        for filename in filenames:
            data, sr = soundfile.read(filename)
            if self._samplerate is None:
                self._samplerate = sr

            if self._samplerate != sr:
                self._clean()
                return False

            self._data_fp.append(data)
            # TODO: load the actual dtype from the file itself
            self._data.append(soundfile.read(filename, dtype="int16"))

        return True

    def load_stream(self, stream, samplerate):
        pass

    def encode(self):
        pass

    def save_file(self, filename):
        pass

    def _clean(self):
        self._data = None
        self._data_fp = None
        self._samplerate = None
