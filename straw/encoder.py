import numpy as np
import soundfile

from straw import lpc


class Encoder:
    _data = None
    _data_fp = None
    _samplerate = None
    _frame_size = None
    _residuals = None

    _frame_duration = None

    # TODO: find the best values for these
    _lpc_order = 8
    _lpc_precision = 12  # bits

    def __init__(self, frame_duration=0.020):
        self._frame_duration = frame_duration

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
                self._frame_size = int(sr * self._frame_duration)

            if self._samplerate != sr:
                self._clean()
                return False

            self._data_fp.append(data)
            # TODO: load the actual dtype from the file itself
            self._data.append(soundfile.read(filename, dtype="int16")[0])

        return True

    def _slice_data_into_frames(self, data):
        # TODO: What to do with the last frame
        # FIXME: We should definitely NOT throw it away like it is done currently!

        frames = []
        for i in range(0, data.shape[0], self._frame_size):
            frames.append(data[i:i + self._frame_size])
        return np.stack(frames[:-1])

    def load_stream(self, stream, samplerate):
        pass

    def encode(self):
        self._residuals = []
        for channel, frames in enumerate(self._data):
            for frame_number, frame in enumerate(frames):
                qlp, quant_level = lpc.compute_qlp(self._data_fp[channel][frame_number],
                                                   self._lpc_order, self._lpc_precision)

                self._residuals.append(lpc.compute_residual(frame, qlp, self._lpc_order, quant_level))

    def save_file(self, filename):
        pass

    def _clean(self):
        self._data = None
        self._data_fp = None
        self._samplerate = None
        self._frame_size = None
        self._residuals = None
