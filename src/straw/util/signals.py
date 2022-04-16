import librosa
import numpy as np


class Signals:
    # @staticmethod
    # def _get_zerocrossing_rate(frame, frame_resolution: int = 256):
    #     lst = []
    #     crossings = librosa.zero_crossings(frame)
    #     for i in range(0, frame.shape[0], frame_resolution):
    #         fr = crossings[i:i + frame_resolution]
    #         shorttime_energy = np.sum(fr) / fr.shape[0]
    #         lst.append(shorttime_energy)
    #     return np.asarray(lst)
    #
    # def _get_frame_limits_zerocrossing(self, channel: int = 0, frame_size: int = 256):
    #     """
    #     Returns a list of indices where frames should start
    #     """
    #     data = self._samplebuffer[channel]
    #     crossings = librosa.zero_crossings(data)
    #     lst = []
    #     for i in range(0, crossings.shape[0], frame_size):
    #         fr = crossings[i:i + frame_size]
    #         crossing_rate = np.sum(fr) / fr.shape[0]
    #         lst.append(crossing_rate)
    #         # print(i, crossing_rate)
    #     tmp = np.asarray(lst)
    #     # vals = tmp[4*64:5*64]
    #     vals = tmp[66 * (4096//frame_size):67 * (4096//frame_size)]
    #     plt.plot(vals)
    #     plt.show()
    #     exit()

    # @staticmethod
    # def _get_shorttime_energy(frame, frame_resolution: int = 128):
    #     frame = frame.astype(np.int64)
    #     lst = []
    #     for i in range(0, frame.shape[0], frame_resolution):
    #         fr = frame[i:i + frame_resolution]
    #         shorttime_energy = np.sum(fr * fr) / fr.shape[0]
    #         lst.append(shorttime_energy)
    #     return np.asarray(lst)

    @staticmethod
    def _add_indice(indice: int, last_indice: int, max_block_size: int):
        lst = []
        while True:
            if indice - last_indice <= max_block_size:
                return lst + [indice]
            else:
                last_indice = last_indice + max_block_size
                lst.append(last_indice)

    @staticmethod
    def get_frame_limits_by_energy(channel_data: np.array,
                                   min_block_size: int = 1 << 11,
                                   treshold: int = 140,
                                   max_block_size: int = 1 << 12):
        """
        Returns a list of indices where frames should start
        """
        treshold *= min_block_size
        data = channel_data.astype(np.int64)
        lst = []
        for i in range(0, data.shape[0], min_block_size):
            fr = data[i:i + min_block_size]
            shorttime_energy = np.sum(fr * fr) / fr.shape[0]
            lst.append(shorttime_energy)
        energies = np.asarray(lst)
        # Find indices where energy crosses a treshold
        # a crossing up means a high energy frame, crossing down a low energy frame
        borders = librosa.zero_crossings(energies - treshold)
        indices = []
        last_indice = 0
        for indice in borders.nonzero()[0]:
            indice *= min_block_size
            indices += Signals._add_indice(indice, last_indice, max_block_size)
            last_indice = indice
        indices += Signals._add_indice(channel_data.shape[0], last_indice, max_block_size)

        # vals = borders[0 * (4096 // frame_resolution):1 * (4096 // frame_resolution)]
        # plt.plot(vals)
        # plt.show()
        # exit()
        return np.asarray(indices)
