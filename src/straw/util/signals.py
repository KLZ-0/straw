import librosa
import numpy as np


class Signals:
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
                                   min_block_size: int = 1 << 10,
                                   treshold: int = 60,
                                   max_block_size: int = 1 << 12):
        """
        Returns a list of indices where frames should start
        """
        treshold *= min_block_size
        data = channel_data.astype(np.int64)
        lst = []
        for i in range(0, data.shape[0], min_block_size):
            fr = data[i:i + min_block_size]
            # shorttime_energy = Signals.signaltonoise(fr)
            shorttime_energy = np.sum(fr * fr) / fr.shape[0]
            lst.append(shorttime_energy)
        energies = np.asarray(lst)
        # Find indices where energy crosses a treshold
        # a crossing up means a high energy frame, crossing down a low energy frame
        # borders = librosa.zero_crossings(energies)
        borders = librosa.zero_crossings(energies - treshold)
        indices = []
        last_indice = 0
        for indice in borders.nonzero()[0]:
            indice *= min_block_size
            indices += Signals._add_indice(indice, last_indice, max_block_size)
            last_indice = indice
        indices += Signals._add_indice(channel_data.shape[0], last_indice, max_block_size)

        return np.asarray(indices)

    @staticmethod
    def signaltonoise(a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))
