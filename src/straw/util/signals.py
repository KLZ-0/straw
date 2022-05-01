import librosa
import numpy as np

from straw.static import Default


def resample_indices(raw_indices: np.array, min_block_size: int, max_block_size: int):
    indices = [0]
    currect_size = 0
    for i in range(1, raw_indices.shape[0]):
        currect_size += raw_indices[i] - raw_indices[i - 1]
        if min_block_size <= currect_size <= max_block_size:
            indices.append(raw_indices[i])
            currect_size = 0

        while currect_size > max_block_size:
            indices.append(indices[-1] + max_block_size)
            currect_size -= max_block_size

    indices[-1] = raw_indices[-1]
    return np.asarray(indices)


class Signals:
    @classmethod
    def get_frame_limits_by_energy(cls,
                                   channel_data: np.array,
                                   min_block_size: int = Default.min_frame_size,
                                   treshold: int = Default.framing_treshold,
                                   max_block_size: int = Default.max_frame_size,
                                   resolution: int = Default.framing_resolution):
        """
        Returns a list of indices where frames should start
        """
        if resolution is None:
            resolution = min_block_size
        energies = cls.get_energies(channel_data, resolution, treshold)
        # Find indices where energy crosses a treshold
        # a crossing up means a high energy frame, crossing down a low energy frame
        # borders = librosa.zero_crossings(energies)
        borders = librosa.zero_crossings(energies - treshold, zero_pos=False)
        all_indices = borders.nonzero()[0] * resolution
        all_indices[-1] = channel_data.shape[0]

        indices = resample_indices(all_indices, min_block_size, max_block_size)

        return indices

    @staticmethod
    def signaltonoise(a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))

    @staticmethod
    def get_energies(data: np.array,
                     resolution: int = Default.framing_resolution,
                     treshold: int = Default.framing_treshold):
        data = data.astype(np.int64)
        lst = []
        for i in range(0, data.shape[0], resolution):
            fr = data[i:i + resolution]
            # shorttime_energy = Signals.signaltonoise(fr)
            shorttime_energy = np.sum(fr * fr) / fr.shape[0]
            lst.append(shorttime_energy)
        lst.append(treshold)
        return np.asarray(lst)
