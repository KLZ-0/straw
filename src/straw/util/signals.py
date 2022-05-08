import numpy as np

from straw.static import Default


def zero_crossings(x, zero_pos: bool = True):
    """
    Similar output to librosa.zero_crossings but more efficient
    :param x: signal
    :param zero_pos: whether to count the first number as a zero crossing
    :return: Zero crossings
    """
    if zero_pos:
        x_sign = np.signbit(x)
    else:
        x_sign = np.sign(x)

    return np.insert(x_sign[1:] != x_sign[:-1], 0, 1)


def resample_indices(raw_indices: np.array, min_block_size: int, max_block_size: int):
    """
    Resample the indices so that the new block sizes are between min_block_size and max_block_size
    :param raw_indices: indices to be resampled
    :param min_block_size: minimal allowed block size
    :param max_block_size: maximal allowed block size
    :return: new indices
    """
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
        Returns a list of indices where frames should start, determined by the short-time energy
        :param channel_data: full data of one channel
        :param min_block_size: minimal allowed block size
        :param treshold: energy threshold
        :param max_block_size: maximal allowed block size
        :param resolution: framing resolution
        :return: Indices of frame borders including the last border
        """
        if resolution is None:
            resolution = min_block_size
        energies = cls.get_energies(channel_data, resolution, treshold)
        # Find indices where energy crosses a treshold
        # a crossing up means a high energy frame, crossing down a low energy frame
        # borders = zero_crossings(energies)
        borders = zero_crossings(energies - treshold, zero_pos=False)
        all_indices = borders.nonzero()[0] * resolution
        all_indices[-1] = channel_data.shape[0]

        indices = resample_indices(all_indices, min_block_size, max_block_size)

        return indices

    @staticmethod
    def get_energies(data: np.array,
                     resolution: int = Default.framing_resolution,
                     treshold: int = Default.framing_treshold):
        """
        Return the energies of a frame
        :param data: frame data
        :param resolution: framing resolution
        :param treshold: energy threshold
        :return: short-term energies in a frame
        """
        data = data.astype(np.int64)
        lst = []
        for i in range(0, data.shape[0], resolution):
            fr = data[i:i + resolution]
            # shorttime_energy = Signals.signaltonoise(fr)
            shorttime_energy = np.sum(fr * fr) / fr.shape[0]
            lst.append(shorttime_energy)
        lst.append(treshold)
        return np.asarray(lst)
