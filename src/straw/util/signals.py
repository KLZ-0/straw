import librosa
import numpy as np


def _find_nearest(array, value, offset=None):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if offset is not None and not (array[idx] - offset <= value <= array[idx] + offset):
        return value
    return array[idx]


def _find_valid_blocksizes(min_block_size, max_block_size) -> np.array:
    blocksizes = []
    i = 1
    while i < min_block_size:
        i <<= 1

    while i <= max_block_size:
        blocksizes.append(i)
        i <<= 1

    return np.asarray(blocksizes)


def resample_indices(raw_indices: np.array,
                     min_block_size: int,
                     max_block_size: int,
                     keep_blocksize_power_of_two: bool):
    valid_blocksizes = _find_valid_blocksizes(min_block_size, max_block_size)
    indices = [0]
    currect_size = 0
    for i in range(1, raw_indices.shape[0]):
        if raw_indices[i] <= indices[-1]:
            continue

        currect_size += raw_indices[i] - raw_indices[i - 1]
        if min_block_size <= currect_size <= max_block_size:
            if keep_blocksize_power_of_two:
                currect_size = _find_nearest(valid_blocksizes, currect_size, offset=50)
            indices.append(indices[-1] + currect_size)
            currect_size = 0

        while currect_size > max_block_size:
            indices.append(indices[-1] + max_block_size)
            currect_size -= max_block_size

    indices[-1] = raw_indices[-1]
    return np.asarray(indices)


class Signals:
    @staticmethod
    def get_frame_limits_by_energy(channel_data: np.array,
                                   min_block_size: int = 1 << 10,
                                   treshold: int = 62000,
                                   max_block_size: int = 1 << 12,
                                   resolution: int = 10,
                                   keep_blocksize_power_of_two: bool = True):
        """
        Returns a list of indices where frames should start
        """
        if resolution is None:
            resolution = min_block_size
        data = channel_data.astype(np.int64)
        lst = []
        for i in range(0, data.shape[0], resolution):
            fr = data[i:i + resolution]
            # shorttime_energy = Signals.signaltonoise(fr)
            shorttime_energy = np.sum(fr * fr) / fr.shape[0]
            lst.append(shorttime_energy)
        lst.append(treshold)
        energies = np.asarray(lst)
        # Find indices where energy crosses a treshold
        # a crossing up means a high energy frame, crossing down a low energy frame
        # borders = librosa.zero_crossings(energies)
        borders = librosa.zero_crossings(energies - treshold, zero_pos=False)
        all_indices = borders.nonzero()[0] * resolution
        all_indices[-1] = channel_data.shape[0]

        indices = resample_indices(all_indices, min_block_size, max_block_size, keep_blocksize_power_of_two)

        return indices

    @staticmethod
    def signaltonoise(a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))
