import numpy as np


class StreamParams:
    sample_rate: int = None
    channels: int = None
    bits_per_sample: int = None
    total_samples: int = None
    md5: bytes = None
    leading_channel: int = None
    lags: np.ndarray  # length equal to number of channels, each value represents lag on a channel
    removed_samples_start = []  # removed samples for each channel from the start
    removed_samples_end = []  # removed samples for each channel from the end
    bias: np.ndarray  # length equal to number of channels, each value represents bias on a channel
    gain: np.ndarray  # length equal to number of channels, each value represents quantized gain factor on a channel
    gain_shift: int = None


class FLACStreamParams(StreamParams):
    min_block_size: int = None
    max_block_size: int = None
    min_frame_size: int = None
    max_frame_size: int = None
