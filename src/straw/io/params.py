import numpy as np


class StreamParams:
    sample_rate: int = None
    channels: int = None
    bits_per_sample: int = None
    total_samples: int = None
    md5: bytes = None
    leading_channel: int = None
    lags: np.ndarray  # length equal to number of channels, each value represents lag on a channel
    bias: np.ndarray  # length equal to number of channels, each value represents bias on a channel


class FLACStreamParams(StreamParams):
    min_block_size: int = None
    max_block_size: int = None
    min_frame_size: int = None
    max_frame_size: int = None
