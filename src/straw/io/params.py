import numpy as np


class StreamParams:
    def alloc_arrays(self):
        if self.channels is None:
            raise ValueError("Channels must be initialized when allocate_arrays is called")
        self.lags = np.zeros(self.channels, dtype=np.int8)  # 4-bit
        self.bias = np.zeros(self.channels, dtype=np.int8)  # 8-bit signed
        self.gain = np.zeros(self.channels, dtype=np.int64)  # quantized floating point number with variable bit width

    sample_rate: int = None
    channels: int = None
    bits_per_sample: int = None
    total_samples: int = None
    total_frames: int = None
    md5: bytes = None
    responsiveness: int = None
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
