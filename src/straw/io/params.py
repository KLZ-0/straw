class StreamParams:
    sample_rate: int = None
    channels: int = None
    bits_per_sample: int = None
    total_samples: int = None
    md5: bytes = None


class FLACStreamParams(StreamParams):
    min_block_size: int = None
    max_block_size: int = None
    min_frame_size: int = None
    max_frame_size: int = None
