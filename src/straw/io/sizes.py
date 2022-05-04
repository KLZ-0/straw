UTF8 = None


class StrawSizes:
    """
    Sizes for the straw format
    Only for fields with at least one known dimension length
    None for unknown values
    All sizes are in bits unless specified otherwise
    """

    class metadata_block_header:  # noqa
        last = 1
        type = 7
        size = 24

    class metadata_block_streaminfo:  # noqa
        samplerate = 20
        channels = UTF8
        bps = 5
        frames = 27
        samples = 36
        md5 = 128
        responsiveness = 8
        leading_channel = UTF8
        shift = 4
        removed_samples = None
        bias = 8
        gain = 12
        gain_shift = 4

    class frame_header:  # noqa
        sync_code = 14
        contains_lpc = 1
        block_size_length = 1
        block_size_log2 = 8
        block_size_exact = 16
        frame_num = UTF8
        frame_bytes = 32
        crc = 8

    class frame_footer:  # noqa
        crc = 16

    class subframe_header:  # noqa
        type = 2

    class subframe_lpc:  # noqa
        contains_lpc = 1
        lpc_order = 5
        lpc_prec = 4
        lpc_shift = 4
        lpc_coeffs = None
        is_coded = 1

    class residual:  # noqa
        param = 4
