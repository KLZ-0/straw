class SubframeType:
    CONSTANT = 0b00
    RAW = 0b01
    LPC = 0b10
    LPC_COMMON = 0b11


col_types = {
    "seq": int,  # int - frame number, if grouped by this then represents a slice across all channels
    "frame": object,  # np.array - raw audio samples (and warmup samples when decoding)
    "channel": int,  # int - channel number or also subframe number
    "qlp": object,  # np.array - quantized lpc coefficients -> len() is the order
    "qlp_precision": float,  # float (because of nans...) - precision of stored LPC coefficients in bits
    "shift": float,  # float (because of nans...) - shift needed (in bits) to revert the quantization
    "residual": object,  # pd.array - the signal residual
    "bps": int,  # int - bits per sample or also the k parameter of rice coding
    "frame_type": int,  # int (enum) - corresponds to one of SUBFRAME_CONSTANT, SUBFRAME_RAW, SUBFRAME_LPC
}

columns = list(col_types.keys())

soundfile_dtype = {
    16: 16,
    24: 32,
    32: 32,
}


class Default:
    min_frame_size = 1 << 11
    max_frame_size = 1 << 13
    framing_treshold = 20000
    framing_resolution = 10
    rice_responsiveness = 20
