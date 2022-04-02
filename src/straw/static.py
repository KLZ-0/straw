columns = [
    "seq",  # int - frame number, if grouped by this then represents a slice across all channels
    "frame",  # np.array - raw audio samples (and warmup samples when decoding)
    "channel",  # int - channel number or also subframe number
    "qlp",  # np.array - quantized lpc coefficients -> len() is the order
    "qlp_precision",  # float (because of nans...) - precision of stored LPC coefficients in bits
    "shift",  # float (because of nans...) - shift needed (in bits) to revert the quantization
    "residual",  # pd.array - the signal residual
    "bps",  # int - bits per sample or also the k parameter of rice coding
]
