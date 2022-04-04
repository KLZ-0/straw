# Design

[IETF](https://www.ietf.org/id/draft-ietf-cellar-flac-02.html)

The base design of Straw is based on [FLAC](https://xiph.org/flac).

Linear prediction (LPC) is used to generate and predict a signal which substracted from the original signal results in a
residual with lower entropy. This residual is then efficiently encoded using rice coding to achieve a similar file size
to FLAC for single channel audio.

In addition to these techniques, Straw also takes advantage of waweform similarities between multiple channels.

In general the differences between channels are a combination of:

- Shift
- Gain
- DC offset

Although other conditions can also affect the signals such as the environment and microphone array material.

# DataFrame

Common columns:

- seq: int - frame number, if grouped by this then represents a slice across all channels
- frame: np.array - raw audio samples (and warmup samples when decoding)
- channel: int - channel number or also subframe number
- qlp: np.array - quantized lpc coefficients -> len() is the order
- qlp_precision: float (because of nans...) - precision of stored LPC coefficients in bits
- shift: float (because of nans...) - shift needed (in bits) to revert the quantization
- residual: pd.array - the signal residual
- bps: int - bits per sample or also the k parameter of rice coding

Encoder specific:

- stream: bitaray representing the rice coded residual samples
- stream_len: length of stream in bits
