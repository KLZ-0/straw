import soundfile

from straw import lpc
from .plotter import plot_list


def fig_lpc():
    data, sr = soundfile.read("inputs/maskoff_tone.wav")

    bs = int(sr * 0.020)
    start = 400
    signal = data[start:start + bs]

    lpc_c = lpc.lpc(signal, 8)

    data, sr = soundfile.read("inputs/maskoff_tone.wav", dtype="int16")
    signal = data[start:start + bs]

    e = lpc.lpc_predict(signal, lpc_c)
    x = lpc.lpc_reconstruct(e, lpc_c)
    # TODO: The residual is float, so quantize something somewhere so that we could actually save space...
    # TODO: the LPC coefficients need to be quantized before computing the residual

    plot_list([signal, x], "lpc_signals.png")
    plot_list([e], "lpc_residual.png")
