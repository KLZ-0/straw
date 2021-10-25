import soundfile
from scipy.signal import lfilter

from straw import lpc
from .plotter import plot_list


def fig_lpc():
    data, sr = soundfile.read("inputs/maskoff_tone.wav")

    bs = int(sr * 0.020)
    start = 400
    signal = data[start:start + bs]

    lpc_c = lpc.lpc(signal, 8)

    print(lpc_c)

    signal2 = lfilter(a=[1], b=lpc_c, x=signal)

    plot_list([signal, signal2], "temp.png")
    plot_list([signal - signal2], "residual.png")
