import math

import numpy as np
import soundfile

from straw import lpc
from .plotter import plot_list


def fig_lpc():
    data, sr = soundfile.read("inputs/maskoff_tone.wav")

    lpc_order = 8
    lpc_precision = 12  # bits

    bs = int(sr * 0.020)
    start = 400
    signal = data[start:start + bs]

    qlp, quant_level = lpc.compute_qlp(signal, lpc_order, lpc_precision)

    data, sr = soundfile.read("inputs/maskoff_tone.wav", dtype="int16")
    signal = data[start:start + bs]

    residual = lpc.compute_residual(signal, qlp, lpc_order, quant_level)
    restored = lpc.restore_signal(residual, qlp, lpc_order, quant_level, signal[:lpc_order])

    plot_list([signal, restored], "lpc_signals.png")
    plot_list([residual], "lpc_residual.png")
