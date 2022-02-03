import straw
from .lpc import LPCPlot


def plot_all(args=None):
    e = straw.Encoder(args)
    e.load_files(args.input_files)
    e.create_frames()
    e.encode()

    LPCPlot(e, args).prediction_comparison()
    LPCPlot(e, args).residual()
