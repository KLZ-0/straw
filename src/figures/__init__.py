import straw
from .corrections import CorrectionsPlot
from .lpc import LPCPlot


def plot_all(args=None):
    e = straw.Encoder(args)
    e.load_files(args.input_files)
    e.create_frames()
    e.encode()

    LPCPlot(e, args).print_lpc_and_qlp()
    LPCPlot(e, args).prediction_comparison()
    LPCPlot(e, args).residual()
    CorrectionsPlot(e, args).shift()
    CorrectionsPlot(e, args).gain()
    CorrectionsPlot(e, args).offset()
    CorrectionsPlot(e, args).all()
