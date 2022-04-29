from pathlib import Path

from .static import show_frame


def plot_all(args=None):
    from straw import Encoder
    from .corrections import CorrectionsPlot
    from .experiments import Experiments
    from .lpc import LPCPlot
    from .rice import RicePlot

    e = Encoder()
    e.load_file(Path(args.input_files[0]))
    e.encode()

    # LPCPlot(e, args).print_lpc_and_qlp()
    # LPCPlot(e, args).prediction_comparison()
    # LPCPlot(e, args).residual()
    # CorrectionsPlot(e, args).shift()
    CorrectionsPlot(e, args).shift_real(corrected=False)
    CorrectionsPlot(e, args).shift_real(corrected=True)
    # CorrectionsPlot(e, args).gain()
    # CorrectionsPlot(e, args).offset()
    # CorrectionsPlot(e, args).all()
    # LPCPlot(e, args).common_lpc_autoc_averaging()
    # LPCPlot(e, args).common_lpc_variances()
    # RicePlot(e, args).interleave()
    # RicePlot(e, args).static_m()
    # RicePlot(e, args).dynamic_m()
    # RicePlot(e, args).k_diff()
    # RicePlot(e, args).rand_comparison()
