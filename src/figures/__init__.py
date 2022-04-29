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

    LPCPlot(e, args).print_lpc_and_qlp()
    LPCPlot(e, args).prediction_comparison("prediction_comparison.pdf")
    LPCPlot(e, args).residual("residual.pdf")
    CorrectionsPlot(e, args).shift("shift.pdf")
    CorrectionsPlot(e, args).gain("gain.pdf")
    CorrectionsPlot(e, args).offset("offset.pdf")
    CorrectionsPlot(e, args).all("all.pdf")
    CorrectionsPlot(e, args).shift_real("shift_real_before.pdf", corrected=False)
    CorrectionsPlot(e, args).shift_real("shift_real_after.pdf", corrected=True)
    LPCPlot(e, args).common_lpc_autoc_averaging("lpc_averaged.png")
    LPCPlot(e, args).common_lpc_variances("lpc_averaged_variances.png")
    RicePlot(e, args).interleave("rice_interleave.pdf")
    RicePlot(e, args).static_m("rice_static_m.pdf")
    RicePlot(e, args).dynamic_m("rice_dynamic_m.pdf")
    RicePlot(e, args).k_diff("rice_k_diff.pdf")
    RicePlot(e, args).rand_comparison("rice_rand_comparison.pdf")
