from pathlib import Path

from .blocking import FrameBlockingPlot
from .static import show_frame


def plot_all(args=None):
    from straw import Encoder
    from .corrections import CorrectionsPlot
    from .experiments import Experiments
    from .lpc import LPCPlot
    from .rice import RicePlot

    ###########
    # Special #
    ###########

    e = Encoder(dynamic_blocksize=True)
    # FrameBlockingPlot(e, args).print_sizes()
    # FrameBlockingPlot(e, args).plot_tresholds("blocking_dynamic_tresholds.pdf")
    RicePlot(e, args).plot_responsiveness("rice_responsiveness.pdf")

    #######################
    # Without corrections #
    #######################

    e = Encoder(do_corrections=())
    e.load_file(Path(args.input_files[0]))
    e.encode()

    # CorrectionsPlot(e, args).shift("shift.pdf")
    # CorrectionsPlot(e, args).gain("gain.pdf")
    # CorrectionsPlot(e, args).offset("offset.pdf")
    # CorrectionsPlot(e, args).all("all.pdf")
    # CorrectionsPlot(e, args).real("corrected_before.pdf")
    # CorrectionsPlot(e, args).real("corrected_shift.pdf", corrected=("shift",))
    # CorrectionsPlot(e, args).real("corrected_gain.pdf", corrected=("gain",))
    # CorrectionsPlot(e, args).real("corrected_bias.pdf", corrected=("bias",))
    # CorrectionsPlot(e, args).real("corrected_all.pdf", corrected=("gain", "shift", "bias"))
    # LPCPlot(e, args).lpc_unit_circle("lpc_unit_circle.pdf")

    ####################
    # With corrections #
    ####################

    e = Encoder()
    e.load_file(Path(args.input_files[0]))
    e.encode()

    # LPCPlot(e, args).print_lpc_and_qlp()
    # LPCPlot(e, args).prediction_comparison("prediction_comparison.pdf")
    # LPCPlot(e, args).residual("residual.pdf")
    # LPCPlot(e, args).common_lpc_autoc_averaging("lpc_averaged.png")
    # LPCPlot(e, args).common_lpc_variances("lpc_averaged_variances.png")
    # RicePlot(e, args).interleave("rice_interleave.pdf")
    # RicePlot(e, args).static_m("rice_static_m.pdf")
    # RicePlot(e, args).dynamic_m("rice_dynamic_m.pdf")
    # RicePlot(e, args).k_diff("rice_k_diff.pdf")
    # RicePlot(e, args).rand_comparison("rice_rand_comparison.pdf")
    # FrameBlockingPlot(e, args).frame_limits("blocking_dynamic.pdf")
    # FrameBlockingPlot(e, args).frame_energy("blocking_dynamic_energy.pdf")
    FrameBlockingPlot(e, args).fixed_blocksize("blocking_fixed.pdf")

    ###########################
    # With corrections + gain #
    ###########################

    # e = Encoder(do_corrections=("gain", "shift", "bias"))
    # e.load_file(Path(args.input_files[0]))
    # e.encode()

    # CorrectionsPlot(e, args).real("corrected_all_global.pdf")
