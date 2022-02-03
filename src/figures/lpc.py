from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# NOTE: Do not import this without checking for args -> seaborn and mpl should be optional dependencies
import straw
from straw.lpc import steps


class LPCPlot:
    def __init__(self, e: straw.Encoder, args=None):
        self._args = args
        self._e = e
        self.fig_dir = Path(getattr(self._args, "fig_dir", "outputs"))
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.fig_show = getattr(self._args, "fig_show", False)

    def prediction_comparison(self):
        frame = self._e.sample_frame()
        # FIXME: magic number order=8
        f = frame["frame"][8:160]
        pred = steps.predict_signal(frame["frame"], frame["qlp"], frame["shift"])[:152]

        df = pd.DataFrame({
            "sample": [i for i in range(len(f))] + [i for i in range(len(pred))],
            "source": ["Original" for _ in range(len(f))] + ["Predicted" for _ in range(len(pred))],
            "value": np.append(f, pred)
        })

        s = sns.relplot(data=df, kind="line", col="source", col_wrap=1, x="sample", y="value", height=2.5, aspect=3)

        s.set_titles("{col_name}")
        s.set_xlabels("Sample")
        s.set_ylabels("Sample value (16-bit)")
        s.tight_layout()

        plt.savefig(self.fig_dir / "prediction_comparison.pdf")

        if self.fig_show:
            plt.show()

    def residual(self):
        frame = self._e.sample_frame()
        # FIXME: magic number order=8
        f = frame["frame"][8:160]
        pred = steps.predict_signal(frame["frame"], frame["qlp"], frame["shift"])[:152]

        df = pd.DataFrame({
            "sample": [i for i in range(len(f))],
            "value": f - pred
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", height=2.5, aspect=3)

        s.set_titles("Residual")
        s.set_xlabels("Sample")
        s.set_ylabels("Sample value (16-bit)")
        s.tight_layout()

        plt.savefig(self.fig_dir / "residual.pdf")

        if self.fig_show:
            plt.show()
