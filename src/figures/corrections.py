import numpy as np
import pandas as pd
import seaborn as sns

from figures.base import BasePlot


class CorrectionsPlot(BasePlot):
    def shift(self):
        frame = self._e.sample_frame()
        f = frame["frame"][8:160]

        df = pd.DataFrame({
            "sample": [i for i in range(len(f))] + [i for i in range(len(f))],
            "value": np.append(f, frame["frame"][:152]),
            "type": ["original" for _ in range(len(f))] + ["shifted" for _ in range(len(f))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="type", dashes=False, height=2.5, aspect=3)

        s.set_titles("Shift")
        s.set_xlabels("Sample")
        s.set_ylabels("Sample value (16-bit)")
        s.tight_layout()

        self.save("shift.pdf")

    def gain(self):
        frame = self._e.sample_frame()
        f = frame["frame"][8:160]

        df = pd.DataFrame({
            "sample": [i for i in range(len(f))] + [i for i in range(len(f))],
            "value": np.append(f, f * 0.8),
            "type": ["original" for _ in range(len(f))] + ["gain" for _ in range(len(f))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="type", dashes=False, height=2.5, aspect=3)

        s.set_titles("Gain")
        s.set_xlabels("Sample")
        s.set_ylabels("Sample value (16-bit)")
        s.tight_layout()

        self.save("gain.pdf")

    def offset(self):
        frame = self._e.sample_frame()
        f = frame["frame"][8:160]

        df = pd.DataFrame({
            "sample": [i for i in range(len(f))] + [i for i in range(len(f))],
            "value": np.append(f, f - 1000),
            "type": ["original" for _ in range(len(f))] + ["offset" for _ in range(len(f))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="type", dashes=False, height=2.5, aspect=3)

        s.set_titles("DC offset")
        s.set_xlabels("Sample")
        s.set_ylabels("Sample value (16-bit)")
        s.tight_layout()

        self.save("offset.pdf")

    def all(self):
        frame = self._e.sample_frame()
        f = frame["frame"][8:160]

        df = pd.DataFrame({
            "sample": [i for i in range(len(f))] + [i for i in range(len(f))],
            "value": np.append(f, frame["frame"][:152] * 0.8 - 1000),
            "type": ["original" for _ in range(len(f))] + ["deformed" for _ in range(len(f))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="type", dashes=False, height=2.5, aspect=3)

        s.set_titles("Deformations")
        s.set_xlabels("Sample")
        s.set_ylabels("Sample value (16-bit)")
        s.tight_layout()

        self.save("all.pdf")
