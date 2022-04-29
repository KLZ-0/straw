import numpy as np
import pandas as pd
import seaborn as sns

from figures.base import BasePlot
from straw.correctors import ShiftCorrector
from straw.io.params import StreamParams


class CorrectionsPlot(BasePlot):
    def shift(self, filename):
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

        self.save(filename)

    @staticmethod
    def _make_multichannel_df(frame: np.array, limits: tuple = None) -> pd.DataFrame:
        if limits is not None:
            frame = frame[:, limits[0]:limits[0] + limits[1]]

        df = {
            "sample": [],
            "value": [],
            "Channel": [],
        }
        for i, subframe in enumerate(frame):
            df["sample"] += [u for u in range(subframe.shape[0])]
            df["value"] += subframe.tolist()
            df["Channel"] += [i for _ in range(subframe.shape[0])]

        return pd.DataFrame(df, copy=False)

    def gain(self, filename):
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

        self.save(filename)

    def offset(self, filename):
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

        self.save(filename)

    def all(self, filename):
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

        self.save(filename)

    def shift_real(self, filename, corrected):
        frame = self._e.samplebuffer_frame_multichannel(seq=4)
        if corrected:
            sc = ShiftCorrector()
            sc.apply(frame, params=StreamParams())
            sc.apply_to_ndarray(frame)

        df = self._make_multichannel_df(frame, limits=(1750, 80))

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="Channel", dashes=False, height=2.5, aspect=3)
        s.set_xlabels("Sample")
        s.set_ylabels("Sample value (16-bit)")
        s.tight_layout()

        if corrected:
            self.save(filename)
        else:
            self.save(filename)
