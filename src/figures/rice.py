import numpy as np
import pandas as pd
import seaborn as sns

from figures.base import BasePlot
from straw.rice import Ricer


class RicePlot(BasePlot):
    def interleave(self):
        frame = self._e.sample_frame()
        signal = frame["frame"]

        df = pd.DataFrame({
            "sample": [i for i in range(len(signal))] + [i for i in range(len(signal))],
            "value": list(signal) + list(self._get_interleaved_signal(signal)),
            "type": ["original" for _ in range(len(signal))] + ["interleaved" for _ in range(len(signal))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="type", height=2.5, aspect=3)
        self.save("rice_interleave.png")

    def static_m(self):
        frame = self._e.sample_frame()
        signal = frame["frame"]

        df = pd.DataFrame({
            "sample": [i for i in range(len(signal))] + [i for i in range(len(signal))],
            "value": list(self._get_interleaved_signal(signal)) + list(self._get_params_static(signal)),
            "type": ["interleaved signal" for _ in range(len(signal))] + ["m parameter" for _ in range(len(signal))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="type", height=2.5, aspect=3)
        self.save("rice_static_m.png")

    def dynamic_m(self):
        frame = self._e.sample_frame()
        signal = frame["frame"]

        df = pd.DataFrame({
            "sample": [i for i in range(len(signal))] + [i for i in range(len(signal))],
            "value": list(self._get_interleaved_signal(signal)) + list(self._get_params_variable(signal)),
            "type": ["signal" for _ in range(len(signal))] + ["static m" for _ in range(len(signal))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="type", height=2.5, aspect=3)
        self.save("rice_dynamic_m.png")

    def k_diff(self):
        frame = self._e.sample_frame()
        signal = frame["frame"]

        df = pd.DataFrame({
            "sample": [i for i in range(len(signal))] + [i for i in range(len(signal))],
            "value": [4 for _ in signal] + list(Ricer.frame_to_kparams(signal, 4, 6)),
            "type": ["static k" for _ in range(len(signal))] + ["dynamic k" for _ in range(len(signal))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="type", height=2.5, aspect=3)
        self.save("rice_k_diff.png")

    @staticmethod
    def _get_params_static(frame: np.ndarray) -> np.ndarray:
        return np.full(frame.shape, 1) << np.asarray([4 for _ in frame])

    @staticmethod
    def _get_params_variable(frame: np.ndarray) -> np.ndarray:
        return np.full(frame.shape, 1) << Ricer.frame_to_kparams(frame, 4)

    @staticmethod
    def _get_interleaved_signal(frame: np.ndarray) -> np.ndarray:
        return Ricer.frame_to_interleaved(frame)
