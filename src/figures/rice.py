import numpy as np
import pandas as pd
import seaborn as sns

from figures.base import BasePlot
from straw.rice import Ricer


class RicePlot(BasePlot):
    def interleave(self, filename):
        frame = self._e.sample_frame()
        signal = frame["frame"]

        df = pd.DataFrame({
            "sample": [i for i in range(len(signal))] + [i for i in range(len(signal))],
            "value": list(signal) + list(self._get_interleaved_signal(signal)),
            "type": ["original" for _ in range(len(signal))] + ["interleaved" for _ in range(len(signal))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="type", height=2.5, aspect=3)
        self.save(filename)

    def static_m(self, filename):
        frame = self._e.sample_frame()
        signal = frame["frame"]

        df = pd.DataFrame({
            "sample": [i for i in range(len(signal))] + [i for i in range(len(signal))],
            "value": list(self._get_interleaved_signal(signal)) + list(self._get_params_static(signal)),
            " ": ["interleaved signal" for _ in range(len(signal))] + ["m parameter" for _ in range(len(signal))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="type", height=2.5, aspect=3)
        self.save(filename)

    def dynamic_m(self, filename):
        frame = self._e.sample_frame()
        signal = frame["frame"]

        df = pd.DataFrame({
            "sample": [i for i in range(len(signal))] + [i for i in range(len(signal))],
            "value": list(self._get_interleaved_signal(signal)) + list(self._get_params_variable(signal)),
            " ": ["Signal" for _ in range(len(signal))] + ["Parameter (m)" for _ in range(len(signal))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue=" ", height=2.5, aspect=3)
        s.set_xlabels("Sample")
        s.set_ylabels("Value (16-bit)")
        s.tight_layout()
        self.save(filename)

    def k_diff(self, filename):
        frame = self._e.sample_frame()
        signal = frame["frame"]

        df = pd.DataFrame({
            "sample": [i for i in range(len(signal))] + [i for i in range(len(signal))],
            "value": [4 for _ in signal] + list(Ricer.frame_to_kparams(signal, 4, 6)),
            "type": ["static k" for _ in range(len(signal))] + ["dynamic k" for _ in range(len(signal))],
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", hue="type", height=2.5, aspect=3)
        self.save(filename)

    def rand_comparison(self, filename):
        from numpy.random import default_rng
        rng = default_rng(seed=[2444, 1248, 5685])
        static = Ricer(adaptive=False)
        dynamic = Ricer(adaptive=True)

        df = {
            "spike_width": [],
            "value": [],
            "Parameter type": [],
        }

        limit = 1000
        spike = 10000
        base_dataset = np.clip(rng.normal(0, limit / 5, size=4096), -limit, limit).astype(np.int64)
        full_window = np.zeros(4096, dtype=np.int64)
        for window_width in range(1, 512, 10):
            window = np.clip(rng.normal(0, spike / 5, size=window_width), -spike, spike).astype(np.int64)
            start_idx = full_window.shape[0] // 2 - window.shape[0] // 2
            full_window[start_idx:start_idx + window.shape[0]] = window
            dataset = base_dataset + full_window
            bps = dynamic.guess_parameter(dataset)
            df["spike_width"].append(window_width)
            df["value"].append(len(dynamic.frame_to_bitstream(dataset, bps)) / 1000)
            df["Parameter type"].append("Adaptive")
            df["spike_width"].append(window_width)
            df["value"].append(len(static.frame_to_bitstream(dataset, bps)) / 1000)
            df["Parameter type"].append("Static")

        df = pd.DataFrame(df)
        s = sns.relplot(data=df, kind="line", x="spike_width", y="value", hue="Parameter type", height=2.5, aspect=3)
        s.set_xlabels("Spike width")
        s.set_ylabels("Bitstream size [kb]")
        s.tight_layout()

        self.save(filename)

    @staticmethod
    def _get_params_static(frame: np.ndarray) -> np.ndarray:
        return np.full(frame.shape, 1) << np.asarray([4 for _ in frame])

    @staticmethod
    def _get_params_variable(frame: np.ndarray) -> np.ndarray:
        return np.full(frame.shape, 1) << Ricer.frame_to_kparams(frame, 4)

    @staticmethod
    def _get_interleaved_signal(frame: np.ndarray) -> np.ndarray:
        return Ricer.frame_to_interleaved(frame)
