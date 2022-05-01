import tempfile
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from figures.base import BasePlot
from straw.static import Default
from straw.util import Signals


class FrameBlockingPlot(BasePlot):
    def frame_limits(self, filename):
        frame = self._e.samplebuffer_frame_multichannel(seq=4)
        limits = Signals.get_frame_limits_by_energy(frame[0], min_block_size=1 << 10)

        df = pd.DataFrame({
            "sample": [i for i in range(frame.shape[1])],
            "value": frame[0]
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", height=2.5, aspect=3)

        for limit_x in limits:
            plt.axvline(x=limit_x, ymin=0, ymax=1, color="red")

        s.set_xlabels("Sample")
        s.set_ylabels("Sample value (16-bit)")
        s.tight_layout()

        self.save(filename)
        print(limits)

    def frame_energy(self, filename):
        frame = self._e.samplebuffer_frame_multichannel(seq=4)

        resolution = 10
        limits = Signals.get_frame_limits_by_energy(frame[0], min_block_size=1 << 10, resolution=resolution)
        energy = Signals.get_energies(frame[0], resolution=resolution)
        treshold = energy[-1]
        energy = energy[:-1]

        df = pd.DataFrame({
            "sample": [i for i in range(energy.shape[0])],
            "value": energy
        })

        s = sns.relplot(data=df, kind="line", x="sample", y="value", height=2.5, aspect=3)

        for limit_x in limits:
            plt.axvline(x=limit_x / resolution, ymin=0, ymax=1, color="red")
        plt.axhline(y=treshold, xmin=0, xmax=1, color="green")

        s.set_xlabels("Sample group (precision = 10)")
        s.set_ylabels("Energy")
        s.tight_layout()

        self.save(filename)

    frame_sizes = [
        (1 << 12, 1 << 12),
        (1 << 10, 1 << 12),
        (1 << 11, 1 << 12),
        (1 << 11, 1 << 13),
        (1 << 11, 1 << 14),
    ]

    def get_stats_for_treshold(self, treshold=Default.framing_treshold):
        self._e.set_framing_treshold(treshold)
        for sizes in self.frame_sizes:
            self._e.set_blocksizes(*sizes)
            self._e.load_file(self._args.input_files[0])
            self._e.encode()
            tmpfile = tempfile.NamedTemporaryFile(delete=True)
            with open(tmpfile.name, "w+b") as f:
                self._e.save_file(f)
                yield sizes, self._e.get_stats(Path(f.name))

    def print_sizes(self):
        for sizes, stats in self.get_stats_for_treshold():
            print(
                f"{sizes[0]} & {sizes[1]} & {stats.frames} & {stats.file_size / 2 ** 20:.2f}\\,MiB & {stats.ratio * 100:.2f}\\,\\% \\\\")

    def plot_tresholds(self, filename):
        run = []
        size = []
        treshold = []
        tress = [1000, 2500, 5000, 10000, 20000, 40000, 60000, 80000]
        for it, tres in enumerate(tress):
            for i, (sizes, stats) in enumerate(self.get_stats_for_treshold(tres)):
                print(f"Processing {it * len(self.frame_sizes) + i + 1} / {len(tress) * len(self.frame_sizes)}")
                run.append(f"MinFS={sizes[0]}, MaxFS={sizes[1]}")
                size.append(stats.file_size / 2 ** 20)
                treshold.append(tres)

        df = pd.DataFrame({
            "Run": run,
            "ratio": size,
            "treshold": treshold
        })

        s = sns.relplot(data=df, kind="line", x="treshold", y="ratio", hue="Run", height=2.5, aspect=3)

        s.set_xlabels("Energy treshold")
        s.set_ylabels("File size [MiB]")
        s.tight_layout()

        self.save(filename)
        print(size)
