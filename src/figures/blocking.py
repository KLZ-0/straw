import tempfile
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from figures.base import BasePlot
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

    def frame_sizes(self, output_file):
        file_sizes = [
            (1 << 12, 1 << 12),
            (1 << 10, 1 << 12),
            (1 << 11, 1 << 12),
            (1 << 11, 1 << 13),
            (1 << 11, 1 << 14),
        ]

        for sizes in file_sizes:
            self._e.set_blocksizes(*sizes)
            self._e.load_file(self._args.input_files[0])
            self._e.encode()
            tmpfile = tempfile.NamedTemporaryFile(delete=True)
            with open(tmpfile.name, "w+b") as f:
                self._e.save_file(f)
                file_size, ratio, frames = self._e.get_stats(Path(f.name))
                print(
                    f"{sizes[0]} & {sizes[1]} & {frames} & {file_size / 2 ** 20:.2f}\\,MiB & {ratio * 100:.2f}\\,\\% \\\\")
