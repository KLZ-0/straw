from pathlib import Path

from matplotlib import pyplot as plt

import straw


class BasePlot:
    def __init__(self, e: straw.Encoder, args=None):
        self._args = args
        self._e = e
        self.fig_dir = Path(getattr(self._args, "fig_dir", "outputs"))
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.fig_show = getattr(self._args, "fig_show", False)

    def save(self, file_name: str):
        plt.savefig(self.fig_dir / file_name)

        if self.fig_show:
            plt.show()
