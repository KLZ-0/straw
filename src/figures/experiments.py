from pathlib import Path

import straw


class Experiments:
    # TODO: make base class
    def __init__(self, e: straw.Encoder, args=None):
        self._args = args
        self._e = e
        self.fig_dir = Path(getattr(self._args, "fig_dir", "outputs"))
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.fig_show = getattr(self._args, "fig_show", False)
