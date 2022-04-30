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

        self.save(filename)
        print(limits)
