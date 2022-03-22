import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# NOTE: Do not import this without checking for args -> seaborn and mpl should be optional dependencies
from figures.baseplot import BasePlot
from straw.lpc import steps


class LPCPlot(BasePlot):
    def print_lpc_and_qlp(self):
        """
        Prints LPC and QLP coefficients
        """
        frame = self._e.sample_frame()
        lpc = steps.compute_lpc(frame["frame"], 8)
        qlp, shift = steps.quantize_lpc(lpc, 12)

        df = pd.DataFrame({
            "LPC": lpc,
            "QLP": qlp
        })

        sys.stderr.flush()
        print("%%%%%%%% INSERT TABLE %%%%%%%%")
        print(df.T.to_latex(caption="Example coefficients for 8-order predictor",
                            position="H", float_format="{:0.3f}".format,
                            header=[f"$a_{i + 1}$" for i in range(len(lpc))], escape=False)
              , end="")
        print("%%%%%%%% INSERT TABLE %%%%%%%%")

    def prediction_comparison(self):
        """
        Shows the original frame and a predicted frame
        """
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

        self.save("prediction_comparison.pdf")

    def residual(self):
        """
        Shows a residual frame
        :return:
        """
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

        self.save("residual.pdf")

    def common_lpc_autoc_averaging(self):
        """
        Shows the residuals after using common LPC coefficients across all channels
        """
        data = self._e.sample_frame_multichannel()
        df = {"x": [], "value": [], "Channel": []}
        for i, ds in enumerate(data["residual"]):
            df["x"] += [i for i in range(len(ds))]
            df["Channel"] += [i for _ in ds]
            df["value"] += list(ds)
        df = pd.DataFrame(df)

        s = sns.relplot(data=df, kind="line", x="x", y="value", hue="Channel", height=2.5, aspect=3)

        plt.title("Frame residuals with common LPC coefficients (averaged autoc)")
        s.set_xlabels("Sample")
        s.set_ylabels("Sample value (16-bit)")
        s.tight_layout()

        self.save("lpc_averaged.png")

        # Also print variances...

        print("Variances:")
        table = pd.DataFrame({
            "channel": data["channel"],
            "variance": data["residual"].apply(np.var),
        })
        table = table.set_index("channel")

        print(table.T.to_latex(caption="Variances",
                               position="H", float_format="{:0.3f}".format, escape=False))
        # print("\n".join([f"{f:.3f}" for f in data["residual"].apply(np.var)]))

    def common_lpc_variances(self):
        """
        Show variances across all frames across for all channels
        """
        df = self._e.get_data()
        df["variance"] = df["residual"].apply(np.var)

        s = sns.relplot(data=df, kind="line", x="seq", y="variance", hue="channel", height=6, aspect=1.8)
        s.set(ylim=(0, 7500))

        plt.title("Residual variance with common LPC coefficients (averaged autoc)")
        s.set_xlabels("Frame")
        s.set_ylabels("Variance")
        s.tight_layout()

        self.save("lpc_averaged_variances.png")
