from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def show_frame(data: pd.DataFrame,
               file_name="tmp.png",
               file_dir="outputs",
               show=True,
               limit=None,
               col_name="frame",
               terminate=True):
    import seaborn as sns
    fig_dir = Path(file_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = {"x": [], "value": [], "Channel": []}
    for i, row in data.iterrows():
        ds = row[col_name]
        if limit:
            ds = ds[:limit]
        df["x"] += [u for u in range(len(ds))]
        df["Channel"] += [row["channel"] for _ in ds]
        df["value"] += list(ds)
    df = pd.DataFrame(df)

    s = sns.relplot(data=df, kind="line", x="x", y="value", hue="Channel", height=2.5, aspect=3)

    plt.title("Frame")
    s.set_xlabels("Sample")
    s.set_ylabels("Sample value (16-bit)")
    s.tight_layout()

    plt.savefig(fig_dir / file_name)
    if show:
        plt.show()

    if terminate:
        exit(0)
