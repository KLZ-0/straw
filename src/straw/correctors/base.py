import numpy as np
import pandas as pd

from straw.io.params import StreamParams


class BaseCorrector:
    def apply(self, df: pd.DataFrame, col_name: str = "frame") -> pd.DataFrame:
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not in dataframe")

        return df

    def global_apply(self, samplebuffer: np.ndarray, params: StreamParams) -> (np.ndarray, np.ndarray):
        return
