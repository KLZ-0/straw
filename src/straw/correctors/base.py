import numpy as np
import pandas as pd


class BaseCorrector:
    def apply(self, df: pd.DataFrame, col_name: str = "frame") -> pd.DataFrame:
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not in dataframe")

        return df

    def global_apply(self, samplebuffer: np.ndarray) -> (np.ndarray, np.ndarray):
        return samplebuffer, np.zeros(samplebuffer.shape[0])
