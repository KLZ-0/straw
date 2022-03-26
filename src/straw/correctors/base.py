import pandas as pd


class BaseCorrector:
    def apply(self, df: pd.DataFrame, col_name: str = "frame"):
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not in dataframe")

        return df
