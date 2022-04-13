import numpy as np
import pandas as pd


class Modifiers:
    @staticmethod
    def localized_sub(x1: np.array, x2: np.array) -> bool:
        """
        Performs localized subtraction on array x1 with the reference array (x2)
        The process is as follows:
            1. A cutoff value is calculated from each given array equal to (x2.max() / (2^3))
            2. Every value below this treshold is saved as is
            3. For every value above this reshold the respective indices value for the reference array is subtracted
            4. The variance of this new array is compared to the original
            5. If the new variance is larger, then the original array is returned instead
        :param x1: array to be modified
        :param x2: reference array
        :return: True if x1 was modified, False otherwise
        """
        diff = x1 - x2
        # too low will cause noise to be decorrelated
        # too high will cause audio that could be decorrelated to not be decorrelated
        if diff.any():
            limits = x2.max() >> 3
            # return x1^x2
            nonzero = np.nonzero(np.abs(x2) > limits)[0]
            if diff[nonzero].var() < x1[nonzero].var():
                x1[nonzero] = diff[nonzero]
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def localized_add(x1: np.array, x2: np.array, was_coded: bool):
        """
        Reverse localized_sub
        :param x1: array to be modified
        :param x2: reference array
        :param was_coded: True if x1 was modified, False otherwise
        :return: None
        """
        if not was_coded:
            return

        limits = x2.max() >> 3
        nonzero = np.nonzero(np.abs(x2) > limits)[0]
        x1[nonzero] = (x2 + x1)[nonzero]


class Decorrelator:
    @staticmethod
    def localized_decorrelate_expander(df: pd.DataFrame, reference: np.ndarray, col_name: str):
        return Modifiers.localized_sub(df[col_name], x2=reference)

    def localized_decorrelate(self, df: pd.DataFrame, col_name: str = "residual"):
        """
        Performs localized decorrelation
        The reference channel is channel 0
        :param df: dataframe slice for one multichannel group
        :param col_name: the name of the column which should be decorrelated
        :return: df with new added columns
        """
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not in dataframe")

        df["was_coded"] = df.apply(self.localized_decorrelate_expander,
                                   reference=df[col_name][df.index[0]],
                                   col_name=col_name,
                                   axis=1)

        return df

    @staticmethod
    def localized_decorrelate_revert_expander(df: pd.DataFrame, reference: np.ndarray, col_name: str):
        Modifiers.localized_add(df[col_name], x2=reference, was_coded=df["was_coded"])

    def localized_decorrelate_revert(self, df: pd.DataFrame, col_name: str = "residual"):
        """
        Reverts localized decorrelation
        The reference channel is channel 0
        For the process description see Decorrelator.localized_decorrelate
        :param df: dataframe slice for one multichannel group
        :param col_name: the name of the column which was decorrelated
        :return: df with new added columns
        """
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not in dataframe")

        df.apply(self.localized_decorrelate_revert_expander,
                 reference=df[col_name][df.index[0]],
                 col_name=col_name,
                 axis=1)

        return df
