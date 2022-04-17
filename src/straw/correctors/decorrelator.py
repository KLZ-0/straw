import numpy as np
import pandas as pd


class Modifiers:
    #############################
    # Subtraction decorrelation #
    #############################
    @staticmethod
    def indicator(x):
        return np.mean(np.abs(x))

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
        if diff.any():
            if Modifiers.indicator(diff) < Modifiers.indicator(x1):
                x1[:] = diff
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

        x1[:] = x2 + x1

    ##########################
    # Mid-side decorrelation #
    ##########################

    @staticmethod
    def transform_midside(x1: np.array, x2: np.array):
        diff = x1 - x2
        mid = (x1 + x2) // 2
        x1[:] = diff
        x2[:] = mid

    @staticmethod
    def transform_midside_reverse(x1: np.array, x2: np.array):
        diff = x1.copy()
        mid = x2.copy()
        x1[:] = mid + np.ceil(diff / 2)
        x2[:] = mid - np.floor(diff // 2)


class Decorrelator:
    #############################
    # Subtraction decorrelation #
    #############################

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

    ##########################
    # Mid-side decorrelation #
    ##########################

    @staticmethod
    def midside_decorrelate(df: pd.DataFrame, col_name: str = "residual"):
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not in dataframe")

        for i in range(0, len(df.index), 2):
            Modifiers.transform_midside(df.loc[df.index[i], col_name], x2=df.loc[df.index[i + 1], col_name])

        return df

    @staticmethod
    def midside_decorrelate_revert(df: pd.DataFrame, col_name: str = "residual"):
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not in dataframe")

        for i in range(0, len(df.index), 2):
            Modifiers.transform_midside_reverse(df.loc[df.index[i], col_name], x2=df.loc[df.index[i + 1], col_name])

        return df
