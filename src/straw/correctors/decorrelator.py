import numpy as np


class Modifiers:
    @staticmethod
    def localized_sub(x1, x2):
        oldx1 = x1.copy()
        diff = x1 - x2
        # too low will cause noise to be decorrelated
        # too high will cause audio that could be decorrelated to not be decorrelated
        if diff.any():
            limits = x2.max() >> 3
            # return x1^x2
            nonzero = np.nonzero(np.abs(x2) > limits)[0]
            x1[nonzero] = diff[nonzero]
            if np.var(x1) > np.var(oldx1):
                return oldx1, False
            else:
                return x1, True
        else:
            return x1, False

    @staticmethod
    def localized_add(x1, x2, was_coded):
        if not was_coded:
            return x1

        limits = x2.max() >> 3
        nonzero = np.nonzero(np.abs(x2) > limits)[0]
        x1[nonzero] = (x2 + x1)[nonzero]
        return x1


class Decorrelator:
    @staticmethod
    def localized_decorrelate_expander(df, reference: np.ndarray, col_name: str):
        return Modifiers.localized_sub(df[col_name], x2=reference)

    def localized_decorrelate(self, df, col_name: str = "residual"):
        """
        Finds the medium channels residual and subtracts it from all other channels
        :return: new residuals
        """
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not in dataframe")

        df[[col_name, "was_coded"]] = df.apply(self.localized_decorrelate_expander,
                                               reference=df[col_name][df.index[0]],
                                               col_name=col_name,
                                               axis=1,
                                               result_type="expand")

        return df

    @staticmethod
    def localized_decorrelate_revert_expander(df, reference: np.ndarray, col_name: str):
        return Modifiers.localized_add(df[col_name], x2=reference, was_coded=df["was_coded"])

    def localized_decorrelate_revert(self, df, col_name: str = "residual"):
        """
        Finds the medium channels residual and subtracts it from all other channels
        :return: new residuals
        """
        if col_name not in df.columns:
            raise ValueError(f"Column '{col_name}' not in dataframe")

        df[col_name] = df.apply(self.localized_decorrelate_revert_expander,
                                reference=df[col_name][df.index[0]],
                                col_name=col_name,
                                axis=1)

        return df
