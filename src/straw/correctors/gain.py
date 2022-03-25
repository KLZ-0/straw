import pandas as pd


class GainCorrector:
    @staticmethod
    def equalize(frame, reference):
        pass

    def apply(self, df: pd.DataFrame):
        """
        Takes dataframe with 1-n channels
        TODO: deal with 1 channel
        :param df:
        :return:
        """

        from figures import show_frame
        show_frame(df[df["channel"].isin([1, 6])])
        exit()
        pass
