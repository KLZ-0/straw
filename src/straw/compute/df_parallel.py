from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

"""
based on https://stackoverflow.com/a/53135031
and https://stackoverflow.com/a/27027632
"""


class ParallelCompute:
    __instance = None

    cpus: int
    _apply_args: tuple
    _apply_kwargs: dict

    @staticmethod
    def get_instance():
        """ Static access method. """
        if ParallelCompute.__instance is None:
            ParallelCompute()
        return ParallelCompute.__instance

    def __init__(self, cpus=cpu_count()):
        """
        Initialize the compute class
        :param cpus: number of CPUs to use when parallel is True, None means use all
        """
        if ParallelCompute.__instance is not None:
            raise RuntimeError("This class is a singleton!")
        else:
            ParallelCompute.__instance = self
        self.cpus = cpus

    def _parallelize(self, data, func):
        data_split = np.array_split(data, self.cpus)
        pool = Pool(self.cpus)
        data = pd.concat(pool.map(func, data_split))
        pool.close()
        pool.join()
        return data

    def _run_on_subset(self, func, data_subset):
        return data_subset.apply(func, args=self._apply_args, **self._apply_kwargs)

    def map(self, data, func, args=(), **kwargs) -> pd.Series:
        """
        Apply the functiom to the given DataFrame or Series in parallel
        :param data: DataFrame or Series to which func will be applied
        :param func: function to apply
        :param args: args to use in apply
        :param kwargs: kwargs to use in apply
        :return: DataFrame or Series with applied data
        """
        self._apply_args = args
        self._apply_kwargs = kwargs
        return self._parallelize(data, partial(self._run_on_subset, func))

    def _group_parallelize(self, data, func):
        with Pool(self.cpus) as p:
            ret_list = p.map(func, [group for name, group in data])
        return pd.concat(ret_list)

    def _group_run_on_subset(self, func, data_subset):
        return func(data_subset, *self._apply_args, **self._apply_kwargs)

    def map_group(self, data, func, args=(), **kwargs) -> pd.DataFrame:
        self._apply_args = args
        self._apply_kwargs = kwargs
        return self._group_parallelize(data, partial(self._group_run_on_subset, func))
