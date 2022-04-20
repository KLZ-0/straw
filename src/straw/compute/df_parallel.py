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
        with Pool(self.cpus) as pool:
            data = pool.map(func, np.array_split(data, self.cpus))
        return pd.concat(data)

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
        chunksize = int(len(data) / self.cpus) * 2
        with Pool(self.cpus + 1) as p:
            ret_list = p.map(func, [group for name, group in data], chunksize=chunksize)
        if not ret_list:
            return None
        elif isinstance(ret_list[0], (pd.Series, pd.DataFrame)):
            return pd.concat(ret_list)
        else:
            return ret_list

    def _group_run_on_subset(self, func, data_subset):
        return func(data_subset, *self._apply_args, **self._apply_kwargs)

    def map_group(self, data, func, args=(), **kwargs) -> pd.DataFrame:
        self._apply_args = args
        self._apply_kwargs = kwargs
        return self._group_parallelize(data, partial(self._group_run_on_subset, func))

    def _ndarray_parallelize(self, data, func):
        with Pool(self.cpus) as p:
            ret_list = p.map(func, data)
        return ret_list

    def map_ndarray(self, data, func, args=(), **kwargs):
        self._apply_args = args
        self._apply_kwargs = kwargs
        return self._ndarray_parallelize(data, partial(self._group_run_on_subset, func))
